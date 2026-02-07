import logging
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Tuple, Optional, Set
import numpy as np
from paddleocr import PaddleOCR

from emulator import Emulator
from image import _png_bytes_to_cv2_image, _crop_image

# thread-local storage to keep PaddleOCR instances per thread
_thread_local = threading.local()

logger = logging.getLogger(f"{__name__}")

class PaddleOCREngine:
    def __init__(self, lang: str):
        tid = threading.get_ident()
        self.logger = logging.getLogger(f"{__name__}.ocr.{tid}")
        self.lang = self._normalize_lang(lang)
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            return_word_box=False
        )

    def _normalize_lang(self, req_lang: str) -> str:
        """
        혹시 모를 안전책. 다양한 언어의 풀네임, 약칭을 paddle ocr에서 사용하는 lang 이름으로 변환
        """
        if not req_lang:
            return "en"
        s = req_lang.strip().lower()
        if s in ("korean", "ko", "kor", "kr"):
            return "korean"
        if s in ("english", "en", "eng"):
            return "en"
        # fallback to the given string (PaddleOCR supports 'ch','japan', etc.)
        return s

    def _parse_paddle_result(self, res) -> List[Tuple[str, Optional[float]]]:
        """
        다양한 형태의 paddle ocr 출력을 해석한다
        """
        lines: List[Tuple[str, Optional[float]]] = []
        if not res:
            return lines

        # If it's a list-like outer and first item has attribute 'json', try that format
        try:
            if isinstance(res, list) and len(res) > 0 and hasattr(res[0], "json"):
                # e.g. paddlex result object: res[0].json['res'] has rec_texts/rec_scores
                obj = res[0]
                j = getattr(obj, "json", None)
                if j and "res" in j and isinstance(j["res"], dict):
                    rec_texts = j["res"].get("rec_texts", [])
                    rec_scores = j["res"].get("rec_scores", [])
                    for i, t in enumerate(rec_texts):
                        sc = float(rec_scores[i]) if i < len(rec_scores) else None
                        lines.append((str(t), sc))
                    return lines
        except Exception:
            pass

        # If it's nested like [ [bbox, (text,score)], ... ] or [ [ ... ], ... ]
        items = None
        if isinstance(res, list) and len(res) > 0:
            # if it's [ [bbox, (text,score)], ... ] directly
            tentative = res
            # sometimes the structure is [ [ ... ], ... ] with an extra wrapper
            if isinstance(res[0], list) and len(res) > 0 and isinstance(res[0][0], list):
                tentative = res[0]
            items = tentative

        if items is not None:
            for item in items:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt_conf = item[1]
                        if isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 1:
                            text = str(txt_conf[0])
                            conf = float(txt_conf[1]) if len(txt_conf) > 1 else None
                        else:
                            text = str(txt_conf)
                            conf = None
                        lines.append((text, conf))
                        continue
                    # fallback stringify
                    lines.append((str(item), None))
                except Exception:
                    continue

        # final fallback: stringify whole result
        if not lines:
            try:
                lines.append((str(res), None))
            except Exception:
                pass

        return lines

    def _run_paddleocr_on_cv2(
            self,
            img: np.ndarray,
    ) -> str:
        """
        cv2 이미지에서 paddle ocr을 수행한다. 결과는 공백으로 join된다
        """

        try:
            res = self.ocr.predict(img)
        except Exception as e:
            self.logger.exception("PaddleOCR inference failed: %s", e)
            raise RuntimeError(f"PaddleOCR inference failed: {e}")

        lines = self._parse_paddle_result(res)
        joined = " ".join([t for t, _ in lines])
        return joined

    def ocr_with_roi_from_png(
            self,
            png_bytes: bytes,
            roi: Optional[List[int]]
    ) -> str:
        # 1) decode
        img = _png_bytes_to_cv2_image(png_bytes)

        # 2) crop if requested
        if roi:
            img = _crop_image(img, roi)

        # 3) run PaddleOCR (cv2 image BGR)
        text = self._run_paddleocr_on_cv2(img=img)
        return text

def get_threadlocal_paddle_ocr_engine(lang: str) -> PaddleOCREngine:
    """
    thread local에 저장된 PaddleOCREngine 객체를 생성, 반환.
    lang을 key로 저장하고 있으며 lang마다 ocr객체를 새로 만든다.
    """
    if not hasattr(_thread_local, "ocr_map"):
        _thread_local.ocr_map = {}

    key = (lang,)
    if key in _thread_local.ocr_map:
        return _thread_local.ocr_map[key]

    # Create new PaddleOCR instance for this thread/lang
    logger.info("Creating PaddleOCREngine instance for thread %s, lang=%s", threading.get_ident(), lang)
    ocr = PaddleOCREngine(lang=lang)
    _thread_local.ocr_map[key] = ocr
    return ocr

def create_paddle_ocr_executors(thread_count: int, possible_languages: List[str]) -> ThreadPoolExecutor:
    """
    모든 ocr 작업 스레드에 대해 ocr객체를 미리 초기화해둔다. 요청 처리에 모델 초기화 딜레이가 없도록 하기 위해.
    """
    logger.info(
        "Initializing %d Paddle OCR executors. Languages: %s. Total OCR engines: %d x %d = %d",
        thread_count,
        ", ".join(possible_languages),
        thread_count,
        len(possible_languages),
        thread_count * len(possible_languages)
    )
    _initialized_threads: Set[int] = set()
    _initialize_lock = threading.Lock()

    ocr_executor = ThreadPoolExecutor(thread_count)

    def _initialize_threadlocal_paddle_ocr_engine():
        tid = threading.get_ident()
        # 현재 스레드에서 모든 언어 모델에 대한 ocr engine을 가져온다
        for lang in possible_languages:
            get_threadlocal_paddle_ocr_engine(lang)
        with _initialize_lock:
            _initialized_threads.add(tid)

    # warm up all threads
    while True:
        with _initialize_lock:
            # 초기화된 스레드 개수가 전체 스레드 개수라면 초기화 완료
            if len(_initialized_threads) >= thread_count:
                break
        # 스레드에서 사용하는 모든 lang에 대해 초기화
        ocr_executor.submit(_initialize_threadlocal_paddle_ocr_engine)
        time.sleep(0.05)

    logger.info("Completed Initialization of all ocr executor threads")
    return ocr_executor