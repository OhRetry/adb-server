import asyncio
import subprocess
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from emulator import Emulator, _get_or_create_device_lock, _get_device_emulator
from ocr import create_paddle_ocr_executors, get_threadlocal_paddle_ocr_engine, PaddleOCREngine
import globals
import yaml

logger = logging.getLogger(f"{__name__}")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # --- Load Config ---
    config_path = Path(__file__).resolve().parent / "config.json"
    globals.app_config = load_config(config_path)
    app_config = globals.app_config

    # --- Initialize Worker Thread ---
    globals.general_executor = ThreadPoolExecutor(max_workers=app_config.general_worker_thread_count)
    globals.ocr_executor = create_paddle_ocr_executors(
        thread_count=app_config.ocr_worker_thread_count,
        possible_languages=app_config.ocr_possible_languages
    )

    with open("logging.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("disable_existing_loggers", False)
    logging.config.dictConfig(cfg)

    yield

    globals.general_executor.shutdown()
    globals.ocr_executor.shutdown()

app = FastAPI(title="ADB Screenshot OCR Server (blocking pipeline -> PaddleOCR)", lifespan=lifespan)

class OCRRequest(BaseModel):
    instance: str = Field(..., description="ADB device id, e.g. 127.0.0.1:5555 or emulator-5554")
    roi: Optional[List[int]] = Field(None, description="Optional ROI(crop image): [x1,y1,x2,y2]")
    lang: Optional[str] = Field("en", description="language. e.g. 'korean', 'en'. default 'en'")

class KeyPressRequest(BaseModel):
    instance: str = Field(..., description="ADB device id, e.g. 127.0.0.1:5555 or emulator-5554")
    key_code: str = Field(..., description="key code to press. e.g. KEYCODE_K")

# ---------- Full blocking pipeline (runs inside ThreadPoolExecutor) ----------
def adb_screencapture_ocr_blocking_pipeline(
        emulator: Emulator,
        lang: str,
        roi: Optional[List[int]]
) -> str:

    """
    Full blocking pipeline executed inside an executor thread:
      1) adb screencap (blocking)
      2) decode PNG -> OpenCV
      3) optional crop
      4) run PaddleOCR (thread-local instance)
    Returns extracted text (possibly empty string).
    Raises RuntimeError on failures.
    """
    # 1) adb screencap
    png_bytes = emulator.screencapture_blocking()

    ocr = get_threadlocal_paddle_ocr_engine(lang=lang)

    # 2) deconde png -> crop -> ocr
    return ocr.ocr_with_roi_from_png(
        png_bytes,
        roi
    )

# ---------- Async endpoint that calls pipeline in executor ----------
@app.post("/ocr")
async def ocr_endpoint(req: OCRRequest):
    device_id = req.instance.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="Missing device instance id")

    # per-device lock to serialize adb calls for same instance
    lock = _get_or_create_device_lock(device_id)

    logger.info("OCR request: device=%s roi=%s lang=%s",
                device_id, req.roi, req.lang)

    # Acquire lock, then run the whole blocking pipeline inside executor
    async with lock:
        loop = asyncio.get_event_loop()
        emulator = _get_device_emulator(device_id)
        try:
            text = await loop.run_in_executor(
                globals.ocr_executor,
                adb_screencapture_ocr_blocking_pipeline,
                emulator,
                req.lang,
                req.roi
            )
        except RuntimeError as e:
            logger.exception("Pipeline RuntimeError for device %s: %s", device_id, e)
            raise HTTPException(status_code=502, detail=str(e))
        except subprocess.TimeoutExpired:
            logger.exception("ADB subprocess timed out for device %s", device_id)
            raise HTTPException(status_code=504, detail="ADB screencap timed out")
        except Exception as e:
            logger.exception("Unexpected error in pipeline for device %s: %s", device_id, e)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    # Return result
    return text

# --- send key event by adb. 안드로이드 14 이전에서는 접근성을 이용한 키이벤트 전송이 불가능하기때문에 adb이용이 필수
@app.post("/send_key_event")
async def send_key_event_endpoint(req: KeyPressRequest):
    device_id = req.instance.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="Missing device instance id")

    # per-device lock to serialize adb calls for same instance
    lock = _get_or_create_device_lock(device_id)

    # Acquire lock, then run the whole blocking pipeline inside executor
    async with lock:
        emulator = _get_device_emulator(device_id)
        emulator.send_key_event(req.key_code)

@app.get("/health")
async def health():
    return "OK"