import asyncio
import subprocess
import logging
import threading
import time

import globals

class Emulator:
    def __init__(self, instance_id, ADB_PATH):
        self.logger = logging.getLogger(f"{__name__}.{instance_id}")
        self.ADB_PATH = ADB_PATH  # PATH에 adb 없으면 전체 경로로 바꿀 것, 예: r"C:\LDPlayer\LDPlayer9\adb.exe"
        self.serial = instance_id
        self.adb_proc = self._start_persistent_shell()

    # Start persistent adb shell (one process per emulator for fast commands)
    def _start_persistent_shell(self):
        """
        adb persistent shell을 생성.
        adb에 연결이 성공했는지는 보장하지 않는다.
        생성 직후에 stdin에 명령을 내리고 직후에 프로세스가 종료된다면 예외 없이 넘어가게 된다.
        """
        cmd = [self.ADB_PATH]
        cmd += ["-s", self.serial, "shell"]
        self.logger.debug("ADB connect command => %s", cmd)
        self.logger.info("Connecting to ADB Shell (%s)", self.serial)
        # 텍스트 모드로 stdin 사용
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
        return p

    def is_adb_alive(self):
        if self.adb_proc is None or self.adb_proc.poll() is not None:
            return False
        return True

    def reconnect_adb(self):
        self.logger.info("Trying to reconnect ADB shell")
        self._dispose_adb()
        self.adb_proc = self._start_persistent_shell()

    def adb_send(self, cmd: str):
        try:
            self.logger.debug("ADB Shell => %s", cmd)
            self.adb_proc.stdin.write(cmd + "\n")
            self.adb_proc.stdin.flush()
        except Exception as e:
            self.logger.exception("Failed to send ADB command")
            # 연결이 죽은 경우 재연결
            # 서버에 연결 후에 디바이스와 adb가 다시 연결되면 기존 adb shell 연결이 끊어진다
            if not self.is_adb_alive():
                self.reconnect_adb()
            raise e
            # try:
            #     self.logger.debug("ADB Shell => %s", cmd)
            #     self.adb_proc.stdin.write(cmd + "\n")
            #     self.adb_proc.stdin.flush()
            # except Exception as e:
            #     self.logger.exception("Failed to resend ADB command")
            #     raise e

    def screencapture_blocking(self) -> bytes:
        """
        Blocking call to run adb exec-out screencap -p and return PNG bytes.
        Raises RuntimeError on failure.
        """
        self.logger.debug("screen capture start")
        cmd = [self.ADB_PATH, "-s", self.serial, "exec-out", "screencap", "-p"]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ADB screencap timed out")
        except FileNotFoundError:
            raise RuntimeError("adb not found on PATH")
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore") if proc.stderr else "<no stderr>"
            raise RuntimeError(f"ADB screencap failed: {err}")
        if not proc.stdout:
            raise RuntimeError("ADB screencap returned empty data")
        return proc.stdout

    def send_key_event(self, key_code: str):
        self.logger.debug("Sending key press event (%s)", key_code)
        cmd = ["input", "keyevent", key_code]
        self.adb_send(" ".join(cmd))

    def dispose(self):
        self._dispose_adb()

    def _dispose_adb(self):
        # None이면 무시
        if self.adb_proc is None:
            return
        # 별도의 스레드에서 기존 adb 프로세스 정리
        self.logger.info("Killing previous ADB subprocess")
        t = threading.Thread(
            target=self._clean_up_adb_proc,
            args=(self.adb_proc, self.logger),
            daemon=True,
            name=f"adb-cleanup-{self.serial}"
        )
        t.start()
        self.adb_proc = None

    @staticmethod
    def _clean_up_adb_proc(proc: subprocess.Popen, logger):
        if proc is None:
            return

        try:
            proc.kill()
        except Exception as e:
            logger.exception("Failed to kill subprocess")

        try:
            proc.wait(timeout=20)
        except Exception as e:
            logger.exception("Failed to wait for subprocess")

        for f in (proc.stdin, ):
            try:
                if f:
                    f.close()
            except Exception as e:
                pass


def _get_or_create_device_lock(device_id: str) -> asyncio.Lock:
    lock = globals.device_locks.get(device_id)
    if lock is None:
        lock = asyncio.Lock()
        globals.device_locks[device_id] = lock
    return lock

# 접근 전 동기화 필수
def _get_device_emulator(device_id: str) -> Emulator:
    emulator = globals.device_emulator.get(device_id)
    if emulator is None:
        emulator = Emulator(device_id, globals.app_config.adb_path)
        globals.device_emulator[device_id] = emulator
    return emulator
