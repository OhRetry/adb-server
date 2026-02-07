from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from config import AppConfig
    from concurrent.futures.thread import ThreadPoolExecutor
    from emulator import Emulator
    import asyncio

app_config: Optional["AppConfig"] = None
general_executor: Optional["ThreadPoolExecutor"] = None
ocr_executor: Optional["ThreadPoolExecutor"] = None
device_locks: Dict[str, "asyncio.Lock"] = {}
device_emulator: Dict[str, "Emulator"] = {}
