from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator
import json

class AppConfig(BaseModel):
    adb_path: str = Field(default="adb")
    general_worker_thread_count: int = Field(default=2, ge=1)
    ocr_worker_thread_count: int = Field(default=2, ge=1)
    ocr_possible_languages: List[str] = Field(default_factory=lambda: ["en"])

def load_config(path: Path) -> AppConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_map = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON format in configuration file: {e}")

    return AppConfig(**config_map)
