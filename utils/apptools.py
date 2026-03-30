from __future__ import annotations

import hashlib
import re
import time
import traceback
from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", s).strip("_")
    return s or "file"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log_debug(msg: str, logs: list[str], run_dir: Optional[Path] = None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    logs.append(line)

    if run_dir is not None:
        try:
            with open(run_dir / "debug.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_exception(e: Exception, logs: list[str], run_dir: Optional[Path] = None) -> None:
    log_debug(f"Exception type: {type(e).__name__}", logs, run_dir)
    log_debug(f"Exception message: {str(e)}", logs, run_dir)
    tb = traceback.format_exc()
    for line in tb.splitlines():
        log_debug(line, logs, run_dir)