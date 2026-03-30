from __future__ import annotations

import logging
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Avoid importing inference modules here to prevent heavy side effects.
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
from utils.apptools import ensure_dir, safe_name, sha256_bytes
from utils.colpali_retriever import retrieve_topk_pages


@dataclass(frozen=True)
class ColPaliHit:
    source_pdf: str
    rank_in_doc: int
    page_num: int
    score: float
    image_path: str


def _fingerprint_path(path: Path) -> str:
    try:
        stat = path.stat()
        token = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}".encode("utf-8")
    except Exception:
        token = str(path).encode("utf-8")
    return sha256_bytes(token)


def collect_colpali_hits(
    file_paths: list[str],
    query: str,
    work_root: Path,
    per_doc_top_k: int = 3,
    global_top_k: int = 3,
    model_name: str = "vidore/colpali-v1.2",
    reuse_index: bool = True,
    strict_reuse: bool = True,
    render_dpi: int = 140,
    logger: Optional[logging.Logger] = None,
) -> list[dict]:
    logger = logger or logging.getLogger("ColPaliRAG")
    work_root = ensure_dir(work_root)
    all_hits: list[dict] = []
    start_all = perf_counter()

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logger.warning("ColPali: file not found: %s", file_path)
            continue

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_PDF_EXTS:
            doc_start = perf_counter()
            fingerprint = _fingerprint_path(path)
            work_dir = ensure_dir(work_root / f"{safe_name(path.stem)}_{fingerprint[:8]}")
            logger.info(
                "ColPali: retrieving from %s (top_k=%s) -> %s",
                path.name,
                per_doc_top_k,
                work_dir,
            )
            hits = retrieve_topk_pages(
                pdf_path=str(path),
                query=query,
                top_k=max(1, per_doc_top_k),
                work_dir=work_dir,
                model_name=model_name,
                reuse_index=reuse_index,
                strict_reuse=strict_reuse,
                render_dpi=render_dpi,
            )
            logger.info("ColPali: retrieval done for %s in %.3fs", path.name, perf_counter() - doc_start)
            for h in hits:
                all_hits.append(
                    {
                        "source_pdf": path.name,
                        "rank_in_doc": h.rank,
                        "page_num": h.page_num,
                        "score": h.score,
                        "image_path": h.image_path,
                    }
                )
        elif suffix in SUPPORTED_IMAGE_EXTS:
            logger.info("ColPali: using image file directly: %s", path.name)
            all_hits.append(
                {
                    "source_pdf": path.name,
                    "rank_in_doc": 1,
                    "page_num": 1,
                    "score": 1.0,
                    "image_path": str(path),
                }
            )
        else:
            logger.warning("ColPali: unsupported file type skipped: %s", file_path)

    all_hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if global_top_k and global_top_k > 0:
        all_hits = all_hits[:global_top_k]
    logger.info("ColPali: total retrieval time %.3fs", perf_counter() - start_all)
    return all_hits
