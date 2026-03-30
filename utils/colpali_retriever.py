from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import logging
from time import perf_counter

from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path

from utils.apptools import ensure_dir


@dataclass
class RetrievalHit:
    rank: int
    page_num: int
    score: float
    image_path: str


def normalize_results(results: Any) -> List[Dict[str, Any]]:
    if results is None:
        return []
    if not isinstance(results, list):
        results = [results]

    out: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            out.append(r)
            continue

        if hasattr(r, "to_dict") and callable(getattr(r, "to_dict")):
            out.append(r.to_dict())
            continue

        d: Dict[str, Any] = {}
        for k in ["doc_id", "page_num", "score", "metadata", "base64"]:
            if hasattr(r, k):
                v = getattr(r, k)
                try:
                    if hasattr(v, "item"):
                        v = v.item()
                except Exception:
                    pass
                d[k] = v

        if not d and hasattr(r, "__dict__"):
            d = dict(r.__dict__)

        out.append(d)

    return out


def get_index_name(pdf_path: str) -> str:
    pdf_name = Path(pdf_path).stem
    return f"{pdf_name}_colpali"


def get_default_index_dir(index_name: str) -> Path:
    return Path(".byaldi") / index_name


def index_exists(index_name: str) -> bool:
    idx_dir = get_default_index_dir(index_name)
    return idx_dir.exists() and any(idx_dir.iterdir())


def load_existing_index(index_name: str, model_name: str):
    """
    Load an existing index with compatibility across different byaldi versions.
    """
    logger = logging.getLogger("ColPaliRetriever")
    logger.info("ColPali: loading existing index: %s", index_name)
    try:
        return RAGMultiModalModel.from_index(index_name)
    except TypeError:
        try:
            return RAGMultiModalModel.from_index(index_name, model_name=model_name)
        except TypeError:
            idx_dir = get_default_index_dir(index_name)
            try:
                return RAGMultiModalModel.from_index(str(idx_dir))
            except TypeError:
                return RAGMultiModalModel.from_index(str(idx_dir), model_name=model_name)


def build_new_index(pdf_path: str, index_name: str, model_name: str):
    logger = logging.getLogger("ColPaliRetriever")
    logger.info("ColPali: building new index: %s", index_name)
    start = perf_counter()
    rag = RAGMultiModalModel.from_pretrained(model_name, verbose=1)
    rag.index(
        input_path=pdf_path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True,
    )
    logger.info("ColPali: index built in %.3fs: %s", perf_counter() - start, index_name)
    return rag


def load_or_build_retriever(
    pdf_path: str,
    model_name: str,
    reuse_index: bool = True,
    strict_reuse: bool = True,
):
    index_name = get_index_name(pdf_path)
    logger = logging.getLogger("ColPaliRetriever")

    if reuse_index and index_exists(index_name):
        try:
            rag = load_existing_index(index_name, model_name)
            logger.info("ColPali: reuse index ok: %s", index_name)
            return rag, index_name, True
        except Exception as exc:
            if strict_reuse:
                logger.exception(
                    "ColPali: strict reuse enabled; failed to load existing index: %s",
                    index_name,
                )
                raise RuntimeError(f"ColPali index load failed (strict reuse): {index_name}") from exc
            idx_dir = get_default_index_dir(index_name)
            if idx_dir.exists():
                import shutil
                shutil.rmtree(idx_dir, ignore_errors=True)

    rag = build_new_index(pdf_path, index_name, model_name)
    return rag, index_name, False


def render_hit_pages(
    pdf_path: str,
    results: List[Dict[str, Any]],
    out_dir: Path,
    dpi: int = 140,
) -> List[RetrievalHit]:
    logger = logging.getLogger("ColPaliRetriever")
    img_dir = ensure_dir(out_dir / "pages")
    hits: List[RetrievalHit] = []

    for rank, hit in enumerate(results, start=1):
        if "page_num" not in hit:
            continue

        page_num = int(hit["page_num"])
        score = float(hit.get("score", 0.0))

        logger.info(
            "ColPali: rendering page %s (rank=%s, score=%.4f, dpi=%s)",
            page_num,
            rank,
            score,
            dpi,
        )
        render_start = perf_counter()
        imgs = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
        )
        logger.info(
            "ColPali: rendered page %s in %.3fs",
            page_num,
            perf_counter() - render_start,
        )
        if not imgs:
            continue

        out = img_dir / f"rank{rank:02d}_page{page_num:04d}_score{score:.4f}.png"
        imgs[0].save(out)

        hits.append(
            RetrievalHit(
                rank=rank,
                page_num=page_num,
                score=score,
                image_path=str(out),
            )
        )

    return hits


def retrieve_topk_pages(
    pdf_path: str,
    query: str,
    top_k: int,
    work_dir: Path,
    model_name: str = "vidore/colpali-v1.2",
    reuse_index: bool = True,
    strict_reuse: bool = True,
    render_dpi: int = 140,
) -> List[RetrievalHit]:
    logger = logging.getLogger("ColPaliRetriever")
    try:
        rag, index_name, loaded_from_cache = load_or_build_retriever(
            pdf_path=pdf_path,
            model_name=model_name,
            reuse_index=reuse_index,
            strict_reuse=strict_reuse,
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        logger.exception("ColPali: load/build index failed for %s: %s", pdf_path, exc)
        raise RuntimeError(f"ColPali index build failed for {pdf_path}") from exc

    if loaded_from_cache:
        print(f"[ColPali] Reusing existing index: {index_name}", flush=True)
    else:
        print(f"[ColPali] Built new index: {index_name}", flush=True)

    try:
        logger.info("ColPali: search start (top_k=%s)", top_k)
        search_start = perf_counter()
        raw = rag.search(query, k=top_k)
        logger.info("ColPali: search done in %.3fs", perf_counter() - search_start)
    except ValueError as e:
        if "No passages provided" in str(e):
            raise RuntimeError(
                f"ColPali retrieval failed: no usable page content found in the index. "
                f"pdf={pdf_path}, index_name={index_name}"
            ) from e
        raise

    results = normalize_results(raw)

    if not results:
        return []

    hits = render_hit_pages(
        pdf_path=pdf_path,
        results=results,
        out_dir=work_dir,
        dpi=render_dpi,
    )
    return hits
