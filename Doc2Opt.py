from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.colpali_rag import collect_colpali_hits
from prompts.critic_prompt import CRITIC_PROMPT_TEMPLATE, REWRITE_PROMPT_TEMPLATE

from pipeline import (
    Pipeline,
    DEFAULT_MAX_PDF_PAGES,
    DEFAULT_PDF_DPI,
    OPENAI_BASE_URL,
    extract_five_elements_source,
    normalize_input_files,
    select_pdf_page_indices,
)


OUTPUT_ROOT = Path("result")

SECTIONS = ["Sets", "Parameters", "Variables", "Objective", "Constraints"]

COLPALI_MODEL = "vidore/colpali-v1.2"
COLPALI_REUSE_INDEX = True
COLPALI_WORK_ROOT = "colpali_cache"
COLPALI_QUERY_HINT = ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Optimization question (optional when auto-generating)",
    )
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--api-key",
        type=str,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=OPENAI_BASE_URL,
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="Input file paths (PDF/image)",
    )
    parser.add_argument(
        "--colpali-per-doc-top-k",
        type=int,
        default=3,
        help="ColPali top-k pages per document.",
    )
    parser.add_argument(
        "--colpali-global-top-k",
        type=int,
        default=3,
        help="ColPali global top-k pages across documents.",
    )
    parser.add_argument(
        "--colpali-render-dpi",
        type=int,
        default=140,
        help="DPI for rendering hit pages.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum judge-regenerate iterations",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.2,
        help="Stop if score improvement between consecutive iterations is below this threshold",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Deprecated: use --min-delta",
    )
    return parser.parse_args()


def _compose_content_only(five_dict: dict) -> str:
    lines: list[str] = []
    for section in SECTIONS:
        entry = five_dict.get(section, "")
        if isinstance(entry, dict):
            content = entry.get("content", "")
        else:
            content = entry
        lines.append(f"## {section}:")
        lines.append(str(content).strip() if content else "")
        lines.append("")
    return "\n".join(lines).strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _parse_judge(judge_json: Optional[dict]) -> tuple[Optional[float], str]:
    if not isinstance(judge_json, dict):
        return None, ""
    score = _safe_float(judge_json.get("score"))
    rationale = judge_json.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return score, rationale.strip()


def _run_judge(
    question_text: str,
    model_text: str,
    pipeline_base: Pipeline,
) -> tuple[Optional[float], str, str]:
    judge_prompt = CRITIC_PROMPT_TEMPLATE.format(
        question=question_text,
        model=model_text,
    )
    judge_response = pipeline_base._chat(
        judge_prompt,
        temperature_override=0.0,
    )
    judge_json = _extract_json(judge_response)
    score, rationale = _parse_judge(judge_json)
    return score, rationale, judge_response


def _run_regen(
    question_text: str,
    model_text: str,
    feedback: str,
    pipeline_base: Pipeline,
    temperature: float,
) -> tuple[str, str]:
    regen_prompt = REWRITE_PROMPT_TEMPLATE.format(
        question=question_text,
        model=model_text,
        feedback=feedback,
    )
    regen_response = pipeline_base._chat(
        regen_prompt,
        temperature_override=max(temperature, 0.7),
    )
    extracted = pipeline_base.extractor.extract_plain_text(regen_response)
    return (extracted or regen_response).strip(), regen_response


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main(
    args: Optional[argparse.Namespace] = None,
) -> str:
    args = args or parse_args()

    if not args.api_key:
        raise ValueError("Missing API key.")
    
    question_text = args.question

    input_files = normalize_input_files(args)
    if not input_files:
        raise ValueError("Requires --files to solve a problem.")


    output_root = OUTPUT_ROOT

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"log_{now_tag}.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    stream_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[file_handler, stream_handler],
        force=True,
    )

    logger = logging.getLogger("Main")

    logger.info("=" * 60)
    logger.info("Input Command: %s", " ".join(sys.argv))
    logger.info("Log file: %s", log_path)
    logger.info("Using input files: %s", input_files)
    logger.info("Model: %s", args.model)
    logger.info("=" * 60)

    pdf_page_indices = select_pdf_page_indices(input_files, DEFAULT_MAX_PDF_PAGES, logger)

    pipeline_base = Pipeline(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_pdf_pages=DEFAULT_MAX_PDF_PAGES,
        pdf_dpi=DEFAULT_PDF_DPI,
        pdf_page_indices=pdf_page_indices,
        rag_enabled=False,
    )

    logger.info("Step0: Load question...")
    logger.info(question_text)

    logger.info("Step1: ColPali multimodal RAG retrieval + five-element extraction...")
    retrieval_query = question_text.strip()
    if COLPALI_QUERY_HINT:
        retrieval_query = f"{retrieval_query} {COLPALI_QUERY_HINT.strip()}".strip()

    retrieval_start = perf_counter()
    hit_items = collect_colpali_hits(
        file_paths=input_files,
        query=retrieval_query,
        work_root=Path(COLPALI_WORK_ROOT) / now_tag,
        per_doc_top_k=args.colpali_per_doc_top_k,
        global_top_k=args.colpali_global_top_k,
        model_name=COLPALI_MODEL,
        reuse_index=bool(COLPALI_REUSE_INDEX),
        render_dpi=args.colpali_render_dpi,
        logger=logging.getLogger("ColPaliRAG"),
    )
    retrieval_sec = perf_counter() - retrieval_start
    logger.info("ColPali retrieval latency: %.3fs", retrieval_sec)

    if hit_items:
        logger.info("ColPali retrieved %s pages (global_top_k=%s).", len(hit_items), args.colpali_global_top_k)
        for idx, h in enumerate(hit_items, 1):
            logger.info(
                "ColPali Hit %s | score=%.4f | doc=%s | page=%s | rank_in_doc=%s | image=%s",
                idx,
                float(h.get("score", 0.0)),
                h.get("source_pdf", ""),
                h.get("page_num", ""),
                h.get("rank_in_doc", ""),
                h.get("image_path", ""),
            )
        image_paths = [h["image_path"] for h in hit_items if h.get("image_path")]
        if image_paths:
            structured_five = pipeline_base.infer_five_elem(
                question_text,
                file_paths=image_paths,
            )
        else:
            logger.warning("ColPali hits missing image paths; falling back to full upload.")
            structured_five = pipeline_base.infer_five_elem(
                question_text,
                file_paths=input_files,
            )
    else:
        logger.warning("ColPali returned no hits; falling back to full upload.")
        structured_five = pipeline_base.infer_five_elem(
            question_text,
            file_paths=input_files,
        )
    logger.info("STEP2: Generate five-elements(COLPALI RAG)")
    logger.info(structured_five)
    structured_dict = extract_five_elements_source(structured_five)
    structured_path = output_root / "Doc2Opt" / f"result_{now_tag}.json"
    save_json(structured_path, {"question": question_text, "five_elements": structured_dict})
    logger.info("Step2 result saved to: %s", structured_path)

    base_model_text = _compose_content_only(structured_dict)

    logger.info("Step3: Iterative judge-regenerate loop...")
    max_iter = max(1, args.max_iter)
    min_delta = args.min_delta
    if args.score_threshold is not None:
        min_delta = args.score_threshold
    history: list[dict] = []
    current_model = base_model_text
    final_score = None
    final_rationale = ""
    stop_reason = "max_iter"
    previous_score = None

    for iteration in range(1, max_iter + 1):
        logger.info("Iteration %s/%s: judging current model...", iteration, max_iter)
        sleep(1)

        model_before_judge = current_model
        score, rationale, judge_response = _run_judge(
            question_text=question_text,
            model_text=model_before_judge,
            pipeline_base=pipeline_base,
        )
        final_score = score
        final_rationale = rationale

        delta = None
        if previous_score is not None and score is not None:
            delta = score - previous_score
        logger.info(
            "Iteration %s | score=%s | delta=%s | rationale=%s",
            iteration,
            score,
            delta,
            rationale,
        )

        next_model = None
        regen_response = ""

        if score is None:
            logger.warning("Iteration %s: judge response missing score, stopping.", iteration)
            stop_reason = "invalid_score"
        elif previous_score is not None and delta is not None and delta < min_delta:
            logger.info(
                "Iteration %s: delta %.4f < min_delta %.4f, stopping.",
                iteration,
                delta,
                min_delta,
            )
            stop_reason = "stop_delta"
        elif iteration >= max_iter:
            stop_reason = "max_iter"
        else:
            logger.info("Iteration %s: regenerating with judge feedback...", iteration)
            next_model, regen_response = _run_regen(
                question_text=question_text,
                model_text=model_before_judge,
                feedback=rationale or "No feedback provided.",
                pipeline_base=pipeline_base,
                temperature=args.temperature,
            )
            if not next_model:
                logger.warning("Iteration %s: regeneration returned empty model, stopping.", iteration)
                current_model = model_before_judge
                stop_reason = "empty_regen"
            else:
                current_model = next_model
                stop_reason = "continue"

        record = {
            "iteration": iteration,
            "question": question_text,
            "model_before_judge": model_before_judge,
            "judge_score": score,
            "delta": delta,
            "judge_rationale": rationale,
            "judge_raw": judge_response,
            "model_after_regen": next_model,
            "regen_raw": regen_response,
            "stop_reason": stop_reason,
        }
        history.append(record)

        iter_path = output_root / "Doc2Opt_Iter" / f"result_{now_tag}_iter{iteration}.json"
        save_json(iter_path, record)

        if stop_reason != "continue":
            break
        previous_score = score

    summary_path = output_root / "Doc2Opt_Summary" / f"result_{now_tag}.json"
    save_json(
        summary_path,
        {
            "question": question_text,
            "initial_model": base_model_text,
            "final_model": current_model,
            "final_score": final_score,
            "final_rationale": final_rationale,
            "history": history,
            "max_iter": max_iter,
            "min_delta": min_delta,
            "stop_reason": stop_reason,
        },
    )
    logger.info("Summary saved to: %s", summary_path)

    logger.info("=" * 60)
    return question_text


if __name__ == "__main__":
    print(f"Starting with command: {' '.join(sys.argv)}")
    args = parse_args()
    main(args=args)
