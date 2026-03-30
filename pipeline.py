from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import random
import sys
import fitz  # PyMuPDF
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional
from time import perf_counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from prompts.generate_prompt import Q2F_with_source
from utils.extract import Extractor


OPENAI_BASE_URL = ""

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SUPPORTED_PDF_EXTS = {".pdf"}

DEFAULT_MAX_PDF_PAGES = 100
DEFAULT_PDF_DPI = 300

class Pipeline:
    """Question -> Five Elements -> Pyomo Code -> Execute."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        pdf_dpi: int = DEFAULT_PDF_DPI,
        pdf_page_indices: Optional[dict[str, list[int]]] = None,
        rag_enabled: bool = True,
        rag_chunk_size: int = 1200,
        rag_chunk_overlap: int = 200,
        rag_top_k: int = 4,
        rag_min_score: float = 0.1,
        rag_max_context_chars: int = 8000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_pdf_pages = max_pdf_pages
        self.pdf_dpi = pdf_dpi
        self.pdf_page_indices = pdf_page_indices or {}
        self.base_url = base_url
        self.extractor = Extractor()
        self.rag_enabled = rag_enabled
        self.rag = (
            RAGRetriever(
                chunk_size=rag_chunk_size,
                chunk_overlap=rag_chunk_overlap,
                top_k=rag_top_k,
                min_score=rag_min_score,
                max_context_chars=rag_max_context_chars,
                max_pdf_pages=max_pdf_pages,
                page_indices_map=self.pdf_page_indices,
                logger=logging.getLogger("RAGRetriever"),
            )
            if rag_enabled
            else None
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

        self.logger = logging.getLogger("Pipeline")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    @staticmethod
    def _sanitize_for_log(obj):
        if isinstance(obj, str):
            if obj.startswith("data:"):
                return "<data_url_omitted>"
            return obj
        if isinstance(obj, list):
            return [Pipeline._sanitize_for_log(v) for v in obj]
        if isinstance(obj, dict):
            sanitized = {}
            for k, v in obj.items():
                if k == "url" and isinstance(v, str) and v.startswith("data:"):
                    sanitized[k] = "<data_url_omitted>"
                else:
                    sanitized[k] = Pipeline._sanitize_for_log(v)
            return sanitized
        return obj

    def _encode_local_image_as_data_url(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(
                f"Unsupported image type: {suffix}. Supported types: {sorted(SUPPORTED_IMAGE_EXTS)}"
            )

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "image/png"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def _pdf_to_image_data_urls(self, file_path: str) -> list[str]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF conversion. Install with: pip install pymupdf")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data_urls: list[str] = []
        zoom = self.pdf_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        with fitz.open(path) as doc:
            total_pages = len(doc)
            if total_pages == 0:
                raise ValueError(f"Empty PDF: {file_path}")

            resolved = str(path.resolve())
            override = self.pdf_page_indices.get(resolved)
            if override:
                page_indices = [idx for idx in override if 0 <= idx < total_pages]
            else:
                pages_to_use = min(total_pages, self.max_pdf_pages)
                page_indices = self._select_pdf_page_indices(total_pages, pages_to_use)
            self.logger.info(
                "Converting PDF to images: %s (pages used: %s/%s, dpi=%s, indices=%s)",
                file_path,
                len(page_indices),
                total_pages,
                self.pdf_dpi,
                [idx + 1 for idx in page_indices],
            )

            for page_idx in page_indices:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes("png")
                encoded = base64.b64encode(img_bytes).decode("utf-8")
                data_urls.append(f"data:image/png;base64,{encoded}")

        return data_urls

    @staticmethod
    def _select_pdf_page_indices(total_pages: int, pages_to_use: int) -> list[int]:
        if pages_to_use <= 0:
            return []
        if pages_to_use >= total_pages:
            return list(range(total_pages))
        if pages_to_use == 1:
            return [0]

        indices: list[int] = []
        for i in range(pages_to_use):
            idx = round(i * (total_pages - 1) / (pages_to_use - 1))
            indices.append(idx)

        seen = set()
        deduped: list[int] = []
        for idx in indices:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
        return deduped

    def _collect_image_data_urls_from_files(self, file_paths: list[str]) -> list[str]:
        all_data_urls: list[str] = []

        for file_path in file_paths:
            suffix = Path(file_path).suffix.lower()
            if suffix in SUPPORTED_PDF_EXTS:
                all_data_urls.extend(self._pdf_to_image_data_urls(file_path))
            elif suffix in SUPPORTED_IMAGE_EXTS:
                all_data_urls.append(self._encode_local_image_as_data_url(file_path))
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. Please provide PDF or image files."
                )

        if not all_data_urls:
            raise ValueError("No valid pages/images extracted from input files.")

        return all_data_urls

    def _build_messages(self, prompt: str, file_paths: Optional[list[str]] = None) -> list[dict]:
        if not file_paths:
            return [{"role": "user", "content": prompt}]

        image_data_urls = self._collect_image_data_urls_from_files(file_paths)

        content_parts: list[dict] = []
        for data_url in image_data_urls:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high",
                    },
                }
            )

        content_parts.append(
            {
                "type": "text",
                "text": (
                    "The following images are pages converted from uploaded PDFs/images. "
                    "Read them carefully and answer based on their content.\n\n"
                    + prompt
                ),
            }
        )

        return [{"role": "user", "content": content_parts}]

    def _chat(
        self,
        prompt: str,
        file_paths: Optional[list[str]] = None,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
        messages = self._build_messages(prompt, file_paths=file_paths)
        completion = self.client.chat.completions.create(
            model=model_override or self.model,
            messages=messages,
            temperature=self.temperature if temperature_override is None else temperature_override,
            max_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        return content or ""

    def _chat_with_metrics(
        self,
        prompt: str,
        file_paths: Optional[list[str]] = None,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> tuple[str, dict]:
        messages = self._build_messages(prompt, file_paths=file_paths)
        start = perf_counter()
        completion = self.client.chat.completions.create(
            model=model_override or self.model,
            messages=messages,
            temperature=self.temperature if temperature_override is None else temperature_override,
            max_tokens=self.max_tokens,
        )
        elapsed = perf_counter() - start
        content = completion.choices[0].message.content or ""

        usage = getattr(completion, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        else:
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

        if prompt_tokens is None:
            if file_paths:
                prompt_tokens = None
            else:
                prompt_tokens = self._estimate_tokens(prompt)
        if completion_tokens is None:
            completion_tokens = self._estimate_tokens(content)
        if total_tokens is None:
            if prompt_tokens is None:
                total_tokens = None
            else:
                total_tokens = prompt_tokens + completion_tokens

        metrics = {
            "latency_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_provided": usage is not None,
        }

        return content, metrics

    @staticmethod
    def _format_metrics(label: str, metrics: dict) -> str:
        usage_note = "api-usage" if metrics.get("usage_provided") else "estimated"
        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")
        total_tokens = metrics.get("total_tokens")
        prompt_str = str(prompt_tokens) if prompt_tokens is not None else "unknown"
        completion_str = str(completion_tokens) if completion_tokens is not None else "unknown"
        total_str = str(total_tokens) if total_tokens is not None else "unknown"
        return (
            f"{label}: latency={metrics.get('latency_sec'):.3f}s, "
            f"prompt_tokens={prompt_str}, "
            f"completion_tokens={completion_str}, "
            f"total_tokens={total_str} ({usage_note})"
        )

    @staticmethod
    def _format_stats(label: str, stats: dict) -> str:
        items = ", ".join(f"{k}={v}" for k, v in stats.items())
        return f"{label}: {items}"

    def _summarize_input_files(self, file_paths: list[str]) -> dict:
        total_bytes = 0
        pdf_pages = 0
        image_files = 0
        other_files = 0

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue
            try:
                total_bytes += path.stat().st_size
            except Exception:
                pass

            suffix = path.suffix.lower()
            if suffix in SUPPORTED_PDF_EXTS:
                if fitz is None:
                    continue
                try:
                    with fitz.open(path) as doc:
                        pages_used = min(len(doc), self.max_pdf_pages)
                        pdf_pages += pages_used
                except Exception:
                    pass
            elif suffix in SUPPORTED_IMAGE_EXTS:
                image_files += 1
            else:
                other_files += 1

        return {
            "files": len(file_paths),
            "input_bytes": total_bytes,
            "pdf_pages_used": pdf_pages,
            "image_files": image_files,
            "other_files": other_files,
        }

    def _build_q2f_prompt(self, question: str) -> str:
        return Q2F_with_source(question)

    def infer_five_elem_direct_prompt(self, prompt: str, file_paths: Optional[list[str]] = None) -> str:
        """
        Use the given prompt directly, without wrapping it again with Q2F / Q2F_with_source.
        This is mainly for the demo app where the prompt is already fully constructed.
        """
        try:
            response = self._chat(prompt, file_paths=file_paths)
            five = self.extractor.extract_plain_text(response)
            if five:
                return five.strip()

            required_markers = [
                "## Sets Content:",
                "## Parameters Content:",
                "## Variables Content:",
                "## Objective Content:",
                "## Constraints Content:",
                "## Math Model Content:",
            ]
            if all(marker in response for marker in required_markers):
                self.logger.warning(
                    "extract_plain_text failed, but structured markers were found. Returning raw response directly."
                )
                return response.strip()

            legacy_markers = [
                "## Sets:",
                "## Parameters:",
                "## Variables:",
                "## Objective:",
                "## Constraints:",
            ]
            if all(marker in response for marker in legacy_markers):
                self.logger.warning(
                    "extract_plain_text failed, but legacy markers were found. Returning raw response directly."
                )
                return response.strip()

            raise RuntimeError("Failed to extract five-element block from direct prompt response.")
        except Exception as e:
            self.logger.error("Failed in infer_five_elem_direct_prompt: %s", e)
            raise

    def infer_five_elem(self, question: str, file_paths: Optional[list[str]] = None) -> str:
        try:
            if self.rag_enabled and file_paths:
                retrieval_start = perf_counter()
                context, selected = self.rag.build_context(question, file_paths)
                retrieval_sec = perf_counter() - retrieval_start
                if context:
                    self.logger.info("RAG enabled: using %s retrieved chunks.", len(selected))
                    self.logger.info("RAG retrieval latency: %.3fs", retrieval_sec)
                    for idx, sc in enumerate(selected, 1):
                        self.logger.info(
                            "RAG Chunk %s | score=%.4f | match=%.3f | source=%s\n%s",
                            idx,
                            sc.score,
                            sc.match,
                            sc.chunk.source,
                            sc.chunk.text,
                        )
                    rag_prompt = self._build_rag_prompt(question, context)
                    response = self._chat(rag_prompt)
                else:
                    self.logger.info("RAG fallback: uploading full files.")
                    if self.rag_benchmark:
                        response, metrics = self._chat_with_metrics(
                            self._build_q2f_prompt(question), file_paths=file_paths
                        )
                        self.logger.info(self._format_metrics("FULL_UPLOAD", metrics))
                    else:
                        response = self._chat(self._build_q2f_prompt(question), file_paths=file_paths)
            else:
                response = self._chat(self._build_q2f_prompt(question), file_paths=file_paths)
            five = self.extractor.extract_plain_text(response)
            if not five:
                raise RuntimeError("Failed to extract five-element block from response.")
            return five.strip()
        except Exception as e:
            self.logger.error(f"Failed to extract five-element block from response:{e}")


    def run(self, question: str, file_paths: Optional[list[str]] = None) -> tuple[str, str, str, str]:
        five_elem = self.infer_five_elem(question, file_paths=file_paths)
        return five_elem
def normalize_input_files(args: argparse.Namespace) -> Optional[list[str]]:
    return args.files if args.files else None


def select_pdf_page_indices(
    file_paths: list[str],
    max_pages: int,
    logger: logging.Logger,
) -> dict[str, list[int]]:
    if fitz is None:
        return {}
    selections: dict[str, list[int]] = {}
    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_PDF_EXTS or not path.exists():
            continue
        with fitz.open(path) as doc:
            total_pages = len(doc)
        if total_pages <= 0:
            continue
        if total_pages <= max_pages:
            indices = list(range(total_pages))
        else:
            start = random.randint(0, total_pages - max_pages)
            indices = list(range(start, start + max_pages))
        resolved = str(path.resolve())
        selections[resolved] = indices
        logger.info(
            "PDF page selection: %s -> %s/%s pages, indices=%s",
            file_path,
            len(indices),
            total_pages,
            [idx + 1 for idx in indices],
        )
    return selections


def extract_five_elements_source(five_elem_str: str) -> dict:
    result = {}
    sections = ["Sets", "Parameters", "Variables", "Objective", "Constraints", "Math Model"]

    for i, section in enumerate(sections):
        content_marker = f"## {section} Content:"
        source_marker = f"## {section} Source:"
        content_idx = five_elem_str.find(content_marker)
        source_idx = five_elem_str.find(source_marker)
        if content_idx == -1 or source_idx == -1 or source_idx < content_idx:
            result[section] = {"content": "EXTRACTION_FAILED", "source": "EXTRACTION_FAILED"}
            continue

        content_start = content_idx + len(content_marker)
        content = five_elem_str[content_start:source_idx].strip()

        source_start = source_idx + len(source_marker)
        end_idx = len(five_elem_str)
        if i + 1 < len(sections):
            next_marker = f"## {sections[i + 1]} Content:"
            next_idx = five_elem_str.find(next_marker, source_start)
            if next_idx != -1:
                end_idx = next_idx

        source = five_elem_str[source_start:end_idx].strip()
        result[section] = {
            "content": content if content else "EXTRACTION_FAILED",
            "source": source if source else "EXTRACTION_FAILED",
        }

    return result


def save_to_json(question: str, five_elements: dict, output_path: Path) -> None:
    data = {
        "question": question,
        "five_elements": five_elements,
    }

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

