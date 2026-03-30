from __future__ import annotations

import re
from typing import Dict, List

from pipeline import (
    Pipeline,
    extract_five_elements_source,
)
from prompts.generate_prompt import Q2F_with_source_demo_v2, Q2F_with_source


def _clean_str(x: str) -> str:
    return (x or "").strip()


def _normalize_one_source(section_name: str, source_text: str) -> str:
    src = _clean_str(source_text)

    if not src or src == "EXTRACTION_FAILED":
        return (
            "Source Type: NL\n"
            f"Explanation: This {section_name.lower()} component is inferred from the user request or the overall modeling context because no valid source block was produced."
        )

    if re.search(r"^Source Type:\s*NL\s*$", src, re.M):
        explanation = re.search(r"Explanation:\s*(.*)", src, re.S)
        exp = explanation.group(1).strip() if explanation else (
            f"This {section_name.lower()} component is inferred from the user request."
        )
        return f"Source Type: NL\nExplanation: {exp}"

    if re.search(r"^Source Type:\s*DOC\s*$", src, re.M):
        doc = re.search(r"Document:\s*(.*)", src)
        page = re.search(r"Page:\s*(.*)", src)
        locator = re.search(r"Locator:\s*(.*)", src)
        evidence = re.search(
            r"Evidence:\s*(.*?)(?=\n(?:Explanation:|Source Type:|Document:|Page:|Locator:)|\Z)",
            src,
            re.S,
        )
        explanation = re.search(r"Explanation:\s*(.*)", src, re.S)

        return (
            "Source Type: DOC\n"
            f"Document: {(doc.group(1).strip() if doc else '')}\n"
            f"Page: {(page.group(1).strip() if page else '')}\n"
            f"Locator: {(locator.group(1).strip() if locator else '')}\n"
            f"Evidence: {(evidence.group(1).strip() if evidence else '')}\n"
            f"Explanation: {(explanation.group(1).strip() if explanation else '')}"
        ).strip()

    doc = re.search(r"Document:\s*(.*)", src)
    page = re.search(r"Page:\s*(.*)", src)
    locator = re.search(r"Locator:\s*(.*)", src)
    evidence = re.search(
        r"Evidence:\s*(.*?)(?=\n(?:Explanation:|Modeling Explanation:|Document:|Page:|Locator:|Source Type:)|\Z)",
        src,
        re.S,
    )
    explanation = re.search(r"(?:Explanation|Modeling Explanation):\s*(.*)", src, re.S)

    doc_val = doc.group(1).strip() if doc else ""
    page_val = page.group(1).strip() if page else ""
    locator_val = locator.group(1).strip() if locator else ""
    evidence_val = evidence.group(1).strip() if evidence else ""
    explanation_val = explanation.group(1).strip() if explanation else ""

    has_real_doc = any(
        v.strip().lower() not in {"", "n/a", "na", "none", "null"}
        for v in [doc_val, page_val, locator_val, evidence_val]
    )
    if has_real_doc:
        return (
            "Source Type: DOC\n"
            f"Document: {doc_val}\n"
            f"Page: {page_val}\n"
            f"Locator: {locator_val}\n"
            f"Evidence: {evidence_val}\n"
            f"Explanation: {explanation_val}"
        ).strip()

    fallback_explanation = explanation_val or src
    return (
        "Source Type: NL\n"
        f"Explanation: {fallback_explanation}"
    )


def _normalize_all_sources(parsed: Dict) -> Dict:
    out = {}
    for section_name, payload in parsed.items():
        if not isinstance(payload, dict):
            out[section_name] = {
                "content": str(payload) if payload is not None else "",
                "source": (
                    "Source Type: NL\n"
                    f"Explanation: This {section_name.lower()} component is inferred from the user request because no structured source block was produced."
                ),
            }
            continue

        out[section_name] = {
            "content": _clean_str(payload.get("content", "")),
            "source": _normalize_one_source(section_name, payload.get("source", "")),
        }
    return out


def _build_hit_metadata_block(hit_items: List[Dict]) -> str:
    """
    Build metadata text for retrieved page images so the VLM can explicitly cite:
    document name + page number + page rank/score.
    """
    lines = []
    for i, h in enumerate(hit_items, start=1):
        source_pdf = h.get("source_pdf", "UnknownDocument")
        page_num = h.get("page_num", "?")
        rank_in_doc = h.get("rank_in_doc", "?")
        score = h.get("score", "?")
        lines.append(
            f"[Image {i}] document={source_pdf}; page={page_num}; rank_in_doc={rank_in_doc}; score={score}"
        )
    return "\n".join(lines)


def generate_five_elements_from_hit_images(
    question: str,
    hit_items: List[Dict],
    api_key: str,
    model: str,
    base_url: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    memory_context: str = "",
) -> Dict:
    """
    Generate a source-aware five-element model + math-model block from retrieved page images.

    Returns:
        {
            "raw_text": "...",
            "parsed": {
                "Sets": {"content": "...", "source": "..."},
                "Parameters": {"content": "...", "source": "..."},
                "Variables": {"content": "...", "source": "..."},
                "Objective": {"content": "...", "source": "..."},
                "Constraints": {"content": "...", "source": "..."},
                "Math Model": {"content": "...", "source": "..."},
            }
        }
    """
    pipeline = Pipeline(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        rag_enabled=False,
        use_source=True,
    )

    if not hit_items:
        raise RuntimeError("No retrieved hit items were provided.")

    image_paths = [h["image_path"] for h in hit_items if h.get("image_path")]
    if not image_paths:
        raise RuntimeError("No valid image paths found in hit items.")

    hit_meta_block = _build_hit_metadata_block(hit_items)


    memory_part = ""

    if memory_context.strip():
        memory_part = (
            "Previous five-element memory from the last round is provided below.\n"
            "Use it as contextual memory to maintain consistency of notation, structure, and modeling choices when helpful.\n"
            "The new question has the highest priority.\n"
            "Revise only the parts that are required by the new question or by the current retrieved evidence.\n"
            "For parts that are not mentioned or affected by the new question, keep them unchanged as much as possible.\n"
            "Do not rewrite the whole model unnecessarily.\n"
            "However, if any previous content conflicts with the new question or the current retrieved evidence, you must correct, replace, or remove it.\n"
            "Do not copy the previous memory blindly.\n\n"
            "Previous five-element memory:\n"
            f"{memory_context.strip()}\n\n"
            "New question:\n"
        )

    five_text = pipeline.infer_five_elem(
        memory_part + question,
        file_paths=image_paths,
    )

    if not five_text:
        raise RuntimeError("The pipeline returned an empty five-element result.")

    parsed = extract_five_elements_source(five_text)
    parsed = _normalize_all_sources(parsed)

    return {
        "raw_text": five_text,
        "parsed": parsed,
    }