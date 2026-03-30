from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from utils.apptools import ensure_dir, safe_name, sha256_bytes, log_debug, log_exception
from utils.colpali_retriever import retrieve_topk_pages
from utils.app_bridge import generate_five_elements_from_hit_images


APP_CONFIG = {
    "retriever_model": "vidore/colpali-v1.2",
    "per_doc_top_k": 3,
    "global_top_k": 3,
    "render_dpi": 140,
    "reuse_index": True,
    "work_root": "/tmp/llmopt_colpali_app",
    "doc_cache_root": "/tmp/llmopt_colpali_app/doc_cache",
    "index_cache_root": "/tmp/llmopt_colpali_app/index_cache",
    "api_base": (
        os.environ.get("MM_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip(),
    "api_key": (os.environ.get("MM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip(),
    "api_model": (os.environ.get("MM_LLM_MODEL") or os.environ.get("LLM_MODEL") or "qwen-vl-max-latest").strip(),
    "temperature": 0.1,
    "max_tokens": 4096,
}

DISPLAY_ORDER = [
    ("Math Model", "Complete Mathematical Model"),
    ("Objective", "Objective"),
    ("Variables", "Variables"),
    ("Constraints", "Constraints"),
    ("Sets", "Sets"),
    ("Parameters", "Parameters"),
]


def init_state():
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []
    if "current_run_dir" not in st.session_state:
        st.session_state["current_run_dir"] = ""
    if "doc_pool" not in st.session_state:
        st.session_state["doc_pool"] = []
    if "chat_records" not in st.session_state:
        st.session_state["chat_records"] = []


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def build_run_dir(doc_pool: List[Dict]) -> Path:
    joined = b"".join([doc["bytes"] for doc in doc_pool])
    bundle_hash = sha256_bytes(joined)[:12] if joined else "empty"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(APP_CONFIG["work_root"]) / f"multi_pdf_{bundle_hash}" / run_id)


def normalize_source_aware_five_tuple(parsed: Dict) -> Dict:
    keys = ["Math Model", "Sets", "Parameters", "Variables", "Objective", "Constraints"]
    out = {}
    for k in keys:
        v = parsed.get(k, {})
        if isinstance(v, dict):
            out[k] = {
                "content": v.get("content", "") or "",
                "source": v.get("source", "") or "",
            }
        else:
            out[k] = {
                "content": str(v) if v is not None else "",
                "source": "",
            }
    return out


def add_uploaded_files_to_pool(uploaded_files):
    if not uploaded_files:
        return

    doc_cache_root = ensure_dir(APP_CONFIG["doc_cache_root"])
    model_root = safe_model_name(APP_CONFIG["retriever_model"])
    index_cache_root = ensure_dir(Path(APP_CONFIG["index_cache_root"]) / model_root)

    existing_hashes = {doc["hash"] for doc in st.session_state["doc_pool"]}

    for f in uploaded_files:
        file_bytes = f.getvalue()
        file_hash = sha256_bytes(file_bytes)
        if file_hash in existing_hashes:
            continue

        pdf_stem = safe_name(Path(f.name).stem)
        cached_pdf_path = doc_cache_root / f"{pdf_stem}_{file_hash[:8]}.pdf"
        if not cached_pdf_path.exists():
            cached_pdf_path.write_bytes(file_bytes)

        index_cache_dir = index_cache_root / file_hash[:16]

        st.session_state["doc_pool"].append(
            {
                "name": f.name,
                "bytes": file_bytes,
                "hash": file_hash,
                "cached_pdf_path": str(cached_pdf_path),
                "index_cache_dir": str(index_cache_dir),
            }
        )
        existing_hashes.add(file_hash)


def clear_doc_pool():
    st.session_state["doc_pool"] = []
    st.session_state["chat_records"] = []


def get_doc_pool_names() -> List[str]:
    return [doc["name"] for doc in st.session_state["doc_pool"]]


def save_selected_pdfs_from_pool(selected_names: List[str], run_dir: Path) -> List[Dict]:
    saved = []

    for doc in st.session_state["doc_pool"]:
        if doc["name"] not in selected_names:
            continue

        saved.append(
            {
                "name": doc["name"],
                "pdf_path": doc["cached_pdf_path"],
                "index_cache_dir": doc["index_cache_dir"],
                "hash": doc["hash"],
            }
        )

    return saved


def build_retrieval_query(user_query: str) -> str:
    retrieval_hint = (
        " Focus on pages useful for electric-power-system mathematical modeling, "
        "especially equations, figures, tables, variable/constraint definitions, "
        "and state-transition or post-contingency operation details."
    )
    return user_query.strip() + retrieval_hint


def collect_hits_from_selected_docs(saved_docs: List[Dict], retrieval_query: str, run_dir: Path) -> List[Dict]:
    all_hits: List[Dict] = []
    logs = st.session_state["debug_logs"]

    for doc in saved_docs:
        doc_name = doc["name"]
        pdf_path = Path(doc["pdf_path"])
        index_cache_dir = Path(doc["index_cache_dir"])

        log_debug(f"Retrieving document: {doc_name} | pdf_path={pdf_path}", logs, run_dir)
        log_debug(f"Index directory: {index_cache_dir}", logs, run_dir)
        log_debug(f"Retrieval query: {retrieval_query}", logs, run_dir)

        hits = retrieve_topk_pages(
            pdf_path=str(pdf_path),
            query=retrieval_query,
            top_k=APP_CONFIG["per_doc_top_k"],
            work_dir=index_cache_dir,
            model_name=APP_CONFIG["retriever_model"],
            reuse_index=APP_CONFIG["reuse_index"],
            render_dpi=APP_CONFIG["render_dpi"],
        )

        for h in hits:
            all_hits.append(
                {
                    "source_pdf": doc_name,
                    "rank_in_doc": h.rank,
                    "page_num": h.page_num,
                    "score": h.score,
                    "image_path": h.image_path,
                }
            )

    all_hits.sort(key=lambda x: float(x["score"]), reverse=True)
    return all_hits[: APP_CONFIG["global_top_k"]]


def split_formula_and_tail_text(text: str):
    if not text or not text.strip():
        return "", ""

    s = text.strip()
    latex_markers = ["\\", "^", "_", r"\sum", r"\min", r"\max", r"\forall", r"\in", r"\leq", r"\geq"]
    if not any(m in s for m in latex_markers):
        return "", s

    for i in range(len(s) - 1, max(len(s) - 120, 0), -1):
        head = s[:i].strip()
        tail = s[i:].strip()
        if not head or not tail:
            continue

        tail_has_letters = bool(re.search(r"[A-Za-z]", tail))
        tail_has_latex = any(x in tail for x in ["\\", "{", "}", "^", "_", "$"])

        if tail_has_letters and not tail_has_latex:
            return head, tail

    return s, ""


def split_latex_and_text_blocks(text: str):
    if not text or not text.strip():
        return []

    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace(r"\begin{align*}", r"\begin{aligned}")
    t = t.replace(r"\end{align*}", r"\end{aligned}")
    t = t.replace(r"\begin{align}", r"\begin{aligned}")
    t = t.replace(r"\end{align}", r"\end{aligned}")

    blocks = []
    itemize_pattern = re.compile(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", re.S)
    pos = 0

    for m in itemize_pattern.finditer(t):
        if m.start() > pos:
            prefix = t[pos:m.start()].strip()
            if prefix:
                blocks.extend(_split_formula_and_plain(prefix))

        item_block = m.group(1).strip()
        items = re.findall(r"\\item\s+(.*?)(?=(\\item|$))", item_block, re.S)
        parsed_items = [it[0].strip() for it in items if it[0].strip()]
        blocks.append(("itemize", parsed_items))
        pos = m.end()

    if pos < len(t):
        suffix = t[pos:].strip()
        if suffix:
            blocks.extend(_split_formula_and_plain(suffix))

    return blocks


def _split_formula_and_plain(text: str):
    patterns = [
        re.compile(r"\\\[(.*?)\\\]", re.S),
        re.compile(r"\$\$(.*?)\$\$", re.S),
        re.compile(r"(\\begin\{aligned\}.*?\\end\{aligned\})", re.S),
    ]

    matches = []
    for p in patterns:
        for m in p.finditer(text):
            matches.append((m.start(), m.end(), m.group(0), m.group(1) if m.lastindex else m.group(0)))

    if not matches:
        return [("text", text)]

    matches.sort(key=lambda x: x[0])

    blocks = []
    pos = 0
    for start, end, raw, inner in matches:
        if start > pos:
            plain = text[pos:start].strip()
            if plain:
                blocks.append(("text", plain))
        formula = inner.strip()
        if formula:
            blocks.append(("latex", formula))
        pos = end

    if pos < len(text):
        plain = text[pos:].strip()
        if plain:
            blocks.append(("text", plain))

    return blocks


def normalize_latex_text(text: str) -> str:
    if not text:
        return ""

    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    m = re.match(r"^\s*```(?:latex|text|plaintext)?\s*\n?(.*?)\n?```\s*$", t, re.S)
    if m:
        t = m.group(1).strip()

    if "\\\\" in t and any(x in t for x in ["\\\\begin", "\\\\sum", "\\\\min", "\\\\max", "\\\\text", "\\\\forall"]):
        t = t.replace("\\\\", "\\")

    t = t.replace(r"\begin{align*}", r"\begin{aligned}")
    t = t.replace(r"\end{align*}", r"\end{aligned}")
    t = t.replace(r"\begin{align}", r"\begin{aligned}")
    t = t.replace(r"\end{align}", r"\end{aligned}")

    m = re.match(r"^\s*\\\[(.*)\\\]\s*$", t, re.S)
    if m:
        t = m.group(1).strip()

    m = re.match(r"^\s*\$\$(.*)\$\$\s*$", t, re.S)
    if m:
        t = m.group(1).strip()

    t = t.strip().strip("$").strip()

    return t

def _is_latex_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    latex_markers = [
        "\\", "^", "_",
        r"\sum", r"\min", r"\max", r"\forall",
        r"\in", r"\leq", r"\geq", r"\text",
        r"\begin", r"\end", r"\mathbf", r"\boldsymbol",
        r"\mathcal", r"\underline", r"\overline",
    ]
    return any(m in t for m in latex_markers)


def _split_aligned_rows(text: str) -> list[str]:
    m = re.search(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", text, re.S)
    if not m:
        return []

    body = m.group(1).strip()
    rows = re.split(r"\\\\\s*", body)
    rows = [r.strip() for r in rows if r.strip()]
    return rows


def _render_aligned_block(text: str) -> bool:
    t = normalize_latex_text(text)
    if not t:
        return False

    try:
        st.latex(t)
        return True
    except Exception:
        pass

    rows = _split_aligned_rows(t)
    if not rows:
        return False

    rendered = False
    for row in rows:
        row = row.strip()
        if not row:
            continue

        row = re.sub(r"^\s*&+", "", row).strip()

        parts = [p.strip() for p in re.split(r"(?<!\\)&{2,}", row) if p.strip()]
        if len(parts) >= 2:
            formula = parts[0].strip().rstrip("\\").strip()
            note = parts[1].strip().rstrip("\\").strip()

            try:
                st.latex(formula)
            except Exception:
                st.code(formula, language="latex")
            if note:
                note_plain = re.sub(r"\\text\{([^{}]*)\}", r"\1", note)
                note_plain = re.sub(r"\s+", " ", note_plain).strip()
                if note_plain:
                    st.markdown(note_plain)
            rendered = True
        else:
            single = parts[0] if parts else row
            single = single.rstrip("\\").strip()
            try:
                st.latex(single)
            except Exception:
                st.code(single, language="latex")
            rendered = True

    return rendered


def _extract_itemize_items(text: str) -> list[str]:
    items = []
    t = text or ""
    for m in re.finditer(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", t, re.S):
        body = m.group(1)
        found = re.findall(r"\\item\s+(.*?)(?=(\\item|$))", body, re.S)
        items.extend([it[0].strip() for it in found if it[0].strip()])
    return items


def _render_itemized_text(text: str) -> bool:
    items = _extract_itemize_items(text)
    if not items:
        return False

    for item in items:
        blocks = _split_formula_and_plain(item)
        text_parts = []
        formula_parts = []

        for kind, value in blocks:
            if kind == "text":
                text_parts.append(value.strip())
            elif kind == "latex":
                formula_parts.append(normalize_latex_text(value))

        if text_parts:
            st.markdown("- " + " ".join([p for p in text_parts if p]))
        else:
            st.markdown("-")

        for f in formula_parts:
            try:
                st.latex(f)
            except Exception:
                st.code(f, language="latex")
    return True


def render_item_with_formula(item: str):
    sub_blocks = _split_formula_and_plain(item)
    text_parts = []
    formula_parts = []

    for kind, value in sub_blocks:
        if kind == "text":
            text_parts.append(value)
        elif kind == "latex":
            formula_parts.append(normalize_latex_text(value))

    if text_parts:
        st.markdown("- " + " ".join(text_parts))
    else:
        st.markdown("-")

    for f in formula_parts:
        try:
            st.latex(f)
        except Exception:
            st.code(f, language="latex")


def looks_like_plain_latex(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    latex_markers = [
        "\\", "^", "_",
        r"\sum", r"\min", r"\max", r"\forall",
        r"\in", r"\leq", r"\geq",
        r"\begin", r"\end",
        r"\boldsymbol", r"\mathcal", r"\text",
        r"\underline", r"\overline",
    ]
    return any(m in t for m in latex_markers)


def render_formula_text(content: str):
    if not content or not content.strip():
        st.info("No content.")
        return

    text = normalize_latex_text(content)

    if r"\begin{itemize}" in text and r"\end{itemize}" in text:
        if _render_itemized_text(text):
            return

    if r"\begin{aligned}" in text and r"\end{aligned}" in text:
        if _render_aligned_block(text):
            return

    if _is_latex_like(text):
        try:
            st.latex(text)
            return
        except Exception:
            pass

    main_formula, tail_text = split_formula_and_tail_text(text)
    main_formula = normalize_latex_text(main_formula)
    tail_text = (tail_text or "").strip()

    if main_formula and _is_latex_like(main_formula):
        try:
            st.latex(main_formula)
            if tail_text:
                st.markdown(tail_text)
            return
        except Exception:
            pass

    rendered_any = False
    blocks = split_latex_and_text_blocks(text)

    for kind, value in blocks:
        if kind == "latex":
            value = normalize_latex_text(value)

            if r"\begin{aligned}" in value and r"\end{aligned}" in value:
                ok = _render_aligned_block(value)
                rendered_any = rendered_any or ok
                if ok:
                    continue

            try:
                st.latex(value)
                rendered_any = True
            except Exception:
                st.code(value, language="latex")
                rendered_any = True

        elif kind == "text":
            st.markdown(value)
            rendered_any = True

        elif kind == "itemize":
            for item in value:
                blocks2 = _split_formula_and_plain(item)
                text_parts = []
                formula_parts = []

                for k2, v2 in blocks2:
                    if k2 == "text":
                        text_parts.append(v2.strip())
                    elif k2 == "latex":
                        formula_parts.append(normalize_latex_text(v2))

                if text_parts:
                    st.markdown("- " + " ".join([p for p in text_parts if p]))
                else:
                    st.markdown("-")

                for f in formula_parts:
                    try:
                        st.latex(f)
                    except Exception:
                        st.code(f, language="latex")
                rendered_any = True

    if not rendered_any:
        if _is_latex_like(text):
            st.code(text, language="latex")
        else:
            st.markdown(text)


def parse_source_blocks(source_text: str) -> Dict[str, str]:
    src = (source_text or "").strip()
    if not src:
        return {"type": "empty", "explanation": ""}

    source_type_match = re.search(r"Source Type:\s*(NL|DOC)", src)
    if not source_type_match:
        return {"type": "raw", "raw": src}

    source_type = source_type_match.group(1).strip()

    if source_type == "NL":
        explanation = re.search(r"Explanation:\s*(.*)", src, re.S)
        return {
            "type": "nl",
            "explanation": explanation.group(1).strip() if explanation else "",
        }

    if source_type == "DOC":
        doc = re.search(r"Document:\s*(.*)", src)
        page = re.search(r"Page:\s*(.*)", src)
        locator = re.search(r"Locator:\s*(.*)", src)
        evidence = re.search(r"Evidence:\s*(.*?)(?=\nExplanation:|\Z)", src, re.S)
        explanation = re.search(r"Explanation:\s*(.*)", src, re.S)
        return {
            "type": "doc",
            "document": doc.group(1).strip() if doc else "",
            "page": page.group(1).strip() if page else "",
            "locator": locator.group(1).strip() if locator else "",
            "evidence": evidence.group(1).strip() if evidence else "",
            "explanation": explanation.group(1).strip() if explanation else "",
        }

    return {"type": "raw", "raw": src}


def render_source_blocks(source_text: str):
    block = parse_source_blocks(source_text)
    btype = block.get("type", "raw")

    if btype == "empty":
        st.caption("No source explanation provided.")
        return

    if btype == "nl":
        st.markdown("**Source Type:** *NL*")
        explanation = block.get("explanation", "").strip()
        if explanation:
            st.markdown("**Explanation**")
            st.markdown(explanation)
        else:
            st.markdown("This component is inferred from the user request.")
        return

    if btype == "doc":
        st.markdown(f"**Source Type:** *DOC*")
        st.markdown(f"**Document:** {block.get('document', '')}")
        st.markdown(f"**Page:** {block.get('page', '')}")
        st.markdown(f"**Locator:** *{block.get('locator', '')}*")

        evidence = block.get("evidence", "").strip()
        if evidence:
            st.markdown("**Evidence**")
            st.info(evidence)

        explanation = block.get("explanation", "").strip()
        if explanation:
            st.markdown("**Modeling Explanation**")
            st.markdown(explanation)
        return

    st.markdown(block.get("raw", ""))

def split_content_lines(content: str) -> List[str]:
    text = (content or "").strip()
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_latex_text(text)

    parts = re.split(r"\\\\\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def render_five_element_content(content: str):
    lines = split_content_lines(content)
    if not lines:
        st.info("No content.")
        return

    for line in lines:
        if r"\begin{aligned}" in line and r"\end{aligned}" in line:
            try:
                st.latex(normalize_latex_text(line))
            except Exception:
                st.code(line, language="latex")
            continue

        if "$" in line:
            st.markdown(f"- {line}")
            continue

        if ":" in line:
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip()

            if _is_latex_like(left):
                st.markdown(f"- ${left}$: {right}")
            else:
                st.markdown(f"- {line}")
            continue

        if _is_latex_like(line):
            try:
                st.latex(line)
            except Exception:
                st.code(line, language="latex")
            continue

        st.markdown(f"- {line}")

def render_math_model_content(content: str):
    text = normalize_latex_text(content)
    if not text:
        st.info("No content.")
        return

    try:
        st.latex(text)
    except Exception:
        st.code(text, language="latex")

def render_component(title: str, block: Dict[str, str], hits: List[Dict], show_source: bool = True):
    st.markdown(f"### {title}")

    content = block.get("content", "")

    if title == "Complete Mathematical Model":
        render_math_model_content(content)
    else:
        render_five_element_content(content)

    if show_source:
        source_text = block.get("source", "")
        with st.expander(f"{title} Source", expanded=False):
            render_source_blocks(source_text)


def render_component_original(title: str, block: Dict[str, str], hits: List[Dict], show_source: bool = True):
    st.markdown(f"### {title}")
    render_formula_text(block.get("content", ""))

    if show_source:
        source_text = block.get("source", "")
        with st.expander(f"{title} Source", expanded=False):
            render_source_blocks(source_text)


def build_tex_document(five: Dict[str, Dict[str, str]]) -> str:
    def norm(x: str) -> str:
        return normalize_latex_text(x)

    math_model = norm(five.get("Math Model", {}).get("content", ""))
    sets_ = norm(five.get("Sets", {}).get("content", ""))
    params = norm(five.get("Parameters", {}).get("content", ""))
    variables = norm(five.get("Variables", {}).get("content", ""))
    objective = norm(five.get("Objective", {}).get("content", ""))
    constraints = norm(five.get("Constraints", {}).get("content", ""))

    return rf"""
\documentclass{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage[margin=1in]{{geometry}}
\begin{{document}}

\section*{{Math Model}}
\[
{math_model}
\]

\section*{{Sets}}
\[
{sets_}
\]

\section*{{Parameters}}
\[
{params}
\]

\section*{{Variables}}
\[
{variables}
\]

\section*{{Objective}}
\[
{objective}
\]

\section*{{Constraints}}
\[
{constraints}
\]

\end{{document}}
""".strip()

def build_math_model_tex(five: Dict[str, Dict[str, str]]) -> str:
    math_model = normalize_latex_text(five.get("Math Model", {}).get("content", ""))

    return rf"""
\documentclass{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage[margin=1in]{{geometry}}
\begin{{document}}

\section*{{Math Model}}
\[
{math_model}
\]

\end{{document}}
""".strip()


def append_chat_record(question: str, selected_pdfs: List[str], hits: List[Dict], parsed: Dict, raw_text: str):
    rid = sha256_bytes(f"{time.time()}_{question}".encode())[:10]
    st.session_state["chat_records"].append(
        {
            "id": rid,
            "time_str": time.strftime("%m-%d %H:%M:%S"),
            "question": question,
            "selected_pdfs": selected_pdfs,
            "hits": hits,
            "parsed": normalize_source_aware_five_tuple(parsed),
            "raw_text": raw_text,
        }
    )


def render_chat_record(record: Dict):
    with st.chat_message("user"):
        st.markdown(
            f"**Question:** {record['question']}\n\n"
            f"**Selected documents:** {', '.join(record['selected_pdfs'])}"
        )

    with st.chat_message("assistant"):
        parsed = normalize_source_aware_five_tuple(record["parsed"])
        hits = record["hits"]

        st.caption(f"Generated at: {record['time_str']}")

        # 1) Five-element first (keep internal order unchanged)
        st.markdown("## Five-Element Model")
        for key, title in DISPLAY_ORDER[1:]:
            with st.container(border=True):
                render_component(title, parsed.get(key, {}), hits, show_source=True)

        # 2) Retrieved pages
        with st.expander("View Retrieved Pages", expanded=False):
            cols = st.columns(2)
            for i, h in enumerate(record["hits"]):
                with cols[i % 2]:
                    st.image(h["image_path"], use_container_width=True)
                    st.caption(
                        f"{h['source_pdf']} | Top {h['rank_in_doc']} | Page {h['page_num']} | score={h['score']:.4f}"
                    )

        # 3) Raw five-element output
        with st.expander("View Raw Five-Element Output", expanded=False):
            st.code(record["raw_text"], language="text")

        # 4) Math model display after raw text
        st.markdown("## Complete Mathematical Model")
        st.caption(
            "Below is the complete mathematical model synthesized from the five-element abstraction. "
            "You may manually revise this final math-model block."
        )
        with st.container(border=True):
            render_component("Complete Mathematical Model", parsed.get("Math Model", {}), hits, show_source=False)

        # 5) Edit only math model content
        with st.expander("Manually Edit the Mathematical Model", expanded=False):
            form_key = f"edit_math_model_form_{record['id']}"
            with st.form(form_key):
                math_model_content = st.text_area(
                    "Math Model Content",
                    value=parsed["Math Model"]["content"],
                    height=260,
                )
                submitted = st.form_submit_button("Update and Re-render")

            if submitted:
                record["parsed"]["Math Model"]["content"] = math_model_content
                st.success("Mathematical model updated.")
                st.rerun()

        tex_text = build_math_model_tex(parsed)
        st.download_button(
            "Download Math Model LaTeX (.tex)",
            data=tex_text,
            file_name=f"math_model_{record['id']}.tex",
            mime="text/plain",
            key=f"download_{record['id']}",
        )


st.set_page_config(page_title="LLMOPT ColPali App", layout="wide")
init_state()

st.title("📄 Multi-Document ColPali Retrieval + Five-Element QA")
st.caption(
    "The document pool is shared across the current session. "
    "For each round, manually select the PDFs to retrieve from. "
    "Supports index reuse, multi-turn QA, source display, and manual editing."
)

st.markdown("## Document Pool")
new_uploaded_files = st.file_uploader(
    "Upload PDF documents (multiple uploads supported)",
    type=["pdf"],
    accept_multiple_files=True,
)

if new_uploaded_files:
    add_uploaded_files_to_pool(new_uploaded_files)

doc_pool_names = get_doc_pool_names()

col1, col2 = st.columns([4, 1])
with col1:
    with st.expander("View Current Session Document Pool", expanded=False):
        if doc_pool_names:
            for name in doc_pool_names:
                st.markdown(f"- {name}")
        else:
            st.caption("No documents have been uploaded yet.")
with col2:
    if st.button("Clear Document Pool", use_container_width=True):
        clear_doc_pool()
        st.rerun()

st.markdown("---")

for record in st.session_state["chat_records"]:
    render_chat_record(record)

st.markdown("## New Question")
selected_names = st.multiselect(
    "Select PDFs for this round",
    options=doc_pool_names,
    default=doc_pool_names,
    key="current_round_selected_pdfs",
)

prompt = st.chat_input(
    "Enter a question for this round, for example: Extract the corresponding five-element model from the selected documents and provide sources for each component."
)

if prompt:
    if not st.session_state["doc_pool"]:
        st.warning("Please upload at least one PDF to the document pool first.")
        st.stop()

    if not selected_names:
        st.warning("Please select at least one PDF for this round.")
        st.stop()

    if not APP_CONFIG["api_key"]:
        st.error("No API key was detected. Please configure the required environment variable before running.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(
            f"**Question:** {prompt}\n\n"
            f"**Documents for this round:** {', '.join(selected_names)}"
        )

    with st.chat_message("assistant"):
        box = st.empty()
        logs = st.session_state["debug_logs"]
        box.markdown(
            "Retrieving relevant pages from the selected documents and generating a source-aware five-element model and complete mathematical model. Please wait..."
        )

        try:
            run_dir = build_run_dir(st.session_state["doc_pool"])
            st.session_state["current_run_dir"] = str(run_dir)
            log_debug(f"Run directory: {run_dir}", logs, run_dir)

            saved_docs = save_selected_pdfs_from_pool(selected_names, run_dir)
            log_debug(f"Selected document count: {len(saved_docs)}", logs, run_dir)

            retrieval_query = build_retrieval_query(prompt)
            hit_dicts = collect_hits_from_selected_docs(saved_docs, retrieval_query, run_dir)
            if not hit_dicts:
                raise RuntimeError("ColPali did not retrieve any valid pages.")

            log_debug(f"Retrieved page count: {len(hit_dicts)}", logs, run_dir)
            log_debug(f"LLM base_url: {APP_CONFIG['api_base']!r}", logs, run_dir)
            log_debug(f"LLM model: {APP_CONFIG['api_model']!r}", logs, run_dir)

            result = generate_five_elements_from_hit_images(
                question=prompt,
                hit_items=hit_dicts,
                api_key=APP_CONFIG["api_key"],
                model=APP_CONFIG["api_model"],
                base_url=APP_CONFIG["api_base"],
                temperature=APP_CONFIG["temperature"],
                max_tokens=APP_CONFIG["max_tokens"],
            )

            parsed = normalize_source_aware_five_tuple(result["parsed"])
            raw_text = result["raw_text"]

            append_chat_record(
                question=prompt,
                selected_pdfs=selected_names,
                hits=hit_dicts,
                parsed=parsed,
                raw_text=raw_text,
            )

            box.success("Generation completed. You may continue with the next question.")
            st.rerun()

        except Exception as e:
            run_dir_str = st.session_state.get("current_run_dir", "")
            run_dir = Path(run_dir_str) if run_dir_str else None
            log_exception(e, logs=logs, run_dir=run_dir)
            box.error(f"Run failed: {e}")