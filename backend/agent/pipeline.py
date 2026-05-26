"""
pipeline.py — Thin adapter: parses PDF input, creates LLM client, runs the agent loop.

The old fixed pipeline has been replaced by a pure ReAct agent (loop.py).
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

from openai import OpenAI

from .loop import run_agent, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ── OpenAI client singleton (keyed by api_key+model to support hot-swap) ────
_client_cache: dict[str, "OpenAI"] = {}

def _get_client(api_key: str) -> "OpenAI":
    if api_key not in _client_cache:
        _client_cache[api_key] = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client_cache[api_key]


def _build_query_parts_from_pdf_struct(pdf_struct: dict) -> list[str]:
    """Build human-readable query segments from structured PDF extraction."""
    if not pdf_struct:
        return []
    parts: list[str] = []
    if pdf_struct.get("title"):
        parts.append(f"[论文标题]: {pdf_struct['title']}")
    if pdf_struct.get("abstract"):
        parts.append(f"[摘要]:\n{pdf_struct['abstract']}")
    if pdf_struct.get("github_links"):
        parts.append(f"[论文代码仓库链接]: {', '.join(pdf_struct['github_links'])}")
    if pdf_struct.get("hf_links"):
        parts.append(f"[HuggingFace 数据集链接]: {', '.join(pdf_struct['hf_links'])}")
    if pdf_struct.get("dataset_section"):
        parts.append(f"[数据集/实验章节节选]:\n{pdf_struct['dataset_section']}")
    if pdf_struct.get("appendix_section"):
        parts.append(f"[附录数据集详情节选]:\n{pdf_struct['appendix_section']}")
    if pdf_struct.get("ref_part"):
        parts.append(f"[相关引用/数据链接]:\n{pdf_struct['ref_part']}")
    return parts


def _build_planner_input(
    text: str,
    user_query: str,
    pdf_struct: dict,
) -> str:
    """Build compact context for planner to reduce noise and token usage."""
    planner_input = text or ""
    if not pdf_struct:
        return planner_input or user_query

    planner_parts: list[str] = []
    if pdf_struct.get("title"):
        planner_parts.append(f"[论文标题]: {pdf_struct['title']}")
    if pdf_struct.get("abstract"):
        planner_parts.append(f"[摘要（前800字）]:\n{pdf_struct['abstract'][:800]}")
    if pdf_struct.get("github_links"):
        planner_parts.append(f"[代码仓库]: {', '.join(pdf_struct['github_links'])}")
    if text:
        planner_parts.append(f"[用户问题]: {text}")
    return "\n\n".join(planner_parts) if planner_parts else text or user_query


async def stream_agent_events(
    text: str = "",
    pdf_path: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
) -> AsyncGenerator[dict, None]:
    """
    Main event-level entry.

    Emits plain event dicts so callers can:
    - wrap as SSE strings (single-paper route)
    - merge/multiplex multiple papers in parallel (multi-paper route)
    """
    query_parts: list[str] = []
    pdf_struct: dict = {}

    if pdf_path:
        try:
            pdf_struct = await asyncio.to_thread(_extract_pdf_structured, pdf_path)
            query_parts.extend(_build_query_parts_from_pdf_struct(pdf_struct))
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")

    if text:
        query_parts.append(text)

    if not query_parts:
        yield {"event": "error", "message": "请提供问题或 PDF 文件"}
        return

    user_query = "\n\n".join(query_parts)

    effective_key = api_key.strip() or os.getenv("DASHSCOPE_API_KEY", "")
    if not effective_key:
        yield {"event": "error", "message": "API Key 未配置，请在页面顶部填写"}
        return

    client = _get_client(effective_key)

    from .loop import run_planner

    planner_input = _build_planner_input(text=text, user_query=user_query, pdf_struct=pdf_struct)
    plan = await run_planner(planner_input, client, model=model)
    yield {
        "event": "agent_thought",
        "turn": 0,
        "text": f"【规划阶段完成】\n\n```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```",
    }

    extended_query = (
        f"【用户原始问题/输入】\n{user_query}\n\n"
        f"【Planner 生成的执行计划】\n请严格参考以下行动建议与优先级：\n"
        f"{json.dumps(plan, ensure_ascii=False, indent=2)}"
    )

    HEARTBEAT_INTERVAL = 15
    queue: asyncio.Queue = asyncio.Queue()

    async def _producer():
        try:
            async for event_dict in run_agent(extended_query, client, model=model):
                await queue.put(event_dict)
        finally:
            await queue.put(None)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            try:
                event_dict = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL)
                if event_dict is None:
                    break
                yield event_dict
            except asyncio.TimeoutError:
                yield {"event": "heartbeat"}
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


async def stream_agent(
    text: str = "",
    pdf_path: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
) -> AsyncGenerator[str, None]:
    """
    Main entry point called by main.py.

    1. Optionally extract PDF text.
    2. Build user_query string.
    3. Create OpenAI client (DashScope-compatible).
    4. Run agent loop and yield SSE strings.
    """

    async for event in stream_agent_events(text=text, pdf_path=pdf_path, api_key=api_key, model=model):
        yield _sse(event)


def _extract_pdf_structured(pdf_path: str) -> dict:
    """
    Structured PDF extraction.
    Returns a dict with keys: title, abstract, dataset_section, appendix_section,
    github_links, hf_links, ref_part.

    Hard limits applied:
    - Max 30 pages processed
    - Max 100,000 chars of raw text
    - References section truncated at 3,000 chars
    """
    import re
    try:
        from pdfminer.high_level import extract_text_to_fp, extract_pages
        from pdfminer.layout import LAParams

        # ── Hard page limit: extract page by page, stop at 30 ────────────
        from pdfminer.high_level import extract_text
        # pdfminer doesn't have a native page-limit API; pass maxpages param
        text = extract_text(pdf_path, maxpages=30)
        # Hard char limit
        if len(text) > 100_000:
            text = text[:100_000]
        logger.info(f"PDF extracted: {len(text)} chars (page-limited to 30)")

        result: dict = {
            "title": "",
            "abstract": "",
            "dataset_section": "",
            "appendix_section": "",
            "github_links": [],
            "hf_links": [],
            "ref_part": "",
        }

        # ── Title ─────────────────────────────────────────────────────────
        _STOP_RE = re.compile(
            r'@|[{]|^\d+$'
            r'|\b(University|Institute|Laboratory|Department|School|College|Center|Centre)\b',
            re.IGNORECASE
        )
        clean_lines = [l.strip() for l in text[:5000].split('\n') if len(l.strip()) > 4]
        title_lines = []
        for line in clean_lines:
            if _STOP_RE.search(line): break
            words = line.split()
            if len(words) >= 8: break
            if line.lower() in ('abstract', 'introduction'): break
            if re.match(r'^[\w.\-:]+$', line) and len(words) <= 2: continue
            if ',' in line and len(words) < 6: break
            title_lines.append(line)
        if title_lines:
            result["title"] = ' '.join(title_lines)[:200]

        # ── Abstract (first 1500 chars) ───────────────────────────────────
        result["abstract"] = text[:1500]

        # ── GitHub & HF links from full text ─────────────────────────────
        gh_raw = re.findall(r'https?://github\.com/[\w\-]+/[\w\-]+', text)
        result["github_links"] = list(dict.fromkeys(u.rstrip('.,;)\'"') for u in gh_raw))[:5]
        hf_raw = re.findall(r'https?://huggingface\.co/datasets/[\w\-/]+', text)
        result["hf_links"] = list(dict.fromkeys(u.rstrip('.,;)\'"') for u in hf_raw))[:5]

        # ── Dataset / Experiment section ──────────────────────────────────
        SECTION_KEYWORDS = [
            r"datasets?\s+and\s+metrics",
            r"experimental\s+(setup|settings?|details)",
            r"\d+\.\d+\s+datasets?(?:\s+and)?",
            r"evaluation\s+datasets?",
            r"benchmarks?\s+and\s+(?:datasets?|metrics)",
            r"data\s+and\s+(?:setup|evaluation|experiments?)",
        ]
        best_match = None
        for kw in SECTION_KEYWORDS:
            m = re.search(kw, text, re.IGNORECASE)
            if m and (best_match is None or m.start() < best_match.start()):
                best_match = m
        if best_match:
            s = best_match.start()
            result["dataset_section"] = text[max(0, s - 200): min(len(text), s + 5000)]
            logger.info(f"Found dataset section at char {s}")
        else:
            result["dataset_section"] = text[1500:6500]

        # ── Appendix dataset details (second half only) ───────────────────
        half = len(text) // 2
        app_pats = [
            r"C\.1\s+Datasets?\s+Details",
            r"Appendix\s+C[^a-z]",
            r"Datasets?\s+Details\s*\n",
            r"Dataset\s+Statistics",
        ]
        for pat in app_pats:
            for m in re.finditer(pat, text[half:], re.IGNORECASE):
                app_start = half + m.start()
                snippet = text[app_start: app_start + 200]
                if re.search(r'(?:\.\s*){4,}|\.\.\.\s*\d+\s*$', snippet):
                    continue  # ToC entry, skip
                result["appendix_section"] = text[app_start: app_start + 2500]
                logger.info(f"Found appendix section at char {app_start}")
                break
            if result["appendix_section"]:
                break

        # ── References — only dataset-related entries, max 3000 chars ────
        ref_section = re.search(r'\bReferences\b', text, re.IGNORECASE)
        if ref_section:
            ref_text = text[ref_section.start(): ref_section.start() + 3000]
            ref_entries = re.split(r'\n\s*\n', ref_text)
            dataset_refs = []
            for entry in ref_entries:
                entry_c = entry.strip()
                if not entry_c:
                    continue
                if re.search(r'dataset|benchmark|corpus|competition|kdd\s+cup|challenge', entry_c, re.IGNORECASE):
                    dataset_refs.append(re.sub(r'\s+', ' ', entry_c)[:200])
            extra_urls = [u.rstrip('.,;)\'"') for u in re.findall(r'https?://\S+', ref_text)
                          if any(kw in u.lower() for kw in [
                              'github', 'zenodo', 'kaggle', 'drive.google', 'figshare', 'data.'])]
            if dataset_refs or extra_urls:
                result["ref_part"] = "\n".join(dataset_refs[:8])
                if extra_urls:
                    result["ref_part"] += "\nURLs: " + ", ".join(list(dict.fromkeys(extra_urls))[:6])

        logger.info(f"Structured extraction complete: title={bool(result['title'])}, "
                    f"gh={len(result['github_links'])}, hf={len(result['hf_links'])}")
        return result

    except Exception as e:
        logger.error(f"PDF structured extraction error: {e}")
        return {}


# Legacy alias kept for safety
def _extract_pdf_smart(pdf_path: str) -> str:
    struct = _extract_pdf_structured(pdf_path)
    parts = []
    if struct.get("title"):    parts.append(f"[论文标题]: {struct['title']}")
    if struct.get("abstract"): parts.append(struct["abstract"])
    if struct.get("github_links"): parts.append(f"[代码仓库]: {', '.join(struct['github_links'])}")
    if struct.get("dataset_section"): parts.append(struct["dataset_section"])
    return "\n\n".join(parts)

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
