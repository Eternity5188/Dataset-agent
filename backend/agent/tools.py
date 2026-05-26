"""
tools.py — Atomic tool primitives for the Dataset Agent.

Each function here represents a SINGLE external API call with minimal
post-processing.  They are the primitive operations that skills.py composes.

Tools (9):
  web_search              — DuckDuckGo general web search
  fetch_webpage_text      — HTTP fetch + HTML stripping (arXiv-aware)
  search_semantic_scholar — Semantic Scholar paper search (rate-limited)
  search_pwc_dataset      — GitHub repo search for dataset repos (by stars)
  search_zenodo           — Zenodo dataset search API
  search_opendatalab      — OpenDataLab (opendatalab.com) dataset search
  get_hf_dataset_configs  — HuggingFace dataset configs/subsets (single call)
  get_zenodo_record       — Zenodo record by record_id (single call)
  get_github_repo_info    — GitHub repo stars / topics / license (single call)

Shared infrastructure (consumed by skills.py):
  _HEADERS, HF_API, HF_DS_SERVER, GH_API, ZENODO_API
  _GH_API, _GH_HEADERS, _DATA_EXTS
  _S2_SEMAPHORE, _get_s2_semaphore
  SPLIT_KEYWORDS, _parse_yaml_frontmatter, _find_splits_section
"""

import asyncio
import logging
import os
import re
from typing import Optional

import httpx

from .config import get_api_key
from .searcher import _extract_data_links

logger = logging.getLogger(__name__)

# ── Shared HTTP constants ─────────────────────────────────────────────────────

_HEADERS      = {"User-Agent": "DatasetDiscoveryAgent/2.0"}
HF_API        = "https://huggingface.co/api"
HF_DS_SERVER  = "https://datasets-server.huggingface.co"
GH_API        = "https://api.github.com"
ZENODO_API    = "https://zenodo.org/api"

# GitHub Contents API
_GH_API     = "https://api.github.com"
_GH_HEADERS = {**_HEADERS, "Accept": "application/vnd.github.v3+json"}


def _hf_headers() -> dict:
    """HuggingFace headers with optional Authorization token from request context."""
    h = {**_HEADERS}
    token = get_api_key("huggingface")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_headers(accept: str = "application/vnd.github+json") -> dict:
    """GitHub headers with optional Bearer token from request context."""
    h = {"User-Agent": _HEADERS["User-Agent"], "Accept": accept}
    token = get_api_key("github")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

# Data file extensions recognised when browsing GitHub directories
_DATA_EXTS = {
    ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".pkl",
    ".txt", ".npz", ".npy", ".h5", ".hdf5", ".pt", ".pth",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".arrow", ".feather",
}

# ── Semantic Scholar rate-limit semaphore ─────────────────────────────────────

_S2_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_s2_semaphore() -> asyncio.Semaphore:
    """Lazily create the S2 semaphore on the running event loop (1 req/s)."""
    global _S2_SEMAPHORE
    if _S2_SEMAPHORE is None:
        _S2_SEMAPHORE = asyncio.Semaphore(1)
    return _S2_SEMAPHORE


# ── HF split keyword mapping ──────────────────────────────────────────────────

SPLIT_KEYWORDS: dict[str, str] = {
    "train":      "train",
    "training":   "train",
    "test":       "test",
    "eval":       "test",
    "held_out":   "test",
    "blind":      "test",
    "unseen":     "test",
    "ood":        "test",
    "validation": "validation",
    "dev":        "validation",
    "val":        "validation",
    "valid":      "validation",
    "in_domain":  "train",
}

# ── README parsing helpers ────────────────────────────────────────────────────


def _parse_yaml_frontmatter(text: str) -> dict:
    """
    Extract YAML frontmatter from a README (--- ... ---) and parse
    dataset_info.splits if present.  Pure-Python, no PyYAML required.
    Returns a dict with whatever we can extract.
    """
    fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not fm_match:
        return {}

    raw = fm_match.group(1)
    result: dict = {}

    # license
    lic = re.search(r"^license:\s*(.+)$", raw, re.MULTILINE)
    if lic:
        result["license"] = lic.group(1).strip()

    # language (list or scalar)
    lang_block = re.search(r"^language:\s*\[([^\]]+)\]", raw, re.MULTILINE)
    if lang_block:
        result["language"] = [l.strip().strip("'\"") for l in lang_block.group(1).split(",")]
    else:
        lang_items = re.findall(r"^language:\s*\n((?:\s+-\s+.+\n?)+)", raw, re.MULTILINE)
        if lang_items:
            result["language"] = re.findall(r"-\s+(.+)", lang_items[0])

    # task_categories
    task_items = re.findall(r"-\s+(task_categories:.+)", raw)
    if task_items:
        result["task_categories"] = [t.replace("task_categories:", "").strip() for t in task_items]

    # dataset_info.splits
    splits_block = re.search(
        r"splits:\s*\n((?:\s+-\s+name:.*\n(?:\s+\w+:.*\n)*)+)",
        raw,
    )
    if splits_block:
        parsed_splits = []
        for entry in re.finditer(
            r"-\s+name:\s*(\S+)(?:[^\-]*?num_examples:\s*(\d+))?",
            splits_block.group(1),
        ):
            s = {"name": entry.group(1)}
            if entry.group(2):
                s["num_examples"] = int(entry.group(2))
            parsed_splits.append(s)
        if parsed_splits:
            result["splits_from_yaml"] = parsed_splits

    return result


def _find_splits_section(text: str) -> str:
    """
    Find and return the 'Dataset Structure' or 'Splits' section of a README.
    Falls back to first 500 chars if not found.
    """
    patterns = [
        r"#{1,3}\s+(?:Dataset\s+)?(?:Structure|Splits|Data\s+Splits|splits)",
        r"#{1,3}\s+(?:数据集结构|划分|splits)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start = m.start()
            return text[start: start + 1200]
    return text[:500]


# ── Atomic tool implementations ───────────────────────────────────────────────


async def web_search(query: str, max_results: int = 6) -> dict:
    """
    General web search using DuckDuckGo (no API key required).
    Returns titles, URLs, and text snippets. Use fetch_webpage_text on the
    most relevant URLs to read full content.
    Best for: finding dataset homepages, official download pages, forum posts,
    GitHub repos, and any web resource not indexed by academic platforms.
    """
    try:
        from ddgs import DDGS

        def _sync_search() -> list:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        hits = await asyncio.to_thread(_sync_search)
        results = []
        for h in hits:
            results.append({
                "title":   h.get("title", ""),
                "url":     h.get("href", ""),
                "snippet": h.get("body", "")[:300],
            })
        return {
            "query":   query,
            "count":   len(results),
            "results": results,
            "hint":    "Call fetch_webpage_text on the most relevant URLs to read full content",
        }
    except Exception as e:
        return {"error": str(e), "results": []}


async def tavily_search(query: str, max_results: int = 6, search_depth: str = "basic") -> dict:
    """
    High-quality web search via Tavily (requires TAVILY_API_KEY).
    Much more reliable than DuckDuckGo for agent workflows — returns curated
    titles, URLs, content snippets, and (optionally) a synthesized answer.

    search_depth:
      - "basic": fast, cheap (default)
      - "advanced": better recall, more API credits used
    """
    api_key = get_api_key("tavily")
    if not api_key:
        return {
            "error": "Tavily 未配置：请在前端高级配置里填写 TAVILY_API_KEY 或设置环境变量。",
            "results": [],
            "hint": "未配置时请改用 web_search（DuckDuckGo）。",
        }

    payload = {
        "api_key":      api_key,
        "query":        query,
        "max_results":  max(1, min(int(max_results or 6), 10)),
        "search_depth": search_depth if search_depth in ("basic", "advanced") else "basic",
        "include_answer": True,
    }
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": _HEADERS["User-Agent"]}) as c:
            r = await c.post("https://api.tavily.com/search", json=payload)
            if r.status_code == 401:
                return {"error": "Tavily API key 无效", "results": []}
            if r.status_code == 429:
                return {"error": "Tavily 限流，请稍后重试或改用 web_search", "results": []}
            if r.status_code != 200:
                return {"error": f"Tavily HTTP {r.status_code}", "results": []}
            data = r.json()
            results = [
                {
                    "title":   hit.get("title", ""),
                    "url":     hit.get("url", ""),
                    "snippet": (hit.get("content") or "")[:300],
                    "score":   hit.get("score"),
                }
                for hit in (data.get("results") or [])[:payload["max_results"]]
            ]
            return {
                "query":   query,
                "count":   len(results),
                "answer":  (data.get("answer") or "")[:600],
                "results": results,
                "hint":    "Tavily 命中后可对最相关的 URL 调用 fetch_webpage_text 读取完整内容。",
            }
    except Exception as e:
        return {"error": str(e), "results": []}


async def fetch_webpage_text(url: str) -> dict:
    """
    Fetch plain text from any https URL. Strips HTML tags.
    Enhancement: arXiv URLs are redirected to export.arxiv.org for clean abstract text.
    Also extracts <meta> citation fields for structured output.
    """
    if not url.startswith(("https://", "http://")):
        return {"error": "Only http/https URLs supported", "text": ""}

    # arXiv special handling
    arxiv_id = None
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        url = f"https://export.arxiv.org/abs/{arxiv_id}"

    try:
        async with httpx.AsyncClient(timeout=14, headers=_HEADERS, follow_redirects=True) as c:
            r = await c.get(url)
            if r.status_code != 200:
                return {"url": url, "error": f"HTTP {r.status_code}", "text": ""}
            text = r.text
            ct   = r.headers.get("content-type", "")
            meta_fields: dict = {}

            if "html" in ct.lower():
                # Extract citation meta tags before stripping
                for field in ("citation_title", "citation_abstract", "citation_author",
                              "citation_arxiv_id", "citation_publication_date",
                              "description"):
                    m = re.search(
                        rf'<meta[^>]+name=["\']?{field}["\']?[^>]+content=["\']([^"\']+)["\']',
                        text, re.I,
                    )
                    if m:
                        meta_fields[field] = m.group(1).strip()

                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.I)
                text = re.sub(r"<style[^>]*>.*?</style>",   "", text, flags=re.DOTALL | re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

            result = {
                "url":          url,
                "content_type": ct,
                "char_length":  len(text),
                "text":         text[:5000],
            }
            if meta_fields:
                result["meta_fields"] = meta_fields
            if arxiv_id:
                result["arxiv_id"] = arxiv_id
            extracted = _extract_data_links(text)
            if any(extracted.values()):
                result["data_links"] = extracted
            return result

    except Exception as e:
        return {"url": url, "error": str(e), "text": ""}


async def search_semantic_scholar(query: str, limit: int = 5) -> dict:
    """
    Search Semantic Scholar for papers about a dataset.
    Papers often contain the official dataset release links and describe splits.

    Without an API key, S2 enforces strict rate limits (~100 req per 5 min for
    the whole IP); with an API key, the per-key quota is much higher and the
    rate-limit pause between requests can be relaxed.

    Set env var SEMANTIC_SCHOLAR_API_KEY to enable the higher quota.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  f"{query} dataset",
        "fields": "title,year,abstract,externalIds,openAccessPdf,fieldsOfStudy,citationCount",
        "limit":  limit,
    }
    api_key = get_api_key("semantic_scholar")
    headers = {**_HEADERS}
    if api_key:
        headers["x-api-key"] = api_key

    sem = _get_s2_semaphore()
    data = None
    last_error = ""
    last_status: Optional[int] = None
    for attempt in range(3):
        try:
            async with sem:
                async with httpx.AsyncClient(timeout=12, headers=headers) as c:
                    r = await c.get(url, params=params)
                last_status = r.status_code
                if r.status_code == 429:
                    wait = max(int(r.headers.get("Retry-After", 0)), 5 * (attempt + 1))
                    last_error = f"HTTP 429 rate limited (attempt {attempt+1}/3, wait {wait}s)"
                    await asyncio.sleep(wait)
                    continue
                if r.status_code != 200:
                    return {
                        "error": f"HTTP {r.status_code}",
                        "papers": [],
                        "hint": (
                            "Semantic Scholar 暂时不可用。可改用 web_search('<论文标题> arxiv') "
                            "或 search_dataset 继续推进。"
                        ),
                    }
                data = r.json()
                await asyncio.sleep(0.4 if api_key else 1.0)
                break
        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))  # 2s, 4s — fit within outer 60s
            continue
    if data is None:
        hint = (
            "Semantic Scholar 限流中（无 API Key 时限制较严）。"
            "建议改用 web_search 搜 '<论文标题> arxiv' / '<数据集名> github' 等关键词，"
            "或直接调 get_paper_code_repos(title=...) 跳过 S2。"
            if last_status == 429 or "429" in last_error
            else "Semantic Scholar 调用失败；改用 web_search 或 search_dataset 推进。"
        )
        return {"error": last_error or "S2 unreachable", "papers": [], "hint": hint}

    papers = []
    for paper in data.get("data", [])[:limit]:
        ext_ids  = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv", "")
        pdf_info = paper.get("openAccessPdf") or {}
        abstract = (paper.get("abstract") or "")[:500]

        paper_url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id else
            f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
        )
        papers.append({
            "title":          paper.get("title", ""),
            "year":           paper.get("year"),
            "url":            paper_url,
            "arxiv_id":       arxiv_id,
            "pdf_url":        pdf_info.get("url", ""),
            "abstract":       abstract,
            "citation_count": paper.get("citationCount", 0),
            "fields":         paper.get("fieldsOfStudy") or [],
        })

    return {
        "query":  query,
        "count":  len(papers),
        "papers": papers,
        "hint":   "Use fetch_webpage_text on paper URLs to find dataset download links",
    }


async def search_pwc_dataset(query: str, limit: int = 5) -> dict:
    """
    Search for datasets on GitHub (PWC API is no longer available).
    Returns repos likely containing the dataset, sorted by stars.
    """
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True, headers=_gh_headers()) as c:
            r = await c.get(
                "https://api.github.com/search/repositories",
                params={"q": f"{query} dataset", "sort": "stars", "per_page": limit},
            )
            if r.status_code != 200:
                return {"count": 0, "datasets": [], "error": f"GitHub search HTTP {r.status_code}"}
            items = r.json().get("items") or []
            datasets = [
                {
                    "name":            item["full_name"],
                    "url":             item["html_url"],
                    "description":     (item.get("description") or "")[:120],
                    "stars":           item.get("stargazers_count", 0),
                    "has_evaluations": False,
                    "test_split_hint": False,
                }
                for item in items
            ]
        return {
            "query":    query,
            "count":    len(datasets),
            "datasets": datasets,
            "hint":     "Results are GitHub repos. Call get_github_readme on promising ones to find dataset download links.",
        }
    except Exception as e:
        return {"count": 0, "datasets": [], "error": str(e)}


async def search_zenodo(query: str, size: int = 6) -> dict:
    """
    Search Zenodo directly for datasets by keyword.
    Returns up to `size` records with title, DOI, access type, and file list.
    """
    try:
        params = {
            "q":    f'"{query}" AND resource_type.type:dataset',
            "size": size,
            "sort": "mostviewed",
        }
        async with httpx.AsyncClient(timeout=12, headers=_HEADERS) as c:
            r = await c.get("https://zenodo.org/api/records", params=params)
            if r.status_code != 200:
                return {"error": f"Zenodo HTTP {r.status_code}"}
            hits = r.json().get("hits", {}).get("hits", [])
            results = []
            for h in hits:
                meta  = h.get("metadata", {})
                rid   = h.get("id")
                files = []
                for f in h.get("files", [])[:5]:
                    files.append({
                        "name": f.get("key", ""),
                        "size": f.get("size"),
                        "download_url": (f.get("links") or {}).get("self") or "",
                    })
                results.append({
                    "record_id": str(rid),
                    "title":     meta.get("title", ""),
                    "doi":       h.get("doi", ""),
                    "access":    meta.get("access_right", ""),
                    "license":   (meta.get("license") or {}).get("id", ""),
                    "files":     files,
                    "url":       f"https://zenodo.org/record/{rid}",
                })
            return {
                "total":   r.json().get("hits", {}).get("total", 0),
                "results": results,
            }
    except Exception as e:
        return {"error": str(e)}


async def search_opendatalab(query: str) -> dict:
    """
    Search OpenDataLab (opendatalab.com) — a major Chinese AI dataset platform
    maintained by Shanghai AI Lab. Covers CV, NLP, autonomous driving datasets.
    Returns dataset names, descriptions, and links.
    """
    try:
        api_url = "https://opendatalab.com/api/dataset/list"
        params  = {"query": query, "page": 1, "pageSize": 8}
        async with httpx.AsyncClient(timeout=12, headers=_HEADERS) as c:
            r = await c.get(api_url, params=params)
            if r.status_code == 200:
                data  = r.json()
                items = data.get("data", {}).get("list", data.get("list", []))
                results = []
                for item in items[:6]:
                    name = item.get("name") or item.get("title") or ""
                    slug = item.get("slug") or item.get("id") or name
                    results.append({
                        "name":        name,
                        "description": (item.get("description") or "")[:200],
                        "url":         f"https://opendatalab.com/OpenDataLab/{slug}",
                        "tags":        item.get("tags", [])[:5],
                    })
                if results:
                    return {"total": len(results), "results": results}

        # Fallback: scrape the search page
        search_url = f"https://opendatalab.com/search?q={query}"
        async with httpx.AsyncClient(timeout=12, headers=_HEADERS, follow_redirects=True) as c:
            r = await c.get(search_url)
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text).strip()
            return {"search_url": search_url, "text_excerpt": text[:2000]}

    except Exception as e:
        return {"error": str(e)}


async def get_hf_dataset_configs(dataset_id: str) -> dict:
    """Get configs/subsets of a HuggingFace dataset (e.g. language variants, domains)."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=_hf_headers()) as c:
            r = await c.get(f"{HF_DS_SERVER}/configs", params={"dataset": dataset_id})
            if r.status_code == 200:
                data    = r.json()
                configs = data.get("configs", [])
                return {
                    "dataset_id":  dataset_id,
                    "num_configs": len(configs),
                    "configs":     [{"config_name": cfg["config_name"]} for cfg in configs[:15]],
                }
    except Exception as e:
        logger.debug(f"HF dataset configs failed ({dataset_id}): {e}")
    return {"dataset_id": dataset_id, "num_configs": 0, "configs": []}


async def get_zenodo_record(record_id: str) -> dict:
    """Get Zenodo record metadata (DOI, license, files, creators) by numeric record ID."""
    try:
        async with httpx.AsyncClient(timeout=8, headers=_HEADERS) as c:
            r = await c.get(f"{ZENODO_API}/records/{record_id}")
            if r.status_code == 200:
                data  = r.json()
                meta  = data.get("metadata", {})
                files = data.get("files", [])
                return {
                    "record_id":   record_id,
                    "doi":         data.get("doi"),
                    "title":       meta.get("title"),
                    "description": (meta.get("description") or "")[:600],
                    "access":      meta.get("access_right"),
                    "license":     (meta.get("license") or {}).get("id"),
                    "creators":    [cr.get("name") for cr in meta.get("creators", [])[:4]],
                    "file_count":  len(files),
                    "files": [
                        {
                            "name": f.get("key"),
                            "size": f.get("size"),
                            "download_url": (f.get("links") or {}).get("self") or "",
                        }
                        for f in files[:6]
                    ],
                    "url":         f"https://zenodo.org/record/{record_id}",
                }
    except Exception as e:
        logger.debug(f"Zenodo record fetch failed ({record_id}): {e}")
    return {"record_id": record_id, "error": "fetch failed"}


async def get_github_repo_info(owner: str, repo: str) -> dict:
    """Get GitHub repo metadata: stars, description, topics, license, last update."""
    gh_h = _gh_headers()
    try:
        async with httpx.AsyncClient(timeout=8, headers=gh_h) as c:
            repo_r, topics_r = await asyncio.gather(
                c.get(f"{GH_API}/repos/{owner}/{repo}"),
                c.get(
                    f"{GH_API}/repos/{owner}/{repo}/topics",
                    headers={**gh_h, "Accept": "application/vnd.github.mercy-preview+json"},
                ),
                return_exceptions=True,
            )
            if not isinstance(repo_r, Exception) and repo_r.status_code == 200:
                data   = repo_r.json()
                topics = (
                    topics_r.json().get("names", [])
                    if not isinstance(topics_r, Exception) and topics_r.status_code == 200
                    else []
                )
                return {
                    "owner":       owner,
                    "repo":        repo,
                    "full_name":   data.get("full_name"),
                    "description": data.get("description"),
                    "stars":       data.get("stargazers_count", 0),
                    "forks":       data.get("forks_count", 0),
                    "language":    data.get("language"),
                    "topics":      topics,
                    "license":     (data.get("license") or {}).get("spdx_id"),
                    "updated_at":  data.get("updated_at"),
                    "url":         data.get("html_url"),
                }
    except Exception as e:
        logger.debug(f"GitHub repo info failed ({owner}/{repo}): {e}")
    return {"owner": owner, "repo": repo, "error": "fetch failed"}
