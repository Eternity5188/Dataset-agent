"""
skills.py — Composed skills for the Dataset Agent.

Each skill orchestrates multiple atomic tools or applies significant business
logic (multi-step search, structured parsing, cross-platform aggregation).

The TOOL_DEFINITIONS and TOOL_FUNCTIONS registries at the bottom of this file
expose ALL callable functions (from both tools.py and this file) to the agent.

Skills (11):
  search_dataset       — multi-platform orchestration (HF/PWC/GitHub/Zenodo/Kaggle)
  search_hf_hub        — exact ID lookup + fuzzy HuggingFace search (combined)
  get_hf_metadata      — parallel splits + info fetch + synthesis
  get_hf_dataset_card  — HF README + YAML frontmatter parsing + section extraction
  get_hf_dataset_files — HF file listing → canonical split name inference
  get_github_readme    — GitHub README + structured link extraction + section prioritization
  get_github_dir       — GitHub directory traversal + data file classification
  get_gdrive_folder    — Google Drive folder listing via embedded view HTML parsing
  compare_datasets     — batch HF metadata for multiple dataset IDs
  get_paper_code_repos — multi-step: optional S2 title lookup → GitHub repo search
  finish               — terminal action: submit final result to user

Atomic tools (9) are defined in tools.py and re-exported here for registration.
"""

import asyncio
import logging
import re
from typing import Any, Optional

import httpx

from .searcher import MultiSourceSearcher, _extract_data_links
from .reader import DatasetPageReader
from .tools import (
    # atomic functions — re-exported for TOOL_FUNCTIONS registration
    web_search, fetch_webpage_text, tavily_search,
    search_semantic_scholar, search_pwc_dataset, search_zenodo, search_opendatalab,
    get_hf_dataset_configs, get_zenodo_record, get_github_repo_info,
    # shared infrastructure consumed by skill implementations below
    _HEADERS, HF_API, HF_DS_SERVER,
    _GH_API, _GH_HEADERS, _DATA_EXTS,
    _hf_headers, _gh_headers,
    SPLIT_KEYWORDS, _parse_yaml_frontmatter, _find_splits_section,
    _get_s2_semaphore,
)
from .config import get_api_key

logger = logging.getLogger(__name__)

_searcher = MultiSourceSearcher()
_reader   = DatasetPageReader()


# ── Skill implementations ─────────────────────────────────────────────────────

async def search_dataset(query: str, sources: Optional[list] = None) -> dict:
    """Search for a dataset across multiple platforms (broad first pass)."""
    if not sources:
        sources = ["HuggingFace", "PapersWithCode", "GitHub", "Zenodo", "Kaggle", "OpenML"]

    spec = {
        "raw_name": query, "canonical": query, "resolved": False,
        "kb_entry": None, "match_type": "none", "match_confidence": 0.0,
        "aliases": [], "context": "", "domain": None,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    camel_spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", query).strip()
    query_variants = [query]
    if slug:
        query_variants.append(slug)
        query_variants.append(slug.replace("-", "_"))
        query_variants.append(f"{slug} dataset")
    if camel_spaced and camel_spaced.lower() != query.lower():
        query_variants.append(camel_spaced)
        query_variants.append(f"{camel_spaced} dataset")
    # Keep user-intent friendly long-tail variants to boost recall in sparse cases
    query_variants.extend([
        f"{query} dataset",
        f"{query} benchmark",
    ])
    dedup_variants: list[str] = []
    seen_lower: set[str] = set()
    for item in query_variants:
        normalized = re.sub(r"\s+", " ", (item or "")).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        dedup_variants.append(normalized)

    plan = {
        "primary_name":   query,
        "query_variants": dedup_variants[:12],
        "source_priority": sources,
        "strategy":       "broad",
    }

    try:
        results = await _searcher.search_all([spec], [plan])
    except Exception as e:
        logger.warning(f"search_dataset error: {e}")
        return {"found": False, "error": str(e), "links": []}

    if not results:
        return {"found": False, "links": [], "sources_hit": []}

    r     = results[0]
    links = r.get("search", {}).get("links", []) or r.get("links", [])
    live  = [l for l in links if l.get("status") == "live"]

    # Collect HF IDs from live links AND from GitHub README extractions
    hf_ids = []
    for l in links:
        m = re.search(r"huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)", l.get("url", ""))
        if m:
            hf_ids.append(m.group(1))
        # Also harvest IDs embedded in GitHub search results
        for gh_hf_id in (l.get("extra") or {}).get("hf_ids", []):
            hf_ids.append(gh_hf_id)

    # Collect PWC evaluation signals
    has_evaluations_signal = any(
        (l.get("extra") or {}).get("has_evaluations", False)
        for l in links if l.get("source") == "PapersWithCode"
    )

    return {
        "found":                  len(live) > 0,
        "live_count":             len(live),
        "sources_hit":            r.get("search", {}).get("sources_hit", []) or r.get("sources_hit", []),
        "hf_ids_found":           list(dict.fromkeys(hf_ids))[:4],
        "has_evaluations_signal": has_evaluations_signal,
        "query_variants_used":    plan["query_variants"],
        "links": [
            {
                "url":    l["url"],
                "label":  l["label"],
                "source": l["source"],
                "extra":  {
                    k: v for k, v in (l.get("extra") or {}).items()
                    if k in ("hf_ids", "zenodo_ids", "arxiv_ids", "has_evaluations", "exact_match")
                },
            }
            for l in links[:8]
        ],
    }


async def search_hf_hub(
    query: str,
    task: Optional[str]     = None,
    language: Optional[str] = None,
    limit: int  = 8,
    sort: str   = "downloads",
) -> dict:
    """
    Search HuggingFace Hub directly.
    Enhancement: if query looks like 'owner/name', attempt exact ID lookup first.
    Returns rich per-dataset info.
    """
    # Exact ID lookup
    exact_result = None
    if "/" in query and re.match(r"^[\w\-\.]+/[\w\-\.]+$", query.strip()):
        try:
            async with httpx.AsyncClient(timeout=8, headers=_hf_headers()) as c:
                r = await c.get(f"{HF_API}/datasets/{query.strip()}")
                if r.status_code == 200:
                    ds     = r.json()
                    ds_id  = ds.get("id", query.strip())
                    tags   = ds.get("tags", [])
                    card   = ds.get("cardData") or {}
                    exact_result = {
                        "id":            ds_id,
                        "url":           f"https://huggingface.co/datasets/{ds_id}",
                        "downloads":     ds.get("downloads", 0),
                        "likes":         ds.get("likes", 0),
                        "license":       card.get("license"),
                        "tasks":         [t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")],
                        "languages":     [t.replace("language:", "") for t in tags if t.startswith("language:")],
                        "last_modified": ds.get("lastModified", ""),
                        "gated":         ds.get("gated", False),
                        "exact_match":   True,
                    }
        except Exception as e:
            logger.debug(f"HF exact lookup failed for '{query}': {e}")

    # Fuzzy search
    params = {"search": query, "limit": limit, "sort": sort, "direction": -1}
    if task:
        params["filter"] = task
    if language:
        params["language"] = language

    try:
        async with httpx.AsyncClient(timeout=10, headers=_hf_headers()) as c:
            r = await c.get(f"{HF_API}/datasets", params=params)
            if r.status_code != 200:
                fuzzy_datasets = []
            else:
                items = r.json()
                fuzzy_datasets = []
                exact_id = (exact_result or {}).get("id", "").lower()
                for ds in items[:limit]:
                    ds_id = ds.get("id", "")
                    if ds_id.lower() == exact_id:
                        continue  # already have it as exact result
                    tags = ds.get("tags", [])
                    card = ds.get("cardData") or {}
                    fuzzy_datasets.append({
                        "id":            ds_id,
                        "url":           f"https://huggingface.co/datasets/{ds_id}",
                        "downloads":     ds.get("downloads", 0),
                        "likes":         ds.get("likes", 0),
                        "license":       card.get("license"),
                        "tasks":         [t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")],
                        "languages":     [t.replace("language:", "") for t in tags if t.startswith("language:")],
                        "last_modified": ds.get("lastModified", ""),
                        "gated":         ds.get("gated", False),
                    })
    except Exception as e:
        return {"error": str(e), "datasets": []}

    datasets = ([exact_result] if exact_result else []) + fuzzy_datasets

    # Filter out obvious shell/fork repos (downloads < 5 AND likes == 0),
    # but keep exact_match results regardless
    filtered = []
    filtered_out = 0
    for ds in datasets:
        is_exact = ds.get("exact_match", False)
        dl = ds.get("downloads", 0) or 0
        likes = ds.get("likes", 0) or 0
        if not is_exact and dl < 5 and likes == 0:
            filtered_out += 1
            continue
        filtered.append(ds)

    result = {"query": query, "count": len(filtered), "datasets": filtered[:limit]}
    if filtered_out:
        result["filtered_out"] = filtered_out
        result["filter_reason"] = "downloads<5 且 likes=0 的空壳/fork 仓库已过滤"
    return result


async def get_hf_metadata(dataset_id: str) -> dict:
    """Get HuggingFace dataset metadata: splits, downloads, license."""
    info, splits = await asyncio.gather(
        _reader._fetch_hf_info(dataset_id),
        _reader._fetch_hf_splits(dataset_id),
        return_exceptions=True,
    )
    if isinstance(info,   Exception): info   = {}
    if isinstance(splits, Exception): splits = []

    # Siblings fallback: when splits API returns empty, infer from file names
    splits_source = "api" if splits else "none"
    if not splits and isinstance(info, dict) and info:
        siblings = info.get("siblings", [])
        if siblings:
            from .tools import SPLIT_KEYWORDS
            inferred: set[str] = set()
            for sib in siblings:
                fname = (sib.get("rfilename") or "").lower()
                for raw_kw, canonical in SPLIT_KEYWORDS.items():
                    if raw_kw in fname:
                        inferred.add(canonical)
            if inferred:
                splits = sorted(inferred)
                splits_source = "siblings"

    result = {
        "dataset_id":    dataset_id,
        "hf_url":        f"https://huggingface.co/datasets/{dataset_id}",
        "splits":        splits,
        "splits_source": splits_source,
        "has_train":     "train" in (splits or []),
        "has_test":      bool({"test", "validation"} & set(splits or [])),
        "downloads":     None,
        "license":       None,
    }
    if isinstance(info, dict) and info:
        dl = info.get("downloads") or info.get("downloadsAllTime")
        result["downloads"] = int(dl) if dl else None
        card = info.get("cardData") or {}
        result["license"] = card.get("license")
    return result


async def get_hf_dataset_card(dataset_id: str) -> dict:
    """
    Fetch the full HuggingFace dataset card (README.md).
    Enhancement:
      - Parse YAML frontmatter → splits_from_yaml (structured split names + sizes)
      - Find "Dataset Structure"/"Splits" section for targeted excerpt
      - Return license / language / task_categories from frontmatter directly
    """
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/README.md"
    try:
        async with httpx.AsyncClient(timeout=12, headers=_hf_headers(), follow_redirects=True) as c:
            r = await c.get(url)
            if r.status_code == 200:
                text = r.text

                # Parse frontmatter
                fm = _parse_yaml_frontmatter(text)

                # Find structure/splits section for excerpt
                splits_excerpt = _find_splits_section(text)

                # Also include beginning of main body (after frontmatter)
                body_start = text.find("\n---", 3)
                body_start = body_start + 4 if body_start != -1 else 0
                body_excerpt = text[body_start: body_start + 2000].strip()

                return {
                    "dataset_id":       dataset_id,
                    "has_card":         True,
                    "char_length":      len(text),
                    # Structured frontmatter fields
                    "license":          fm.get("license"),
                    "languages":        fm.get("language", []),
                    "task_categories":  fm.get("task_categories", []),
                    "splits_from_yaml": fm.get("splits_from_yaml", []),
                    # Targeted excerpts
                    "splits_section":   splits_excerpt,
                    "body_excerpt":     body_excerpt[:1500],
                    # Legacy booleans
                    "has_train_section": bool(re.search(r"\btrain\b",                    text, re.I)),
                    "has_splits_info":   bool(re.search(r"splits?|partition",            text, re.I)),
                    "has_citation":      bool(re.search(r"bibtex|citation|@article|@inproceedings", text, re.I)),
                    "has_license_info":  bool(re.search(r"licen[sc]e",                  text, re.I)),
                }
    except Exception as e:
        logger.debug(f"Dataset card fetch failed ({dataset_id}): {e}")
    return {"dataset_id": dataset_id, "has_card": False, "excerpt": ""}


async def get_hf_dataset_files(dataset_id: str) -> dict:
    """
    List files in a HuggingFace dataset repo; infer split structure from filenames.
    Enhancement: extended SPLIT_KEYWORDS covering val/held_out/blind/unseen/ood etc.
    Returns canonical split names (train/validation/test) rather than raw file names.
    """
    try:
        async with httpx.AsyncClient(timeout=10, headers=_hf_headers()) as c:
            r = await c.get(f"{HF_API}/datasets/{dataset_id}/tree/main")
            if r.status_code == 200:
                all_files  = r.json()
                data_exts  = {"parquet", "csv", "json", "jsonl", "arrow", "tsv", "txt", "gz"}
                data_files = []
                raw_hints:       set[str] = set()
                canonical_hints: set[str] = set()

                for f in all_files:
                    if f.get("type") != "file":
                        continue
                    path = f.get("path", "")
                    ext  = path.rsplit(".", 1)[-1].lower().rstrip("gz").rstrip(".")
                    if ext in data_exts or any(path.endswith("." + e) for e in data_exts):
                        size = f.get("size", 0)
                        data_files.append({"path": path, "size_bytes": size})
                        path_lower = path.lower()
                        for raw_kw, canonical in SPLIT_KEYWORDS.items():
                            if raw_kw in path_lower:
                                raw_hints.add(raw_kw)
                                canonical_hints.add(canonical)

                return {
                    "dataset_id":              dataset_id,
                    "total_files":             len(all_files),
                    "data_files":              data_files[:24],
                    "split_hints_from_files":  sorted(canonical_hints),
                    "raw_split_keywords_found": sorted(raw_hints),
                    "has_train_files":         "train" in canonical_hints,
                    "has_test_files":          "test"  in canonical_hints,
                    "has_validation_files":    "validation" in canonical_hints,
                }
    except Exception as e:
        logger.debug(f"HF dataset files failed ({dataset_id}): {e}")
    return {"dataset_id": dataset_id, "data_files": [], "error": "fetch failed"}


async def get_github_readme(owner: str, repo: str) -> dict:
    """
    Get GitHub repository README.
    Enhancements:
      - URL normalization: strips /tree/..., /blob/..., /issues, /pull, /releases paths
      - 4000-char excerpt (was 2000)
      - Structured data_links: HF dataset IDs, Zenodo record IDs, arXiv IDs
      - has_test_hint, has_validation_hint added alongside existing has_train_hint
      - Prioritized excerpt: if a "data" / "download" / "splits" section found, show that
    """
    # Normalize: strip sub-paths like /tree/main, /blob/master/..., /issues, /releases
    _strip = re.compile(r'/(tree|blob|issues|pull|releases|actions|discussions|commit|compare|wiki)(/?.*)?$')
    owner = re.sub(r'[^A-Za-z0-9_.\-]', '', owner)
    repo  = re.sub(r'[^A-Za-z0-9_.\-]', '', repo.split('/')[0])  # guard against owner/repo/extra

    readme = await _reader._fetch_github_readme(owner, repo)
    if not readme:
        return {"owner": owner, "repo": repo, "has_content": False, "excerpt": "", "data_links": {}}

    lower = readme.lower()

    train_hint = any(kw in lower for kw in [
        "train split", "training data", "train.json", "train.csv",
        "train.parquet", "training set", "trainset",
    ])
    test_hint = any(kw in lower for kw in [
        "test split", "test set", "testset", "test.json", "test.csv",
        "test.parquet", "evaluation set", "held-out",
    ])
    validation_hint = any(kw in lower for kw in [
        "validation", "val split", "dev split", "dev.json", "val.json",
        "development set",
    ])

    # Extract structured links
    data_links = _extract_data_links(readme)

    # Find dataset/download section for prioritized excerpt
    section_excerpt = ""
    for section_pat in [
        r"#{1,3}\s+(?:Data|Dataset|Download|Splits|Usage)",
        r"#{1,3}\s+(?:数据|下载|使用)",
    ]:
        m = re.search(section_pat, readme, re.IGNORECASE)
        if m:
            section_excerpt = readme[m.start(): m.start() + 1500]
            break

    excerpt = section_excerpt if section_excerpt else readme[:4000]

    return {
        "owner":              owner,
        "repo":               repo,
        "has_content":        True,
        "has_train_hint":     train_hint,
        "has_test_hint":      test_hint,
        "has_validation_hint": validation_hint,
        "data_links":         data_links,   # {hf_ids, zenodo, arxiv, gh_repos}
        "excerpt":            excerpt,
    }


async def get_github_dir(repo_path: str, path: str = "") -> dict:
    """
    List files in a GitHub repository directory (or read a small file's content).
    repo_path: 'owner/repo'  OR  full GitHub URL like
               https://github.com/owner/repo/tree/main/data
    path:      sub-path inside the repo (overrides path parsed from URL).
    """
    import base64

    # Parse repo_path if it's a full URL
    m = re.match(r"https?://github\.com/([\w\-]+/[\w\-]+)(?:/tree/[^/]+/(.*))?", repo_path)
    if m:
        owner_repo = m.group(1)
        url_path   = (m.group(2) or "").rstrip("/")
        if not path:
            path = url_path
    else:
        owner_repo = repo_path.strip("/")

    api_url = f"{_GH_API}/repos/{owner_repo}/contents/{path}"
    try:
        async with httpx.AsyncClient(timeout=12, headers=_gh_headers("application/vnd.github.v3+json"), follow_redirects=True) as c:
            r = await c.get(api_url)
            if r.status_code == 404:
                return {"error": f"路径 '{path}' 不存在于 {owner_repo}", "repo": owner_repo}
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "repo": owner_repo}

            data = r.json()

            # ── Directory listing ──────────────────────────────────────────
            if isinstance(data, list):
                items = []
                for item in data:
                    entry = {
                        "name":         item["name"],
                        "type":         item["type"],   # "file" or "dir"
                        "size":         item.get("size", 0),
                        "path":         item["path"],
                        "download_url": item.get("download_url"),
                    }
                    items.append(entry)

                data_files = [
                    it for it in items
                    if it["type"] == "file"
                    and any(it["name"].lower().endswith(e) for e in _DATA_EXTS)
                ]
                sub_dirs = [it for it in items if it["type"] == "dir"]

                return {
                    "repo":       owner_repo,
                    "path":       path or "/",
                    "type":       "directory",
                    "file_count": len(items),
                    "items":      items[:60],
                    "data_files": data_files,
                    "sub_dirs":   [d["name"] for d in sub_dirs],
                }

            # ── Single file ────────────────────────────────────────────────
            if isinstance(data, dict) and data.get("type") == "file":
                content_b64 = data.get("content", "")
                preview = ""
                if content_b64 and data.get("size", 0) < 512_000:  # < 512 KB
                    try:
                        raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                        # For CSV/TSV show header + first few rows
                        lines = raw.split("\n")
                        preview = "\n".join(lines[:30])
                    except Exception:
                        preview = "(binary file)"
                return {
                    "repo":         owner_repo,
                    "path":         path,
                    "type":         "file",
                    "name":         data["name"],
                    "size":         data.get("size", 0),
                    "download_url": data.get("download_url"),
                    "content_preview": preview[:3000],
                }

    except Exception as e:
        return {"repo": owner_repo, "path": path, "error": str(e)}

    return {"repo": owner_repo, "path": path, "error": "unexpected response"}


async def get_gdrive_folder(url: str) -> dict:
    """
    List files inside a public Google Drive folder using the embedded folder view
    endpoint (no JavaScript required). Extracts file names and detects dataset files.
    """
    # Extract folder ID from various URL formats
    folder_id = None
    m = re.search(r"/folders/([\w-]+)", url)
    if m:
        folder_id = m.group(1)
    else:
        m = re.search(r"[?&]id=([\w-]+)", url)
        if m:
            folder_id = m.group(1)

    if not folder_id:
        return {"error": "无法从 URL 中提取 folder ID", "url": url}

    embed_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    try:
        async with httpx.AsyncClient(timeout=15, headers=_HEADERS, follow_redirects=True) as c:
            r = await c.get(embed_url)
            if r.status_code != 200:
                return {"folder_id": folder_id, "error": f"HTTP {r.status_code}", "url": embed_url}

            html = r.text

            # Check for login redirect (private folder)
            if "accounts.google.com" in r.url.host if hasattr(r.url, 'host') else "accounts.google.com" in str(r.url):
                return {"folder_id": folder_id, "error": "文件夹需要登录（非公开）", "url": url}

            # Parse file names: the embedded view uses class="flip-entry-title"
            file_names: list[str] = []
            for m in re.finditer(r'class="flip-entry-title"[^>]*>(.*?)</div>', html, re.DOTALL):
                name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                if name:
                    file_names.append(name)

            # Fallback: aria-label on entries
            if not file_names:
                file_names = [
                    n for n in re.findall(r'aria-label="([^"]+)"', html)
                    if not n.startswith(("Folder", "Google Drive"))
                ]

            # Fallback: plain text scan
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain).strip()

            # Classify dataset-like files
            data_exts = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".pkl",
                         ".txt", ".npz", ".npy", ".h5", ".zip", ".tar", ".gz"}
            data_files = [f for f in file_names
                          if any(f.lower().endswith(e) for e in data_exts)]

            return {
                "folder_id":   folder_id,
                "url":         url,
                "file_count":  len(file_names),
                "files":       file_names[:50],        # cap at 50
                "data_files":  data_files,
                "text_excerpt": plain[:1000] if not file_names else "",
            }
    except Exception as e:
        return {"folder_id": folder_id, "url": url, "error": str(e)}


async def compare_datasets(dataset_ids: list[str]) -> dict:
    """
    NEW: Batch-fetch metadata for multiple HuggingFace dataset IDs and return
    a side-by-side comparison. Useful when the agent has multiple candidates and
    wants to quickly pick the best one.
    """
    if not dataset_ids:
        return {"error": "No dataset IDs provided", "datasets": []}
    if len(dataset_ids) > 8:
        dataset_ids = dataset_ids[:8]

    results = await asyncio.gather(
        *[get_hf_metadata(did) for did in dataset_ids],
        return_exceptions=True,
    )
    comparison = []
    for did, res in zip(dataset_ids, results):
        if isinstance(res, Exception):
            comparison.append({"dataset_id": did, "error": str(res)})
        else:
            comparison.append(res)

    return {
        "count":    len(comparison),
        "datasets": comparison,
        "hint":     "Compare splits/downloads/license across candidates",
    }


async def get_paper_code_repos(arxiv_id: str = "", title: str = "") -> dict:
    """
    Find code repositories for a paper on GitHub.
    Uses the paper title (or fetches it from Semantic Scholar) to search GitHub for repos.
    Call get_github_readme on the returned repos to find dataset download links.

    At least one of `arxiv_id` or `title` must be provided.
    """
    import re as _re
    arxiv_id = (arxiv_id or "").strip()
    title = (title or "").strip()
    arxiv_clean = _re.sub(r'v\d+$', '', arxiv_id)

    if not arxiv_id and not title:
        return {
            "arxiv_id": "",
            "repos": [],
            "error": "请提供 arxiv_id 或 title 至少一个参数。",
            "hint": "示例：get_paper_code_repos(title='Paper Title Here')",
        }

    gh_headers = _gh_headers()
    s2_api_key = get_api_key("semantic_scholar")
    s2_headers = {"User-Agent": "DatasetAgent/1.0"}
    if s2_api_key:
        s2_headers["x-api-key"] = s2_api_key
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
            # Step 1: get paper title if not provided
            paper_title = title
            if not paper_title and arxiv_clean:
                sem = _get_s2_semaphore()
                async with sem:
                    for attempt in range(3):
                        rs = await c.get(
                            f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_clean}",
                            params={"fields": "title"},
                            headers=s2_headers,
                        )
                        if rs.status_code == 200:
                            paper_title = rs.json().get("title", "")
                            await asyncio.sleep(0.4 if s2_api_key else 1.1)
                            break
                        if rs.status_code != 429:
                            break
                        await asyncio.sleep(5 * (attempt + 1))

            if not paper_title:
                return {
                    "arxiv_id": arxiv_id,
                    "repos": [],
                    "hint": (
                        "无法从 arXiv ID 获取论文标题（可能是 S2 限流或 ID 错误）。"
                        "请直接传入 title 参数重试，例如 "
                        "get_paper_code_repos(title='完整论文标题')。"
                    ),
                }

            # Step 2: build search query from title
            # Strategy: acronym from method name (before colon) + domain keywords (after colon)
            _stopwords = {'and','the','for','with','via','of','in','on','an','a','to','from',
                          'using','based','towards','learning','exploring','towards','efficient'}
            if ":" in paper_title:
                method_part, domain_part = paper_title.split(":", 1)
            else:
                method_part, domain_part = paper_title, ""

            # Try to form an acronym from the method name (capitalize initials of ALL words)
            method_words = _re.findall(r'[A-Za-z]+', method_part)
            # Include all words for acronym (e.g. "Expand And Compress" → "EAC")
            acronym = "".join(w[0].upper() for w in method_words) if len(method_words) >= 2 else ""

            # Domain keywords: nouns/adjectives from after colon
            domain_kw = [w for w in _re.findall(r'[A-Za-z]{4,}', domain_part)
                         if w.lower() not in _stopwords][:3]

            if acronym and len(acronym) >= 2:
                # Use acronym + domain context: e.g. "EAC continual spatio-temporal forecasting"
                query_parts = [acronym] + domain_kw
            else:
                # Fallback: key content words from method part
                query_parts = [w for w in method_words if w.lower() not in _stopwords][:3] + domain_kw
            query = " ".join(query_parts[:5])
            if not query.strip():
                query = paper_title[:60]

            # Step 3: search GitHub repos
            rg = await c.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "updated", "per_page": 8},
                headers=gh_headers,
            )
            if rg.status_code != 200:
                return {"arxiv_id": arxiv_id, "repos": [], "error": f"GitHub search HTTP {rg.status_code}", "query": query}

            items = rg.json().get("items") or []
            repos = [
                {
                    "url":         f"https://github.com/{item['full_name']}",
                    "is_official": False,        # GitHub search can't confirm officialness
                    "stars":       item.get("stargazers_count", 0),
                    "description": (item.get("description") or "")[:120],
                }
                for item in items
            ]
            return {
                "arxiv_id": arxiv_id,
                "title":    paper_title,
                "query":    query,
                "repos":    repos,
                "hint":     "Call get_github_readme on the top repo(s) to find dataset download links (data_links.cloud_links). Pick repos whose description matches the paper.",
            }
    except Exception as e:
        return {"arxiv_id": arxiv_id, "repos": [], "error": str(e)}


async def verify_hf_dataset(dataset_id: str, expected_name: str = "") -> dict:
    """
    One-shot comprehensive HuggingFace dataset verification.
    Combines get_hf_metadata + get_hf_dataset_files + get_hf_dataset_card
    into a single tool call with a structured verdict.

    Returns splits (from API, YAML, and file inference), downloads, license,
    data file count, name relevance score, and an overall quality verdict.
    """
    metadata_task = get_hf_metadata(dataset_id)
    files_task = get_hf_dataset_files(dataset_id)
    card_task = get_hf_dataset_card(dataset_id)

    metadata, files_info, card_info = await asyncio.gather(
        metadata_task, files_task, card_task,
        return_exceptions=True,
    )
    if isinstance(metadata, Exception):
        metadata = {}
    if isinstance(files_info, Exception):
        files_info = {}
    if isinstance(card_info, Exception):
        card_info = {}

    # Merge splits from all three sources
    api_splits = set(metadata.get("splits") or [])
    file_splits = set(files_info.get("split_hints_from_files") or [])
    yaml_splits = {s.get("name", "") for s in (card_info.get("splits_from_yaml") or [])}
    all_splits = sorted(api_splits | file_splits | yaml_splits - {""})

    downloads = metadata.get("downloads")
    license_info = metadata.get("license") or card_info.get("license")
    data_file_count = len(files_info.get("data_files") or [])
    has_card = card_info.get("has_card", False)

    # Quality signals
    issues: list[str] = []
    quality = "good"

    if not all_splits:
        issues.append("所有来源均未发现 splits（API / 文件名 / YAML）")
        quality = "suspect"

    if downloads is not None and downloads < 10:
        issues.append(f"下载量极低 ({downloads})，可能是空壳或 fork")
        if not all_splits:
            quality = "likely_invalid"

    if not has_card:
        issues.append("无 README/数据集卡片")

    if data_file_count == 0:
        issues.append("仓库中无数据文件")
        quality = "likely_invalid"

    # Name relevance check
    relevance_score = 0.0
    if expected_name:
        relevance_score = _name_relevance(expected_name, dataset_id)
        if relevance_score < 0.2:
            issues.append(
                f"名称相关性极低 ({relevance_score:.2f}): "
                f"'{expected_name}' vs '{dataset_id}'"
            )
            quality = "likely_invalid"

    return {
        "dataset_id":       dataset_id,
        "verified":         quality == "good",
        "quality":          quality,
        "splits":           all_splits,
        "splits_sources": {
            "api":   sorted(api_splits),
            "files": sorted(file_splits),
            "yaml":  sorted(yaml_splits - {""}),
        },
        "has_train":        "train" in all_splits,
        "has_test":         "test" in all_splits,
        "has_validation":   "validation" in all_splits,
        "downloads":        downloads,
        "license":          license_info,
        "data_file_count":  data_file_count,
        "has_card":         has_card,
        "name_relevance":   relevance_score if expected_name else None,
        "issues":           issues,
        "hf_url":           f"https://huggingface.co/datasets/{dataset_id}",
    }


def _name_relevance(query_name: str, candidate_id: str) -> float:
    """
    Compute 0.0–1.0 relevance between a dataset name and a candidate HF ID / repo name.
    Uses token overlap ratio (Jaccard-like).
    """
    def _tokenize(s: str) -> set[str]:
        return {t for t in re.split(r"[\s\-_/\.]+", s.lower()) if len(t) >= 2}

    q_tokens = _tokenize(query_name)
    c_tokens = _tokenize(candidate_id)
    if not q_tokens or not c_tokens:
        return 0.0

    # Exact containment check (e.g. "FollowIR" in "jhu-clsp/FollowIR")
    if query_name.lower().replace("-", "").replace("_", "") in \
       candidate_id.lower().replace("-", "").replace("_", ""):
        return 1.0

    overlap = q_tokens & c_tokens
    if not overlap:
        return 0.0

    return len(overlap) / max(len(q_tokens), len(c_tokens))


async def validate_dataset_result(
    datasets: list[dict],
    structured_memory: dict,
) -> tuple[list[dict], list[str]]:
    """
    Validate agent's final dataset submission. Returns (cleaned_datasets, warnings).

    Checks per dataset:
      1. Name relevance: fuzzy match dataset name vs hf_id / links
      2. HF shell detection: downloads < 10 + no splits → likely empty fork
      3. Confidence audit: low confidence + empty verified_by → needs more evidence
      4. Domain consistency: planner domain vs HF task_categories mismatch
    """
    warnings: list[str] = []
    cleaned: list[dict] = []
    confirmed_splits = structured_memory.get("confirmed_splits", {})
    planner_domain = structured_memory.get("planner_domain")

    for ds in datasets:
        ds_name = ds.get("name", "")
        hf_id = ds.get("hf_id", "")
        confidence = ds.get("confidence", 0.5)
        verified_by = ds.get("verified_by") or []
        splits = ds.get("splits") or []
        downloads = ds.get("downloads")
        issues: list[str] = []

        # 1. Name relevance check
        if hf_id and ds_name:
            relevance = _name_relevance(ds_name, hf_id)
            if relevance < 0.2:
                issues.append(
                    f"名称相关性极低 ({relevance:.1f}): '{ds_name}' vs HF '{hf_id}'，"
                    "可能不是同一个数据集"
                )
                ds["confidence"] = min(confidence, 0.3)

        # 2. HF shell/fork detection
        if hf_id:
            mem_splits = confirmed_splits.get(hf_id, [])
            is_empty = not splits and not mem_splits
            is_low_downloads = downloads is not None and downloads < 10

            if is_empty and is_low_downloads:
                issues.append(
                    f"HF 空壳检测: '{hf_id}' downloads={downloads}, splits 为空 — "
                    "极可能是空壳仓库或 fork，非原始数据集"
                )
                ds["confidence"] = min(ds.get("confidence", confidence), 0.2)
                ds.pop("hf_id", None)
            elif is_empty and downloads is None:
                issues.append(
                    f"HF '{hf_id}' splits 为空且未确认下载量，建议验证"
                )

        # 3. Confidence audit
        if ds.get("confidence", confidence) < 0.3 and not verified_by:
            issues.append(
                f"低可信度 ({ds.get('confidence', confidence):.1f}) 且无验证证据 "
                "(verified_by 为空)，需要更多交叉验证"
            )

        # 4. Domain consistency (if planner provided domain hint)
        if planner_domain and hf_id:
            hf_splits_key = confirmed_splits.get(hf_id)
            if hf_splits_key is not None and not hf_splits_key:
                issues.append(
                    f"Planner 标注领域为 '{planner_domain}'，但 HF '{hf_id}' "
                    "无 splits 数据，可能是不同领域的同名数据集"
                )

        if issues:
            warnings.extend(issues)
            ds.setdefault("_validation_issues", issues)

        cleaned.append(ds)

    return cleaned, warnings


async def finish(datasets: list = None, reason: str = "") -> dict:
    """Terminal action — submit final dataset list to the user."""
    return {"__finish__": True, "datasets": datasets or [], "reason": reason}


# ── Skill registry & JSON schemas ────────────────────────────────────────────

TOOL_FUNCTIONS: dict[str, Any] = {
    "search_dataset":           search_dataset,
    "search_hf_hub":            search_hf_hub,
    "search_semantic_scholar":  search_semantic_scholar,
    "search_pwc_dataset":       search_pwc_dataset,
    "get_hf_metadata":          get_hf_metadata,
    "get_hf_dataset_card":      get_hf_dataset_card,
    "get_hf_dataset_files":     get_hf_dataset_files,
    "get_hf_dataset_configs":   get_hf_dataset_configs,
    "get_github_readme":        get_github_readme,
    "get_github_repo_info":     get_github_repo_info,
    "get_zenodo_record":        get_zenodo_record,
    "fetch_webpage_text":       fetch_webpage_text,
    "web_search":               web_search,
    "tavily_search":            tavily_search,
    "get_gdrive_folder":        get_gdrive_folder,
    "get_github_dir":           get_github_dir,
    "search_zenodo":            search_zenodo,
    "search_opendatalab":       search_opendatalab,
    "compare_datasets":         compare_datasets,
    "get_paper_code_repos":     get_paper_code_repos,
    "verify_hf_dataset":        verify_hf_dataset,
    "finish":                   finish,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_dataset",
            "description": (
                "跨平台广泛搜索数据集（HuggingFace / PapersWithCode / GitHub / Zenodo / Kaggle / OpenML）。"
                "返回链接列表、命中来源、HF ID 列表，以及 PapersWithCode evaluation 信号（has_evaluations_signal）。"
                "GitHub 结果中附带从 README 提取的 HF 数据集 ID（extra.hf_ids）。首选工具，用于初步发现候选数据集。"
                "自动生成 camelCase 拆分、缩写展开等多种查询变体（query_variants_used 字段可见）。"
                "建议与 search_hf_hub 同时并行调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，如 'FollowIR dataset' 或 'instruction following evaluation'",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，限定平台: HuggingFace, PapersWithCode, GitHub, Zenodo, Kaggle, OpenML",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hf_hub",
            "description": (
                "直接在 HuggingFace Hub 搜索，支持任务类别/语言过滤，按下载量排序。"
                "增强：若 query 形如 'owner/name'，会先做精确 ID 直查（速度更快、结果更准）；"
                "downloads 不足时自动切换 trending 排序，补充热门候选。"
                "建议与 search_dataset 同时并行调用作为第一步。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":    {"type": "string", "description": "搜索词，或精确的 HF 数据集 ID（owner/name）"},
                    "task":     {"type": "string", "description": "HuggingFace 任务标签，如 task_categories:text-classification"},
                    "language": {"type": "string", "description": "语言代码，如 en, zh"},
                    "limit":    {"type": "integer", "description": "返回数量上限，默认 8"},
                    "sort":     {"type": "string", "description": "排序方式: downloads（默认）或 likes"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": (
                "在 Semantic Scholar 搜索与数据集相关的论文。"
                "返回论文标题、arXiv ID、摘要、引用量。"
                "当其他工具找不到数据集时，论文页往往含有官方数据集发布链接。"
                "返回的 arxiv_id 可传给 fetch_webpage_text 读取论文摘要。"
                "⚠️ 无 API Key 时 S2 限流严格，若返回 error/hint 含 '429' 或 '限流'，"
                "请改用 web_search('<论文标题> arxiv') 或直接 get_paper_code_repos(title=...) 兜底。"
                "可设置环境变量 SEMANTIC_SCHOLAR_API_KEY 提高调用配额。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "数据集名称或相关论文关键词"},
                    "limit": {"type": "integer", "description": "返回论文数量，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pwc_dataset",
            "description": (
                "在 PapersWithCode 搜索数据集，并获取每个数据集的 evaluation 信息。"
                "has_evaluations=True 是测试集存在的强信号（有 leaderboard 几乎等于有 test split）。"
                "与 search_dataset 相比，提供更结构化的 PWC 数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "数据集名称"},
                    "limit": {"type": "integer", "description": "返回数量，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hf_metadata",
            "description": (
                "获取 HuggingFace 数据集的核心元数据：splits（train/test/validation）、"
                "下载量、许可证。发现 HF 数据集 ID 后立即调用，是确认训练集是否存在的最快方法。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "HuggingFace 数据集 ID，格式 owner/name，如 jhu-clsp/FollowIR",
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hf_dataset_card",
            "description": (
                "获取 HuggingFace 数据集的完整 README（数据集卡片）。"
                "增强：直接解析 YAML frontmatter，返回 splits_from_yaml（含每个 split 的样本数）、"
                "license、language、task_categories 等结构化字段，无需 LLM 从文本里解析。"
                "同时返回 splits_section（定位到数据集结构段落的摘要）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "HuggingFace 数据集 ID，如 owner/name"},
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hf_dataset_files",
            "description": (
                "列出 HuggingFace 数据集仓库中的所有文件，从文件名推断 splits 结构。"
                "增强：扩展 split 关键词（val/held_out/blind/unseen/ood 等），"
                "返回规范化 split 名（train/validation/test）而非原始文件名。"
                "适合 splits API 返回为空时的兜底方案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "HuggingFace 数据集 ID"},
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hf_dataset_configs",
            "description": (
                "获取 HuggingFace 数据集的所有 configs（子集/配置），"
                "如多语言数据集的各语言配置、多任务数据集的各任务配置。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "HuggingFace 数据集 ID"},
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_github_readme",
            "description": (
                "获取 GitHub 仓库的 README 内容（最多 4000 字，优先显示数据/下载相关段落）。"
                "增强：返回结构化 data_links（HF 数据集 ID、Zenodo record ID、arXiv ID），"
                "以及 cloud_links（Baidu Pan / Dropbox / OSF / figshare / Box / OneDrive / S3 / 直接下载链接）。"
                "返回 has_test_hint、has_validation_hint 等信号。"
                "发现 cloud_links 非空 → 立刻 finish，将链接写入数据集 links 字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub 用户名或组织名"},
                    "repo":  {"type": "string", "description": "仓库名"},
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_github_repo_info",
            "description": (
                "获取 GitHub 仓库的完整元数据：stars、forks、描述、topics、许可证、最近更新时间。"
                "适合评估数据集活跃度和可信度。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub 用户名或组织名"},
                    "repo":  {"type": "string", "description": "仓库名"},
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_zenodo_record",
            "description": (
                "获取 Zenodo 记录的详细信息：DOI、标题、描述、访问权限、许可证、文件列表。"
                "每个文件条目包含直接下载链接（download_url），open access 记录可直接写入 finish 的 links 字段。"
                "适合学术数据集的详情查询（从 search_dataset 返回的 Zenodo 链接中提取 record_id）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Zenodo 记录 ID（纯数字），从 URL zenodo.org/record/{id} 提取",
                    },
                },
                "required": ["record_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "使用 DuckDuckGo 搜索任意网页（无需 API Key），返回标题、URL 和摘要片段。"
                "适用场景：查找数据集官网、官方下载页面、GitHub 仓库、论坛讨论、竞赛页面等学术平台之外的资源。"
                "找到 URL 后，用 fetch_webpage_text 读取完整内容。"
                "与 search_dataset/search_hf_hub 互补：当专用学术搜索无结果时，用此工具做通用网络搜索。"
                "**若 Tavily 已配置，请优先使用 tavily_search，结果质量更高、限流更少。**"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，例如 'PEMS-Stream dataset download github' 或 'DDGPrompt dataset site:github.com'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 6，最大 10",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "高质量 agent 级网络搜索（需配置 TAVILY_API_KEY）。"
                "相比 DuckDuckGo，Tavily 召回更稳定、片段更长、还会返回综合回答（answer 字段）。"
                "**当用户配置了 Tavily 时优先使用本工具替代 web_search。**"
                "若返回 error 提示未配置，则降级到 web_search。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，建议含数据集名 + 关键限定词",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 6，最大 10",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "basic（默认，快） / advanced（更深，消耗更多额度）",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage_text",
            "description": (
                "获取任意 HTTPS URL 的纯文本内容（自动去除 HTML 标签）。"
                "增强：arXiv URL 自动改用 export.arxiv.org 返回干净摘要；"
                "提取 <meta> citation 字段（标题/摘要/作者）；"
                "自动提取页面中的 HF/Zenodo/arXiv 链接（data_links 字段）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标 URL（必须以 https:// 开头）",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gdrive_folder",
            "description": (
                "列出公开 Google Drive 文件夹内的所有文件名。"
                "当找到 drive.google.com/drive/folders/... 链接时，用此工具确认文件夹里具体包含哪些数据集文件。"
                "不需要登录，使用嵌入式文件夹视图端点（无 JavaScript）。"
                "返回 files（文件名列表）、data_files（数据集相关文件）、file_count。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Google Drive 文件夹 URL，如 https://drive.google.com/drive/folders/xxx",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_github_dir",
            "description": (
                "列出 GitHub 仓库某目录下的文件，或读取单个小文件的内容（CSV 前几行、JSON 预览等）。"
                "当 README 或代码中提到 data/、processed_data/、datasets/ 等子目录时，用此工具探索里面有哪些数据文件。"
                "repo_path 可以是 'owner/repo' 或完整 GitHub URL（如 https://github.com/owner/repo/tree/main/data）。"
                "返回 items（目录列表）、data_files（数据集相关文件）、sub_dirs（子目录名），文件类型包含 csv/npy/json/parquet 等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "'owner/repo' 或完整 GitHub URL（含 tree/main/path 部分）",
                    },
                    "path": {
                        "type": "string",
                        "description": "仓库内子路径（可选，URL 里已包含路径时可省略），如 'data' 或 'processed_data'",
                    },
                },
                "required": ["repo_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_zenodo",
            "description": (
                "在 Zenodo（欧洲开放科学数据仓库）按关键词搜索数据集。"
                "适合找学术论文配套的标注数据集、实验数据集。"
                "返回 record_id、title、DOI、access（open/restricted）、files 列表、URL。"
                "每个文件条目含 download_url，可直接写入 finish 的 links 字段（open access 时）。"
                "找到后可用 get_zenodo_record 获取完整下载链接列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，如 'traffic flow dataset' 或 'EAC Stream'",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回结果数量，默认 6，最多 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_opendatalab",
            "description": (
                "在 OpenDataLab（opendatalab.com，上海 AI 实验室）搜索数据集。"
                "覆盖大量计算机视觉、NLP、自动驾驶等中文及国际数据集，是 HuggingFace 之外的重要平台。"
                "返回数据集名称、描述、链接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词，如 'PEMS traffic' 或 'air quality dataset'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_datasets",
            "description": (
                "批量获取多个 HuggingFace 数据集 ID 的元数据并并排对比。"
                "当发现多个候选数据集时，用此工具一次性获取所有候选的 splits/downloads/license，"
                "无需逐个调用 get_hf_metadata。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HuggingFace 数据集 ID 列表，最多 8 个",
                    },
                },
                "required": ["dataset_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_code_repos",
            "description": (
                "搜索论文的 GitHub 代码仓库。"
                "arxiv_id 与 title **至少传一个**（推荐传 title，速度最快、最稳定）。"
                "返回仓库列表（含 stars、描述、GitHub URL）。"
                "**在 PDF 模式下建议调用**，然后对描述最匹配的仓库调用 get_github_readme，"
                "从 README 的 data_links.cloud_links 中找数据集下载链接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "论文的 arXiv ID，如 '2410.12593' 或 '2208.04360v2'（无 arxiv ID 时留空字符串）",
                    },
                    "title": {
                        "type": "string",
                        "description": "论文标题（推荐传入；提供后跳过 Semantic Scholar 直接搜 GitHub，更稳定）",
                    },
                },
                # No `required` here — but at least one of arxiv_id/title is needed (runtime-validated).
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hf_dataset",
            "description": (
                "一站式验证 HuggingFace 数据集：并行获取 metadata + 文件列表 + 数据集卡片，"
                "合并三个来源的 splits 信息（API / 文件名推断 / YAML frontmatter），"
                "检测空壳仓库（downloads 极低 + 无数据文件）、名称不匹配等问题。"
                "返回 quality 字段（good / suspect / likely_invalid）和 issues 列表。"
                "**替代** 分别调用 get_hf_metadata + get_hf_dataset_files + get_hf_dataset_card，"
                "只需一次工具调用即可完成完整验证。"
                "建议在 search_hf_hub 或 search_dataset 返回 HF 候选后立即使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "HuggingFace 数据集 ID，格式 owner/name",
                    },
                    "expected_name": {
                        "type": "string",
                        "description": "期望的数据集名称（用于名称相关性检查），可选",
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "完成检索，提交最终结论。"
                "reason 字段必须直接回答用户的问题（如：该数据集有/没有 test split，依据是...）。"
                "当对结论有充分证据，或工具调用次数接近上限时调用（即使结果为空也要调用）。"
            ),
            "parameters": {
                "type": "object",
                "required": ["reason", "datasets"],
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "对用户整体问题的结论性回答（一两句话）。"
                            "例：若用户问的是『找出所有数据集』，则写："
                            "'论文共使用 3 个数据集：PEMS-Stream、Air-Stream、Energy-Stream，"
                            "均可通过论文 GitHub 仓库提供的 Google Drive 链接下载。'"
                            "若用户问的是『是否有 test split』，则写确认结果及依据。"
                            "《勿拿 splits 信息填充 reason》——用户没有问时不要提『训练集』『评估』等标签。"
                        ),
                    },
                    "datasets": {
                        "type": "array",
                        "description": "找到的数据集列表（可为空数组）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":      {"type": "string",  "description": "数据集名称"},
                                "hf_id":     {"type": "string",  "description": "HuggingFace ID（owner/name）"},
                                "splits":    {"type": "array", "items": {"type": "string"}, "description": "实际确认的 splits 列表（只在用户问了 splits 时才填）"},
                                "has_train": {"type": "boolean", "description": "是否有 train split（只在用户问了训练集时才填）"},
                                "confidence": {
                                    "type": "number", 
                                    "description": "0.0~1.0 之间的可信度分数，表示数据来源的准确性确信度。如交叉比对了多个来源可填较高。"
                                },
                                "verified_by": {
                                    "type": "array", 
                                    "items": {"type": "string"}, 
                                    "description": "交叉验证此数据集所依赖的方式，如 ['github readme 链接', 'hf splits 数据量匹配']"
                                },
                                "downloads": {"type": "integer", "description": "HF 下载量（可选）"},
                                "license":   {"type": "string",  "description": "许可证（可选）"},
                                "links": {
                                    "type": "array",
                                    "description": (
                                        "数据集的所有下载/访问链接。**必须**将 get_github_readme 返回的 "
                                        "data_links.cloud_links 中的每条 URL 都写入此字段，格式示例：\n"
                                        "[{\"url\": \"https://drive.google.com/file/d/xxx\", "
                                        "\"label\": \"PEMS-Stream 数据集下载\", \"source\": \"GitHub README\"},"
                                        " {\"url\": \"https://github.com/Onedean/EAC\", "
                                        "\"label\": \"官方代码与数据\", \"source\": \"论文代码仓库\"}]"
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url":    {"type": "string"},
                                            "label":  {"type": "string"},
                                            "source": {"type": "string"},
                                        },
                                    },
                                },
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "直接回答用户的问题 + 证据来源。"
                                        "用户问的是『找出所有数据集』，只说数据集名称和获取方式，不要提 splits。"
                                        "用户问的是『是否有 test split』，写明 splits 确认结果及依据。"
                                    ),
                                },
                            },
                            "required": ["name", "reason"],
                        },
                    },
                },
                "required": ["reason", "datasets"],
            },
        },
    },
]