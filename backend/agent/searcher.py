"""
searcher.py
Step 4: Multi-Source Parallel Search (Agent-style)

Search flow:
  1. Build a retrieval plan per dataset (query variants + source priority)
  2. Run source searches concurrently, priority-first with early exit
  3. Merge and deduplicate evidence links (HF ID-aware dedup)
  4. Verify top links quickly for accessibility

Enhancements over v1:
  - LRU/TTL cache per (query, source) to avoid redundant network calls
  - HuggingFace: exact "owner/name" ID lookup before fuzzy search
  - GitHub: auto-extract HF/Zenodo/arXiv links from README in one pass
  - Dedup: HF datasets keyed by owner/name (not raw URL), multi-source merge
  - search_one: priority-ordered execution with early exit (≥3 live hits)
  - PapersWithCode: fetch evaluations to signal test-split existence
  - Semantic Scholar: paper search to find dataset-linked arXiv papers
"""

import asyncio
import base64
import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

TIMEOUT = 10.0
MAX_QUERY_VARIANTS = 14

SOURCE_AUTHORITY = {
    "HuggingFace":      85,
    "PapersWithCode":   82,
    "Zenodo":           80,
    "OpenML":           78,
    "SemanticScholar":  72,
    "GitHub":           60,
    "Kaggle":           55,
    "Web":              40,
}

# ── Simple TTL cache ─────────────────────────────────────────────────────────

class _TTLCache:
    """In-memory TTL cache keyed by (query, source). Thread-safe enough for asyncio."""
    def __init__(self, ttl_seconds: int = 600, maxsize: int = 256):
        self._store: dict[tuple, tuple[float, object]] = {}
        self._ttl = ttl_seconds
        self._maxsize = maxsize

    def get(self, key: tuple):
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return val

    def set(self, key: tuple, value):
        if len(self._store) >= self._maxsize:
            # evict oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[key] = (time.monotonic(), value)


_cache = _TTLCache(ttl_seconds=600, maxsize=512)


def _cached(source: str):
    """Decorator: cache async search results by (frozenset(queries), source)."""
    def decorator(fn):
        async def wrapper(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
            key = (frozenset(queries), source)
            hit = _cache.get(key)
            if hit is not None:
                logger.debug(f"Cache hit: {source} {queries[:2]}")
                return hit
            result = await fn(self, queries, client)
            _cache.set(key, result)
            return result
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ── Helper: extract structured links from text ───────────────────────────────

def _extract_data_links(text: str) -> dict:
    """Extract HF dataset IDs, Zenodo record IDs, arXiv IDs, and cloud storage links from any text."""
    hf_ids    = list(dict.fromkeys(re.findall(r"huggingface\.co/datasets/([\w\-]+/[\w\-]+)", text)))
    zenodo    = list(dict.fromkeys(re.findall(r"zenodo\.org/(?:record|doi)/(\d+)", text)))
    arxiv     = list(dict.fromkeys(re.findall(r"arxiv\.org/abs/([\d.]+)", text)))
    gh_repos  = list(dict.fromkeys(re.findall(r"github\.com/([\w\-]+/[\w\-]+)(?:/|$)", text)))

    # Cloud storage / direct download links often used for dataset distribution
    cloud_links: list[str] = []
    _cloud_pats = [
        r"https?://drive\.google\.com/\S+",
        r"https?://(?:www\.)?dropbox\.com/\S+",
        r"https?://(?:1drv\.ms|onedrive\.live\.com)/\S+",
        r"https?://\S+\.s3(?:\.[\w-]+)?\.amazonaws\.com/\S+",
        r"https?://storage\.googleapis\.com/\S+",
        r"https?://\S+\.(?:zip|tar\.gz|tgz|tar\.bz2|7z|jsonl?\.gz|csv\.gz|parquet)",
    ]
    seen: set[str] = set()
    for pat in _cloud_pats:
        for m in re.finditer(pat, text, re.IGNORECASE):
            url = m.group(0).rstrip(".,;)\"'")
            # Skip if a longer URL covering this one already collected
            if any(existing.startswith(url) for existing in seen):
                continue
            # Remove any shorter prefix already collected
            seen = {u for u in seen if not url.startswith(u)}
            cloud_links = [u for u in cloud_links if not url.startswith(u)]
            seen.add(url)
            cloud_links.append(url)

    return {
        "hf_ids":      hf_ids[:6],
        "zenodo":      zenodo[:4],
        "arxiv":       arxiv[:4],
        "gh_repos":    gh_repos[:6],
        "cloud_links": cloud_links[:8],
    }


class MultiSourceSearcher:

    def __init__(self, github_token: Optional[str] = None):
        # Token argument takes priority. Falls back to per-request config (see
        # ._github_token()) so the singleton instance in skills.py can still
        # honor user-provided tokens.
        self.github_token = github_token
        self.headers = {"User-Agent": "DatasetRetrievalAgent/2.0"}

    def _github_token(self) -> Optional[str]:
        if self.github_token:
            return self.github_token
        try:
            from .config import get_api_key
            return get_api_key("github") or None
        except Exception:
            return None

    def _hf_token(self) -> Optional[str]:
        try:
            from .config import get_api_key
            return get_api_key("huggingface") or None
        except Exception:
            return None

    @staticmethod
    def _append_unique(target: list[str], value: str, seen_lower: set[str]) -> None:
        value = re.sub(r"\s+", " ", (value or "")).strip()
        if not value:
            return
        key = value.lower()
        if key in seen_lower:
            return
        seen_lower.add(key)
        target.append(value)

    def _expand_term(self, term: str) -> list[str]:
        """
        Expand a term into robust query variants:
        - normalize separators
        - split camelCase/PascalCase
        """
        term = (term or "").strip()
        if not term:
            return []
        expanded: list[str] = [term]

        # e.g. DDGPrompt -> DDG Prompt
        camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", term).strip()
        if camel_split and camel_split.lower() != term.lower():
            expanded.append(camel_split)

        # e.g. follow_ir / follow-ir -> follow ir
        plain = re.sub(r"[-_]+", " ", term).strip()
        if plain and plain.lower() != term.lower():
            expanded.append(plain)

        return expanded

    def build_search_plan(self, dataset: dict, intent: Optional[dict] = None) -> dict:
        """Create an explicit retrieval plan for one dataset."""
        intent = intent or {}
        canonical = dataset.get("canonical", "")
        raw_name  = dataset.get("raw_name", "")
        domain    = dataset.get("domain")

        aliases = list(dataset.get("aliases", []))
        seed_terms = []
        for term in [canonical, raw_name, *aliases]:
            term = (term or "").strip()
            if term and term.lower() not in [t.lower() for t in seed_terms]:
                seed_terms.append(term)

        dedup_queries: list[str] = []
        seen_lower: set[str] = set()
        for term in seed_terms[:6]:
            for base in self._expand_term(term):
                self._append_unique(dedup_queries, base, seen_lower)
                self._append_unique(dedup_queries, f"{base} dataset", seen_lower)
                self._append_unique(dedup_queries, f"{base} benchmark", seen_lower)
                if domain:
                    self._append_unique(dedup_queries, f"{base} {domain} dataset", seen_lower)

        for kw in intent.get("keywords", [])[:6]:
            kw = (kw or "").strip()
            self._append_unique(dedup_queries, kw, seen_lower)

        if intent.get("needs_training_split"):
            training_queries = []
            for q in dedup_queries[:5]:
                training_queries.extend([
                    f"{q} training set", f"{q} train split",
                    f"{q} open source",  f"{q} github",
                    f"{q} huggingface",
                ])
            for q in training_queries:
                self._append_unique(dedup_queries, q, seen_lower)

        if intent.get("needs_training_split"):
            source_priority = ["HuggingFace", "GitHub", "Kaggle", "PapersWithCode", "OpenML", "Zenodo"]
        else:
            source_priority = ["HuggingFace", "PapersWithCode", "Zenodo", "OpenML", "GitHub", "Kaggle"]

        return {
            "primary_name":   canonical or raw_name,
            "query_variants": dedup_queries[:MAX_QUERY_VARIANTS],
            "source_priority": source_priority,
            "strategy": "training-first" if intent.get("needs_training_split") else "live-first",
        }

    def plan_all(self, resolved_datasets: list[dict], intent: Optional[dict] = None) -> list[dict]:
        return [self.build_search_plan(ds, intent=intent) for ds in resolved_datasets]

    # ── Individual source searchers ──────────────────────────────────────────

    @_cached("HuggingFace")
    async def search_huggingface(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        """
        Search HuggingFace.
        Enhancement: if any query looks like 'owner/name', attempt exact ID lookup first
        and prepend result with boosted auth_score.
        """
        results = []
        exact_ids_tried: set[str] = set()

        hf_headers = {**self.headers}
        hf_token = self._hf_token()
        if hf_token:
            hf_headers["Authorization"] = f"Bearer {hf_token}"

        # Pass 1: exact ID lookup for owner/name queries
        for query in queries:
            if "/" in query and re.match(r"^[\w\-\.]+/[\w\-\.]+$", query.strip()):
                ds_id = query.strip()
                if ds_id in exact_ids_tried:
                    continue
                exact_ids_tried.add(ds_id)
                try:
                    r = await client.get(
                        f"https://huggingface.co/api/datasets/{ds_id}",
                        headers=hf_headers, timeout=TIMEOUT,
                    )
                    if r.status_code == 200:
                        ds = r.json()
                        results.append({
                            "source":     "HuggingFace",
                            "url":        f"https://huggingface.co/datasets/{ds_id}",
                            "label":      ds_id,
                            "auth_score": SOURCE_AUTHORITY["HuggingFace"] + 10,  # exact match bonus
                            "status":     "live",   # 200 == live
                            "extra": {
                                "downloads":    ds.get("downloads", 0),
                                "likes":        ds.get("likes", 0),
                                "exact_match":  True,
                                "query":        query,
                            },
                        })
                except Exception as e:
                    logger.debug(f"HF exact lookup failed for '{ds_id}': {e}")

        # Pass 2: fuzzy search
        url = "https://huggingface.co/api/datasets"
        for query in queries[:4]:
            try:
                params = {"search": query, "limit": 5}
                r = await client.get(url, params=params, headers=hf_headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                for ds in r.json()[:3]:
                    ds_id = ds.get("id", "")
                    if not ds_id or ds_id in exact_ids_tried:
                        continue
                    results.append({
                        "source":     "HuggingFace",
                        "url":        f"https://huggingface.co/datasets/{ds_id}",
                        "label":      ds_id,
                        "auth_score": SOURCE_AUTHORITY["HuggingFace"],
                        "status":     "unknown",
                        "extra": {
                            "downloads": ds.get("downloads", 0),
                            "likes":     ds.get("likes", 0),
                            "query":     query,
                        },
                    })
            except Exception as e:
                logger.warning(f"HuggingFace search failed for '{query}': {e}")
        return results

    @_cached("PapersWithCode")
    async def search_paperswithcode(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        """
        Search PapersWithCode.
        Enhancement: for top result, also fetch evaluations to detect test-split existence.
        """
        results = []
        url = "https://paperswithcode.com/api/v1/datasets/"
        top_pwc_ids: list[str] = []

        for query in queries[:4]:
            try:
                params = {"q": query, "page": 1}
                r = await client.get(url, params=params, headers=self.headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                data = r.json()
                for ds in data.get("results", [])[:3]:
                    pwc_id = ds.get("id", "")
                    entry = {
                        "source":     "PapersWithCode",
                        "url":        ds.get("url") or f"https://paperswithcode.com/dataset/{pwc_id}",
                        "label":      ds.get("name") or query,
                        "auth_score": SOURCE_AUTHORITY["PapersWithCode"],
                        "status":     "unknown",
                        "extra": {
                            "paper_count":    ds.get("paper_count", 0),
                            "full_name":      ds.get("full_name", ""),
                            "has_evaluations": False,
                            "query":          query,
                        },
                    }
                    results.append(entry)
                    if pwc_id and len(top_pwc_ids) < 2:
                        top_pwc_ids.append(pwc_id)
            except Exception as e:
                logger.warning(f"PapersWithCode search failed for '{query}': {e}")

        # Fetch evaluations for top results (strong signal for test split existence)
        async def _fetch_evals(pwc_id: str) -> bool:
            try:
                r = await client.get(
                    f"https://paperswithcode.com/api/v1/datasets/{pwc_id}/evaluations/",
                    headers=self.headers, timeout=6.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    return bool(data.get("count", 0) or data.get("results"))
            except Exception:
                pass
            return False

        if top_pwc_ids:
            eval_flags = await asyncio.gather(*[_fetch_evals(pid) for pid in top_pwc_ids], return_exceptions=True)
            # Backfill has_evaluations into results
            idx = 0
            for entry in results:
                if idx >= len(top_pwc_ids):
                    break
                pid = top_pwc_ids[idx]
                if pid and entry["url"].endswith(pid):
                    flag = eval_flags[idx]
                    entry["extra"]["has_evaluations"] = bool(flag) if not isinstance(flag, Exception) else False
                    idx += 1

        return results

    @_cached("GitHub")
    async def search_github(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        """
        Search GitHub.
        Enhancement: for each found repo, immediately fetch README and extract
        HF/Zenodo/arXiv links so downstream agent doesn't need a second tool call.
        """
        results = []
        url = "https://api.github.com/search/repositories"
        headers = {**self.headers, "Accept": "application/vnd.github+json"}
        gh_token = self._github_token()
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"

        for query in queries[:3]:
            try:
                q = f'"{query}" dataset in:name,description,readme'
                params = {"q": q, "sort": "stars", "per_page": 4}
                r = await client.get(url, params=params, headers=headers, timeout=TIMEOUT)
                if r.status_code == 403:
                    logger.warning("GitHub API rate limit hit; consider setting GITHUB_TOKEN")
                    continue
                if r.status_code != 200:
                    continue

                data = r.json()
                for repo in data.get("items", [])[:3]:
                    full_name     = repo.get("full_name", query)
                    default_branch = repo.get("default_branch", "main")

                    # Fetch README in the same pass
                    data_links: dict = {}
                    try:
                        readme_url = (
                            f"https://raw.githubusercontent.com/{full_name}/"
                            f"{default_branch}/README.md"
                        )
                        rr = await client.get(readme_url, headers=self.headers, timeout=8.0)
                        if rr.status_code == 200:
                            data_links = _extract_data_links(rr.text)
                    except Exception:
                        pass

                    results.append({
                        "source":     "GitHub",
                        "url":        repo.get("html_url", ""),
                        "label":      full_name,
                        "auth_score": SOURCE_AUTHORITY["GitHub"],
                        "status":     "unknown",
                        "extra": {
                            "stars":       repo.get("stargazers_count", 0),
                            "description": (repo.get("description") or "")[:120],
                            "hf_ids":      data_links.get("hf_ids", []),
                            "zenodo_ids":  data_links.get("zenodo", []),
                            "arxiv_ids":   data_links.get("arxiv", []),
                            "query":       query,
                        },
                    })
            except Exception as e:
                logger.warning(f"GitHub search failed for '{query}': {e}")
        return results

    @_cached("Zenodo")
    async def search_zenodo(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        results = []
        url = "https://zenodo.org/api/records"
        for query in queries[:3]:
            try:
                params = {
                    "q":    f'"{query}" AND resource_type.type:dataset',
                    "size": 3,
                    "sort": "mostviewed",
                }
                r = await client.get(url, params=params, headers=self.headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                for record in r.json().get("hits", {}).get("hits", [])[:2]:
                    metadata  = record.get("metadata", {})
                    record_id = record.get("id")
                    if not record_id:
                        continue
                    results.append({
                        "source":     "Zenodo",
                        "url":        f"https://zenodo.org/record/{record_id}",
                        "label":      (metadata.get("title") or query)[:90],
                        "auth_score": SOURCE_AUTHORITY["Zenodo"],
                        "status":     "unknown",
                        "extra": {
                            "doi":    record.get("doi", ""),
                            "access": metadata.get("access_right", ""),
                            "query":  query,
                        },
                    })
            except Exception as e:
                logger.warning(f"Zenodo search failed for '{query}': {e}")
        return results

    @_cached("Kaggle")
    async def search_kaggle(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        results = []
        url = "https://www.kaggle.com/api/v1/datasets/list"
        for query in queries[:3]:
            try:
                params = {"search": query, "page": 1, "pageSize": 3}
                r = await client.get(url, params=params, headers=self.headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                for ds in r.json()[:2]:
                    ref = ds.get("ref", "")
                    if not ref:
                        continue
                    results.append({
                        "source":     "Kaggle",
                        "url":        f"https://www.kaggle.com/datasets/{ref}",
                        "label":      ds.get("title", ref),
                        "auth_score": SOURCE_AUTHORITY["Kaggle"],
                        "status":     "unknown",
                        "extra": {
                            "votes": ds.get("voteCount", 0),
                            "size":  ds.get("totalBytes", 0),
                            "query": query,
                        },
                    })
            except Exception as e:
                logger.warning(f"Kaggle search failed for '{query}': {e}")
        return results

    @_cached("OpenML")
    async def search_openml(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        results = []
        for query in queries[:3]:
            try:
                safe_name = query.replace(" ", "%20")
                url = f"https://www.openml.org/api/v1/json/data/list/data_name/{safe_name}/limit/3"
                r = await client.get(url, headers=self.headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                datasets = r.json().get("data", {}).get("dataset", [])
                if isinstance(datasets, dict):
                    datasets = [datasets]
                for ds in datasets[:2]:
                    did = ds.get("did")
                    if not did:
                        continue
                    results.append({
                        "source":     "OpenML",
                        "url":        f"https://www.openml.org/d/{did}",
                        "label":      ds.get("name") or query,
                        "auth_score": SOURCE_AUTHORITY["OpenML"],
                        "status":     "unknown",
                        "extra": {
                            "version":   ds.get("version", ""),
                            "instances": ds.get("NumberOfInstances", ""),
                            "query":     query,
                        },
                    })
            except Exception as e:
                logger.warning(f"OpenML search failed for '{query}': {e}")
        return results

    @_cached("SemanticScholar")
    async def search_semantic_scholar(self, queries: list[str], client: httpx.AsyncClient) -> list[dict]:
        """
        NEW: Search Semantic Scholar for papers about the dataset.
        Papers often contain official dataset release links.
        No API key required.
        """
        results = []
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        for query in queries[:2]:
            try:
                params = {
                    "query":  f"{query} dataset",
                    "fields": "title,year,externalIds,openAccessPdf,fieldsOfStudy",
                    "limit":  3,
                }
                r = await client.get(url, params=params, timeout=TIMEOUT)
                if r.status_code != 200:
                    continue
                for paper in r.json().get("data", [])[:3]:
                    title    = paper.get("title", "")
                    ext_ids  = paper.get("externalIds", {})
                    arxiv_id = ext_ids.get("ArXiv", "")
                    pdf_info = paper.get("openAccessPdf") or {}
                    pdf_url  = pdf_info.get("url", "")

                    paper_url = (
                        f"https://arxiv.org/abs/{arxiv_id}"
                        if arxiv_id else
                        f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
                    )
                    results.append({
                        "source":     "SemanticScholar",
                        "url":        paper_url,
                        "label":      title[:90],
                        "auth_score": SOURCE_AUTHORITY["SemanticScholar"],
                        "status":     "unknown",
                        "extra": {
                            "year":      paper.get("year"),
                            "arxiv_id":  arxiv_id,
                            "pdf_url":   pdf_url,
                            "fields":    paper.get("fieldsOfStudy", []),
                            "query":     query,
                        },
                    })
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
        return results

    # ── Deduplication (HF ID-aware) ──────────────────────────────────────────

    def _hf_key(self, url: str) -> Optional[str]:
        """Extract 'owner/name' from a HuggingFace datasets URL as dedup key."""
        m = re.search(r"huggingface\.co/datasets/([\w\-\.]+/[\w\-\.]+)", url)
        return m.group(1) if m else None

    def normalize_url(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            path   = parsed.path.rstrip("/")
            clean  = parsed._replace(query="", fragment="", path=path)
            return urlunparse(clean)
        except Exception:
            return url

    def dedupe_links(self, links: list[dict]) -> list[dict]:
        """
        Deduplicate links.
        HuggingFace entries: keyed by 'owner/name' (not raw URL), merging evidence
        from multiple sources into a single entry.
        Other entries: keyed by normalized URL + source.
        """
        hf_merged:    dict[str, dict] = {}   # hf_id -> merged entry
        other_merged: dict[str, dict] = {}   # url|source -> entry

        for link in links:
            raw_url = link.get("url", "")
            hf_id   = self._hf_key(raw_url)

            if hf_id:
                if hf_id not in hf_merged:
                    hf_merged[hf_id] = {
                        **link,
                        "url":            f"https://huggingface.co/datasets/{hf_id}",
                        "hf_id":          hf_id,
                        "sources_merged": [link.get("source", "HuggingFace")],
                    }
                else:
                    existing = hf_merged[hf_id]
                    # Keep best auth_score
                    if link.get("auth_score", 0) > existing.get("auth_score", 0):
                        hf_merged[hf_id] = {
                            **link,
                            "url":            existing["url"],
                            "hf_id":          hf_id,
                            "sources_merged": existing.get("sources_merged", []) + [link.get("source", "")],
                        }
                    else:
                        src = link.get("source", "")
                        if src and src not in existing.get("sources_merged", []):
                            existing.setdefault("sources_merged", []).append(src)
                        # Merge extras
                        existing["extra"] = {**existing.get("extra", {}), **link.get("extra", {})}
            else:
                normalized = self.normalize_url(raw_url)
                if not normalized:
                    continue
                key = f"{normalized}|{link.get('source', 'Web')}"
                if key not in other_merged:
                    other_merged[key] = {**link, "url": normalized}
                else:
                    if link.get("auth_score", 0) > other_merged[key].get("auth_score", 0):
                        other_merged[key] = {**link, "url": normalized}
                    else:
                        prev_extra = other_merged[key].get("extra", {})
                        curr_extra = link.get("extra", {})
                        other_merged[key]["extra"] = {**prev_extra, **curr_extra}

        all_results = list(hf_merged.values()) + list(other_merged.values())
        all_results.sort(key=lambda x: x.get("auth_score", 0), reverse=True)
        return all_results

    # ── URL verification ─────────────────────────────────────────────────────

    async def verify_url(self, url: str, client: httpx.AsyncClient) -> str:
        def _classify(status_code: int) -> str:
            if status_code < 400:
                return "live"
            if status_code in (401, 403):
                return "auth_required"
            return "dead"

        try:
            r = await client.head(url, timeout=5.0, follow_redirects=True)
            status = _classify(r.status_code)
            # Some endpoints reject HEAD even when GET works (405/429/4xx gateway behavior)
            if r.status_code not in (405, 429) and status != "dead":
                return status
        except Exception:
            pass

        try:
            r = await client.get(
                url,
                timeout=6.0,
                follow_redirects=True,
                headers={"Range": "bytes=0-0"},
            )
            return _classify(r.status_code)
        except Exception:
            return "unknown"

    async def verify_top_links(self, links: list[dict], top_k: int = 4) -> list[dict]:
        if not links:
            return links
        top_links  = links[:top_k]
        tail_links = links[top_k:]
        async with httpx.AsyncClient() as client:
            statuses = await asyncio.gather(
                *[self.verify_url(link["url"], client) for link in top_links],
                return_exceptions=True,
            )
        checked = [
            {**link, "status": status if isinstance(status, str) else "unknown"}
            for link, status in zip(top_links, statuses)
        ]
        return checked + tail_links

    # ── Core search orchestration ────────────────────────────────────────────

    async def search_one(self, dataset: dict, plan: Optional[dict] = None) -> dict:
        """
        Run planned multi-source search for a single dataset.

        Enhancement: sources are tried in source_priority order. After the top-2
        sources complete, if ≥3 live-status links already found, skip remaining
        lower-priority sources to save network round-trips.
        """
        plan    = plan or self.build_search_plan(dataset)
        queries = plan.get("query_variants", [])
        priority: list[str] = plan.get("source_priority", [
            "HuggingFace", "PapersWithCode", "Zenodo", "OpenML", "GitHub", "Kaggle",
        ])

        source_fn_map = {
            "HuggingFace":     self.search_huggingface,
            "PapersWithCode":  self.search_paperswithcode,
            "GitHub":          self.search_github,
            "Zenodo":          self.search_zenodo,
            "Kaggle":          self.search_kaggle,
            "OpenML":          self.search_openml,
            "SemanticScholar": self.search_semantic_scholar,
        }

        all_links: list[dict] = []

        async with httpx.AsyncClient() as client:
            # Run top-2 priority sources first (they're the most authoritative)
            top_sources  = [s for s in priority if s in source_fn_map][:2]
            rest_sources = [s for s in priority if s in source_fn_map][2:]
            # Always include SemanticScholar if not already in priority list
            if "SemanticScholar" not in priority:
                rest_sources.append("SemanticScholar")

            top_results = await asyncio.gather(
                *[source_fn_map[s](queries, client) for s in top_sources],
                return_exceptions=True,
            )
            for result in top_results:
                if isinstance(result, list):
                    all_links.extend(result)

            # Early exit: if we already have ≥3 live-ish links, skip remaining sources
            live_count = sum(1 for l in all_links if l.get("status") in ("live", "unknown"))
            hf_count   = sum(1 for l in all_links if l.get("source") == "HuggingFace")

            if live_count >= 3 and hf_count >= 1 and rest_sources:
                logger.debug(
                    f"Early exit after top-2 sources: {live_count} live links, "
                    f"{hf_count} HF entries. Skipping: {rest_sources}"
                )
            else:
                rest_results = await asyncio.gather(
                    *[source_fn_map[s](queries, client) for s in rest_sources],
                    return_exceptions=True,
                )
                for result in rest_results:
                    if isinstance(result, list):
                        all_links.extend(result)

        deduped_links  = self.dedupe_links(all_links)
        verified_links = await self.verify_top_links(deduped_links, top_k=4)
        sources_hit    = sorted({link.get("source", "") for link in verified_links if link.get("source")})

        return {
            "name":             dataset.get("canonical") or dataset.get("raw_name"),
            "plan":             plan,
            "links":            verified_links,
            "sources_searched": top_sources + (rest_sources if live_count < 3 else []),
            "sources_hit":      sources_hit,
        }

    async def search_all(self, resolved_datasets: list[dict], plans: Optional[list[dict]] = None) -> list[dict]:
        """Search for all resolved datasets concurrently with per-dataset plans."""
        plans = plans or self.plan_all(resolved_datasets)
        if len(plans) < len(resolved_datasets):
            plans = plans + [
                self.build_search_plan(ds)
                for ds in resolved_datasets[len(plans):]
            ]
        tasks = [
            self.search_one(ds, plan=plan)
            for ds, plan in zip(resolved_datasets, plans)
        ]
        search_results = await asyncio.gather(*tasks)
        output = []
        for ds, search in zip(resolved_datasets, search_results):
            output.append({**ds, "search": search})
        return output