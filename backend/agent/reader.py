"""
reader.py
Deep page reading for dataset discovery.

Fetches real metadata (splits, downloads, license) directly from platform APIs:
  - HuggingFace dataset info API + splits API
  - GitHub README (via API)
"""

import asyncio
import base64
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class DatasetPageReader:
    """Reads dataset platform pages to extract verified metadata."""

    HF_INFO_URL = "https://huggingface.co/api/datasets/{dataset_id}"
    HF_SPLITS_URL = "https://datasets-server.huggingface.co/splits?dataset={dataset_id}"
    GH_README_URL = "https://api.github.com/repos/{owner}/{repo}/readme"

    def __init__(self, github_token: Optional[str] = None):
        self.gh_headers = {
            "Accept": "application/vnd.github.v3+json",
            **({"Authorization": f"token {github_token}"} if github_token else {}),
        }

    # ── URL parsers ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_hf_id(url: str) -> Optional[str]:
        """Extract 'owner/name' from a HuggingFace datasets URL."""
        m = re.search(r"huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)", url)
        return m.group(1) if m else None

    @staticmethod
    def parse_github_repo(url: str) -> Optional[tuple[str, str]]:
        """Extract (owner, repo) from a GitHub URL."""
        m = re.search(r"github\.com/([^/?#\s]+)/([^/?#\s]+?)(?:\.git)?(?:[/?#]|$)", url)
        return (m.group(1), m.group(2)) if m else None

    # ── Single-source fetchers ───────────────────────────────────────────────

    async def _fetch_hf_info(self, dataset_id: str) -> dict:
        url = self.HF_INFO_URL.format(dataset_id=dataset_id)
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    return r.json()
        except Exception as e:
            logger.debug(f"HF info failed ({dataset_id}): {e}")
        return {}

    async def _fetch_hf_splits(self, dataset_id: str) -> list[str]:
        url = self.HF_SPLITS_URL.format(dataset_id=dataset_id)
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    data = r.json()
                    return sorted({s["split"] for s in data.get("splits", [])})
        except Exception as e:
            logger.debug(f"HF splits failed ({dataset_id}): {e}")
        return []

    async def _fetch_github_readme(self, owner: str, repo: str) -> str:
        url = self.GH_README_URL.format(owner=owner, repo=repo)
        try:
            async with httpx.AsyncClient(timeout=8, headers=self.gh_headers) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    b64 = r.json().get("content", "")
                    return base64.b64decode(b64.replace("\n", "")).decode("utf-8", errors="replace")[:5000]
        except Exception as e:
            logger.debug(f"GitHub README failed ({owner}/{repo}): {e}")
        return ""

    # ── Main enrichment entry point ──────────────────────────────────────────

    async def enrich(self, canonical: str, links: list[dict]) -> dict:
        """
        Deep-read the top links for a dataset and return verified metadata.

        Returns:
            {
                "hf_id": str | None,
                "splits": list[str],          # e.g. ["train", "test"]
                "has_train_split": bool,
                "downloads": int | None,
                "license": str | None,
                "confirmed_facts": list[str],  # Human-readable verified facts
                "relevance_ok": bool,          # False = likely irrelevant result
            }
        """
        result: dict = {
            "hf_id": None,
            "splits": [],
            "has_train_split": False,
            "downloads": None,
            "license": None,
            "confirmed_facts": [],
            "relevance_ok": True,
        }

        live_links = [l for l in links if l.get("status") == "live"]

        # ── Step 1: Try HuggingFace (best metadata source) ──────────────────
        hf_seen: set[str] = set()
        for link in live_links[:6]:
            hf_id = self.parse_hf_id(link.get("url", ""))
            if not hf_id or hf_id in hf_seen:
                continue
            hf_seen.add(hf_id)

            info, splits = await asyncio.gather(
                self._fetch_hf_info(hf_id),
                self._fetch_hf_splits(hf_id),
                return_exceptions=True,
            )
            if isinstance(info, Exception):
                info = {}
            if isinstance(splits, Exception):
                splits = []

            # Relevance gate: dataset ID should contain the canonical name substring
            canonical_lower = canonical.lower().replace("-", "").replace("_", "").replace(" ", "")
            hf_id_lower = hf_id.lower().replace("-", "").replace("_", "").replace(" ", "")
            if canonical_lower not in hf_id_lower and hf_id_lower not in canonical_lower:
                # Try matching individual tokens
                tokens = re.split(r"[\s\-_/]", canonical.lower())
                sig_tokens = [t for t in tokens if len(t) >= 4]
                if sig_tokens and not any(t in hf_id_lower for t in sig_tokens):
                    logger.debug(f"Skipping irrelevant HF dataset: {hf_id} for {canonical}")
                    result["relevance_ok"] = False
                    continue

            result["hf_id"] = hf_id

            if splits:
                result["splits"] = splits
                result["has_train_split"] = "train" in splits
                result["confirmed_facts"].append(
                    f"HuggingFace ({hf_id}) splits: {', '.join(splits)}"
                )
            else:
                # Fallback: check file names for train clues
                siblings = info.get("siblings", []) if isinstance(info, dict) else []
                fnames = [s.get("rfilename", "").lower() for s in siblings]
                if any("train" in f for f in fnames):
                    result["has_train_split"] = True
                    result["confirmed_facts"].append("HuggingFace 文件列表包含训练数据文件")

            if isinstance(info, dict):
                dl = info.get("downloads") or info.get("downloadsAllTime")
                if dl:
                    result["downloads"] = int(dl)
                    result["confirmed_facts"].append(f"下载量: {result['downloads']:,}")
                card = info.get("cardData") or {}
                if card.get("license"):
                    result["license"] = card["license"]
                    result["confirmed_facts"].append(f"许可证: {card['license']}")

            break  # Got HF info, stop

        # ── Step 2: Try GitHub README (if no HF found or no facts yet) ──────
        if not result["confirmed_facts"]:
            for link in live_links[:6]:
                repo = self.parse_github_repo(link.get("url", ""))
                if not repo:
                    continue
                owner, name = repo
                readme = await self._fetch_github_readme(owner, name)
                if not readme:
                    continue

                lower = readme.lower()
                has_train = any(
                    kw in lower for kw in [
                        "train split", "training data", "train.json", "train.csv",
                        "train.parquet", "training set", "trainset",
                    ]
                )
                if has_train:
                    result["has_train_split"] = True
                    result["confirmed_facts"].append("GitHub README 提到训练数据集")

                # Relevance check: canonical name should appear somewhere in README
                if canonical.lower() not in lower and len(canonical) >= 4:
                    result["relevance_ok"] = False

                break

        return result
