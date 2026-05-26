"""
loop.py — Pure ReAct agent loop for dataset discovery.

The LLM is in the driver's seat:
  Observe user query → choose tool → execute → observe result → repeat
  until the agent calls finish() or hits the turn limit.

Enhancements over v1:
  1. Parallel tool execution: multiple tool_calls in one LLM response are now
     executed concurrently via asyncio.gather (not sequentially).
  2. Context window compression: tool results > 1500 chars are intelligently
     trimmed before being written to message history (full result still yielded
     via SSE for the frontend).
  3. Confidence detection: after get_hf_metadata / get_hf_dataset_card return
     complete split info, a hint is injected prompting the LLM to finish early.
  4. Structured scratchpad: after each tool round a brief state-card is appended
     to message history so the LLM stays oriented without re-reading raw JSON.
  5. Tiered retry: 401/403 → immediate error; 429 → wait 3s retry once;
     5xx/timeout → exponential backoff (1s, 2s), max 2 retries.
"""

import asyncio
import json
import logging
import re
import time
from typing import AsyncGenerator, Optional

from openai import OpenAI

from .skills import TOOL_DEFINITIONS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen-max"

SYSTEM_PROMPT = """\
你是一名数据集研究专家。给定用户的问题，你需要自主调用工具进行调查，最终给出有证据支撑的准确答案。

## ⚠️ 最重要规则：本 Agent 只能通过调用 `finish` 工具输出结果
**直接输出文字回复是被明确禁止的。** 无论证据是否完整，当调查到位时，你必须调用 `finish` 工具——而不是写 Markdown 文字总结。即便某些数据集没有找到下载链接，也应当调用 `finish`，在 links 字段留空或填写已知的来源页面 URL，而不是写一段文字说"建议进一步搜索"。

## 核心检索策略（提升智能度与稳定性）

1. **Query Rewriting (查询重写与拓展)**
   在你执行广搜工具前（如 `search_dataset`, `web_search`, `search_semantic_scholar`），**千万不要死板地仅使用原始缩写**。例如遇到 "DDGPrompt"，你需要尝试展开为 "Data-centric Prompt Tuning Dynamic Graphs dataset"，或者加上 `site:github.com` 等高级检索语法增强召回率。

2. **交叉验证 (Cross-Verification)**
   不要找到一个同名的 HF repo 就立刻认定它是原论文数据集（有可能是网友 fork 或二次处理版本）。你必须进行**交叉比对**：
   - 论文描述的特征维度 / 数据规模（节点、边）是否与仓库一致？
   - 论文里的 splits 与代码仓库 README 或 HF metadata 中是一致的吗？
   - GitHub official README 里是否显式引用了该 HF 链接？

3. **平台选择（根据数据集类型）**
   - 中文 / CV / 自动驾驶类数据集 → 优先 search_opendatalab + search_hf_hub
   - 学术小型数据集、专项标注集 → 优先 search_zenodo（已修复多词零结果 bug）
   - 竞赛数据集（KDD Cup / Kaggle 等）→ web_search 效果更好
   - 大规模 NLP 数据集 → search_hf_hub 首选
## 第一要务：理解用户真正在问什么

用户的问题多种多样，不要假设：
- "是否有训练集" → 确认 train split 是否存在
- "是否有测试集" → 确认 test split 是否存在
- "有哪些数据集" → 列出所有相关数据集及其来源链接（**无需强调 splits**，除非用户明确问）
- "数据量多少" → 找 downloads / 文件大小
- "许可证" → 找 license 信息

**读懂问题，然后去找那个具体的答案。**

## 调查方式：完全自主

你没有固定流程。每次工具调用结束后，根据你得到的信息，自己判断：
- 还有哪些未知？
- 哪条线索最值得追踪？
- 当前证据是否已经足够回答用户的问题？

**第一步建议同时并行调用 search_dataset、search_hf_hub、search_semantic_scholar**——这三个工具可以并发执行，无需等待彼此的结果，大幅拓宽覆盖面。
**例外**：若输入含 [论文节选]，请优先走下方 "当输入包含 [论文节选]" 的专用流程，而不是盲目并行广搜。

## ⚡ 快速结束条件（立刻 finish，无需额外搜索）
- **get_github_readme 返回 data_links.cloud_links 非空** → 云盘/直接下载链接已找到，立刻 finish，cloud_links 写入每个数据集的 links 字段
- **Zenodo 记录有 open access 文件** → 下载链接已有，立刻 finish
- **HF metadata 返回完整 splits 列表** → splits 问题已答，立刻 finish

可以对同一工具用不同参数多次调用，可以在平台之间自由跳跃，
可以随时改变调查方向。直到你有充分把握为止。

当你找到多个候选 HF 数据集时，优先使用 compare_datasets 工具批量获取元数据，
而不是逐个调用 get_hf_metadata。

**不允许的行为**：
- 一次搜索无结果就 finish
- 找到 GitHub 链接却不去读 README（get_github_readme 已内置链接提取，会直接给你 HF ID 且会提取所有 cloud_links）
- 找到 **cloud_links**（Google Drive/Dropbox/OneDrive/S3/直接下载 zip）却不将其写入 finish 的 links 字段
- 对 Google Drive / Dropbox / OneDrive / S3 等**云存储链接**调用 fetch_webpage_text（这类链接需要 JavaScript 无法有效抓取）；Google Drive 文件夹请用 get_gdrive_folder
- 找到 cloud_links 后继续发起额外搜索（**已经有下载链接就是有答案，立刻 finish**）
- 用户没有问 splits/训练集/测试集时，主动去验证 HF splits（浪费调用次数）
- 用模糊措辞回避用户的实际问题
- **将 `get_hf_metadata` 返回 `splits=[]` 的 HF URL 写入 finish 的 links 字段**——`splits=[]` 是红牌，说明 HF 上同名数据集不是论文使用的版本（`huggingface.co/datasets/Wikipedia` 是文本语料库，与时序图研究毫无关系）；必须改用 web_search 找真正来源
- **未经 search 工具验证就直接用数据集名称调用 get_hf_metadata**——Wikipedia/Reddit/MOOC/LastFM 等通用名在 HF 上虽有同名数据集，但几乎从不是学术 benchmark 论文实际使用的版本；必须先 search 后再用返回的具体 ID 去查 metadata
- **在 get_hf_metadata 返回 splits=[] 后继续调用 get_hf_dataset_configs / get_hf_dataset_files** ——这是在用错误的 HF ID 无效循环，应立即转向 web_search 或 get_paper_code_repos

## finish 时的要求

最终必须给出经得起推敲的数据，包含**可信度审查**：
- **`confidence`**: 给每个数据集赋予 0.0~1.0 的可信度分数。
- **`verified_by`**: 明确填写验证证据（如 `["通过官方 GitHub README 的 HF 下载链接比对", "样本数 500k 匹配论文描述"]`）。

**`reason` 字段必须直接回答用户的问题**——根据用户实际问了什么来写：

| 用户的问题 | reason 应该写什么 |
|---|---|
| 找出所有数据集 / 获取方式 | 该数据集的 HF/GitHub/DOI 链接及简要说明，**不必提 splits** |
| 是否有训练集/测试集 | 确认 split 是否存在及依据 |
| 数据量多少 | 下载量/样本数 |
| 找不到 | 在 HF / GitHub / PWC 均未找到公开数据集，原因：... |

`splits` 和 `has_train` 字段：如实填写已确认的信息；若未调查 splits，留空数组即可，**不要为了填满字段而乱猜**。

## 工具参考
| 工具 | 用途 |
|------|------|
| search_dataset | 跨平台广搜（HF/PWC/GitHub/Zenodo/Kaggle）；GitHub 结果含 README 里提取的 HF ID |
| search_hf_hub | HuggingFace 精确搜索，支持 owner/name 直查 |
| search_zenodo | 直接在 Zenodo 搜索学术数据集（返回 record_id/DOI/文件列表）|
| search_opendatalab | 搜索 OpenDataLab（上海AI实验室，覆盖大量 CV/NLP/自动驾驶数据集）|
| tavily_search | **优先使用**的高质量网络搜索（需 TAVILY_API_KEY；返回 results + answer 综合回答）|
| web_search | DuckDuckGo 通用网络搜索（无需 API Key，作为 tavily 兜底）|
| search_semantic_scholar | 找引用该数据集的相关论文，从其他论文找数据发布/下载渠道 |
| search_pwc_dataset | GitHub 数据集仓库搜索（按 stars 排序）|
| verify_hf_dataset | **推荐**：一站式验证 HF 数据集（合并 metadata+files+card，检测空壳/fork，返回 quality 判定）|
| get_hf_metadata | 快速确认 splits / 下载量 / 许可证（含 siblings 文件名推断兜底）|
| get_hf_dataset_card | HF README + YAML frontmatter（直接给出结构化 splits_from_yaml）|
| get_hf_dataset_files | 从文件名推断 splits（当 splits API 返回空时）|
| get_hf_dataset_configs | 多语言/多任务数据集的子集列表 |
| compare_datasets | 批量对比多个 HF 数据集的 splits/downloads/license |
| get_paper_code_repos | 通过 arXiv ID 搜索论文在 GitHub 上的代码仓库 |
| get_github_readme | GitHub README（自动提取 HF/Zenodo/arXiv 链接，以及 Google Drive 等云存储链接）|
| get_github_dir | 列出 GitHub 仓库某目录的文件，或读取小文件内容（CSV/NPY/JSON 预览）|
| get_github_repo_info | GitHub stars/topics/许可证 |
| get_zenodo_record | 按 record_id 获取 Zenodo 详细信息和下载链接 |
| get_gdrive_folder | 列出公开 Google Drive 文件夹内所有文件，确认包含哪些数据集文件 |
| fetch_webpage_text | 读任意网页（arXiv 自动使用 export API 返回干净摘要）|
| finish | 提交最终结论（只在有充分证据时调用）|

## 当输入包含 [论文节选] 时

你拿到的是一篇论文的节选文本（通常包括摘要、实验设置章节、附录数据集详情以及相关引用）。你的任务是找出论文中**用于实验的具体数据集**及其获取方式。

**必须执行的步骤**：

1. **仔细阅读论文节选**，识别其中提及的所有数据集/基准名称
   - 常见位置：小节标题如 "4.1 Datasets and Metrics"、表格、"We evaluate on X, Y, Z"
   - 注意区分：论文本身的系统/方法名称 vs 用于实验的基准数据集

2. **优先捷径：使用论文自身的代码仓库链接**（速度最快！）
   - 若节选中有 **[论文代码仓库链接]** 段落，**只调用 `get_github_readme`**，不要同时并行发起任何 search 搜索
   - 等待 README 结果返回后再判断：
     - 若 `data_links.cloud_links` 里有 Google Drive 文件夹链接 → 调用 `get_gdrive_folder` 确认文件夹包含哪些数据集，然后 finish
     - 若 `data_links.cloud_links` 是直接下载链接（zip/csv 等） → 直接 finish，无需额外搜索
     - 若 README 中有 `data/`、`processed_data/`、`datasets/` 等子目录路径 → 调用 `get_github_dir` 探索该目录，验证是否有数据文件
     - 若 README 没有任何数据链接/目录 → 再进入步骤 3
   - `get_github_readme` 返回的 `data_links.cloud_links` = 数据集云存储下载链接
   - `data_links.gh_repos` = README 中列出的其他相关仓库
   - 若节选中**没有** [论文代码仓库链接]，但有 **[论文标题]** → 立即调用 `get_paper_code_repos(arxiv_id="", title="<完整标题>")`  
     找到结果后再调 `get_github_readme`；若没有 [论文标题] 则跳到步骤 3

3. **没有直接仓库链接时，多平台并行搜索**（HF 之外同样重要！）
   - 以下平台**同等优先级并行搜索**：
     - `search_hf_hub` — HuggingFace
     - `search_dataset` — 跨平台广搜（内含 Kaggle/Zenodo/OpenML）
     - `search_zenodo` — Zenodo 学术数据集（论文配套数据集常在此）
     - `search_opendatalab` — OpenDataLab（中文/国际 AI 数据集）
     - `tavily_search`（如已配置 TAVILY_API_KEY，优先使用）或 `web_search` — 通用网络搜索（当数据集名罕见、学术平台无结果时必用；如 `"DDGPrompt dataset site:github.com"`）
   - 找到 GitHub 仓库后：先读 README，再用 `get_github_dir` 探索 `data/` 或 `datasets/` 子目录
   - 找到 Zenodo 结果后：用 `get_zenodo_record` 获取实际下载链接

4. **逐个数据集搜索下载链接**（多个数据集时必须执行）
   对于论文中提到但尚未找到下载链接的每个数据集，**分别独立搜索**：
   - `web_search("<数据集名> dataset download")` — 找官网或仓库
   - `web_search("<数据集名> dataset github site:github.com")` — 找 GitHub 数据仓库
   - `web_search("<数据集名> dataset huggingface")` — 找 HF 镜像
   - 典型模式：很多数据集没有独立页面，但在 **引入该数据集的原始论文** 的 GitHub 仓库 README 里有下载脚本或链接
   - 若数据集名很通用（如 "Wikipedia"、"Reddit"），必须加上论文场景限定词再搜索（如 `"Wikipedia temporal graph dataset jodie download"`）

5. **若多平台搜索仍无结果，从相关论文找获取渠道**
5. **若多平台搜索仍无结果，从相关论文找获取渠道**
   - `search_semantic_scholar` 每次只调一个查询（S2 有速率限制，禁止并发）
   - 搜索 "数据集名称 dataset"，找引用或发布该数据集的论文
   - 从返回的论文中找其代码仓库（`get_paper_code_repos`）→ 读 README → 可能有下载说明
   - 若论文节选中有 arXiv ID → 直接 `get_paper_code_repos(arxiv_id)` 找代码仓库

6. **调用 finish 时的注意事项**：
   - **共享链接**：若一个 GitHub README 包含多个数据集的下载链接，将该链接写入**每个**数据集的 `links` 字段，不要只写到其中一个
   - 用户问"获取方式" / "找出数据集" → 填写 HuggingFace 直链、GitHub 链接、Zenodo DOI、竞赛页面或官方网站
   - 用户问"是否有 test split" → 填写实际确认的 splits 列表及依据
   - 禁止回复模板化内容如"该数据集包含 XX split，来自 get_hf_metadata"

**禁止事项**：
- `search_semantic_scholar` 并发调用（S2 有严格速率限制，并发调用全部 429 等于白费）
- 只搜 HuggingFace 就放弃——必须同时尝试 Zenodo、OpenDataLab、GitHub 等多个平台
- 搜索论文的系统/方法名称作为数据集（论文叫 "LaSER" ≠ 有数据集叫 "laser"）
- **`fetch_webpage_text` 返回 HTTP 403 后，继续访问同一域名的其他 URL**——403 = 该网站封锁爬虫，应立即改用 web_search 或其他工具
- **用 `fetch_webpage_text` 读取数据集官网（wikipedia.org / reddit.com / imdb.com 等）**——这类网站的首页/百科页面无法提供研究数据集的下载链接；应搜 GitHub 仓库、Zenodo 记录或 HuggingFace 页面
- 找到 GitHub 代码仓库后只看 README，不探索 data/ 子目录（如果 README 没有链接）
- 找到 GitHub README 中的 cloud_links 后不写入 finish 的 links 字段
- 若节选文本未包含数据集相关内容，立即回复："节选文本未包含数据集和实验设置部分，无法确认具体数据集"
- 盲目返回空数组 — 必须至少尝试 3 次工具调用才能确认无法找到
"""


# ── Context compression ───────────────────────────────────────────────────────

def _compress_tool_result(tool_name: str, result: dict) -> dict:
    """
    Trim large tool results before writing to message history.
    The full result is still yielded to the frontend; this is history-only.
    """
    raw = json.dumps(result, ensure_ascii=False, default=str)
    if len(raw) <= 1500:
        return result

    # Tool-specific trimming strategies
    if tool_name == "get_hf_dataset_card":
        # Keep structured frontmatter fields + splits_section only
        return {k: v for k, v in result.items() if k in (
            "dataset_id", "has_card", "license", "languages", "task_categories",
            "splits_from_yaml", "splits_section", "has_splits_info",
            "has_train_section", "has_citation",
        )}

    if tool_name == "fetch_webpage_text":
        compressed = {k: v for k, v in result.items() if k != "text"}
        text = result.get("text", "")
        compressed["text"] = text[:2000]
        if "data_links" in result:
            compressed["data_links"] = result["data_links"]
        return compressed

    if tool_name == "search_hf_hub":
        # Strip tags/languages from each dataset entry; keep id/downloads/splits
        compressed = dict(result)
        compressed["datasets"] = [
            {k: v for k, v in ds.items() if k in ("id", "url", "downloads", "license", "gated", "exact_match")}
            for ds in result.get("datasets", [])
        ]
        return compressed

    if tool_name in ("search_dataset", "search_pwc_dataset"):
        # Keep top 5 links, drop verbose extras
        compressed = dict(result)
        links = result.get("links", [])
        compressed["links"] = [
            {k: v for k, v in l.items() if k in ("url", "label", "source")}
            for l in links[:5]
        ]
        return compressed

    if tool_name == "web_search":
        # Keep top 6 results, truncate snippets to 250 chars
        compressed = dict(result)
        compressed["results"] = [
            {**r, "snippet": r.get("snippet", "")[:250]}
            for r in result.get("results", [])[:6]
        ]
        return compressed

    if tool_name == "search_zenodo":
        # Keep top 4 records, truncate descriptions, preserve download URLs
        compressed = dict(result)
        records = []
        for rec in result.get("results", [])[:4]:
            r_c = {k: v for k, v in rec.items() if k in ("record_id", "title", "doi", "access", "url")}
            r_c["files"] = rec.get("files", [])[:3]
            records.append(r_c)
        compressed["results"] = records
        return compressed

    if tool_name == "search_semantic_scholar":
        # Keep top 4 papers, trim abstracts to 300 chars
        compressed = dict(result)
        compressed["papers"] = [
            {**p, "abstract": p.get("abstract", "")[:300]}
            for p in result.get("papers", [])[:4]
        ]
        return compressed

    if tool_name == "get_github_readme":
        # Keep data_links + hint booleans; trim excerpt
        compressed = {k: v for k, v in result.items() if k != "excerpt"}
        excerpt = result.get("excerpt", "")
        compressed["excerpt"] = excerpt[:1200]
        return compressed

    # Generic fallback: Semantic Compression instead of hard cut
    compressed = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 5:
            compressed[k] = v[:5] + [{"_truncated": f"{len(v)-5} more items"}]
        elif isinstance(v, str) and len(v) > 1000:
            compressed[k] = v[:1000] + "…[truncated]"
        else:
            compressed[k] = v
    final_str = json.dumps(compressed, ensure_ascii=False, default=str)
    if len(final_str) > 2000:
        return {"__truncated__": True, "raw_prefix": final_str[:1500] + "…[truncated]"}
    return compressed


# ── Confidence detection ──────────────────────────────────────────────────────

def _confidence_hint(tool_name: str, result: dict) -> Optional[str]:
    """
    Return a hint string to inject after a tool result when evidence looks complete.
    Returns None if no hint is warranted.
    """
    if tool_name == "get_hf_metadata":
        splits = result.get("splits", [])
        ds_id  = result.get("dataset_id", "")
        downloads = result.get("downloads")
        if splits:
            hint = f"[提示] get_hf_metadata 已返回完整 splits 列表：{splits}。"
            if downloads is not None and downloads < 10:
                hint += (
                    f"\n[⚠️ 低可信度] downloads={downloads}，下载量极低，"
                    "可能是空壳仓库或 fork，建议用 verify_hf_dataset 交叉验证，"
                    "或用 web_search 确认这是否为原始数据集。"
                )
            else:
                hint += "如果这足以回答用户的问题，请直接调用 finish。"
            return hint
        else:
            return (
                f"[⚠️ 红牌警告] get_hf_metadata('{ds_id}') 返回 splits=[]，"
                "说明这不是论文使用的正确数据集版本（HF 上的 Wikipedia/Reddit 等是文本语料库，"
                "与时序图/学术 benchmark 数据集完全不同）。"
                "禁止将此 HF URL 写入 finish 的 links 字段。"
                "请立即改用 web_search 搜索 '<数据集名> dataset download github' 找到真正来源，"
                "或调用 get_paper_code_repos 找论文代码仓库。"
            )

    if tool_name == "verify_hf_dataset":
        quality = result.get("quality", "")
        ds_id = result.get("dataset_id", "")
        issues = result.get("issues", [])
        if quality == "likely_invalid":
            return (
                f"[🚫 验证失败] verify_hf_dataset('{ds_id}') 判定为 likely_invalid。\n"
                f"问题：{'; '.join(issues)}\n"
                "禁止将此 HF URL 写入 finish。请改用 web_search 搜索真正来源。"
            )
        elif quality == "suspect":
            return (
                f"[⚠️ 可疑] verify_hf_dataset('{ds_id}') 判定为 suspect。\n"
                f"问题：{'; '.join(issues)}\n"
                "建议用 web_search 交叉验证后再决定是否写入 finish。"
            )
        elif quality == "good":
            splits = result.get("splits", [])
            return (
                f"[✅ 验证通过] verify_hf_dataset('{ds_id}') 判定为 good，"
                f"splits={splits}。可以直接 finish。"
            )

    if tool_name == "get_hf_dataset_card":
        yaml_splits = result.get("splits_from_yaml", [])
        if yaml_splits:
            names = [s.get("name", "") for s in yaml_splits]
            return (
                f"[提示] README YAML frontmatter 中包含完整的 splits 定义：{names}。"
                "如果这足以回答用户的问题，请直接调用 finish。"
            )

    if tool_name == "get_github_readme":
        dl = result.get("data_links") or {}
        cloud_links = dl.get("cloud_links", [])
        hf_ids = dl.get("hf_ids", [])
        if cloud_links:
            return (
                f"[⚡ 快速结束] get_github_readme 已找到 {len(cloud_links)} 个云存储/直接下载链接：\n"
                + "\n".join(f"  - {u}" for u in cloud_links[:5])
                + "\n立刻调用 finish，将这些链接写入每个数据集的 links 字段。无需再搜索。"
            )
        if hf_ids:
            return (
                f"[提示] GitHub README 中发现 HF 数据集 ID：{hf_ids}。"
                "建议调用 get_hf_metadata 确认 splits，然后 finish。"
            )

    if tool_name == "get_zenodo_record":
        access = result.get("access", "")
        files  = result.get("files", [])
        if access == "open" and files:
            dl_urls = [f.get("download_url") for f in files if f.get("download_url")]
            if dl_urls:
                return (
                    f"[⚡ 快速结束] Zenodo 记录为 open access，已有 {len(dl_urls)} 个直接下载链接。"
                    "立刻调用 finish，将 download_url 写入 links 字段。"
                )

    if tool_name == "fetch_webpage_text":
        error = result.get("error", "")
        url   = result.get("url", "")
        text  = result.get("text", "")
        if "403" in error:
            return (
                f"[🚫 网站屏蔽] {url} 返回 HTTP 403（禁止访问），该网站封锁了自动抓取。"
                "立即停止访问该域名的其他 URL。"
                "改用 web_search 搜索该数据集的 GitHub 仓库、Zenodo 记录或 HuggingFace 页面。"
            )
        if not text.strip():
            return (
                f"[⚠️ 空页面] {url} 返回了空内容，可能需要 JavaScript 渲染。"
                "不要再重试该 URL，改用其他搜索工具。"
            )

    if tool_name in ("search_pwc_dataset", "search_dataset"):
        if result.get("has_evaluations_signal"):
            return (
                "[提示] PapersWithCode 显示该数据集有 evaluation 记录（has_evaluations=True），"
                "说明它在学术界被广泛使用。"
            )

    if tool_name == "search_hf_hub":
        datasets = result.get("datasets", [])
        low_dl_exact = [
            ds for ds in datasets
            if ds.get("exact_match") and (ds.get("downloads", 0) or 0) < 10
        ]
        if low_dl_exact:
            ids = [ds.get("id", "") for ds in low_dl_exact]
            return (
                f"[⚠️ 低可信度精确匹配] 以下 HF 数据集虽然 ID 精确匹配，但下载量极低：{ids}。"
                "可能是空壳仓库或 fork，建议用 verify_hf_dataset 验证后再使用。"
            )
        if result.get("filtered_out"):
            return (
                f"[提示] 已自动过滤 {result['filtered_out']} 个空壳/fork 仓库 "
                "(downloads<5 且 likes=0)。"
            )

    return None


# ── Planner entity tracking ───────────────────────────────────────────────────

def _update_entity_status(
    structured_memory: dict,
    tool_name: str,
    args: dict,
    result: dict,
) -> None:
    """
    Update entity_status based on tool results. An entity progresses:
      pending -> discovered (found in search results)
      discovered -> verified (confirmed via metadata/card/files)
    """
    entities = structured_memory.get("planner_entities", [])
    if not entities:
        return
    entity_status = structured_memory.get("entity_status", {})

    result_text = json.dumps(result, ensure_ascii=False, default=str).lower()

    for entity in entities:
        current = entity_status.get(entity, "pending")
        entity_lower = entity.lower()
        tokens = [t for t in re.split(r"[\s\-_/]+", entity_lower) if len(t) >= 3]

        if current == "pending":
            has_match = entity_lower in result_text or (
                tokens and all(t in result_text for t in tokens)
            )
            if has_match and tool_name in (
                "search_dataset", "search_hf_hub", "web_search", "tavily_search",
                "search_zenodo", "search_opendatalab", "search_pwc_dataset",
                "search_semantic_scholar", "get_github_readme",
            ):
                entity_status[entity] = "discovered"

        elif current == "discovered":
            if tool_name in ("get_hf_metadata", "get_hf_dataset_card",
                             "get_hf_dataset_files", "verify_hf_dataset",
                             "get_zenodo_record"):
                query_text = json.dumps(args, ensure_ascii=False, default=str).lower()
                if entity_lower in query_text or any(t in query_text for t in tokens):
                    entity_status[entity] = "verified"

    structured_memory["entity_status"] = entity_status


# ── Scratchpad state card ────────────────────────────────────────────────────

def _build_scratchpad(
    turn: int,
    tool_name: str,
    result: dict,
    structured_memory: dict,
) -> str:
    """
    Build a brief state annotation for the agent to stay oriented.
    This acts as a Structured Evidence Memory Retrieval layer.
    """
    lines = [f"[当前进度 · 第 {turn} 轮 · {tool_name}]"]

    # 1. Negative Evidence (prevent repetitive failures)
    failed = structured_memory.get("failed_queries", set())
    if failed:
        lines.append(f"❌ 已确认无结果的搜索词: {', '.join(list(failed)[-3:])}")
    blocked_domains = structured_memory.get("blocked_domains", set())
    if blocked_domains:
        lines.append(f"🚫 已屏蔽域名（返回403/空页面，禁止继续访问）: {', '.join(blocked_domains)}")

    # 2. Confirmed Datasets & Splits
    known_hf_ids = structured_memory.get("known_hf_ids", set())
    confirmed_splits = structured_memory.get("confirmed_splits", {})
    verified_links = structured_memory.get("verified_links", {})
    cloud_links_found = structured_memory.get("cloud_links_found", set())

    if known_hf_ids:
        lines.append(f"✅ 已发现 HF 数据集候选: {', '.join(sorted(known_hf_ids)[:5])}")

    for ds_id, splits in confirmed_splits.items():
        lines.append(f"✅ 已验证 splits [{ds_id}]: {splits}")

    if cloud_links_found:
        lines.append(f"⚡ 已发现云存储/直接下载链接 ({len(cloud_links_found)} 个): {', '.join(list(cloud_links_found)[:3])}")
        lines.append("→ 立刻 finish，将 cloud_links 写入每个数据集的 links 字段！")
    else:
        for entity, links in verified_links.items():
            if links:
                lines.append(f"🔗 有效下载链接 [{entity}]: {', '.join(list(links)[:2])}")

    # 3. Planner entity status tracking
    entity_status = structured_memory.get("entity_status", {})
    if entity_status:
        pending_entities = [e for e, s in entity_status.items() if s == "pending"]
        discovered_entities = [e for e, s in entity_status.items() if s == "discovered"]
        verified_entities = [e for e, s in entity_status.items() if s == "verified"]

        if verified_entities:
            lines.append(f"🟢 已验证实体: {', '.join(verified_entities)}")
        if discovered_entities:
            lines.append(f"🟡 已发现但未验证: {', '.join(discovered_entities)}")
        if pending_entities:
            lines.append(
                f"🔴 尚未覆盖的实体: {', '.join(pending_entities)} "
                "— 请对这些实体执行搜索！"
            )

    # 4. What this tool just revealed
    if tool_name == "get_hf_metadata":
        splits = result.get("splits", [])
        src = result.get("splits_source", "")
        if splits:
            extra = f" (来源: {src})" if src and src != "api" else ""
            lines.append(f"👉 本轮新增: splits = {splits}{extra}")
        else:
            lines.append("👉 本轮结果: splits 为空，建议用 verify_hf_dataset 一站式验证")
    elif tool_name == "verify_hf_dataset":
        quality = result.get("quality", "")
        splits = result.get("splits", [])
        issues = result.get("issues", [])
        lines.append(f"👉 验证结果: quality={quality}, splits={splits}")
        if issues:
            lines.append(f"   问题: {'; '.join(issues[:3])}")
    elif tool_name in ("search_dataset", "search_hf_hub"):
        hf_ids = result.get("hf_ids_found", [])
        if hf_ids:
            lines.append(f"👉 本轮新增 HF ID 候选: {hf_ids}")
        else:
            lines.append(f"👉 本轮搜索命中: {result.get('count', 0)} 条记录，未直接发现 HF ID")
    elif tool_name == "get_github_readme":
        dl = result.get("data_links", {})
        cloud = dl.get("cloud_links", [])
        hf = dl.get("hf_ids", [])
        if cloud:
            lines.append(f"⚡ GitHub README 包含 cloud_links：{cloud[:3]}")
        elif hf:
            lines.append(f"GitHub README 包含 HF 链接：{hf}")
        else:
            lines.append("GitHub README 未找到数据链接，可尝试 get_github_dir 探索 data/ 子目录")
    elif tool_name == "get_hf_dataset_card":
        yaml_splits = [s.get("name") for s in result.get("splits_from_yaml", [])]
        if yaml_splits:
            lines.append(f"YAML frontmatter splits：{yaml_splits}")
    elif tool_name == "search_zenodo":
        total = result.get("total", 0)
        if total:
            lines.append(f"👉 Zenodo 命中 {total} 条记录，可调 get_zenodo_record 获取下载链接")
        else:
            lines.append("👉 Zenodo 无结果，尝试 web_search 或 search_opendatalab")

    # Next steps suggestion (context-aware)
    if not confirmed_splits and not known_hf_ids and not cloud_links_found:
        if pending_entities:
            lines.append(f"下一步建议：对未覆盖实体 {pending_entities[:3]} 执行 search_hf_hub 或 web_search")
        else:
            lines.append("下一步建议：用 verify_hf_dataset 一站式验证候选，或继续多平台搜索")
    elif discovered_entities:
        lines.append(f"下一步建议：对已发现的 {discovered_entities[:3]} 调用 verify_hf_dataset 或 get_hf_metadata 验证")

    # Tool Budget warnings
    tool_usage = structured_memory.get("tool_usage", {})
    # TOOL_BUDGETS is defined in run_agent scope; read from memory if stored there
    budgets = structured_memory.get("_tool_budgets", {})
    budget_warnings = []
    for t, budget in budgets.items():
        used = tool_usage.get(t, 0)
        if used >= budget:
            budget_warnings.append(f"⚠️ {t}: 已用 {used}/{budget} 次，请改用其他工具")
        elif used == budget - 1:
            budget_warnings.append(f"⚡ {t}: 剩余 1 次配额")
    if budget_warnings:
        lines.append("\n".join(budget_warnings))

    return "\n".join(lines)


# ── Retry logic ───────────────────────────────────────────────────────────────

async def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    turn: int,
):
    """
    Call the LLM with tiered retry logic:
      - 401/403/invalid_api_key → immediate permanent error (no retry)
      - 429 → wait 3s, retry once
      - 5xx / network timeout → exponential backoff (1s, 2s), max 2 retries
    Returns (response, error_message). Exactly one of them is None.
    """
    max_retries = 2
    backoff_seconds = [1, 2]

    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(model=model, messages=messages, max_tokens=3000)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = await asyncio.to_thread(
                client.chat.completions.create,
                **kwargs,
            )
            return response, None

        except Exception as e:
            error_str = str(e)

            # Extract human-readable detail from DashScope JSON error
            _m = re.search(r"'message':\s*'([^']+)'", error_str)
            detail = _m.group(1) if _m else error_str[:200]

            # Permanent errors — do not retry
            if "401" in error_str or "invalid_api_key" in error_str:
                return None, f"API Key 无效：{detail}"
            if "403" in error_str:
                return None, f"无权限（可能未开通 {model}）：{detail}"

            # Rate limit — wait 3s, retry once
            if "429" in error_str:
                if attempt < 1:
                    logger.warning(f"Rate limited (turn {turn}), waiting 3s before retry…")
                    await asyncio.sleep(3)
                    continue
                return None, "API 限流，重试后仍失败，请稍后再试"

            # Model not found — permanent
            if "model" in error_str.lower() and (
                "not found" in error_str.lower() or "does not exist" in error_str.lower()
            ):
                return None, f"模型不存在，请检查账号是否开通 {model}：{detail}"

            # 5xx / network — exponential backoff
            if attempt < max_retries:
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                logger.warning(f"LLM call failed (turn {turn}, attempt {attempt + 1}): {e}. Retrying in {wait}s…")
                await asyncio.sleep(wait)
                continue

            # Exhausted retries
            return None, f"LLM 调用失败（已重试 {max_retries} 次）：{detail}"

    return None, "未知错误"


# ── Planner Layer ─────────────────────────────────────────────────────────────

PLANNER_PROMPT = """\
你是一个高级 Dataset Research Planner。
你的任务是根据用户的输入（可能是纯文本提问，也可能是包含论文提取文本的长段落），制定一个**执行计划（Execution Plan）**。

请遵循以下分析与格式输出要求：
1. **分析意图与核心实体**：提取输入中提到的论文名、缩写、具体数据集要求（如"有测试集"），去除无关废话的句子。
2. **需要收集的证据(Need)**：明确目前还需要寻找什么。
3. **优先搜索策略排序(Priority Order)**：按最高效的路径排序，例如若输入中有 GitHub 链接，则优先读 README；如果没有链接但有标题，优先用论文标题搜索。

请以严格的 JSON 格式输出，不要包含反引号或 markdown 格式，只输出 JSON。

示例格式：
{
  "extracted_entities": ["Dataset A", "Model B"],
  "paper_title": "如果存在的话",
  "has_direct_code_link": false,
  "evidence_needs": ["HuggingFace URL", "License type", "Splits validation"],
  "priority_routing": ["search_hf_hub", "search_dataset", "web_search"],
  "reasoning_steps": [
     "首先在 HuggingFace 直接搜索", 
     "如果找不到，展开全称搜索 web_search"
  ]
}
"""

async def run_planner(user_query: str, client: OpenAI, model: str = DEFAULT_MODEL) -> dict:
    """
    Run the planner step to generate an execution plan.
    """
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"请为以下查询制定计划：\n\n{user_query}"}
    ]
    response, error = await _call_llm_with_retry(client, model, messages, tools=[], turn=0)
    if error or not response:
        return {"error": error or "Planner failed"}
        
    content = response.choices[0].message.content or "{}"
    try:
        # cleanup potential markdown
        import re
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content).strip()
        plan = json.loads(content)
        return plan
    except Exception as e:
        logger.warning(f"Planner JSON decode failed: {e}. Raw content: {content}")
        return {"raw_plan": content, "error": "JSON parse error"}


# ── Main agent loop ───────────────────────────────────────────────────────────

async def run_agent(
    user_query: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_tool_calls: int = 30,
) -> AsyncGenerator[dict, None]:
    """
    Core agent loop.

    Yields dicts with shape:
      {"event": "agent_thought", "turn": int, "text": str}
      {"event": "tool_call",    "tool": str, "args": dict, "call_num": int}
      {"event": "tool_result",  "tool": str, "result": dict, "call_num": int}
      {"event": "done",         "results": list[dict]}
      {"event": "error",        "message": str}
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    call_num = 0
    turn     = 0
    finish_rejections = 0  # reject at most once to avoid infinite loops
    text_correction_injected = False  # inject at most once when agent skips finish

    # Tracking for scratchpad & Structured Evidence Memory
    structured_memory = {
        "known_hf_ids":      set(),
        "confirmed_splits":  {},     # hf_id -> splits
        "verified_links":    {},     # hf_id/dataset_name -> set of links
        "failed_queries":    set(),  # queries that yielded 0 results
        "tool_usage":        {},     # tool_name -> int (call count)
        "cloud_links_found": set(),  # direct download / cloud-storage URLs found
        "blocked_domains":   set(),  # domains that returned 403/empty
        "planner_entities":  [],     # entities extracted by planner
        "entity_status":     {},     # entity -> "pending" | "discovered" | "verified"
        "planner_domain":    None,   # domain hint from planner (e.g. "时序图")
    }

    # Extract planner entities from query (planner JSON is embedded in user_query)
    _plan_match = re.search(r"【Planner 生成的执行计划】.*?(\{.*\})", user_query, re.DOTALL)
    if _plan_match:
        try:
            _plan_json = json.loads(_plan_match.group(1))
            _entities = _plan_json.get("extracted_entities", [])
            if isinstance(_entities, list):
                structured_memory["planner_entities"] = [
                    e.strip() for e in _entities if isinstance(e, str) and e.strip()
                ]
                structured_memory["entity_status"] = {
                    e: "pending" for e in structured_memory["planner_entities"]
                }
            _domain = _plan_json.get("domain") or _plan_json.get("paper_domain")
            if _domain:
                structured_memory["planner_domain"] = str(_domain)
        except (json.JSONDecodeError, TypeError):
            pass

    # Per-tool call budgets (soft limit: shows warning in scratchpad)
    TOOL_BUDGETS: dict[str, int] = {
        "web_search":              5,
        "search_semantic_scholar": 2,
        "fetch_webpage_text":      3,   # reduced: 403 sites waste budget
        "search_hf_hub":           5,
    }
    structured_memory["_tool_budgets"] = TOOL_BUDGETS

    while call_num < max_tool_calls:
        turn += 1

        # ── Ask LLM for next action ──────────────────────────────────────
        response, error_msg = await _call_llm_with_retry(
            client, model, messages, TOOL_DEFINITIONS, turn
        )
        if error_msg:
            yield {"event": "error", "message": error_msg}
            return

        choice = response.choices[0]
        msg    = choice.message

        # Serialize for message history
        msg_dict: dict = {"role": "assistant"}
        if msg.content:
            msg_dict["content"] = msg.content
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if msg.content:
            yield {"event": "agent_thought", "turn": turn, "text": msg.content}

        if not msg.tool_calls:
            # Agent produced text without calling finish — inject a correction and retry once
            if msg.content and not text_correction_injected:
                text_correction_injected = True
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ 错误：你输出了文字回复，但本 Agent 只能通过调用 finish 工具输出结论，"
                        "不允许直接输出文字。请立即调用 finish 工具，将你刚才分析出的数据集信息填入 "
                        "datasets 字段。如果某个数据集没有找到下载链接，links 字段可以填写来源页面 "
                        "URL 或留空，但不得跳过 finish 调用。"
                    ),
                })
                continue
            yield {"event": "done", "results": []}
            return

        # ── Execute tool calls in PARALLEL ──────────────────────────────
        # Parse all tool calls first
        parsed_calls: list[tuple[str, dict, str]] = []   # (tool_name, args, tc_id)
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            parsed_calls.append((tool_name, args, tc.id))

        # Yield tool_call events (before execution), record per-call call_num
        call_nums: list[int] = []
        for tool_name, args, _tc_id in parsed_calls:
            call_num += 1
            call_nums.append(call_num)
            yield {"event": "tool_call", "tool": tool_name, "args": args, "call_num": call_num}
            if call_num >= max_tool_calls:
                break
        
        # Tool Cache (avoid redundant API calls within the same session)
        tool_cache = structured_memory.setdefault("_tool_cache", {})

        # Execute all calls concurrently
        async def _execute(tool_name: str, args: dict) -> dict:
            fn = TOOL_FUNCTIONS.get(tool_name)
            if fn is None:
                return {"error": f"未知工具: {tool_name}"}

            # Track tool usage (before cache check, to reflect intent)
            tool_usage = structured_memory.setdefault("tool_usage", {})
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            # 403 domain auto-blocking: reject fetch_webpage_text to known-blocked domains
            if tool_name == "fetch_webpage_text":
                blocked = structured_memory.get("blocked_domains", set())
                req_url = args.get("url", "")
                if req_url and blocked:
                    try:
                        from urllib.parse import urlparse as _urlparse
                        req_domain = _urlparse(req_url).netloc
                        if req_domain and req_domain in blocked:
                            return {
                                "error": f"域名 {req_domain} 已被标记为屏蔽（此前返回 403 或空页面），"
                                         "请改用 web_search 或 search_dataset 搜索该资源。",
                                "url": req_url,
                                "_blocked": True,
                            }
                    except Exception:
                        pass

            args_str = json.dumps(args, sort_keys=True)
            cache_key = f"{tool_name}::{args_str}"
            if cache_key in tool_cache:
                _cache_turn = tool_cache.get(f"{cache_key}::_turn", "?")
                return {
                    "_cached": True,
                    "_cache_hint": (
                        f"⚠️ 此调用 ({tool_name}) 与第 {_cache_turn} 轮完全相同，"
                        "结果不会改变。请换用不同的搜索词、其他工具、或不同参数重试。"
                    ),
                    **tool_cache[cache_key],
                }
                
            # S2 needs more time for its own retry loop (12s×3 + sleeps ≈ 43s)
            _TOOL_TIMEOUTS: dict[str, float] = {
                "search_semantic_scholar": 60.0,
            }
            _timeout = _TOOL_TIMEOUTS.get(tool_name, 30.0)
            try:
                res = await asyncio.wait_for(fn(**args), timeout=_timeout)
                tool_cache[cache_key] = res
                tool_cache[f"{cache_key}::_turn"] = turn
                return res
            except asyncio.TimeoutError:
                logger.error(f"Tool {tool_name} timed out after {_timeout}s")
                return {"error": f"执行超时 (Timeout after {int(_timeout)}s)"}
            except TypeError as e:
                return {"error": f"参数错误: {e}"}
            except Exception as e:
                logger.warning(f"Tool '{tool_name}' raised: {e}")
                return {"error": str(e)}

        results_list = await asyncio.gather(
            *[_execute(tn, a) for tn, a, _ in parsed_calls],
            return_exceptions=True,
        )

        # Process results — two passes to keep message order valid:
        # Pass 1: yield SSE events + append ALL tool messages first (protocol requires
        #         tool messages to be consecutive after the assistant tool_calls message).
        # Pass 2: append hints / check finish (only after all tool messages are in history).
        finish_payload = None
        pending_hints: list[str] = []
        resolved: list[tuple[str, dict, str, dict]] = []  # (tool_name, args, tc_id, result)

        for (tool_name, args, tc_id), result, cn in zip(parsed_calls, results_list, call_nums):
            if isinstance(result, Exception):
                result = {"error": str(result)}

            # Yield full result to frontend immediately (SSE)
            yield {"event": "tool_result", "tool": tool_name, "result": result, "call_num": cn}

            # Update Structured Evidence Memory
            if result.get("count") == 0:
                q = args.get("query")
                if q: structured_memory["failed_queries"].add(q)

            if tool_name == "fetch_webpage_text":
                error = result.get("error", "")
                url   = result.get("url", "")
                if ("403" in error or not result.get("text", "").strip()) and url:
                    # Extract domain and mark it as blocked
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            structured_memory["blocked_domains"].add(domain)
                    except Exception:
                        pass
                
            if tool_name == "get_hf_metadata":
                ds_id  = args.get("dataset_id", "")
                splits = result.get("splits", [])
                if ds_id:
                    structured_memory["known_hf_ids"].add(ds_id)
                    hf_url = result.get("hf_url")
                    if hf_url:
                        structured_memory["verified_links"].setdefault(ds_id, set()).add(hf_url)
                if splits and ds_id:
                    structured_memory["confirmed_splits"][ds_id] = splits
            elif tool_name in ("search_dataset", "search_hf_hub"):
                for hid in result.get("hf_ids_found", []):
                    structured_memory["known_hf_ids"].add(hid)
                for ds in result.get("datasets", []):
                    if ds.get("id"):
                        structured_memory["known_hf_ids"].add(ds["id"])
                        url = ds.get("url")
                        if url:
                            structured_memory["verified_links"].setdefault(ds["id"], set()).add(url)
            elif tool_name == "get_github_readme":
                dl = result.get("data_links") or {}
                for hid in dl.get("hf_ids", []):
                    structured_memory["known_hf_ids"].add(hid)
                    structured_memory["verified_links"].setdefault(hid, set()).add(f"https://huggingface.co/datasets/{hid}")
                for cl in dl.get("cloud_links", []):
                    structured_memory["cloud_links_found"].add(cl)
                    structured_memory["verified_links"].setdefault("cloud_links", set()).add(cl)

            # Update planner entity status based on tool results
            _update_entity_status(structured_memory, tool_name, args, result)

            # Append compressed tool message to history (pass 1)
            compressed = _compress_tool_result(tool_name, result)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc_id,
                "content":      json.dumps(compressed, ensure_ascii=False, default=str),
            })

            resolved.append((tool_name, args, tc_id, result))

        # Pass 2: hints + finish detection (after ALL tool messages are in history)
        for tool_name, args, tc_id, result in resolved:
            hint = _confidence_hint(tool_name, result)
            if hint:
                pending_hints.append(hint)

            # Inject strong warning on cache hit or blocked domain
            if result.get("_cache_hint"):
                pending_hints.append(result["_cache_hint"])
            if result.get("_blocked"):
                pending_hints.append(
                    f"[🚫 域名拦截] {args.get('url', '')} 的域名已被自动屏蔽，"
                    "此前该域名返回 403 或空页面。请改用 web_search 或其他工具。"
                )

            if tool_name == "finish" and isinstance(result, dict) and result.get("__finish__"):
                finish_payload = result

        # Append hints as a single combined user message (avoid noisy multi-message injection)
        if pending_hints:
            messages.append({"role": "user", "content": "\n".join(pending_hints)})

        # Scratchpad state card (one per turn, summarising all tool results this round)
        if parsed_calls:
            last_tool_name, _, _ = parsed_calls[-1]
            last_result = results_list[-1] if not isinstance(results_list[-1], Exception) else {}
            scratchpad = _build_scratchpad(
                turn, last_tool_name, last_result,
                structured_memory,
            )
            messages.append({"role": "assistant", "content": scratchpad})

        if finish_payload:
            datasets = finish_payload.get("datasets", [])

            # ── Auto-backfill links from structured_memory ────────────────
            verified_links = structured_memory.get("verified_links", {})
            cloud_links_found = structured_memory.get("cloud_links_found", set())

            for ds in datasets:
                if not ds.get("links"):
                    ds_name_lower = ds.get("name", "").lower()
                    backfilled = []
                    # Prefer cloud_links for ALL datasets (shared download source)
                    if cloud_links_found:
                        backfilled.extend(list(cloud_links_found))
                    else:
                        for entity, links in verified_links.items():
                            if entity == "cloud_links" or entity.lower() in ds_name_lower or ds_name_lower in entity.lower():
                                backfilled.extend(list(links))
                    if backfilled:
                        ds["links"] = list(dict.fromkeys(backfilled))[:3]
                        logger.info(f"Auto-backfilled links for '{ds.get('name')}': {ds['links']}")

            # ── Validation gateway: structural quality checks ──────────
            from .skills import validate_dataset_result
            try:
                datasets, val_warnings = await validate_dataset_result(
                    datasets, structured_memory
                )
                finish_payload["datasets"] = datasets
                if val_warnings and finish_rejections < 2:
                    # Only reject for validation issues once
                    severe = [w for w in val_warnings if "空壳" in w or "极低" in w]
                    if severe and finish_rejections == 0:
                        finish_rejections += 1
                        messages.append({"role": "user", "content": (
                            f"【验证网关】数据集质量检查发现以下问题：\n"
                            + "\n".join(f"- {w}" for w in severe) +
                            "\n请针对有问题的数据集重新搜索正确版本"
                            "（用 web_search 找原始来源），或降低 confidence 后重新 finish。"
                        )})
                        finish_payload = None
                        continue
            except Exception as e:
                logger.warning(f"Validation gateway error: {e}")

            empty_ds = [d["name"] for d in datasets if not d.get("links")]
            has_any  = any(d.get("links") for d in datasets)

            # Reject (1): agent submitted empty list — must search more
            if not datasets and finish_rejections == 0:
                finish_rejections += 1
                messages.append({"role": "user", "content": (
                    "【拒绝提交】你返回了空的数据集列表，但论文节选中提到了数据集。"
                    "请继续搜索：对每个数据集名称单独用 web_search 搜索 "
                    "'<数据集名> dataset download github'，或使用 get_paper_code_repos "
                    "找到原始论文的代码仓库（通常含有数据下载说明）。"
                )})
                finish_payload = None
            # Reject (2): some datasets have links but others don't — try to complete (once)
            elif empty_ds and has_any and finish_rejections == 0:
                finish_rejections += 1
                hint = (
                    f"【链接完整性检查】以下数据集没有 links：{', '.join(empty_ds)}。"
                    "请针对每个缺链接的数据集，用 web_search 搜索 "
                    "'<数据集名> dataset download site:github.com' 或 "
                    "'<数据集名> dataset huggingface'，找到后写入 links 并重新 finish。"
                    "若搜索后仍找不到，直接 finish（空 links 可接受）。"
                )
                messages.append({"role": "user", "content": hint})
                finish_payload = None
            else:
                yield {
                    "event":   "done",
                    "results": finish_payload.get("datasets", []),
                    "reason":  finish_payload.get("reason", ""),
                }
                return

        if call_num >= max_tool_calls:
            break

    yield {"event": "done", "results": [], "message": "已达最大工具调用次数，检索结束"}