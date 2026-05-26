"""
main.py
FastAPI server for Dataset Retrieval Agent

Endpoints:
  POST /api/search         → streaming SSE search (text input)
  POST /api/search/pdf     → streaming SSE search (PDF upload)
  GET  /api/health         → health check
  GET  /api/kb             → list knowledge base entries
"""

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.pipeline import stream_agent, stream_agent_events
from agent.config import set_api_keys, reset_api_keys, configured_keys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dataset Retrieval Agent",
    description="Paper-driven dataset discovery system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    text: str
    options: Optional[dict] = {}


# ── Streaming helper ───────────────────────────────────────────────────────

def _extract_api_keys(
    x_api_key: Optional[str],
    x_github_token: Optional[str],
    x_hf_token: Optional[str],
    x_s2_key: Optional[str],
    x_tavily_key: Optional[str],
) -> dict:
    """Build the per-request API key bundle from inbound headers."""
    return {
        "dashscope":        (x_api_key or "").strip(),
        "github":           (x_github_token or "").strip(),
        "huggingface":      (x_hf_token or "").strip(),
        "semantic_scholar": (x_s2_key or "").strip(),
        "tavily":           (x_tavily_key or "").strip(),
    }


async def _stream(text: str = "", pdf_path: str = "", api_key: str = "",
                  api_keys: Optional[dict] = None):
    """Wraps stream_agent and appends a stream_end sentinel."""
    token = set_api_keys(api_keys or {"dashscope": api_key})
    try:
        async for chunk in stream_agent(text=text, pdf_path=pdf_path, api_key=api_key):
            yield chunk
            await asyncio.sleep(0)
    except Exception as e:
        logger.exception("Agent error")
        yield f'data: {{"event":"error","message":{json.dumps(str(e))}}}\n\n'
    finally:
        reset_api_keys(token)
        yield 'data: {"event":"stream_end"}\n\n'


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _stream_multi_pdfs(
    question: str,
    pdf_items: list[tuple[int, str, str]],
    api_key: str = "",
    api_keys: Optional[dict] = None,
    max_concurrency: int = 3,
):
    """
    Run multi-paper retrieval in parallel and multiplex SSE events.

    pdf_items: [(paper_index, file_name, file_path), ...]
    """
    if not pdf_items:
        yield _sse({"event": "error", "message": "请至少上传 1 个 PDF"})
        yield _sse({"event": "stream_end"})
        return

    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(max(1, max_concurrency))
    bundle = api_keys or {"dashscope": api_key}

    async def _worker(paper_index: int, file_name: str, file_path: str):
        await queue.put({
            "event": "paper_start",
            "paper_index": paper_index,
            "paper_name": file_name,
        })
        token = set_api_keys(bundle)
        try:
            async with sem:
                async for ev in stream_agent_events(
                    text=question,
                    pdf_path=file_path,
                    api_key=api_key,
                ):
                    ev["paper_index"] = paper_index
                    ev["paper_name"] = file_name
                    await queue.put(ev)
        except Exception as e:
            logger.exception("Multi-paper worker error")
            await queue.put({
                "event": "error",
                "paper_index": paper_index,
                "paper_name": file_name,
                "message": str(e),
            })
        finally:
            reset_api_keys(token)
            await queue.put({
                "event": "paper_end",
                "paper_index": paper_index,
                "paper_name": file_name,
            })
            await queue.put({"event": "__worker_done__"})

    tasks = [
        asyncio.create_task(_worker(paper_index=i, file_name=name, file_path=path))
        for i, name, path in pdf_items
    ]

    active_workers = len(tasks)
    try:
        while active_workers > 0:
            ev = await queue.get()
            if ev.get("event") == "__worker_done__":
                active_workers -= 1
                continue
            yield _sse(ev)
            await asyncio.sleep(0)
        yield _sse({"event": "multi_done", "paper_count": len(pdf_items)})
    except Exception as e:
        logger.exception("Multi stream failed")
        yield _sse({"event": "error", "message": str(e)})
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        yield _sse({"event": "stream_end"})


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "server_keys": configured_keys(),  # which keys are set via env on server side
    }


@app.get("/api/config")
async def config_status():
    """Return server-side key presence so the UI knows which fields are pre-filled."""
    return {"server_keys": configured_keys()}


@app.post("/api/test-key")
async def test_key(x_api_key: Optional[str] = Header(None)):
    """Quick LLM ping to validate an API key — returns {ok, model, error}."""
    from openai import OpenAI
    key = (x_api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        return {"ok": False, "error": "未提供 API Key"}
    client = OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        r = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=3,
        )
        return {"ok": True, "model": r.model}
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}


@app.post("/api/search")
async def search_text(
    req: SearchRequest,
    x_api_key: Optional[str] = Header(None),
    x_github_token: Optional[str] = Header(None),
    x_hf_token: Optional[str] = Header(None),
    x_s2_key: Optional[str] = Header(None),
    x_tavily_key: Optional[str] = Header(None),
):
    """
    Search by text input (abstract, paper text, or dataset name).
    Returns Server-Sent Events stream.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is required")

    api_keys = _extract_api_keys(x_api_key, x_github_token, x_hf_token, x_s2_key, x_tavily_key)
    effective_key = api_keys["dashscope"] or os.getenv("DASHSCOPE_API_KEY", "")
    return StreamingResponse(
        _stream(text=req.text, api_key=effective_key, api_keys=api_keys),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/search/pdf")
async def search_pdf(
    file: UploadFile = File(...),
    paper_title: str = Form(""),
    question: str = Form(""),   # user's question alongside the PDF (optional but recommended)
    x_api_key: Optional[str] = Header(None),
    x_github_token: Optional[str] = Header(None),
    x_hf_token: Optional[str] = Header(None),
    x_s2_key: Optional[str] = Header(None),
    x_tavily_key: Optional[str] = Header(None),
):
    """
    Search by PDF upload.
    Returns Server-Sent Events stream.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    api_keys = _extract_api_keys(x_api_key, x_github_token, x_hf_token, x_s2_key, x_tavily_key)
    effective_key = api_keys["dashscope"] or os.getenv("DASHSCOPE_API_KEY", "")

    async def stream_and_cleanup():
        try:
            async for chunk in _stream(text=question, pdf_path=tmp_path,
                                       api_key=effective_key, api_keys=api_keys):
                yield chunk
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/search/pdfs")
async def search_pdfs(
    files: list[UploadFile] = File(...),
    question: str = Form(""),
    x_api_key: Optional[str] = Header(None),
    x_github_token: Optional[str] = Header(None),
    x_hf_token: Optional[str] = Header(None),
    x_s2_key: Optional[str] = Header(None),
    x_tavily_key: Optional[str] = Header(None),
):
    """
    Search with multiple PDF files in parallel.
    Returns SSE with per-paper events:
      - paper_index
      - paper_name
      - per-paper tool/agent/done events
    """
    if not files:
        raise HTTPException(status_code=400, detail="至少上传 1 个 PDF 文件")
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="单次最多支持 8 个 PDF 文件")

    import shutil
    import tempfile

    tmp_paths: list[str] = []
    pdf_items: list[tuple[int, str, str]] = []
    for idx, file in enumerate(files):
        filename = file.filename or f"paper_{idx + 1}.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"仅支持 PDF 文件: {filename}")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_paths.append(tmp.name)
            pdf_items.append((idx, filename, tmp.name))

    api_keys = _extract_api_keys(x_api_key, x_github_token, x_hf_token, x_s2_key, x_tavily_key)
    effective_key = api_keys["dashscope"] or os.getenv("DASHSCOPE_API_KEY", "")

    async def stream_and_cleanup():
        try:
            async for chunk in _stream_multi_pdfs(
                question=question,
                pdf_items=pdf_items,
                api_key=effective_key,
                api_keys=api_keys,
            ):
                yield chunk
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root():
    return {"message": "Dataset Retrieval Agent API", "docs": "/docs"}
