"""
fg_sync/proxy.py
----------------
Thin async HTTP proxy that sits between any Ollama client and Ollama itself.

  Client → localhost:11435 (fg-proxy) → localhost:11434 (Ollama)

On every POST /api/chat or POST /api/generate, the proxy:
  1. Reads and buffers the full request body
  2. Forwards it to Ollama, collecting the full response (streaming supported)
  3. Appends a capture record to capture.jsonl
  4. Returns the original Ollama response to the client — unmodified

On all other requests, the proxy is a transparent passthrough.
The injector modifies the request body BEFORE forwarding (system prompt prefix).

CRITICAL: ~/.ollama/logs/server.log is a diagnostic log — NOT conversation data.
          This proxy is the ONLY reliable way to capture conversation payloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import aiofiles
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route, Mount

from fg_sync.config import ProxyConfig
from fg_sync.injector import Injector

logger = logging.getLogger("fg_sync.proxy")

# Capture endpoints — any POST to these paths is intercepted
CAPTURE_PATHS = {"/api/chat", "/api/generate"}


class FgProxy:
    """
    Async HTTP reverse proxy + conversation capture.

    Parameters
    ----------
    config : ProxyConfig
    injector : Injector
        Reads current ruleset.json and injects system prompt prefix.
    """

    def __init__(self, config: ProxyConfig, injector: Injector):
        self.config = config
        self.injector = injector
        self._capture_lock = asyncio.Lock()

        # Ensure capture dir exists
        config.capture_path.parent.mkdir(parents=True, exist_ok=True)

        # httpx async client — reused across requests
        self._client = httpx.AsyncClient(
            base_url=f"http://{config.ollama_host}:{config.ollama_port}",
            timeout=httpx.Timeout(300.0),  # allow long generations
        )

        # Build Starlette app
        self.app = Starlette(
            routes=[
                Route("/{path:path}", self._handle, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
            ]
        )

    # ------------------------------------------------------------------
    # Core handler
    # ------------------------------------------------------------------

    async def _handle(self, request: Request) -> Response:
        path = "/" + request.path_params["path"]
        method = request.method
        headers = dict(request.headers)
        # Remove hop-by-hop headers
        for h in ("host", "transfer-encoding", "connection"):
            headers.pop(h, None)

        body = await request.body()

        # Intercept conversation endpoints
        if method == "POST" and path in CAPTURE_PATHS:
            return await self._intercept(request, path, headers, body)

        # Transparent passthrough for everything else
        return await self._passthrough(method, path, headers, body, request.query_params)

    async def _intercept(
        self,
        request: Request,
        path: str,
        headers: dict,
        body: bytes,
    ) -> Response:
        """Capture, inject system prompt, forward, capture response."""
        session_id = str(uuid.uuid4())
        t_start = time.monotonic()

        # Parse request body
        try:
            req_json = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("Non-JSON body on %s — passthrough only", path)
            return await self._passthrough(
                "POST", path, headers, body, request.query_params
            )

        model = req_json.get("model", "unknown")
        is_streaming = req_json.get("stream", True)  # Ollama streams by default

        # --- Inject behavioral ruleset into system prompt ---
        req_json = self.injector.inject(req_json, path)
        modified_body = json.dumps(req_json).encode()

        # --- Forward to Ollama ---
        if is_streaming:
            return await self._stream_intercept(
                session_id, path, headers, modified_body, req_json, model, t_start
            )
        else:
            return await self._blocking_intercept(
                session_id, path, headers, modified_body, req_json, model, t_start
            )

    async def _stream_intercept(
        self,
        session_id: str,
        path: str,
        headers: dict,
        body: bytes,
        req_json: dict,
        model: str,
        t_start: float,
    ) -> StreamingResponse:
        """Handle streaming response: buffer chunks, capture, re-stream to client."""
        collected_chunks: list[dict] = []
        assistant_text = ""

        async def generate() -> AsyncIterator[bytes]:
            nonlocal assistant_text
            async with self._client.stream("POST", path, content=body, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield (line + "\n").encode()
                        try:
                            chunk = json.loads(line)
                            collected_chunks.append(chunk)
                            # /api/chat streaming format
                            if "message" in chunk:
                                assistant_text += chunk["message"].get("content", "")
                            # /api/generate streaming format
                            elif "response" in chunk:
                                assistant_text += chunk.get("response", "")
                        except json.JSONDecodeError:
                            pass

            # Capture after stream completes
            duration_ms = int((time.monotonic() - t_start) * 1000)
            tokens_prompt = collected_chunks[-1].get("prompt_eval_count", 0) if collected_chunks else 0
            tokens_completion = collected_chunks[-1].get("eval_count", 0) if collected_chunks else 0
            await self._write_capture(
                session_id, model, path, req_json, assistant_text,
                tokens_prompt, tokens_completion, duration_ms
            )

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    async def _blocking_intercept(
        self,
        session_id: str,
        path: str,
        headers: dict,
        body: bytes,
        req_json: dict,
        model: str,
        t_start: float,
    ) -> Response:
        """Handle non-streaming response."""
        resp = await self._client.post(path, content=body, headers=headers)
        duration_ms = int((time.monotonic() - t_start) * 1000)

        try:
            resp_json = resp.json()
        except Exception:
            return Response(content=resp.content, status_code=resp.status_code,
                            headers=dict(resp.headers))

        assistant_text = ""
        if "message" in resp_json:
            assistant_text = resp_json["message"].get("content", "")
        elif "response" in resp_json:
            assistant_text = resp_json.get("response", "")

        tokens_prompt = resp_json.get("prompt_eval_count", 0)
        tokens_completion = resp_json.get("eval_count", 0)

        await self._write_capture(
            session_id, model, path, req_json, assistant_text,
            tokens_prompt, tokens_completion, duration_ms
        )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-encoding")},
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _passthrough(
        self,
        method: str,
        path: str,
        headers: dict,
        body: bytes,
        query_params,
    ) -> Response:
        """Transparent passthrough — no capture, no modification."""
        url = path
        if query_params:
            url += "?" + str(query_params)
        resp = await self._client.request(method, url, content=body, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-encoding")},
        )

    # ------------------------------------------------------------------
    # Capture writer
    # ------------------------------------------------------------------

    async def _write_capture(
        self,
        session_id: str,
        model: str,
        path: str,
        req_json: dict,
        assistant_text: str,
        tokens_prompt: int,
        tokens_completion: int,
        duration_ms: int,
    ) -> None:
        """Append one JSONL record to capture.jsonl."""
        from datetime import datetime, timezone

        # Extract conversation messages
        messages = req_json.get("messages", [])
        if not messages and "prompt" in req_json:
            # /api/generate format — wrap as messages
            messages = [
                {"role": "user", "content": req_json["prompt"]},
                {"role": "assistant", "content": assistant_text},
            ]
        else:
            # Append assistant reply to messages list
            messages = list(messages) + [{"role": "assistant", "content": assistant_text}]

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "model": model,
            "endpoint": path,
            "messages": messages,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "duration_ms": duration_ms,
            "fg_injected": self.injector.is_active(),
        }

        line = json.dumps(record, ensure_ascii=False) + "\n"

        async with self._capture_lock:
            async with aiofiles.open(self.config.capture_path, "a", encoding="utf-8") as f:
                await f.write(line)

        logger.debug(
            "Captured session=%s model=%s tokens=%d/%d",
            session_id[:8], model, tokens_prompt, tokens_completion
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self):
        logger.info(
            "fg-proxy listening on port %d → Ollama %s:%d",
            self.config.listen_port, self.config.ollama_host, self.config.ollama_port
        )

    async def shutdown(self):
        await self._client.aclose()
        logger.info("fg-proxy shut down")
