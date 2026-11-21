#!/usr/bin/env python3
"""Minimal MCP stdio adapter around the official SparseRead core tools.

SPARSEREAD_REPO points at a checkout of Liuzedooong/SparseReading. One MCP
process owns one SparseRead orchestrator, preserving episodes and raw_refs.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import sys

repo = Path(os.environ["SPARSEREAD_REPO"]).resolve()
sys.path.insert(0, str(repo / "packages/sparseread-core/src"))
from sparseread import SparseRead  # noqa: E402
from sparseread.config import SparseReadConfig  # noqa: E402

runtime = SparseRead(SparseReadConfig(mode=os.environ.get("SPARSEREAD_MODE", "auto"), workspace=os.getcwd()))
tools = {tool.name: tool for tool in runtime.tools()}


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def schemas() -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {**tool.parameters, "type": "object"},
        }
        for tool in tools.values()
    ]


def handle(request: dict) -> None:
    method = request.get("method")
    request_id = request.get("id")
    if method and method.startswith("notifications/"):
        return
    if method == "initialize":
        result = {
            "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "official-sparseread", "version": "0.1.4"},
        }
    elif method == "tools/list":
        result = {"tools": schemas()}
    elif method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        try:
            if name not in tools:
                raise ValueError(f"unknown SparseRead tool: {name}")
            text = asyncio.run(tools[name].execute(**params.get("arguments", {})))
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as error:
            result = {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
    elif method == "ping":
        result = {}
    elif method == "shutdown":
        result = {}
    else:
        if request_id is not None:
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}})
        return
    if request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


for line in sys.stdin:
    try:
        request = json.loads(line)
        if isinstance(request, dict):
            handle(request)
    except Exception as error:
        print(f"official SparseRead MCP adapter: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
