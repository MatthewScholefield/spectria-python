"""HTTP/SSE server for serving training logs to Spectra."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .writer import RunWriter


def _allowed_origins() -> list[str]:
    """Exact-match allowed origins."""
    return [
        "https://matthewscholefield.github.io",
    ]


_LOCALHOST_ORIGIN_REGEX = r"http://localhost:\d+"


def create_app(logdir: str | Path):
    """Create a Starlette app serving Spectra training logs."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    logdir = Path(logdir)

    def list_projects(request: Request) -> JSONResponse:
        projects = []
        if logdir.exists():
            for d in sorted(logdir.iterdir()):
                if d.is_dir():
                    run_count = sum(1 for _ in d.iterdir() if _.is_dir())
                    projects.append({"name": d.name, "run_count": run_count})
        return JSONResponse(projects)

    def list_runs(request: Request) -> JSONResponse:
        project_name = request.path_params["name"]
        project_dir = logdir / project_name
        if not project_dir.exists():
            return JSONResponse([], status_code=404)

        runs = []
        for run_dir in sorted(project_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            events_path = run_dir / "events.jsonl"
            header = RunWriter.read_header(events_path)

            # Determine status by checking if the file is still being written to
            status = "completed"
            if events_path.exists():
                mtime = events_path.stat().st_mtime
                if time.time() - mtime < 120:  # active within last 2 minutes
                    status = "running"

            runs.append({
                "run_id": run_dir.name,
                "baseline": header.get("baseline") if header else None,
                "status": status,
                "config": header.get("config", {}) if header else {},
            })
        return JSONResponse(runs)

    def get_run_data(request: Request) -> JSONResponse:
        project_name = request.path_params["name"]
        run_id = request.path_params["run"]
        events_path = logdir / project_name / run_id / "events.jsonl"
        rows = RunWriter.read_rows(events_path)
        return JSONResponse(rows)

    async def stream_events(request: Request) -> StreamingResponse:
        project_name = request.path_params["name"]
        run_id = request.path_params["run"]
        events_path = logdir / project_name / run_id / "events.jsonl"

        async def generate():
            # First, send all existing rows
            rows = RunWriter.read_rows(events_path)
            for row in rows:
                yield f"event: row\ndata: {json.dumps(row, default=str)}\n\n"

            # Then poll for new rows
            sent_count = len(rows)
            idle_ticks = 0

            while True:
                await asyncio.sleep(0.5)

                current_rows = RunWriter.read_rows(events_path)
                new_rows = current_rows[sent_count:]

                if new_rows:
                    idle_ticks = 0
                    for row in new_rows:
                        yield f"event: row\ndata: {json.dumps(row, default=str)}\n\n"
                    sent_count = len(current_rows)
                else:
                    idle_ticks += 1

                # Check if training is still running (file was modified recently)
                if events_path.exists():
                    mtime = events_path.stat().st_mtime
                    if time.time() - mtime > 120 and sent_count > 0:
                        # File hasn't been modified in 2 minutes — likely done
                        yield f"event: complete\ndata: {{}}\n\n"
                        break

                # Safety: stop after 24 hours
                if idle_ticks > 172800:
                    yield f"event: complete\ndata: {{}}\n\n"
                    break

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app = Starlette(
        routes=[
            Route("/api/projects", endpoint=list_projects),
            Route("/api/projects/{name}/runs", endpoint=list_runs),
            Route("/api/projects/{name}/runs/{run}/data", endpoint=get_run_data),
            Route("/api/projects/{name}/runs/{run}/events", endpoint=stream_events),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=_allowed_origins(),
                allow_origin_regex=_LOCALHOST_ORIGIN_REGEX,
                allow_methods=["GET"],
            ),
        ],
    )
    return app
