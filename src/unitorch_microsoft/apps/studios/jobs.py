# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import asyncio
import aiosqlite
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


class JobSummary(BaseModel):
    id: str
    name: str
    description: str = ""


class JobInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    status: str = "pending"
    progress: int = 0
    created_at: str = ""
    updated_at: str = ""
    logs: str = ""


class JobIdRequest(BaseModel):
    id: str


INIT_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    progress INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    logs TEXT NOT NULL DEFAULT ''
);
"""


@register_fastapi("microsoft/apps/studios/jobs")
class StudioJobsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/jobs")
        router = config.getoption("router", "/microsoft/apps/studio/jobs")
        self._db_path = config.getoption("db_path", "studio_jobs.db")

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("", self.list_jobs, methods=["GET"])
        self._router.add_api_route("/details", self.get_details, methods=["POST"])
        self._router.add_api_route("/cancel", self.cancel_job, methods=["POST"])
        self._router.add_api_route("/restart", self.restart_job, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()
        self._running = False
        self._db_initialized = False

    @property
    def router(self):
        return self._router

    def start(self):
        self._running = True
        return "running"

    def stop(self):
        self._running = False
        return "stopped"

    def status(self):
        return "running" if self._running else "stopped"

    async def _get_db(self):
        db = await aiosqlite.connect(self._db_path)
        if not self._db_initialized:
            await db.executescript(INIT_SQL)
            await db.commit()
            self._db_initialized = True
        db.row_factory = aiosqlite.Row
        return db

    async def list_jobs(self):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id, name, description FROM jobs") as cursor:
                rows = await cursor.fetchall()
            return [
                JobSummary(id=r[0], name=r[1], description=r[2])
                for r in rows
            ]
        finally:
            await db.close()

    async def get_details(self, request: JobIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, status, progress, created_at, updated_at, logs FROM jobs WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "job not found"}
            return JobInfo(
                id=row[0], name=row[1], description=row[2],
                status=row[3], progress=row[4],
                created_at=row[5], updated_at=row[6], logs=row[7],
            )
        finally:
            await db.close()

    async def cancel_job(self, request: JobIdRequest):
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, status, progress, created_at, logs FROM jobs WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "job not found"}

            logs = row[6] + "\nJob was cancelled."
            await db.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ?, logs = ? WHERE id = ?",
                (now, logs, request.id),
            )
            await db.commit()

            return JobInfo(
                id=row[0], name=row[1], description=row[2],
                status="cancelled", progress=row[4],
                created_at=row[5], updated_at=now, logs=logs,
            )
        finally:
            await db.close()

    async def restart_job(self, request: JobIdRequest):
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, created_at FROM jobs WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "job not found"}

            logs = "Job restarted..."
            await db.execute(
                "UPDATE jobs SET status = 'running', progress = 0, updated_at = ?, logs = ? WHERE id = ?",
                (now, logs, request.id),
            )
            await db.commit()

            return JobInfo(
                id=row[0], name=row[1], description=row[2],
                status="running", progress=0,
                created_at=row[3], updated_at=now, logs=logs,
            )
        finally:
            await db.close()
