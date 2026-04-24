# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import asyncio
import aiosqlite
from fastapi import APIRouter
from pydantic import BaseModel
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


class ReportSummary(BaseModel):
    id: str
    name: str
    description: str = ""


class ReportInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    content: str = ""


class ReportIdRequest(BaseModel):
    id: str


INIT_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT ''
);
"""


@register_fastapi("microsoft/apps/studios/reports")
class StudioReportsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/reports")
        router = config.getoption("router", "/microsoft/apps/studio/reports")
        self._db_path = config.getoption("db_path", "studio_reports.db")

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("", self.list_reports, methods=["GET"])
        self._router.add_api_route("/details", self.get_details, methods=["POST"])
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

    async def list_reports(self):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id, name, description FROM reports") as cursor:
                rows = await cursor.fetchall()
            return [
                ReportSummary(id=r[0], name=r[1], description=r[2])
                for r in rows
            ]
        finally:
            await db.close()

    async def get_details(self, request: ReportIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, created_at, updated_at, content FROM reports WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "report not found"}
            return ReportInfo(
                id=row[0], name=row[1], description=row[2],
                created_at=row[3], updated_at=row[4], content=row[5],
            )
        finally:
            await db.close()
