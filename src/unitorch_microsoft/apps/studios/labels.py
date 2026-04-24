# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import asyncio
import aiosqlite
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


class LabelSummary(BaseModel):
    id: str
    name: str
    description: str = ""


class LabelerInfo(BaseModel):
    id: str
    name: str
    completed: int = 0


class LabelStats(BaseModel):
    total: int = 0
    completed: int = 0
    pending: int = 0
    partial_completed: int = 0
    agreement: float = 0.0
    labelers: List[LabelerInfo] = Field(default_factory=list)


class LabelInfo(BaseModel):
    id: str
    name: str
    type: str = "classification"
    description: str = ""
    stats: LabelStats = Field(default_factory=LabelStats)


class LabelIdRequest(BaseModel):
    id: str


INIT_SQL = """
CREATE TABLE IF NOT EXISTS labels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'classification',
    description TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}'
);
"""


@register_fastapi("microsoft/apps/studios/labels")
class StudioLabelsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/labels")
        router = config.getoption("router", "/microsoft/apps/studio/labels")
        self._db_path = config.getoption("db_path", "studio_labels.db")

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("", self.list_labels, methods=["GET"])
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

    async def list_labels(self):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id, name, description FROM labels") as cursor:
                rows = await cursor.fetchall()
            return [
                LabelSummary(id=r[0], name=r[1], description=r[2])
                for r in rows
            ]
        finally:
            await db.close()

    async def get_details(self, request: LabelIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, type, description, stats FROM labels WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "label not found"}
            return LabelInfo(
                id=row[0], name=row[1], type=row[2], description=row[3],
                stats=LabelStats(**json.loads(row[4])),
            )
        finally:
            await db.close()
