# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import uuid
import asyncio
import aiosqlite
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReportSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


class ReportInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    content: str = ""               # markdown content
    dataset_ids: List[str] = Field(default_factory=list)
    job_ids: List[str] = Field(default_factory=list)
    label_task_ids: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


# --- request models ---

class ReportIdRequest(BaseModel):
    id: str


class ReportCreateRequest(BaseModel):
    name: str
    description: str = ""
    content: str = ""
    dataset_ids: List[str] = Field(default_factory=list)
    job_ids: List[str] = Field(default_factory=list)
    label_task_ids: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ReportUpdateRequest(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    dataset_ids: Optional[List[str]] = None
    job_ids: Optional[List[str]] = None
    label_task_ids: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

INIT_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    dataset_ids     TEXT NOT NULL DEFAULT '[]',
    job_ids         TEXT NOT NULL DEFAULT '[]',
    label_task_ids  TEXT NOT NULL DEFAULT '[]',
    extra           TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_report_info(row) -> ReportInfo:
    return ReportInfo(
        id=row[0], name=row[1], description=row[2], content=row[3],
        dataset_ids=json.loads(row[4]),
        job_ids=json.loads(row[5]),
        label_task_ids=json.loads(row[6]),
        extra=json.loads(row[7]),
        created_at=row[8], updated_at=row[9],
    )


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------

@register_fastapi("microsoft/apps/studios/reports")
class StudioReportsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/reports")
        router = config.getoption("router", "/microsoft/apps/studios/reports")
        studios_folder = config.getoption("studios_folder", "studios")
        dbs_folder = os.path.join(studios_folder, "dbs")
        os.makedirs(dbs_folder, exist_ok=True)
        self._db_path = os.path.join(dbs_folder, "reports.db")

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("", self.list_reports, methods=["GET"])
        self._router.add_api_route("/create", self.create_report, methods=["POST"])
        self._router.add_api_route("/get", self.get_report, methods=["POST"])
        self._router.add_api_route("/update", self.update_report, methods=["POST"])
        self._router.add_api_route("/delete", self.delete_report, methods=["POST"])
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
        return db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list_reports(self):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, created_at, updated_at FROM reports"
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                ReportSummary(
                    id=r[0], name=r[1], description=r[2],
                    created_at=r[3], updated_at=r[4],
                )
                for r in rows
            ]
        finally:
            await db.close()

    async def create_report(self, request: ReportCreateRequest):
        """Create a markdown report with optional references to datasets, jobs, and label tasks."""
        report_id = str(uuid.uuid4())
        now = _now()
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO reports "
                "(id, name, description, content, dataset_ids, job_ids, label_task_ids, extra, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report_id, request.name, request.description, request.content,
                    json.dumps(request.dataset_ids),
                    json.dumps(request.job_ids),
                    json.dumps(request.label_task_ids),
                    json.dumps(request.extra),
                    now, now,
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return await self.get_report(ReportIdRequest(id=report_id))

    async def get_report(self, request: ReportIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, content, dataset_ids, job_ids, label_task_ids, extra, created_at, updated_at "
                "FROM reports WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="report not found")
            return _build_report_info(row)
        finally:
            await db.close()

    async def update_report(self, request: ReportUpdateRequest):
        """Update a report. Only provided fields are overwritten; `extra` is merged."""
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, content, dataset_ids, job_ids, label_task_ids, extra, created_at, updated_at "
                "FROM reports WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="report not found")

            info = _build_report_info(row)
            name = request.name if request.name is not None else info.name
            description = request.description if request.description is not None else info.description
            content = request.content if request.content is not None else info.content
            dataset_ids = request.dataset_ids if request.dataset_ids is not None else info.dataset_ids
            job_ids = request.job_ids if request.job_ids is not None else info.job_ids
            label_task_ids = request.label_task_ids if request.label_task_ids is not None else info.label_task_ids
            if request.extra is not None:
                info.extra.update(request.extra)

            now = _now()
            await db.execute(
                "UPDATE reports SET name=?, description=?, content=?, dataset_ids=?, job_ids=?, "
                "label_task_ids=?, extra=?, updated_at=? WHERE id=?",
                (
                    name, description, content,
                    json.dumps(dataset_ids),
                    json.dumps(job_ids),
                    json.dumps(label_task_ids),
                    json.dumps(info.extra),
                    now, request.id,
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return await self.get_report(ReportIdRequest(id=request.id))

    async def delete_report(self, request: ReportIdRequest):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id FROM reports WHERE id=?", (request.id,)) as cursor:
                if not await cursor.fetchone():
                    raise HTTPException(status_code=404, detail="report not found")
            await db.execute("DELETE FROM reports WHERE id=?", (request.id,))
            await db.commit()
        finally:
            await db.close()
        return {"id": request.id, "deleted": True}
