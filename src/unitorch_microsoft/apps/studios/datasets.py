# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import asyncio
import aiosqlite
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str = ""


class DatasetColumnInfo(BaseModel):
    name: str
    type: str
    description: str = ""


class DatasetSplits(BaseModel):
    train: int = 0
    validation: int = 0
    test: int = 0


class DatasetDetails(BaseModel):
    splits: DatasetSplits = Field(default_factory=DatasetSplits)
    columns: List[DatasetColumnInfo] = Field(default_factory=list)
    labels_distributions: Dict[str, float] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    rows: int = 0
    size: str = ""
    created_at: str = ""
    updated_at: str = ""
    details: DatasetDetails = Field(default_factory=DatasetDetails)


class DatasetIdRequest(BaseModel):
    id: str


class DatasetPreviewRequest(BaseModel):
    id: str
    split: str = "train"
    start: int = 0
    limit: int = 5


class DatasetPreview(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    number_columns: List[str] = Field(default_factory=list)
    image_columns: List[str] = Field(default_factory=list)
    video_columns: List[str] = Field(default_factory=list)


INIT_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    rows INTEGER NOT NULL DEFAULT 0,
    size TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS dataset_data (
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    data TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (dataset_id, split, row_index),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
"""


@register_fastapi("microsoft/apps/studios/datasets")
class StudioDatasetsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/datasets")
        router = config.getoption("router", "/microsoft/apps/studio/datasets")
        self._db_path = config.getoption("db_path", "studio_datasets.db")

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("", self.list_datasets, methods=["GET"])
        self._router.add_api_route("/details", self.get_details, methods=["POST"])
        self._router.add_api_route("/preview", self.get_preview, methods=["POST"])
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

    async def list_datasets(self):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id, name, description FROM datasets") as cursor:
                rows = await cursor.fetchall()
            return [
                DatasetSummary(id=r[0], name=r[1], description=r[2])
                for r in rows
            ]
        finally:
            await db.close()

    async def get_details(self, request: DatasetIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, rows, size, created_at, updated_at, details FROM datasets WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return {"error": "dataset not found"}
            return DatasetInfo(
                id=row[0], name=row[1], description=row[2],
                rows=row[3], size=row[4], created_at=row[5], updated_at=row[6],
                details=DatasetDetails(**json.loads(row[7])),
            )
        finally:
            await db.close()

    async def get_preview(self, request: DatasetPreviewRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT details FROM datasets WHERE id = ?",
                (request.id,),
            ) as cursor:
                ds_row = await cursor.fetchone()
            if not ds_row:
                return {"error": "dataset not found"}

            details = json.loads(ds_row[0])
            columns_info = details.get("columns", [])
            col_names = [c["name"] for c in columns_info]
            number_cols = [c["name"] for c in columns_info if c.get("type") in ("integer", "float", "number")]
            image_cols = [c["name"] for c in columns_info if c.get("type") == "image"]
            video_cols = [c["name"] for c in columns_info if c.get("type") == "video"]

            async with db.execute(
                "SELECT data FROM dataset_data WHERE dataset_id = ? AND split = ? AND row_index >= ? ORDER BY row_index LIMIT ?",
                (request.id, request.split, request.start, request.limit),
            ) as cursor:
                data_rows = await cursor.fetchall()

            rows = [json.loads(r[0]) for r in data_rows]

            return DatasetPreview(
                columns=col_names,
                rows=rows,
                number_columns=number_cols,
                image_columns=image_cols,
                video_columns=video_cols,
            )
        finally:
            await db.close()
