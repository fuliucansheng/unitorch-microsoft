# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import os
import json
import uuid
import asyncio
import aiosqlite
import pandas as pd
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DatasetColumnInfo(BaseModel):
    name: str
    type: str
    description: str = ""


class DatasetMeta(BaseModel):
    """Mutable dataset-level metadata."""
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: str = ""
    metrics: List[str] = Field(default_factory=list)
    label_columns: List[str] = Field(default_factory=list)
    columns: List[DatasetColumnInfo] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    type: str = "local"
    rows: int = 0
    created_at: str = ""
    updated_at: str = ""


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    type: str = "local"
    rows: int = 0
    size: str = ""
    created_at: str = ""
    updated_at: str = ""
    meta: DatasetMeta = Field(default_factory=DatasetMeta)


# --- request models ---

class DatasetIdRequest(BaseModel):
    id: str


class DatasetMetaUpdateRequest(BaseModel):
    id: str
    meta: DatasetMeta


class SampleListRequest(BaseModel):
    dataset_id: str
    split: Optional[str] = None
    start: int = 0
    limit: int = 20
    include_meta: bool = True


class SampleGetRequest(BaseModel):
    dataset_id: str
    sample_id: str


class SampleMetaSetRequest(BaseModel):
    """Set (upsert) meta key-value pairs for one or more samples."""
    dataset_id: str
    # {sample_id: {key: value, ...}}
    records: Dict[str, Dict[str, Any]]


class SampleMetaGetRequest(BaseModel):
    """Get all meta for a list of sample_ids (empty = entire dataset)."""
    dataset_id: str
    sample_ids: Optional[List[str]] = None
    keys: Optional[List[str]] = None


class SampleMetaDeleteRequest(BaseModel):
    """Delete specific meta keys for given sample_ids (empty keys = delete all meta for sample)."""
    dataset_id: str
    sample_ids: List[str]
    keys: Optional[List[str]] = None


class MetaFilter(BaseModel):
    """A single filter condition on a meta key.

    Supported operators:
      eq / ne          – equal / not equal
      gt / gte / lt / lte – numeric comparison
      in / not_in      – value in a list
      exists / not_exists – key presence (value field ignored)
      contains         – substring match (string values)
    """
    key: str
    op: str = "eq"   # eq | ne | gt | gte | lt | lte | in | not_in | exists | not_exists | contains
    value: Optional[Any] = None


class SampleSelectRequest(BaseModel):
    """Select samples whose meta satisfies ALL provided filters (AND logic)."""
    dataset_id: str
    filters: List[MetaFilter]
    split: Optional[str] = None
    start: int = 0
    limit: int = 100
    include_meta: bool = True


# --- response models ---

class SampleRecord(BaseModel):
    sample_id: str
    split: Optional[str] = None
    data: Dict[str, Any]
    meta: Dict[str, Any] = Field(default_factory=dict)


class SampleListResponse(BaseModel):
    total: int
    start: int
    limit: int
    samples: List[SampleRecord]


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

INIT_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'local',
    rows INTEGER NOT NULL DEFAULT 0,
    size TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS dataset_samples (
    dataset_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, sample_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS dataset_sample_meta (
    dataset_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT 'null',
    PRIMARY KEY (dataset_id, sample_id, key),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pandas_dtype_to_col_type(dtype) -> str:
    kind = dtype.kind
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        return "float"
    return "string"


def _infer_col_type_from_series(series: "pd.Series") -> str:
    col_type = _pandas_dtype_to_col_type(series.dtype)
    if col_type == "string":
        sample = series.dropna().head(5).astype(str)
        if not sample.empty:
            lower = sample.iloc[0].lower()
            if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
                return "image"
            if any(lower.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return "video"
    return col_type


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_dataset_info(row, meta_dict: dict) -> DatasetInfo:
    return DatasetInfo(
        id=row[0], name=row[1], description=row[2], type=row[3],
        rows=row[4], size=row[5], created_at=row[6], updated_at=row[7],
        meta=DatasetMeta(**meta_dict),
    )


async def _fetch_sample_meta(
    db: aiosqlite.Connection,
    dataset_id: str,
    sample_ids: Optional[List[str]] = None,
    keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return {sample_id: {key: value}} for the given filters."""
    result: Dict[str, Dict[str, Any]] = {}
    if sample_ids is not None and len(sample_ids) == 0:
        return result

    clauses = ["dataset_id = ?"]
    params: list = [dataset_id]
    if sample_ids is not None:
        placeholders = ",".join("?" * len(sample_ids))
        clauses.append(f"sample_id IN ({placeholders})")
        params.extend(sample_ids)
    if keys is not None and keys:
        placeholders = ",".join("?" * len(keys))
        clauses.append(f"key IN ({placeholders})")
        params.extend(keys)

    sql = f"SELECT sample_id, key, value FROM dataset_sample_meta WHERE {' AND '.join(clauses)}"
    async with db.execute(sql, params) as cursor:
        async for sid, k, v in cursor:
            result.setdefault(sid, {})[k] = json.loads(v)
    return result


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------

@register_fastapi("microsoft/apps/studios/datasets")
class StudioDatasetsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/datasets")
        router = config.getoption("router", "/microsoft/apps/studios/datasets")
        studios_folder = config.getoption("studios_folder", "studios")
        dbs_folder = os.path.join(studios_folder, "dbs")
        os.makedirs(dbs_folder, exist_ok=True)
        self._db_path = os.path.join(dbs_folder, "datasets.db")

        self._router = APIRouter(prefix=router)
        # dataset CRUD
        self._router.add_api_route("", self.list_datasets, methods=["GET"])
        self._router.add_api_route("/create", self.create_dataset, methods=["POST"])
        self._router.add_api_route("/upload", self.upload_split, methods=["POST"])
        self._router.add_api_route("/get", self.get_dataset, methods=["POST"])
        self._router.add_api_route("/delete", self.delete_dataset, methods=["POST"])
        # dataset meta
        self._router.add_api_route("/meta/update", self.update_dataset_meta, methods=["POST"])
        # sample queries
        self._router.add_api_route("/sample/list", self.list_samples, methods=["POST"])
        self._router.add_api_route("/sample/get", self.get_sample, methods=["POST"])
        self._router.add_api_route("/sample/select", self.select_samples, methods=["POST"])
        # sample meta CRUD
        self._router.add_api_route("/sample/meta/set", self.set_sample_meta, methods=["POST"])
        self._router.add_api_route("/sample/meta/get", self.get_sample_meta, methods=["POST"])
        self._router.add_api_route("/sample/meta/delete", self.delete_sample_meta, methods=["POST"])
        self._router.add_api_route("/sample/meta/list", self.list_sample_meta, methods=["POST"])
        # service control
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
    # Dataset CRUD
    # ------------------------------------------------------------------

    async def list_datasets(self):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, rows, created_at, updated_at FROM datasets"
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                DatasetSummary(
                    id=r[0], name=r[1], description=r[2], type=r[3],
                    rows=r[4], created_at=r[5], updated_at=r[6],
                )
                for r in rows
            ]
        finally:
            await db.close()

    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = "",
        splits: Optional[str] = None,
        files: List[UploadFile] = File(...),
    ):
        """Create a new dataset from one or more TSV files.

        ``splits`` is a JSON string mapping split name to a list of filenames,
        e.g. ``'{"train": ["train1.tsv", "train2.tsv"], "validate": ["val.tsv"], "test": ["test.tsv"]}'``.
        Every uploaded filename must appear in exactly one split list.
        If ``splits`` is omitted, all files are assigned to ``"train"``.
        """
        # Build filename → split_name mapping
        if splits:
            try:
                splits_dict: Dict[str, List[str]] = json.loads(splits)
            except Exception:
                raise HTTPException(status_code=400, detail="'splits' must be a valid JSON object")
            filename_to_split: Dict[str, str] = {}
            for split_name, filenames in splits_dict.items():
                for fn in filenames:
                    if fn in filename_to_split:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Filename '{fn}' appears in more than one split",
                        )
                    filename_to_split[fn] = split_name
            # Validate all uploaded files are covered
            for f in files:
                if f.filename not in filename_to_split:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file '{f.filename}' is not listed in 'splits'",
                    )
        else:
            filename_to_split = {f.filename: "test" for f in files}

        # Parse TSV files and validate consistent headers
        file_dfs: List[tuple] = []   # (split_name, df)
        total_bytes = 0
        for f in files:
            raw = await f.read()
            total_bytes += len(raw)
            df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, keep_default_na=False)
            if df.empty and len(df.columns) == 0:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty")
            file_dfs.append((filename_to_split[f.filename], df))

        ref_header = list(file_dfs[0][1].columns)
        for i, (_, df) in enumerate(file_dfs[1:], start=1):
            header = list(df.columns)
            if header != ref_header:
                raise HTTPException(
                    status_code=400,
                    detail=f"File #{i+1} header {header} does not match first file header {ref_header}",
                )

        all_dfs = [df for _, df in file_dfs]
        combined = pd.concat(all_dfs, ignore_index=True)
        columns_info = [
            DatasetColumnInfo(name=col, type=_infer_col_type_from_series(combined[col]))
            for col in combined.columns
        ]

        total_rows = len(combined)
        size_str = (
            f"{total_bytes / 1024:.1f} KB"
            if total_bytes < 1024 * 1024
            else f"{total_bytes / 1024 / 1024:.1f} MB"
        )
        now = _now()
        dataset_id = str(uuid.uuid4())

        meta = DatasetMeta(columns=columns_info)

        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO datasets (id, name, description, type, rows, size, created_at, updated_at, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, name, description or "", "local", total_rows, size_str, now, now, meta.model_dump_json()),
            )
            for split_name, df in file_dfs:
                for _, row in df.iterrows():
                    sample_id = str(uuid.uuid4())
                    await db.execute(
                        "INSERT INTO dataset_samples (dataset_id, sample_id, data) VALUES (?, ?, ?)",
                        (dataset_id, sample_id, json.dumps(row.to_dict())),
                    )
                    await db.execute(
                        "INSERT INTO dataset_sample_meta (dataset_id, sample_id, key, value) VALUES (?, ?, 'split', ?)",
                        (dataset_id, sample_id, json.dumps(split_name)),
                    )
            await db.commit()
        finally:
            await db.close()

        return DatasetInfo(
            id=dataset_id, name=name, description=description or "",
            type="local", rows=total_rows, size=size_str,
            created_at=now, updated_at=now, meta=meta,
        )

    async def upload_split(
        self,
        dataset_id: str,
        split: Optional[str] = "train",
        files: List[UploadFile] = File(...),
    ):
        """Append rows from one or more TSV files into an existing dataset split.

        All uploaded files must share the same header as the existing dataset.
        The dataset's ``rows`` and ``updated_at`` fields are updated accordingly.
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, rows, size, created_at, updated_at, meta FROM datasets WHERE id = ?",
                (dataset_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="dataset not found")

            existing_meta = DatasetMeta(**json.loads(row[8]))
            existing_col_names = [c.name for c in existing_meta.columns]

            dfs: List[pd.DataFrame] = []
            total_bytes = 0
            for f in files:
                raw = await f.read()
                total_bytes += len(raw)
                df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, keep_default_na=False)
                if df.empty and len(df.columns) == 0:
                    raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty")
                header = list(df.columns)
                if existing_col_names and header != existing_col_names:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{f.filename}' header {header} does not match dataset header {existing_col_names}",
                    )
                dfs.append(df)

            new_rows = sum(len(df) for df in dfs)
            split_name = split or "train"
            now = _now()

            for df in dfs:
                for _, r in df.iterrows():
                    sample_id = str(uuid.uuid4())
                    await db.execute(
                        "INSERT INTO dataset_samples (dataset_id, sample_id, data) VALUES (?, ?, ?)",
                        (dataset_id, sample_id, json.dumps(r.to_dict())),
                    )
                    await db.execute(
                        "INSERT INTO dataset_sample_meta (dataset_id, sample_id, key, value) VALUES (?, ?, 'split', ?)",
                        (dataset_id, sample_id, json.dumps(split_name)),
                    )

            updated_rows = row[4] + new_rows
            existing_bytes = 0
            if row[5].endswith("MB"):
                existing_bytes = int(float(row[5][:-3]) * 1024 * 1024)
            elif row[5].endswith("KB"):
                existing_bytes = int(float(row[5][:-3]) * 1024)
            total_bytes += existing_bytes
            size_str = (
                f"{total_bytes / 1024:.1f} KB"
                if total_bytes < 1024 * 1024
                else f"{total_bytes / 1024 / 1024:.1f} MB"
            )
            await db.execute(
                "UPDATE datasets SET rows=?, size=?, updated_at=? WHERE id=?",
                (updated_rows, size_str, now, dataset_id),
            )
            await db.commit()

            async with db.execute(
                "SELECT id, name, description, type, rows, size, created_at, updated_at, meta FROM datasets WHERE id = ?",
                (dataset_id,),
            ) as cursor:
                updated = await cursor.fetchone()
            return _build_dataset_info(updated, json.loads(updated[8]))
        finally:
            await db.close()

    async def get_dataset(self, request: DatasetIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, rows, size, created_at, updated_at, meta FROM datasets WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="dataset not found")
            return _build_dataset_info(row, json.loads(row[8]))
        finally:
            await db.close()

    async def delete_dataset(self, request: DatasetIdRequest):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id FROM datasets WHERE id = ?", (request.id,)) as cursor:
                if not await cursor.fetchone():
                    raise HTTPException(status_code=404, detail="dataset not found")
            await db.execute("DELETE FROM dataset_sample_meta WHERE dataset_id = ?", (request.id,))
            await db.execute("DELETE FROM dataset_samples WHERE dataset_id = ?", (request.id,))
            await db.execute("DELETE FROM datasets WHERE id = ?", (request.id,))
            await db.commit()
        finally:
            await db.close()
        return {"id": request.id, "deleted": True}

    # ------------------------------------------------------------------
    # Dataset meta update
    # ------------------------------------------------------------------

    async def update_dataset_meta(self, request: DatasetMetaUpdateRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, rows, size, created_at, updated_at, meta FROM datasets WHERE id = ?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="dataset not found")

            existing = DatasetMeta(**json.loads(row[8]))
            patch = request.meta

            # Merge: only overwrite fields that are explicitly set in the patch
            name = patch.name if patch.name is not None else row[1]
            description = patch.description if patch.description is not None else row[2]
            if patch.task_type:
                existing.task_type = patch.task_type
            if patch.metrics:
                existing.metrics = patch.metrics
            if patch.label_columns:
                existing.label_columns = patch.label_columns
            if patch.columns:
                existing.columns = patch.columns
            if patch.extra:
                existing.extra.update(patch.extra)

            now = _now()
            await db.execute(
                "UPDATE datasets SET name=?, description=?, updated_at=?, meta=? WHERE id=?",
                (name, description, now, existing.model_dump_json(), request.id),
            )
            await db.commit()

            async with db.execute(
                "SELECT id, name, description, type, rows, size, created_at, updated_at, meta FROM datasets WHERE id = ?",
                (request.id,),
            ) as cursor:
                updated = await cursor.fetchone()
            return _build_dataset_info(updated, json.loads(updated[8]))
        finally:
            await db.close()

    # ------------------------------------------------------------------
    # Sample queries (read-only on data)
    # ------------------------------------------------------------------

    async def list_samples(self, request: SampleListRequest):
        db = await self._get_db()
        try:
            # Total count — filter by split via meta table when requested
            if request.split:
                count_sql = (
                    "SELECT COUNT(*) FROM dataset_samples s "
                    "JOIN dataset_sample_meta m ON s.dataset_id=m.dataset_id AND s.sample_id=m.sample_id "
                    "WHERE s.dataset_id=? AND m.key='split' AND m.value=?"
                )
                count_params: list = [request.dataset_id, json.dumps(request.split)]
            else:
                count_sql = "SELECT COUNT(*) FROM dataset_samples WHERE dataset_id = ?"
                count_params = [request.dataset_id]
            async with db.execute(count_sql, count_params) as cursor:
                total = (await cursor.fetchone())[0]

            # Paginated rows
            if request.split:
                sql = (
                    "SELECT s.sample_id, s.data FROM dataset_samples s "
                    "JOIN dataset_sample_meta m ON s.dataset_id=m.dataset_id AND s.sample_id=m.sample_id "
                    "WHERE s.dataset_id=? AND m.key='split' AND m.value=? LIMIT ? OFFSET ?"
                )
                params: list = [request.dataset_id, json.dumps(request.split), request.limit, request.start]
            else:
                sql = "SELECT sample_id, data FROM dataset_samples WHERE dataset_id=? LIMIT ? OFFSET ?"
                params = [request.dataset_id, request.limit, request.start]

            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

            sample_ids = [r[0] for r in rows]
            meta_map: Dict[str, Dict[str, Any]] = {}
            if sample_ids:
                meta_map = await _fetch_sample_meta(db, request.dataset_id, sample_ids)

            samples = [
                SampleRecord(
                    sample_id=r[0],
                    split=meta_map.get(r[0], {}).get("split") if not request.include_meta else None,
                    data=json.loads(r[1]),
                    meta=meta_map.get(r[0], {}) if request.include_meta else {},
                )
                for r in rows
            ]
            # When include_meta is True, split is already inside meta; also expose it on the top-level field
            for s in samples:
                if request.include_meta:
                    s.split = s.meta.get("split")
            return SampleListResponse(total=total, start=request.start, limit=request.limit, samples=samples)
        finally:
            await db.close()

    async def get_sample(self, request: SampleGetRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT sample_id, data FROM dataset_samples WHERE dataset_id = ? AND sample_id = ?",
                (request.dataset_id, request.sample_id),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="sample not found")
            meta_map = await _fetch_sample_meta(db, request.dataset_id, [request.sample_id])
            meta = meta_map.get(row[0], {})
            return SampleRecord(
                sample_id=row[0],
                split=meta.get("split"),
                data=json.loads(row[1]),
                meta=meta,
            )
        finally:
            await db.close()

    # ------------------------------------------------------------------
    # Sample meta CRUD
    # ------------------------------------------------------------------

    async def set_sample_meta(self, request: SampleMetaSetRequest):
        """Upsert meta key-value pairs for one or more samples."""
        db = await self._get_db()
        try:
            async with db.execute("SELECT id FROM datasets WHERE id = ?", (request.dataset_id,)) as cursor:
                if not await cursor.fetchone():
                    raise HTTPException(status_code=404, detail="dataset not found")
            now = _now()
            for sample_id, kv in request.records.items():
                for key, value in kv.items():
                    await db.execute(
                        "INSERT INTO dataset_sample_meta (dataset_id, sample_id, key, value) VALUES (?, ?, ?, ?)"
                        " ON CONFLICT(dataset_id, sample_id, key) DO UPDATE SET value=excluded.value",
                        (request.dataset_id, sample_id, key, json.dumps(value)),
                    )
            await db.execute(
                "UPDATE datasets SET updated_at=? WHERE id=?", (now, request.dataset_id)
            )
            await db.commit()
        finally:
            await db.close()
        return {"dataset_id": request.dataset_id, "updated_samples": len(request.records)}

    async def get_sample_meta(self, request: SampleMetaGetRequest):
        """Get meta for specified samples (and optionally specific keys)."""
        db = await self._get_db()
        try:
            meta_map = await _fetch_sample_meta(
                db, request.dataset_id, request.sample_ids, request.keys
            )
        finally:
            await db.close()
        return meta_map

    async def delete_sample_meta(self, request: SampleMetaDeleteRequest):
        """Delete meta keys for given samples. If keys is None, delete all meta for those samples."""
        db = await self._get_db()
        try:
            placeholders = ",".join("?" * len(request.sample_ids))
            if request.keys:
                key_placeholders = ",".join("?" * len(request.keys))
                await db.execute(
                    f"DELETE FROM dataset_sample_meta WHERE dataset_id=? AND sample_id IN ({placeholders}) AND key IN ({key_placeholders})",
                    [request.dataset_id] + request.sample_ids + request.keys,
                )
            else:
                await db.execute(
                    f"DELETE FROM dataset_sample_meta WHERE dataset_id=? AND sample_id IN ({placeholders})",
                    [request.dataset_id] + request.sample_ids,
                )
            await db.commit()
        finally:
            await db.close()
        return {"dataset_id": request.dataset_id, "deleted_samples": len(request.sample_ids)}

    async def list_sample_meta(self, request: SampleMetaGetRequest):
        """List all distinct meta keys present in the dataset (or for given sample_ids)."""
        db = await self._get_db()
        try:
            if request.sample_ids:
                placeholders = ",".join("?" * len(request.sample_ids))
                sql = f"SELECT DISTINCT key FROM dataset_sample_meta WHERE dataset_id=? AND sample_id IN ({placeholders})"
                params = [request.dataset_id] + request.sample_ids
            else:
                sql = "SELECT DISTINCT key FROM dataset_sample_meta WHERE dataset_id=?"
                params = [request.dataset_id]
            async with db.execute(sql, params) as cursor:
                keys = [r[0] for r in await cursor.fetchall()]
        finally:
            await db.close()
        return {"dataset_id": request.dataset_id, "keys": keys}

    async def select_samples(self, request: SampleSelectRequest):
        """Select samples whose meta satisfies ALL filters (AND logic).

        Strategy: for each filter, compute the matching sample_id set via SQL,
        then intersect all sets in Python. Finally fetch sample rows and meta.
        The ``split`` shorthand on the request is automatically converted into
        a ``split eq <value>`` filter so that split is treated uniformly as meta.
        """
        db = await self._get_db()
        try:
            # Merge the convenience split field into filters
            filters = list(request.filters)
            if request.split:
                filters.append(MetaFilter(key="split", op="eq", value=request.split))

            # --- build candidate sets per filter ---
            candidate_sets: List[set] = []
            for f in filters:
                op = f.op.lower()
                if op in ("exists", "not_exists"):
                    async with db.execute(
                        "SELECT DISTINCT sample_id FROM dataset_sample_meta WHERE dataset_id=? AND key=?",
                        (request.dataset_id, f.key),
                    ) as cursor:
                        ids = {r[0] for r in await cursor.fetchall()}
                    candidate_sets.append(ids if op == "exists" else None)
                    if op == "not_exists":
                        # handled below via complement
                        candidate_sets[-1] = ("NOT_EXISTS", ids)
                else:
                    async with db.execute(
                        "SELECT sample_id, value FROM dataset_sample_meta WHERE dataset_id=? AND key=?",
                        (request.dataset_id, f.key),
                    ) as cursor:
                        rows = await cursor.fetchall()

                    matched: set = set()
                    for sid, raw in rows:
                        try:
                            v = json.loads(raw)
                        except Exception:
                            v = raw
                        try:
                            if op == "eq" and v == f.value:
                                matched.add(sid)
                            elif op == "ne" and v != f.value:
                                matched.add(sid)
                            elif op == "gt" and v > f.value:
                                matched.add(sid)
                            elif op == "gte" and v >= f.value:
                                matched.add(sid)
                            elif op == "lt" and v < f.value:
                                matched.add(sid)
                            elif op == "lte" and v <= f.value:
                                matched.add(sid)
                            elif op == "in" and v in f.value:
                                matched.add(sid)
                            elif op == "not_in" and v not in f.value:
                                matched.add(sid)
                            elif op == "contains" and isinstance(v, str) and isinstance(f.value, str) and f.value in v:
                                matched.add(sid)
                        except TypeError:
                            pass
                    candidate_sets.append(matched)

            # --- intersect all sets (handle NOT_EXISTS tuples) ---
            # first collect all sample_ids in this dataset as universe for not_exists
            if any(isinstance(s, tuple) for s in candidate_sets):
                async with db.execute(
                    "SELECT DISTINCT sample_id FROM dataset_samples WHERE dataset_id=?",
                    (request.dataset_id,),
                ) as cursor:
                    universe = {r[0] for r in await cursor.fetchall()}
            else:
                universe = None

            matched_ids: Optional[set] = None
            for s in candidate_sets:
                if isinstance(s, tuple):  # NOT_EXISTS
                    complement = universe - s[1]
                    matched_ids = complement if matched_ids is None else matched_ids & complement
                else:
                    matched_ids = s if matched_ids is None else matched_ids & s

            if matched_ids is None:
                matched_ids = set()

            total = len(matched_ids)
            paginated_ids = sorted(matched_ids)[request.start: request.start + request.limit]

            if not paginated_ids:
                return SampleListResponse(total=total, start=request.start, limit=request.limit, samples=[])

            # --- fetch sample data ---
            placeholders = ",".join("?" * len(paginated_ids))
            sql = f"SELECT sample_id, data FROM dataset_samples WHERE dataset_id=? AND sample_id IN ({placeholders})"
            async with db.execute(sql, [request.dataset_id] + paginated_ids) as cursor:
                sample_rows = await cursor.fetchall()

            meta_map: Dict[str, Dict[str, Any]] = {}
            if request.include_meta:
                meta_map = await _fetch_sample_meta(db, request.dataset_id, paginated_ids)

            # preserve sorted order
            order = {sid: i for i, sid in enumerate(paginated_ids)}
            samples = sorted(
                [
                    SampleRecord(
                        sample_id=r[0],
                        split=meta_map.get(r[0], {}).get("split"),
                        data=json.loads(r[1]),
                        meta=meta_map.get(r[0], {}),
                    )
                    for r in sample_rows
                ],
                key=lambda s: order.get(s.sample_id, 0),
            )
            return SampleListResponse(total=total, start=request.start, limit=request.limit, samples=samples)
        finally:
            await db.close()
