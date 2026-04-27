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

# --- display field config ---

class DisplayField(BaseModel):
    """One column/field shown in the annotation UI."""
    key: str                        # data or meta key name
    label: str = ""                 # human-readable label
    source: str = "data"            # "data" | "meta"
    type: str = "text"              # text | image | video | number | json
    width: Optional[int] = None     # pixel width hint for table columns


# --- label field config ---

class LabelField(BaseModel):
    """One field that annotators must fill in."""
    key: str                                    # field key name stored in label dict
    label: str = ""                             # human-readable label shown in UI
    type: str = "text"                          # text | select | multiselect | number | bool
    options: List[str] = Field(default_factory=list)  # allowed values for select / multiselect
    required: bool = True                       # whether the field must be filled before submit
    default: Optional[str] = None              # default value pre-filled in UI


# --- labeler ---

class LabelerInfo(BaseModel):
    id: str
    name: str
    assigned: int = 0
    completed: int = 0


# --- label task ---

class LabelTaskSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    dataset_id: str = ""
    status: str = "pending"         # pending | active | completed | archived
    created_at: str = ""
    updated_at: str = ""


class LabelTaskInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    dataset_id: str = ""
    status: str = "pending"
    display_fields: List[DisplayField] = Field(default_factory=list)
    label_fields: List[LabelField] = Field(default_factory=list)  # fields annotators must fill in
    ui_html: str = ""               # custom HTML template for the annotation UI
    labelers: List[LabelerInfo] = Field(default_factory=list)
    total_samples: int = 0
    completed_samples: int = 0
    created_at: str = ""
    updated_at: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


# --- request models ---

class LabelIdRequest(BaseModel):
    id: str


class LabelTaskCreateRequest(BaseModel):
    name: str
    description: str = ""
    dataset_id: str
    display_fields: List[DisplayField] = Field(default_factory=list)
    label_fields: List[LabelField] = Field(default_factory=list)
    ui_html: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


class LabelTaskUpdateRequest(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    display_fields: Optional[List[DisplayField]] = None
    label_fields: Optional[List[LabelField]] = None
    ui_html: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class AssignRequest(BaseModel):
    """Assign samples to one or more labelers.

    If ``sample_ids`` is empty, all unassigned samples in the dataset are
    distributed evenly across ``labeler_ids``.
    """
    task_id: str
    labeler_ids: List[str]
    sample_ids: List[str] = Field(default_factory=list)


class SubmitLabelRequest(BaseModel):
    task_id: str
    labeler_id: str
    sample_id: str
    label: Dict[str, Any]
    comment: str = ""


class RandomSampleRequest(BaseModel):
    """Get a sample (random or specific) with its data and meta for the annotation UI."""
    task_id: str
    sample_id: Optional[str] = None    # None = pick a random sample from the dataset


class ExportRequest(BaseModel):
    """Export annotations merged by sample_id."""
    task_id: str
    labeler_ids: Optional[List[str]] = None   # None = all labelers
    include_unfinished: bool = False


# --- response models ---

class SampleForAnnotation(BaseModel):
    sample_id: str
    data: Dict[str, Any]
    meta: Dict[str, Any]


class LabelProgressResponse(BaseModel):
    task_id: str
    total_samples: int
    completed_samples: int
    labelers: List[LabelerInfo]


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

INIT_SQL = """
CREATE TABLE IF NOT EXISTS label_tasks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    dataset_id      TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    display_fields  TEXT NOT NULL DEFAULT '[]',
    label_fields    TEXT NOT NULL DEFAULT '[]',
    ui_html         TEXT NOT NULL DEFAULT '',
    labelers        TEXT NOT NULL DEFAULT '[]',
    total_samples   INTEGER NOT NULL DEFAULT 0,
    completed_samples INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT '',
    extra           TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS label_assignments (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    labeler_id  TEXT NOT NULL,
    sample_id   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    label       TEXT NOT NULL DEFAULT '{}',
    comment     TEXT NOT NULL DEFAULT '',
    labeled_at  TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (task_id) REFERENCES label_tasks(id),
    UNIQUE (task_id, labeler_id, sample_id)
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_task_info(row) -> LabelTaskInfo:
    return LabelTaskInfo(
        id=row[0], name=row[1], description=row[2], dataset_id=row[3],
        status=row[4],
        display_fields=[DisplayField(**f) for f in json.loads(row[5])],
        label_fields=[LabelField(**f) for f in json.loads(row[6])],
        ui_html=row[7],
        labelers=[LabelerInfo(**lb) for lb in json.loads(row[8])],
        total_samples=row[9], completed_samples=row[10],
        created_at=row[11], updated_at=row[12],
        extra=json.loads(row[13]),
    )


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------

@register_fastapi("microsoft/apps/studios/labels")
class StudioLabelsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/labels")
        router = config.getoption("router", "/microsoft/apps/studios/labels")
        studios_folder = config.getoption("studios_folder", "studios")
        dbs_folder = os.path.join(studios_folder, "dbs")
        os.makedirs(dbs_folder, exist_ok=True)
        self._db_path = os.path.join(dbs_folder, "labels.db")
        self._datasets_db_path = os.path.join(dbs_folder, "datasets.db")

        self._router = APIRouter(prefix=router)
        # task CRUD
        self._router.add_api_route("", self.list_tasks, methods=["GET"])
        self._router.add_api_route("/create", self.create_task, methods=["POST"])
        self._router.add_api_route("/get", self.get_task, methods=["POST"])
        self._router.add_api_route("/update", self.update_task, methods=["POST"])
        self._router.add_api_route("/delete", self.delete_task, methods=["POST"])
        # assignment & annotation
        self._router.add_api_route("/assign", self.assign_samples, methods=["POST"])
        self._router.add_api_route("/sample/random", self.random_sample, methods=["POST"])
        self._router.add_api_route("/sample/submit", self.submit_label, methods=["POST"])
        # progress & export
        self._router.add_api_route("/progress", self.get_progress, methods=["POST"])
        self._router.add_api_route("/export", self.export_labels, methods=["POST"])
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

    async def _get_datasets_db(self):
        return await aiosqlite.connect(self._datasets_db_path)

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    async def list_tasks(self):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, dataset_id, status, created_at, updated_at FROM label_tasks"
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                LabelTaskSummary(
                    id=r[0], name=r[1], description=r[2], dataset_id=r[3],
                    status=r[4], created_at=r[5], updated_at=r[6],
                )
                for r in rows
            ]
        finally:
            await db.close()

    async def create_task(self, request: LabelTaskCreateRequest):
        """Create a label task tied to a dataset.

        ``display_fields`` controls which data/meta columns are shown in the
        annotation UI.  ``label_fields`` lists the keys that annotators must
        fill in.  ``ui_html`` is an optional custom HTML template.
        """
        task_id = str(uuid.uuid4())
        now = _now()

        # Count total samples in the referenced dataset
        ddb = await self._get_datasets_db()
        try:
            async with ddb.execute(
                "SELECT COUNT(*) FROM dataset_samples WHERE dataset_id=?",
                (request.dataset_id,),
            ) as cursor:
                total = (await cursor.fetchone())[0]
        finally:
            await ddb.close()

        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO label_tasks "
                "(id, name, description, dataset_id, status, display_fields, label_fields, "
                "ui_html, labelers, total_samples, completed_samples, created_at, updated_at, extra) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, '[]', ?, 0, ?, ?, ?)",
                (
                    task_id, request.name, request.description, request.dataset_id,
                    json.dumps([f.model_dump() for f in request.display_fields]),
                    json.dumps([f.model_dump() for f in request.label_fields]),
                    request.ui_html,
                    total, now, now,
                    json.dumps(request.extra),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return await self.get_task(LabelIdRequest(id=task_id))

    async def get_task(self, request: LabelIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, dataset_id, status, display_fields, label_fields, "
                "ui_html, labelers, total_samples, completed_samples, created_at, updated_at, extra "
                "FROM label_tasks WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")
            return _build_task_info(row)
        finally:
            await db.close()

    async def update_task(self, request: LabelTaskUpdateRequest):
        """Update mutable fields of a label task (name, description, display_fields,
        label_fields, ui_html, extra). Only provided fields are overwritten."""
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, dataset_id, status, display_fields, label_fields, "
                "ui_html, labelers, total_samples, completed_samples, created_at, updated_at, extra "
                "FROM label_tasks WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")

            info = _build_task_info(row)
            name = request.name if request.name is not None else info.name
            description = request.description if request.description is not None else info.description
            display_fields = request.display_fields if request.display_fields is not None else info.display_fields
            label_fields = request.label_fields if request.label_fields is not None else info.label_fields
            ui_html = request.ui_html if request.ui_html is not None else info.ui_html
            if request.extra is not None:
                info.extra.update(request.extra)

            now = _now()
            await db.execute(
                "UPDATE label_tasks SET name=?, description=?, display_fields=?, label_fields=?, "
                "ui_html=?, updated_at=?, extra=? WHERE id=?",
                (
                    name, description,
                    json.dumps([f.model_dump() for f in display_fields]),
                    json.dumps([f.model_dump() for f in label_fields]),
                    ui_html, now, json.dumps(info.extra), request.id,
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return await self.get_task(LabelIdRequest(id=request.id))

    async def delete_task(self, request: LabelIdRequest):
        db = await self._get_db()
        try:
            async with db.execute("SELECT id FROM label_tasks WHERE id=?", (request.id,)) as cursor:
                if not await cursor.fetchone():
                    raise HTTPException(status_code=404, detail="label task not found")
            await db.execute("DELETE FROM label_assignments WHERE task_id=?", (request.id,))
            await db.execute("DELETE FROM label_tasks WHERE id=?", (request.id,))
            await db.commit()
        finally:
            await db.close()
        return {"id": request.id, "deleted": True}

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    async def assign_samples(self, request: AssignRequest):
        """Assign samples to labelers.

        If ``sample_ids`` is empty, all samples in the dataset are distributed
        evenly across ``labeler_ids`` (round-robin).  Existing assignments for
        the same (labeler, sample) pair are ignored (upsert-style).
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, dataset_id, labelers FROM label_tasks WHERE id=?",
                (request.task_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")

            dataset_id = row[1]
            labelers: List[Dict] = json.loads(row[2])

            # Resolve sample list
            if request.sample_ids:
                sample_ids = request.sample_ids
            else:
                ddb = await self._get_datasets_db()
                try:
                    async with ddb.execute(
                        "SELECT sample_id FROM dataset_samples WHERE dataset_id=?",
                        (dataset_id,),
                    ) as cursor:
                        sample_ids = [r[0] for r in await cursor.fetchall()]
                finally:
                    await ddb.close()

            # Upsert assignments (round-robin across labelers)
            now = _now()
            new_assignments = 0
            for i, sample_id in enumerate(sample_ids):
                labeler_id = request.labeler_ids[i % len(request.labeler_ids)]
                async with db.execute(
                    "SELECT id FROM label_assignments WHERE task_id=? AND labeler_id=? AND sample_id=?",
                    (request.task_id, labeler_id, sample_id),
                ) as cursor:
                    exists = await cursor.fetchone()
                if not exists:
                    await db.execute(
                        "INSERT INTO label_assignments (id, task_id, labeler_id, sample_id, status) "
                        "VALUES (?, ?, ?, ?, 'pending')",
                        (str(uuid.uuid4()), request.task_id, labeler_id, sample_id),
                    )
                    new_assignments += 1

            # Sync labelers list in task
            labeler_map = {lb["id"]: lb for lb in labelers}
            for labeler_id in request.labeler_ids:
                if labeler_id not in labeler_map:
                    labeler_map[labeler_id] = {"id": labeler_id, "name": labeler_id, "assigned": 0, "completed": 0}

            # Recalculate assigned counts
            for labeler_id in labeler_map:
                async with db.execute(
                    "SELECT COUNT(*) FROM label_assignments WHERE task_id=? AND labeler_id=?",
                    (request.task_id, labeler_id),
                ) as cursor:
                    labeler_map[labeler_id]["assigned"] = (await cursor.fetchone())[0]

            await db.execute(
                "UPDATE label_tasks SET labelers=?, status='active', updated_at=? WHERE id=?",
                (json.dumps(list(labeler_map.values())), now, request.task_id),
            )
            await db.commit()
        finally:
            await db.close()

        return {"task_id": request.task_id, "new_assignments": new_assignments}

    # ------------------------------------------------------------------
    # Annotation UI helpers
    # ------------------------------------------------------------------

    async def random_sample(self, request: RandomSampleRequest):
        """Return a sample's data + meta for the annotation UI.

        If ``sample_id`` is provided, returns that specific sample.
        Otherwise picks a random sample from the task's dataset.
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT dataset_id FROM label_tasks WHERE id=?", (request.task_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")
            dataset_id = row[0]
        finally:
            await db.close()

        ddb = await self._get_datasets_db()
        try:
            if request.sample_id:
                sample_id = request.sample_id
            else:
                # Exclude samples that already have at least one completed label for this task.
                db2 = await self._get_db()
                try:
                    async with db2.execute(
                        "SELECT sample_id FROM label_assignments "
                        "WHERE task_id=? AND status='completed'",
                        (request.task_id,),
                    ) as cursor:
                        labeled_ids = {r[0] for r in await cursor.fetchall()}
                finally:
                    await db2.close()

                if labeled_ids:
                    placeholders = ",".join("?" * len(labeled_ids))
                    sql = (
                        f"SELECT sample_id FROM dataset_samples "
                        f"WHERE dataset_id=? AND sample_id NOT IN ({placeholders}) "
                        f"ORDER BY RANDOM() LIMIT 1"
                    )
                    params = [dataset_id] + list(labeled_ids)
                else:
                    sql = "SELECT sample_id FROM dataset_samples WHERE dataset_id=? ORDER BY RANDOM() LIMIT 1"
                    params = [dataset_id]

                async with ddb.execute(sql, params) as cursor:
                    r = await cursor.fetchone()
                if not r:
                    raise HTTPException(status_code=404, detail="no unlabeled samples in dataset")
                sample_id = r[0]

            async with ddb.execute(
                "SELECT data FROM dataset_samples WHERE dataset_id=? AND sample_id=?",
                (dataset_id, sample_id),
            ) as cursor:
                sample_row = await cursor.fetchone()
            if not sample_row:
                raise HTTPException(status_code=404, detail="sample not found in dataset")

            data = json.loads(sample_row[0])

            meta: Dict[str, Any] = {}
            async with ddb.execute(
                "SELECT key, value FROM dataset_sample_meta WHERE dataset_id=? AND sample_id=?",
                (dataset_id, sample_id),
            ) as cursor:
                async for key, val in cursor:
                    meta[key] = json.loads(val)
        finally:
            await ddb.close()

        return SampleForAnnotation(sample_id=sample_id, data=data, meta=meta)

    async def submit_label(self, request: SubmitLabelRequest):
        """Record a labeler's annotation for a sample.

        No pre-existing assignment is required — any labeler_id is accepted.
        If an assignment for (task, labeler, sample) already exists it is
        updated; otherwise a new one is inserted.  ``completed_samples`` on the
        parent task is recalculated after each submission.
        """
        db = await self._get_db()
        try:
            # Verify the task exists
            async with db.execute(
                "SELECT labelers FROM label_tasks WHERE id=?", (request.task_id,)
            ) as cursor:
                task_row = await cursor.fetchone()
            if not task_row:
                raise HTTPException(status_code=404, detail="label task not found")

            now = _now()

            # Upsert the assignment
            async with db.execute(
                "SELECT id FROM label_assignments WHERE task_id=? AND labeler_id=? AND sample_id=?",
                (request.task_id, request.labeler_id, request.sample_id),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing:
                await db.execute(
                    "UPDATE label_assignments SET status='completed', label=?, comment=?, labeled_at=? "
                    "WHERE task_id=? AND labeler_id=? AND sample_id=?",
                    (
                        json.dumps(request.label), request.comment, now,
                        request.task_id, request.labeler_id, request.sample_id,
                    ),
                )
            else:
                await db.execute(
                    "INSERT INTO label_assignments (id, task_id, labeler_id, sample_id, status, label, comment, labeled_at) "
                    "VALUES (?, ?, ?, ?, 'completed', ?, ?, ?)",
                    (
                        str(uuid.uuid4()), request.task_id, request.labeler_id, request.sample_id,
                        json.dumps(request.label), request.comment, now,
                    ),
                )

            # Sync labelers list in task (add if new, increment completed count)
            labelers = json.loads(task_row[0])
            labeler_map = {lb["id"]: lb for lb in labelers}
            if request.labeler_id not in labeler_map:
                labeler_map[request.labeler_id] = {
                    "id": request.labeler_id,
                    "name": request.labeler_id,
                    "assigned": 0,
                    "completed": 0,
                }
            lb = labeler_map[request.labeler_id]
            if not existing:
                lb["assigned"] = lb.get("assigned", 0) + 1
                lb["completed"] = lb.get("completed", 0) + 1

            # Count distinct samples labeled (by at least one labeler)
            async with db.execute(
                "SELECT COUNT(DISTINCT sample_id) FROM label_assignments "
                "WHERE task_id=? AND status='completed'",
                (request.task_id,),
            ) as cursor:
                completed_samples = (await cursor.fetchone())[0]

            await db.execute(
                "UPDATE label_tasks SET labelers=?, completed_samples=?, updated_at=? WHERE id=?",
                (json.dumps(list(labeler_map.values())), completed_samples, now, request.task_id),
            )
            await db.commit()
        finally:
            await db.close()

        return {"task_id": request.task_id, "sample_id": request.sample_id, "status": "completed"}

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    async def get_progress(self, request: LabelIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT total_samples, completed_samples, labelers FROM label_tasks WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")
            labelers = [LabelerInfo(**lb) for lb in json.loads(row[2])]
        finally:
            await db.close()

        return LabelProgressResponse(
            task_id=request.id,
            total_samples=row[0],
            completed_samples=row[1],
            labelers=labelers,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_labels(self, request: ExportRequest):
        """Export annotations merged by sample_id.

        Returns a dict keyed by ``sample_id``.  Each value contains the
        original sample ``data``, its ``meta``, and per-labeler results under
        ``labels`` — a dict keyed by ``labeler_id`` with ``label`` (dict) and
        ``comment`` fields.  Unfinished assignments are included only when
        ``include_unfinished`` is True.
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT dataset_id FROM label_tasks WHERE id=?", (request.task_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="label task not found")
            dataset_id = row[0]

            clauses = ["task_id=?"]
            params: list = [request.task_id]
            if request.labeler_ids:
                placeholders = ",".join("?" * len(request.labeler_ids))
                clauses.append(f"labeler_id IN ({placeholders})")
                params.extend(request.labeler_ids)
            if not request.include_unfinished:
                clauses.append("status='completed'")

            sql = (
                f"SELECT labeler_id, sample_id, label, comment, labeled_at, status "
                f"FROM label_assignments WHERE {' AND '.join(clauses)}"
            )
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        finally:
            await db.close()

        # Build {sample_id: {labeler_id: {...}}}
        sample_map: Dict[str, Dict[str, Any]] = {}
        for labeler_id, sample_id, label_json, comment, labeled_at, status in rows:
            sample_map.setdefault(sample_id, {})[labeler_id] = {
                "label": json.loads(label_json),
                "comment": comment,
                "labeled_at": labeled_at,
                "status": status,
            }

        if not sample_map:
            return {}

        # Fetch sample data + meta from datasets DB
        ddb = await self._get_datasets_db()
        try:
            sample_data: Dict[str, Dict] = {}
            sample_meta: Dict[str, Dict] = {}
            placeholders = ",".join("?" * len(sample_map))
            async with ddb.execute(
                f"SELECT sample_id, data FROM dataset_samples "
                f"WHERE dataset_id=? AND sample_id IN ({placeholders})",
                [dataset_id] + list(sample_map.keys()),
            ) as cursor:
                async for sid, data_json in cursor:
                    sample_data[sid] = json.loads(data_json)
            async with ddb.execute(
                f"SELECT sample_id, key, value FROM dataset_sample_meta "
                f"WHERE dataset_id=? AND sample_id IN ({placeholders})",
                [dataset_id] + list(sample_map.keys()),
            ) as cursor:
                async for sid, key, val in cursor:
                    sample_meta.setdefault(sid, {})[key] = json.loads(val)
        finally:
            await ddb.close()

        return {
            sample_id: {
                "data": sample_data.get(sample_id, {}),
                "meta": sample_meta.get(sample_id, {}),
                "labels": labeler_results,
            }
            for sample_id, labeler_results in sample_map.items()
        }
