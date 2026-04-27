# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import uuid
import signal
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

class JobSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    type: str = ""
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""


class JobInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    type: str = ""
    dataset_ids: List[str] = Field(default_factory=list)
    command: str = ""
    workdir: str = ""
    status: str = "pending"
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)
    log: str = ""


# --- request models ---

class JobIdRequest(BaseModel):
    id: str


class JobCreateRequest(BaseModel):
    name: str
    description: str = ""
    type: str = ""
    dataset_ids: List[str] = Field(default_factory=list)
    command: str
    workdir: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class JobLogsRequest(BaseModel):
    id: str
    tail: Optional[int] = 50   # None = all lines


# --- response models ---

class JobLogsResponse(BaseModel):
    id: str
    status: str
    lines: List[str]
    total_lines: int


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

INIT_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    type        TEXT NOT NULL DEFAULT '',
    dataset_ids TEXT NOT NULL DEFAULT '[]',
    command     TEXT NOT NULL DEFAULT '',
    workdir     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    pid         INTEGER,
    exit_code   INTEGER,
    log         TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT '',
    started_at  TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    extra       TEXT NOT NULL DEFAULT '{}'
);
"""


_LOG_FILENAME = "job.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path(workdir: str) -> str:
    return os.path.join(workdir, _LOG_FILENAME)


def _build_job_info(row) -> JobInfo:
    return JobInfo(
        id=row[0], name=row[1], description=row[2], type=row[3],
        dataset_ids=json.loads(row[4]), command=row[5], workdir=row[6],
        status=row[7], pid=row[8], exit_code=row[9],
        log=row[10],
        created_at=row[11], updated_at=row[12],
        started_at=row[13], finished_at=row[14],
        extra=json.loads(row[15]),
    )


def _read_log_lines(workdir: str, tail: Optional[int] = None) -> List[str]:
    path = _log_path(workdir)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if tail is not None:
        lines = lines[-tail:]
    return lines


def _append_log(workdir: str, text: str):
    path = _log_path(workdir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------

@register_fastapi("microsoft/apps/studios/jobs")
class StudioJobsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/jobs")
        router = config.getoption("router", "/microsoft/apps/studios/jobs")
        studios_folder = config.getoption("studios_folder", "studios")
        dbs_folder = os.path.join(studios_folder, "dbs")
        os.makedirs(dbs_folder, exist_ok=True)
        self._db_path = os.path.join(dbs_folder, "jobs.db")

        self._router = APIRouter(prefix=router)
        # job CRUD
        self._router.add_api_route("", self.list_jobs, methods=["GET"])
        self._router.add_api_route("/create", self.create_job, methods=["POST"])
        self._router.add_api_route("/get", self.get_job, methods=["POST"])
        self._router.add_api_route("/cancel", self.cancel_job, methods=["POST"])
        self._router.add_api_route("/restart", self.restart_job, methods=["POST"])
        self._router.add_api_route("/delete", self.delete_job, methods=["POST"])
        # logs
        self._router.add_api_route("/logs", self.get_logs, methods=["POST"])
        # service control
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

        self._lock = asyncio.Lock()
        self._running = False
        self._db_initialized = False
        # pid → asyncio.Task, tracked for cancel/cleanup
        self._process_tasks: Dict[str, asyncio.Task] = {}

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
    # Internal: run subprocess and stream logs
    # ------------------------------------------------------------------

    async def _run_job(self, job_id: str, command: str, workdir: str):
        """Launch the command in workdir, stream stdout+stderr to job.log,
        persist log into DB, and update job status/exit_code on completion."""
        db = await self._get_db()
        now = _now()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await db.execute(
                "UPDATE jobs SET status='running', pid=?, started_at=?, updated_at=? WHERE id=?",
                (proc.pid, now, now, job_id),
            )
            await db.commit()

            start_line = f"[{now}] Job started (pid={proc.pid}): {command}"
            _append_log(workdir, start_line)
            await db.execute(
                "UPDATE jobs SET log=log||?, updated_at=? WHERE id=?",
                (start_line + "\n", now, job_id),
            )
            await db.commit()

            # stream output line by line
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                _append_log(workdir, text)
                ts = _now()
                await db.execute(
                    "UPDATE jobs SET log=log||?, updated_at=? WHERE id=?",
                    (text + "\n", ts, job_id),
                )
                await db.commit()

            await proc.wait()
            exit_code = proc.returncode
            finished = _now()
            final_status = "completed" if exit_code == 0 else "failed"
            final_line = f"[{finished}] Job {final_status} (exit_code={exit_code})"
            _append_log(workdir, final_line)
            await db.execute(
                "UPDATE jobs SET status=?, exit_code=?, finished_at=?, updated_at=?, pid=NULL, "
                "log=log||? WHERE id=?",
                (final_status, exit_code, finished, finished, final_line + "\n", job_id),
            )
            await db.commit()
        except asyncio.CancelledError:
            # cancel_job already killed the process and updated DB
            pass
        except Exception as exc:
            err_time = _now()
            err_line = f"[{err_time}] Runner error: {exc}"
            _append_log(workdir, err_line)
            await db.execute(
                "UPDATE jobs SET status='failed', updated_at=?, pid=NULL, log=log||? WHERE id=?",
                (err_time, err_line + "\n", job_id),
            )
            await db.commit()
        finally:
            await db.close()
            self._process_tasks.pop(job_id, None)


    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    async def list_jobs(self):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, status, created_at, updated_at FROM jobs"
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                JobSummary(
                    id=r[0], name=r[1], description=r[2], type=r[3],
                    status=r[4], created_at=r[5], updated_at=r[6],
                )
                for r in rows
            ]
        finally:
            await db.close()

    async def create_job(self, request: JobCreateRequest):
        """Create and immediately launch a job.

        A working directory is created under ``workdir_root/<job_id>/`` unless
        ``workdir`` is explicitly provided.  The command is executed there via
        a shell subprocess; stdout and stderr are merged and written to
        ``job.log`` inside the workdir.
        """
        job_id = str(uuid.uuid4())
        now = _now()

        workdir = request.workdir
        os.makedirs(workdir, exist_ok=True)

        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO jobs (id, name, description, type, dataset_ids, command, workdir, "
                "status, log, created_at, updated_at, started_at, finished_at, extra) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '', ?, ?, '', '', ?)",
                (
                    job_id, request.name, request.description, request.type,
                    json.dumps(request.dataset_ids), request.command, workdir,
                    now, now, json.dumps(request.extra),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        # launch async subprocess
        task = asyncio.create_task(self._run_job(job_id, request.command, workdir))
        self._process_tasks[job_id] = task

        return await self.get_job(JobIdRequest(id=job_id))

    async def get_job(self, request: JobIdRequest):
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, name, description, type, dataset_ids, command, workdir, "
                "status, pid, exit_code, log, created_at, updated_at, started_at, finished_at, extra "
                "FROM jobs WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")
            return _build_job_info(row)
        finally:
            await db.close()

    async def cancel_job(self, request: JobIdRequest):
        """Cancel a running job by sending SIGTERM to its process group."""
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, status, pid, workdir FROM jobs WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")

            job_status, pid, workdir = row[1], row[2], row[3]
            if job_status not in ("pending", "running"):
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot cancel a job with status '{job_status}'",
                )

            now = _now()
            # kill process if running
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

            # cancel asyncio task
            task = self._process_tasks.pop(request.id, None)
            if task:
                task.cancel()

            cancel_line = f"[{now}] Job cancelled by user."
            _append_log(workdir, cancel_line)
            await db.execute(
                "UPDATE jobs SET status='cancelled', pid=NULL, updated_at=?, finished_at=?, "
                "log=log||? WHERE id=?",
                (now, now, cancel_line + "\n", request.id),
            )
            await db.commit()

            async with db.execute(
                "SELECT id, name, description, type, dataset_ids, command, workdir, "
                "status, pid, exit_code, log, created_at, updated_at, started_at, finished_at, extra "
                "FROM jobs WHERE id=?",
                (request.id,),
            ) as cursor:
                updated = await cursor.fetchone()
            return _build_job_info(updated)
        finally:
            await db.close()

    async def restart_job(self, request: JobIdRequest):
        """Restart a job that is completed / failed / cancelled.

        Clears the old log file, resets status to pending, then re-launches
        the same command in the same workdir.
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT id, status, command, workdir FROM jobs WHERE id=?",
                (request.id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")

            job_status, command, workdir = row[1], row[2], row[3]
            if job_status == "running":
                raise HTTPException(
                    status_code=400,
                    detail="job is already running; cancel it first",
                )

            # clear old log file and DB log
            log_path = _log_path(workdir)
            if os.path.exists(log_path):
                os.remove(log_path)

            now = _now()
            await db.execute(
                "UPDATE jobs SET status='pending', pid=NULL, exit_code=NULL, log='', "
                "started_at='', finished_at='', updated_at=? WHERE id=?",
                (now, request.id),
            )
            await db.commit()
        finally:
            await db.close()

        # re-launch
        task = asyncio.create_task(self._run_job(request.id, command, workdir))
        self._process_tasks[request.id] = task

        return await self.get_job(JobIdRequest(id=request.id))

    async def delete_job(self, request: JobIdRequest):
        """Delete a job record. Running jobs must be cancelled first."""
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT status FROM jobs WHERE id=?", (request.id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")
            if row[0] == "running":
                raise HTTPException(
                    status_code=400,
                    detail="job is running; cancel it before deleting",
                )
            await db.execute("DELETE FROM jobs WHERE id=?", (request.id,))
            await db.commit()
        finally:
            await db.close()
        return {"id": request.id, "deleted": True}

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    async def get_logs(self, request: JobLogsRequest):
        """Return log lines for a job.

        - While the job is **running**: returns the latest ``tail`` lines
          (default 50).  Pass ``tail=null`` to get all lines so far.
        - When the job is **completed / failed / cancelled**: ``tail`` is
          ignored and all lines are returned.
        """
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT status, workdir FROM jobs WHERE id=?", (request.id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")
            job_status, workdir = row[0], row[1]
        finally:
            await db.close()

        # For terminal states return full log regardless of tail param
        tail = None if job_status in ("completed", "failed", "cancelled") else request.tail
        all_lines = _read_log_lines(workdir, tail=None)
        display_lines = all_lines if tail is None else all_lines[-tail:] if tail else all_lines

        return JobLogsResponse(
            id=request.id,
            status=job_status,
            lines=display_lines,
            total_lines=len(all_lines),
        )
