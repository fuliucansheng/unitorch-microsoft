# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

"""
Lightweight Python wrapper for the Studios API endpoints.

Covers: Chats, Datasets, Jobs, Labels, Reports, Utils.

Usage:
    from unitorch_microsoft.apps.skills.studios import StudiosClient

    client = StudiosClient("http://127.0.0.1:5000")

    # --- Chats ---
    session = client.new_session()
    session_id = session["new_session_id"]
    reply = client.chat(session_id, "Analyze dataset X")

    # --- Datasets ---
    ds = client.create_dataset("my_ds", ["train.tsv"], splits={"train": ["train.tsv"]})
    samples = client.list_samples(ds["id"], split="train", limit=50)

    # --- Jobs ---
    job = client.create_job("train", "python train.py", workdir="/ws/jobs/train")
    logs = client.job_logs(job["id"])

    # --- Labels ---
    task = client.create_label_task("QA", dataset_id=ds["id"], label_fields=["quality"])
    client.assign_labels(task["id"], labeler_ids=["user1"])

    # --- Reports ---
    report = client.create_report("Weekly", content="# Analysis\n...")
    client.update_report(report["id"], content="# Analysis v2\n...")
"""

import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional


class StudiosClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: int = 120):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _get(self, path: str, **params) -> Any:
        resp = httpx.get(self._url(path), params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Any = None, **kwargs) -> Any:
        resp = httpx.post(self._url(path), json=json, timeout=self._timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # ==================================================================
    # Chats
    # ==================================================================

    def get_commands(self) -> List[dict]:
        """List available chat commands."""
        return self._get("/microsoft/apps/studios/chats/commands")

    def get_entities(self) -> List[dict]:
        """List available entities (datasets, jobs, labels, reports)."""
        return self._get("/microsoft/apps/studios/chats/entities")

    def get_models(self) -> List[dict]:
        """List available LLM models connected to opencode."""
        return self._get("/microsoft/apps/studios/chats/models")

    def new_session(
        self,
        session_id: Optional[str] = None,
    ) -> dict:
        """Create a new chat session (or fork an existing one).

        Returns {"new_session_id", "workspace", "welcome_message"}.
        """
        payload: dict = {}
        if session_id:
            payload["session_id"] = session_id
        return self._post("/microsoft/apps/studios/chats/new", json=payload)

    def chat(
        self,
        session_id: str,
        message: str,
        mode: str = "plan",
        model_id: str = "",
        provider_id: str = "",
        entities: Optional[List[dict]] = None,
    ) -> str:
        """Send a message and return the assistant's reply text.

        entities: list of {"type": "dataset"|"job"|"label"|"report", "id": "..."}
        """
        result = self._post(
            "/microsoft/apps/studios/chats/completions",
            json={
                "session_id": session_id,
                "message": {"role": "user", "content": message},
                "mode": mode,
                "model_id": model_id,
                "provider_id": provider_id,
                "entities": entities or [],
                "stream": False,
            },
        )
        return result.get("content", "")

    def get_history(self, session_id: str) -> dict:
        """Return full message history for a session."""
        return self._get("/microsoft/apps/studios/chats/history", session_id=session_id)

    def get_sessions(self) -> List[dict]:
        """List all chat sessions."""
        return self._get("/microsoft/apps/studios/chats/sessions")

    def rename_session(self, session_id: str, name: str) -> dict:
        """Rename a chat session."""
        return self._post("/microsoft/apps/studios/chats/name", json={"session_id": session_id, "name": name})

    def delete_session(self, session_id: str) -> dict:
        """Delete a chat session."""
        return self._post("/microsoft/apps/studios/chats/delete", json={"session_id": session_id})

    # ==================================================================
    # Datasets
    # ==================================================================

    def list_datasets(self) -> List[dict]:
        """List all datasets."""
        return self._get("/microsoft/apps/studios/datasets")

    def create_dataset(
        self,
        name: str,
        file_paths: List[str],
        description: str = "",
        splits: Optional[Dict[str, List[str]]] = None,
    ) -> dict:
        """Create a dataset from TSV files.

        splits: {"train": ["f1.tsv"], "test": ["f2.tsv"]}  (optional, defaults to "test")
        """
        import json as _json
        params: dict = {"name": name}
        if description:
            params["description"] = description
        if splits:
            params["splits"] = _json.dumps(splits)

        files = [
            ("files", (Path(p).name, open(p, "rb"), "text/tab-separated-values"))
            for p in file_paths
        ]
        try:
            resp = httpx.post(
                self._url("/microsoft/apps/studios/datasets/create"),
                params=params,
                files=files,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for _, (_, fobj, _) in files:
                try:
                    fobj.close()
                except Exception:
                    pass

    def upload_split(
        self,
        dataset_id: str,
        file_paths: List[str],
        split: str = "train",
    ) -> dict:
        """Append TSV rows to an existing dataset split."""
        files = [
            ("files", (Path(p).name, open(p, "rb"), "text/tab-separated-values"))
            for p in file_paths
        ]
        try:
            resp = httpx.post(
                self._url("/microsoft/apps/studios/datasets/upload"),
                params={"dataset_id": dataset_id, "split": split},
                files=files,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for _, (_, fobj, _) in files:
                try:
                    fobj.close()
                except Exception:
                    pass

    def get_dataset(self, dataset_id: str) -> dict:
        """Get full info for a dataset."""
        return self._post("/microsoft/apps/studios/datasets/get", json={"id": dataset_id})

    def delete_dataset(self, dataset_id: str) -> dict:
        """Delete a dataset and all its samples."""
        return self._post("/microsoft/apps/studios/datasets/delete", json={"id": dataset_id})

    def update_dataset_meta(self, dataset_id: str, meta: dict) -> dict:
        """Update dataset-level metadata (partial update)."""
        return self._post(
            "/microsoft/apps/studios/datasets/meta/update",
            json={"id": dataset_id, "meta": meta},
        )

    def list_samples(
        self,
        dataset_id: str,
        split: Optional[str] = None,
        start: int = 0,
        limit: int = 20,
        include_meta: bool = True,
    ) -> dict:
        """List samples with pagination. Returns {"total", "start", "limit", "samples"}."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/list",
            json={
                "dataset_id": dataset_id,
                "split": split,
                "start": start,
                "limit": limit,
                "include_meta": include_meta,
            },
        )

    def get_sample(self, dataset_id: str, sample_id: str) -> dict:
        """Get a single sample by ID."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/get",
            json={"dataset_id": dataset_id, "sample_id": sample_id},
        )

    def select_samples(
        self,
        dataset_id: str,
        filters: List[dict],
        split: Optional[str] = None,
        start: int = 0,
        limit: int = 100,
        include_meta: bool = True,
    ) -> dict:
        """Filter samples by meta conditions (AND logic).

        filters: [{"key": "score", "op": "gte", "value": 0.9}, ...]
        ops: eq | ne | gt | gte | lt | lte | in | not_in | exists | not_exists | contains
        """
        return self._post(
            "/microsoft/apps/studios/datasets/sample/select",
            json={
                "dataset_id": dataset_id,
                "filters": filters,
                "split": split,
                "start": start,
                "limit": limit,
                "include_meta": include_meta,
            },
        )

    def set_sample_meta(self, dataset_id: str, records: Dict[str, Dict[str, Any]]) -> dict:
        """Upsert meta key-value pairs. records: {sample_id: {key: value}}."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/meta/set",
            json={"dataset_id": dataset_id, "records": records},
        )

    def get_sample_meta(
        self,
        dataset_id: str,
        sample_ids: Optional[List[str]] = None,
        keys: Optional[List[str]] = None,
    ) -> dict:
        """Get meta for samples. Returns {sample_id: {key: value}}."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/meta/get",
            json={"dataset_id": dataset_id, "sample_ids": sample_ids, "keys": keys},
        )

    def delete_sample_meta(
        self,
        dataset_id: str,
        sample_ids: List[str],
        keys: Optional[List[str]] = None,
    ) -> dict:
        """Delete meta keys for given samples. If keys is None, deletes all meta."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/meta/delete",
            json={"dataset_id": dataset_id, "sample_ids": sample_ids, "keys": keys},
        )

    def list_sample_meta_keys(
        self,
        dataset_id: str,
        sample_ids: Optional[List[str]] = None,
    ) -> dict:
        """List all distinct meta keys in the dataset."""
        return self._post(
            "/microsoft/apps/studios/datasets/sample/meta/list",
            json={"dataset_id": dataset_id, "sample_ids": sample_ids},
        )

    # ==================================================================
    # Jobs
    # ==================================================================

    def list_jobs(self) -> List[dict]:
        """List all jobs."""
        return self._get("/microsoft/apps/studios/jobs")

    def create_job(
        self,
        name: str,
        command: str,
        workdir: str,
        description: str = "",
        job_type: str = "",
        dataset_ids: Optional[List[str]] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Create and immediately launch a job.

        workdir is required and managed by the agent (e.g. a session workspace subfolder).
        """
        return self._post(
            "/microsoft/apps/studios/jobs/create",
            json={
                "name": name,
                "description": description,
                "type": job_type,
                "dataset_ids": dataset_ids or [],
                "command": command,
                "workdir": workdir,
                "extra": extra or {},
            },
        )

    def get_job(self, job_id: str) -> dict:
        """Get full info for a job."""
        return self._post("/microsoft/apps/studios/jobs/get", json={"id": job_id})

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a pending or running job (sends SIGTERM)."""
        return self._post("/microsoft/apps/studios/jobs/cancel", json={"id": job_id})

    def restart_job(self, job_id: str) -> dict:
        """Restart a completed/failed/cancelled job."""
        return self._post("/microsoft/apps/studios/jobs/restart", json={"id": job_id})

    def delete_job(self, job_id: str) -> dict:
        """Delete a job record (must not be running)."""
        return self._post("/microsoft/apps/studios/jobs/delete", json={"id": job_id})

    def job_logs(self, job_id: str, tail: Optional[int] = 50) -> dict:
        """Get log lines for a job.

        tail=None returns all lines. For finished jobs all lines are always returned.
        Returns {"id", "status", "lines", "total_lines"}.
        """
        return self._post(
            "/microsoft/apps/studios/jobs/logs",
            json={"id": job_id, "tail": tail},
        )

    # ==================================================================
    # Labels
    # ==================================================================

    def list_label_tasks(self) -> List[dict]:
        """List all label tasks."""
        return self._get("/microsoft/apps/studios/labels")

    def create_label_task(
        self,
        name: str,
        dataset_id: str,
        description: str = "",
        display_fields: Optional[List[dict]] = None,
        label_fields: Optional[List[dict]] = None,
        ui_html: str = "",
        extra: Optional[dict] = None,
    ) -> dict:
        """Create a label task tied to a dataset.

        display_fields: columns shown in the annotation UI.
            [{"key": "image_url", "label": "Image", "source": "data", "type": "image"},
             {"key": "split",     "label": "Split",  "source": "meta", "type": "text"}]
            source: "data" | "meta"
            type:   text | image | video | number | json

        label_fields: fields annotators must fill in.
            Each field is a dict with the following keys:
              key      (str)        – field key stored in the label dict
              label    (str)        – human-readable label shown in the UI
              type     (str)        – text | select | multiselect | number | bool
              options  (list[str])  – allowed values for select / multiselect types
              required (bool)       – must be filled before submit (default true)
              default  (str|null)   – pre-filled default value

            Example:
            [
                {"key": "quality",  "label": "Quality",  "type": "select",
                 "options": ["excellent", "good", "fair", "poor"], "required": True},
                {"key": "is_valid", "label": "Is Valid", "type": "bool",        "options": []},
                {"key": "tags",     "label": "Tags",     "type": "multiselect",
                 "options": ["blurry", "cropped", "watermark"], "required": False},
                {"key": "comment",  "label": "Comment",  "type": "text",        "options": [], "required": False},
            ]
        """
        return self._post(
            "/microsoft/apps/studios/labels/create",
            json={
                "name": name,
                "description": description,
                "dataset_id": dataset_id,
                "display_fields": display_fields or [],
                "label_fields": label_fields or [],
                "ui_html": ui_html,
                "extra": extra or {},
            },
        )

    def get_label_task(self, task_id: str) -> dict:
        """Get full info for a label task.

        Returns LabelTaskInfo with label_fields as a list of field dicts
        (key, label, type, options, required, default).
        """
        return self._post("/microsoft/apps/studios/labels/get", json={"id": task_id})

    def update_label_task(
        self,
        task_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        display_fields: Optional[List[dict]] = None,
        label_fields: Optional[List[dict]] = None,
        ui_html: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Update mutable fields of a label task (partial update).

        Only provided (non-None) fields are overwritten. extra is merged.
        label_fields must be the full replacement list of field dicts when provided.
        """
        payload: dict = {"id": task_id}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if display_fields is not None:
            payload["display_fields"] = display_fields
        if label_fields is not None:
            payload["label_fields"] = label_fields
        if ui_html is not None:
            payload["ui_html"] = ui_html
        if extra is not None:
            payload["extra"] = extra
        return self._post("/microsoft/apps/studios/labels/update", json=payload)

    def delete_label_task(self, task_id: str) -> dict:
        """Delete a label task and all its assignments."""
        return self._post("/microsoft/apps/studios/labels/delete", json={"id": task_id})

    def assign_labels(
        self,
        task_id: str,
        labeler_ids: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> dict:
        """Assign samples to labelers (round-robin if sample_ids is empty)."""
        return self._post(
            "/microsoft/apps/studios/labels/assign",
            json={
                "task_id": task_id,
                "labeler_ids": labeler_ids,
                "sample_ids": sample_ids or [],
            },
        )

    def random_sample_for_annotation(
        self,
        task_id: str,
        sample_id: Optional[str] = None,
    ) -> dict:
        """Get a sample for annotation UI. Returns {"sample_id", "data", "meta"}."""
        return self._post(
            "/microsoft/apps/studios/labels/sample/random",
            json={"task_id": task_id, "sample_id": sample_id},
        )

    def submit_label(
        self,
        task_id: str,
        labeler_id: str,
        sample_id: str,
        label: dict,
        comment: str = "",
    ) -> dict:
        """Submit an annotation for a sample."""
        return self._post(
            "/microsoft/apps/studios/labels/sample/submit",
            json={
                "task_id": task_id,
                "labeler_id": labeler_id,
                "sample_id": sample_id,
                "label": label,
                "comment": comment,
            },
        )

    def label_progress(self, task_id: str) -> dict:
        """Get annotation progress. Returns {"task_id", "total_samples", "completed_samples", "labelers"}."""
        return self._post("/microsoft/apps/studios/labels/progress", json={"id": task_id})

    def export_labels(
        self,
        task_id: str,
        labeler_ids: Optional[List[str]] = None,
        include_unfinished: bool = False,
    ) -> dict:
        """Export annotations merged by sample_id.

        Returns {sample_id: {"data", "meta", "labels": {labeler_id: {"label", "comment", ...}}}}.
        """
        return self._post(
            "/microsoft/apps/studios/labels/export",
            json={
                "task_id": task_id,
                "labeler_ids": labeler_ids,
                "include_unfinished": include_unfinished,
            },
        )

    # ==================================================================
    # Reports
    # ==================================================================

    def list_reports(self) -> List[dict]:
        """List all reports."""
        return self._get("/microsoft/apps/studios/reports")

    def create_report(
        self,
        name: str,
        content: str = "",
        description: str = "",
        dataset_ids: Optional[List[str]] = None,
        job_ids: Optional[List[str]] = None,
        label_task_ids: Optional[List[str]] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Create a markdown report."""
        return self._post(
            "/microsoft/apps/studios/reports/create",
            json={
                "name": name,
                "description": description,
                "content": content,
                "dataset_ids": dataset_ids or [],
                "job_ids": job_ids or [],
                "label_task_ids": label_task_ids or [],
                "extra": extra or {},
            },
        )

    def get_report(self, report_id: str) -> dict:
        """Get full content of a report."""
        return self._post("/microsoft/apps/studios/reports/get", json={"id": report_id})

    def update_report(self, report_id: str, **fields) -> dict:
        """Update a report (partial update, extra is merged).

        Keyword args: name, description, content, dataset_ids, job_ids, label_task_ids, extra
        """
        return self._post(
            "/microsoft/apps/studios/reports/update",
            json={"id": report_id, **fields},
        )

    def delete_report(self, report_id: str) -> dict:
        """Delete a report."""
        return self._post("/microsoft/apps/studios/reports/delete", json={"id": report_id})

    # ==================================================================
    # Utils
    # ==================================================================

    def upload_file(self, file_path: str) -> dict:
        """Upload a file to the statics store.

        Returns {"path": "<server_path>", "filename": "<original_name>", "size": <bytes>}.
        The returned path can be used as a reference in datasets, reports, etc.
        """
        with open(file_path, "rb") as f:
            resp = httpx.post(
                self._url("/microsoft/apps/studios/utils/upload"),
                files={"file": (Path(file_path).name, f)},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json()
