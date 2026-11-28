"""Абстрактный слой хранения состояния и очереди задач + реализация на SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class StateStore(ABC):
    """Абстрактное хранилище сессий, очереди задач и метаданных карантина."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Имя бэкенда хранилища (например, 'sqlite')."""

    @abstractmethod
    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Создаёт или обновляет состояние сессии."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает данные сессии по ID или None."""

    @abstractmethod
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает список последних сессий (сводка)."""

    @abstractmethod
    def list_session_headers(self) -> List[Dict[str, Any]]:
        """Возвращает заголовки всех сессий (для TTL-очистки)."""

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Удаляет сессию по ID."""

    @abstractmethod
    def enqueue_job(
        self,
        job_id: str,
        session_id: str,
        job_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ставит новую задачу в очередь со статусом 'queued'."""

    @abstractmethod
    def claim_next_job(self) -> Optional[Dict[str, Any]]:
        """Атомарно забирает следующую задачу из очереди и переводит её в 'running'."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает задачу по job_id."""

    @abstractmethod
    def get_job_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает последнюю задачу по session_id."""

    @abstractmethod
    def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        """Обновляет статус задачи."""

    @abstractmethod
    def recover_stuck_jobs(self, max_attempts: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Восстанавливает задачи, оставшиеся в 'running' после падения процесса."""

    @abstractmethod
    def count_active_jobs(self) -> int:
        """Возвращает количество задач в статусах 'queued' и 'running'."""

    @abstractmethod
    def save_quarantine_item(self, item_id: str, data: Dict[str, Any]) -> None:
        """Создаёт или обновляет метаданные записи карантина."""

    @abstractmethod
    def get_quarantine_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает метаданные записи карантина или None."""

    @abstractmethod
    def list_quarantine_items(self, status_filter: str = "pending") -> List[Dict[str, Any]]:
        """Возвращает список записей карантина (pending|approved|rejected|all)."""

    @abstractmethod
    def delete_quarantine_item(self, item_id: str) -> bool:
        """Удаляет метаданные записи карантина."""

    @abstractmethod
    def close(self) -> None:
        """Закрывает соединение с хранилищем."""


class SQLiteStateStore(StateStore):
    """Потокобезопасное хранилище состояния и очереди задач на SQLite (WAL)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @property
    def backend_name(self) -> str:
        return "sqlite"

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=10000;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                    ON sessions (created_at DESC, session_id DESC);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    job_type TEXT NOT NULL DEFAULT 'single',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    payload_json TEXT NOT NULL,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs (status, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_jobs_session_id
                    ON jobs (session_id);

                CREATE TABLE IF NOT EXISTS quarantine (
                    item_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    exported_at TEXT,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_quarantine_status_submitted
                    ON quarantine (status, submitted_at DESC);
                """
            )

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        now_iso = _now_iso()
        created_at = str(data.get("created_at") or now_iso)
        status = str(data.get("status") or "unknown")
        job_id = data.get("job_id")
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, job_id, status, created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        job_id = excluded.job_id,
                        status = excluded.status,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at,
                        data_json = excluded.data_json
                    """,
                    (session_id, job_id, status, created_at, now_iso, payload),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["data_json"])

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT data_json FROM sessions
                ORDER BY created_at DESC, session_id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in rows:
            data = json.loads(row["data_json"])
            items.append({
                "session_id": data["session_id"],
                "job_id": data.get("job_id"),
                "source_filename": data.get("source_filename", ""),
                "created_at": data.get("created_at", ""),
                "status": data.get("status", "unknown"),
                "progress": data.get("progress", 0),
                "truncated": bool(data.get("truncated", False)),
                "finalized": bool(data.get("finalized", False)),
                "finalized_at": data.get("finalized_at"),
                "summary": data.get("summary", {}),
            })
        return items

    def list_session_headers(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, job_id, status, created_at, updated_at
                FROM sessions
                ORDER BY created_at DESC, session_id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                self._conn.execute("COMMIT")
                return cur.rowcount > 0
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "session_id": row["session_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
            "error": row["error"],
            "attempts": int(row["attempts"]),
        }

    def enqueue_job(
        self,
        job_id: str,
        session_id: str,
        job_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        now_iso = _now_iso()
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, session_id, job_type, status, created_at,
                        started_at, finished_at, payload_json, error, attempts
                    )
                    VALUES (?, ?, ?, 'queued', ?, NULL, NULL, ?, NULL, 0)
                    ON CONFLICT(job_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        job_type = excluded.job_type,
                        status = 'queued',
                        created_at = excluded.created_at,
                        started_at = NULL,
                        finished_at = NULL,
                        payload_json = excluded.payload_json,
                        error = NULL
                    """,
                    (job_id, session_id, job_type, now_iso, payload_json),
                )
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
                return self._row_to_job(row)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def claim_next_job(self) -> Optional[Dict[str, Any]]:
        now_iso = _now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC, rowid ASC
                    LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        started_at = ?,
                        attempts = attempts + 1,
                        error = NULL
                    WHERE job_id = ?
                    """,
                    (now_iso, row["job_id"]),
                )
                updated = self._conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (row["job_id"],),
                ).fetchone()
                self._conn.execute("COMMIT")
                return self._row_to_job(updated)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def get_job_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM jobs
                WHERE session_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        now_iso = _now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if finished:
                    self._conn.execute(
                        """
                        UPDATE jobs
                        SET status = ?, error = ?, finished_at = ?
                        WHERE job_id = ?
                        """,
                        (status, error, now_iso, job_id),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE jobs
                        SET status = ?, error = ?
                        WHERE job_id = ?
                        """,
                        (status, error, job_id),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def recover_stuck_jobs(self, max_attempts: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        max_att = max(1, int(max_attempts))
        now_iso = _now_iso()
        requeued: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE status = 'running'"
                ).fetchall()
                for row in rows:
                    eff_attempts = max(int(row["attempts"]), 1)
                    if eff_attempts < max_att:
                        err = "Задача возвращена в очередь после аварийного завершения процесса."
                        self._conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'queued',
                                started_at = NULL,
                                attempts = ?,
                                error = ?
                            WHERE job_id = ?
                            """,
                            (eff_attempts, err, row["job_id"]),
                        )
                        updated = self._conn.execute(
                            "SELECT * FROM jobs WHERE job_id = ?",
                            (row["job_id"],),
                        ).fetchone()
                        requeued.append(self._row_to_job(updated))
                    else:
                        err = (
                            f"Задача прервана аварийным завершением процесса "
                            f"(исчерпано попыток: {eff_attempts}/{max_att})."
                        )
                        self._conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'failed',
                                finished_at = ?,
                                attempts = ?,
                                error = ?
                            WHERE job_id = ?
                            """,
                            (now_iso, eff_attempts, err, row["job_id"]),
                        )
                        updated = self._conn.execute(
                            "SELECT * FROM jobs WHERE job_id = ?",
                            (row["job_id"],),
                        ).fetchone()
                        failed.append(self._row_to_job(updated))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {"requeued": requeued, "failed": failed}

    def count_active_jobs(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()
        return int(row["cnt"]) if row is not None else 0

    def save_quarantine_item(self, item_id: str, data: Dict[str, Any]) -> None:
        status = str(data.get("status") or "pending")
        submitted_at = str(data.get("submitted_at") or _now_iso())
        reviewed_at = data.get("reviewed_at")
        exported_at = data.get("exported_at")
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO quarantine (
                        item_id, status, submitted_at, reviewed_at, exported_at, data_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        status = excluded.status,
                        submitted_at = excluded.submitted_at,
                        reviewed_at = excluded.reviewed_at,
                        exported_at = excluded.exported_at,
                        data_json = excluded.data_json
                    """,
                    (item_id, status, submitted_at, reviewed_at, exported_at, payload),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_quarantine_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM quarantine WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["data_json"])

    def list_quarantine_items(self, status_filter: str = "pending") -> List[Dict[str, Any]]:
        with self._lock:
            if status_filter == "all":
                rows = self._conn.execute(
                    "SELECT data_json FROM quarantine ORDER BY submitted_at DESC, item_id DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT data_json FROM quarantine
                    WHERE status = ?
                    ORDER BY submitted_at DESC, item_id DESC
                    """,
                    (status_filter,),
                ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def delete_quarantine_item(self, item_id: str) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "DELETE FROM quarantine WHERE item_id = ?",
                    (item_id,),
                )
                self._conn.execute("COMMIT")
                return cur.rowcount > 0
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def migrate_from_disk(self, sessions_dir: Path, quarantine_dir: Path) -> Dict[str, int]:
        """Мягкая одноразовая миграция существующих session.json и item.json в SQLite."""
        migrated_sessions = 0
        migrated_quarantine = 0
        now_iso = _now_iso()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if sessions_dir.exists():
                    for entry in sessions_dir.iterdir():
                        meta_file = entry / "session.json"
                        if not meta_file.is_file():
                            continue
                        try:
                            data = json.loads(meta_file.read_text(encoding="utf-8"))
                            sid = str(data.get("session_id") or entry.name)
                            created_at = str(
                                data.get("created_at")
                                or datetime.fromtimestamp(meta_file.stat().st_mtime).isoformat(timespec="seconds")
                            )
                            cur = self._conn.execute(
                                """
                                INSERT OR IGNORE INTO sessions
                                    (session_id, job_id, status, created_at, updated_at, data_json)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    sid,
                                    data.get("job_id"),
                                    str(data.get("status") or "unknown"),
                                    created_at,
                                    now_iso,
                                    json.dumps(data, ensure_ascii=False),
                                ),
                            )
                            migrated_sessions += cur.rowcount
                        except Exception:
                            continue

                for bucket in ("pending", "approved", "rejected"):
                    bucket_dir = quarantine_dir / bucket
                    if not bucket_dir.exists():
                        continue
                    for entry in bucket_dir.iterdir():
                        meta_file = entry / "item.json"
                        if not meta_file.is_file():
                            continue
                        try:
                            data = json.loads(meta_file.read_text(encoding="utf-8"))
                            qid = str(data.get("item_id") or entry.name)
                            data["status"] = bucket
                            submitted_at = str(data.get("submitted_at") or now_iso)
                            cur = self._conn.execute(
                                """
                                INSERT OR IGNORE INTO quarantine
                                    (item_id, status, submitted_at, reviewed_at, exported_at, data_json)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    qid,
                                    bucket,
                                    submitted_at,
                                    data.get("reviewed_at"),
                                    data.get("exported_at"),
                                    json.dumps(data, ensure_ascii=False),
                                ),
                            )
                            migrated_quarantine += cur.rowcount
                        except Exception:
                            continue

                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return {"sessions": migrated_sessions, "quarantine": migrated_quarantine}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
