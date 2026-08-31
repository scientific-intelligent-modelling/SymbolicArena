"""Claude 批处理的 SQLite WAL 控制面。

每次物理调用在进程启动前占用一次全局预算。控制面只保存紧凑状态；原始请求、
stdout、stderr 和结构化结果由调用器写入不可变审计文件。
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence


class StateContractError(RuntimeError):
    """状态转换、预算或任务契约不合法。"""


@dataclass(frozen=True)
class TaskSpec:
    evaluation_key: str
    logical_id: str
    task_type: str
    condition: str
    priority: int
    input_hash: str
    prompt_version: str
    schema_version: str
    dependencies: tuple[str, ...]

    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["dependencies"] = list(self.dependencies)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class AttemptLease:
    attempt_id: str
    evaluation_key: str
    attempt_number: int
    lease_expires_at: float


class TaskStateStore:
    """支持并发 worker、断点恢复和硬预算的任务状态库。"""

    def __init__(
        self,
        path: str | Path,
        *,
        attempt_cap: int = 23700,
        max_attempts_per_task: int = 3,
    ) -> None:
        if attempt_cap <= 0:
            raise StateContractError("attempt_cap 必须为正整数")
        if max_attempts_per_task <= 0:
            raise StateContractError("max_attempts_per_task 必须为正整数")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempt_cap = int(attempt_cap)
        self.max_attempts_per_task = int(max_attempts_per_task)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._write_transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    evaluation_key TEXT PRIMARY KEY,
                    logical_id TEXT NOT NULL UNIQUE,
                    task_type TEXT NOT NULL,
                    condition_name TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    input_hash TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL,
                    last_error_class TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_ready
                    ON tasks(state, condition_name, priority, logical_id);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    evaluation_key TEXT NOT NULL REFERENCES tasks(evaluation_key),
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reserved_at REAL NOT NULL,
                    lease_expires_at REAL NOT NULL,
                    finished_at REAL,
                    error_class TEXT,
                    retryable INTEGER,
                    UNIQUE(evaluation_key, attempt_number)
                );
                CREATE TABLE IF NOT EXISTS frozen_results (
                    evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
                    result_path TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    frozen_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_key TEXT,
                    attempt_id TEXT,
                    event_type TEXT NOT NULL,
                    event_at REAL NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )
            existing = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            desired = {
                "attempt_cap": str(self.attempt_cap),
                "max_attempts_per_task": str(self.max_attempts_per_task),
                "schema_version": "state.v1",
            }
            for key, value in desired.items():
                previous = existing.get(key)
                if previous is not None and previous != value:
                    raise StateContractError(
                        f"状态库参数漂移: {key} 原为 {previous!r}，当前请求 {value!r}"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                    (key, value),
                )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        event_at: float,
        evaluation_key: str | None = None,
        attempt_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO events(
                   evaluation_key, attempt_id, event_type, event_at, details_json
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                evaluation_key,
                attempt_id,
                event_type,
                float(event_at),
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            ),
        )

    def register_task(self, spec: TaskSpec, *, now: float | None = None) -> None:
        if not spec.evaluation_key or not spec.logical_id:
            raise StateContractError("evaluation_key 和 logical_id 不得为空")
        timestamp = time.time() if now is None else float(now)
        spec_json = spec.canonical_json()
        dependencies_json = json.dumps(list(spec.dependencies), ensure_ascii=False)
        with self._write_transaction() as connection:
            existing = connection.execute(
                "SELECT spec_json FROM tasks WHERE evaluation_key = ? OR logical_id = ?",
                (spec.evaluation_key, spec.logical_id),
            ).fetchone()
            if existing is not None:
                if existing["spec_json"] != spec_json:
                    raise StateContractError(
                        f"任务 {spec.logical_id!r} 已存在，但契约内容发生漂移"
                    )
                return
            connection.execute(
                """INSERT INTO tasks(
                       evaluation_key, logical_id, task_type, condition_name,
                       priority, input_hash, prompt_version, schema_version,
                       dependencies_json, spec_json, state, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    spec.evaluation_key,
                    spec.logical_id,
                    spec.task_type,
                    spec.condition,
                    int(spec.priority),
                    spec.input_hash,
                    spec.prompt_version,
                    spec.schema_version,
                    dependencies_json,
                    spec_json,
                    timestamp,
                    timestamp,
                ),
            )
            self._event(
                connection,
                event_type="task_registered",
                event_at=timestamp,
                evaluation_key=spec.evaluation_key,
                details={"logical_id": spec.logical_id},
            )

    def attempts_reserved(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()
        return int(row["count"])

    def task_state(self, evaluation_key: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
        if row is None:
            raise StateContractError(f"未知任务: {evaluation_key}")
        return str(row["state"])

    def _dependencies_frozen(
        self, connection: sqlite3.Connection, dependencies_json: str
    ) -> bool:
        dependencies = json.loads(dependencies_json)
        for dependency in dependencies:
            row = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key = ?", (dependency,)
            ).fetchone()
            if row is None or row["state"] != "frozen":
                return False
        return True

    def next_ready_key(self, *, allowed_conditions: Sequence[str]) -> str | None:
        if not allowed_conditions:
            return None
        placeholders = ",".join("?" for _ in allowed_conditions)
        query = f"""SELECT evaluation_key, dependencies_json
                    FROM tasks
                    WHERE state IN ('pending', 'retry_wait')
                      AND condition_name IN ({placeholders})
                    ORDER BY priority ASC, logical_id ASC"""
        with self._connect() as connection:
            rows = connection.execute(query, tuple(allowed_conditions)).fetchall()
            for row in rows:
                if self._dependencies_frozen(connection, row["dependencies_json"]):
                    return str(row["evaluation_key"])
        return None

    def reserve_attempt(
        self,
        evaluation_key: str,
        *,
        now: float | None = None,
        lease_seconds: float = 1800.0,
    ) -> AttemptLease:
        timestamp = time.time() if now is None else float(now)
        if lease_seconds <= 0:
            raise StateContractError("lease_seconds 必须为正数")
        lease_expires_at = timestamp + float(lease_seconds)
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
            if row is None:
                raise StateContractError(f"未知任务: {evaluation_key}")
            if row["state"] not in {"pending", "retry_wait"}:
                raise StateContractError(
                    f"任务 {evaluation_key} 当前状态 {row['state']!r}，不可预占"
                )
            if not self._dependencies_frozen(connection, row["dependencies_json"]):
                raise StateContractError(f"任务 {evaluation_key} 的依赖尚未全部冻结")
            count = int(row["attempt_count"])
            if count >= self.max_attempts_per_task:
                raise StateContractError(f"任务 {evaluation_key} 已达到单任务尝试上限")
            global_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            )
            if global_count >= self.attempt_cap:
                raise StateContractError(
                    f"全局尝试预算已耗尽: {global_count}/{self.attempt_cap}"
                )
            attempt_number = count + 1
            attempt_id = f"{evaluation_key}.a{attempt_number:02d}"
            connection.execute(
                """INSERT INTO attempts(
                       attempt_id, evaluation_key, attempt_number, status,
                       reserved_at, lease_expires_at
                   ) VALUES (?, ?, ?, 'running', ?, ?)""",
                (
                    attempt_id,
                    evaluation_key,
                    attempt_number,
                    timestamp,
                    lease_expires_at,
                ),
            )
            connection.execute(
                """UPDATE tasks
                   SET state='running', attempt_count=?, lease_expires_at=?, updated_at=?
                   WHERE evaluation_key=?""",
                (attempt_number, lease_expires_at, timestamp, evaluation_key),
            )
            self._event(
                connection,
                event_type="attempt_reserved",
                event_at=timestamp,
                evaluation_key=evaluation_key,
                attempt_id=attempt_id,
                details={"attempt_number": attempt_number},
            )
        return AttemptLease(
            attempt_id=attempt_id,
            evaluation_key=evaluation_key,
            attempt_number=attempt_number,
            lease_expires_at=lease_expires_at,
        )

    def finish_failure(
        self,
        attempt_id: str,
        *,
        error_class: str,
        retryable: bool,
        now: float | None = None,
    ) -> str:
        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None or attempt["status"] != "running":
                raise StateContractError(f"attempt {attempt_id!r} 不处于 running 状态")
            task = connection.execute(
                "SELECT * FROM tasks WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone()
            global_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            )
            can_retry = (
                bool(retryable)
                and int(task["attempt_count"]) < self.max_attempts_per_task
                and global_count < self.attempt_cap
            )
            next_state = "retry_wait" if can_retry else "exhausted"
            connection.execute(
                """UPDATE attempts
                   SET status='failed', finished_at=?, error_class=?, retryable=?
                   WHERE attempt_id=?""",
                (timestamp, error_class, int(bool(retryable)), attempt_id),
            )
            connection.execute(
                """UPDATE tasks
                   SET state=?, lease_expires_at=NULL, last_error_class=?, updated_at=?
                   WHERE evaluation_key=?""",
                (next_state, error_class, timestamp, attempt["evaluation_key"]),
            )
            self._event(
                connection,
                event_type="attempt_failed",
                event_at=timestamp,
                evaluation_key=attempt["evaluation_key"],
                attempt_id=attempt_id,
                details={
                    "error_class": error_class,
                    "retryable": bool(retryable),
                    "next_state": next_state,
                },
            )
        return next_state

    def freeze_result(
        self,
        attempt_id: str,
        *,
        result_path: str,
        result_sha256: str,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        if not result_path or not result_sha256:
            raise StateContractError("冻结结果必须包含路径和 SHA-256")
        with self._write_transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None or attempt["status"] != "running":
                raise StateContractError(f"attempt {attempt_id!r} 不处于 running 状态")
            existing = connection.execute(
                "SELECT attempt_id FROM frozen_results WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone()
            if existing is not None:
                raise StateContractError(
                    f"任务 {attempt['evaluation_key']} 已由 {existing['attempt_id']} 冻结"
                )
            connection.execute(
                """INSERT INTO frozen_results(
                       evaluation_key, attempt_id, result_path, result_sha256, frozen_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    attempt["evaluation_key"],
                    attempt_id,
                    result_path,
                    result_sha256,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE attempts SET status='accepted', finished_at=? WHERE attempt_id=?",
                (timestamp, attempt_id),
            )
            connection.execute(
                """UPDATE tasks
                   SET state='frozen', lease_expires_at=NULL,
                       last_error_class=NULL, updated_at=?
                   WHERE evaluation_key=?""",
                (timestamp, attempt["evaluation_key"]),
            )
            self._event(
                connection,
                event_type="result_frozen",
                event_at=timestamp,
                evaluation_key=attempt["evaluation_key"],
                attempt_id=attempt_id,
                details={"result_path": result_path, "result_sha256": result_sha256},
            )

    def frozen_result(self, evaluation_key: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT result_path, result_sha256, attempt_id
                   FROM frozen_results WHERE evaluation_key = ?""",
                (evaluation_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "result_path": str(row["result_path"]),
            "result_sha256": str(row["result_sha256"]),
            "attempt_id": str(row["attempt_id"]),
        }

    def recover_expired_leases(self, *, now: float | None = None) -> list[str]:
        timestamp = time.time() if now is None else float(now)
        recovered: list[str] = []
        with self._write_transaction() as connection:
            rows = connection.execute(
                """SELECT a.*, t.attempt_count
                   FROM attempts a
                   JOIN tasks t ON t.evaluation_key = a.evaluation_key
                   WHERE a.status='running' AND a.lease_expires_at < ?""",
                (timestamp,),
            ).fetchall()
            global_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            )
            for row in rows:
                can_retry = (
                    int(row["attempt_count"]) < self.max_attempts_per_task
                    and global_count < self.attempt_cap
                )
                next_state = "retry_wait" if can_retry else "exhausted"
                connection.execute(
                    """UPDATE attempts
                       SET status='failed', finished_at=?, error_class='lease_expired', retryable=1
                       WHERE attempt_id=?""",
                    (timestamp, row["attempt_id"]),
                )
                connection.execute(
                    """UPDATE tasks
                       SET state=?, lease_expires_at=NULL,
                           last_error_class='lease_expired', updated_at=?
                       WHERE evaluation_key=?""",
                    (next_state, timestamp, row["evaluation_key"]),
                )
                self._event(
                    connection,
                    event_type="lease_expired",
                    event_at=timestamp,
                    evaluation_key=row["evaluation_key"],
                    attempt_id=row["attempt_id"],
                    details={"next_state": next_state},
                )
                recovered.append(str(row["attempt_id"]))
        return recovered

