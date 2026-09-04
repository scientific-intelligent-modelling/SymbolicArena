"""Claude 批处理的 SQLite WAL 控制面。

每次物理调用在进程启动前占用一次全局预算。控制面只保存紧凑状态；原始请求、
stdout、stderr 和结构化结果由调用器写入不可变审计文件。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
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


@dataclass(frozen=True)
class PredecessorAttemptManifest:
    path: str
    sha256: str
    attempt_count: int


@dataclass(frozen=True)
class TaskSupersession:
    predecessor_evaluation_key: str
    successor: TaskSpec
    identity: str
    reason: str
    predecessor_plan_sha256: str
    successor_plan_sha256: str


@dataclass(frozen=True)
class TaskRetirement:
    exhausted_evaluation_key: str
    replacement: TaskSpec
    identity: str
    reason: str
    exhausted_plan_sha256: str
    replacement_plan_sha256: str


class TaskStateStore:
    """支持并发 worker、断点恢复和硬预算的任务状态库。"""

    def __init__(
        self,
        path: str | Path,
        *,
        attempt_cap: int = 23700,
        logical_task_cap: int = 15800,
        max_attempts_per_task: int = 3,
        attempt_offset: int = 0,
        predecessor_attempt_manifest: PredecessorAttemptManifest | None = None,
        allow_schema_upgrade: bool = False,
    ) -> None:
        if attempt_cap <= 0:
            raise StateContractError("attempt_cap 必须为正整数")
        if logical_task_cap <= 0:
            raise StateContractError("logical_task_cap 必须为正整数")
        if max_attempts_per_task <= 0:
            raise StateContractError("max_attempts_per_task 必须为正整数")
        if attempt_offset < 0:
            raise StateContractError("attempt_offset 不能为负数")
        if predecessor_attempt_manifest is not None:
            if not predecessor_attempt_manifest.path:
                raise StateContractError("predecessor_attempt_manifest.path 不得为空")
            if not predecessor_attempt_manifest.sha256:
                raise StateContractError("predecessor_attempt_manifest.sha256 不得为空")
            if predecessor_attempt_manifest.attempt_count < 0:
                raise StateContractError("predecessor_attempt_manifest.attempt_count 不能为负数")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempt_cap = int(attempt_cap)
        self.logical_task_cap = int(logical_task_cap)
        self.max_attempts_per_task = int(max_attempts_per_task)
        self.requested_attempt_offset = int(attempt_offset)
        self.predecessor_attempt_manifest = predecessor_attempt_manifest
        self.allow_schema_upgrade = bool(allow_schema_upgrade)
        self.predecessor_attempt_manifest_path = (
            predecessor_attempt_manifest.path if predecessor_attempt_manifest is not None else ""
        )
        self.predecessor_attempt_manifest_sha256 = (
            predecessor_attempt_manifest.sha256 if predecessor_attempt_manifest is not None else ""
        )
        self.predecessor_attempt_count = (
            int(predecessor_attempt_manifest.attempt_count)
            if predecessor_attempt_manifest is not None
            else 0
        )
        self.attempt_offset = self.predecessor_attempt_count
        self._write_lock = threading.RLock()
        self._configure_database()
        self._initialize()

    def _configure_database(self) -> None:
        """只在 store 初始化时设置持久 journal 模式，避免并发连接反复争锁。"""
        with sqlite3.connect(self.path, timeout=60.0) as connection:
            connection.execute("PRAGMA busy_timeout = 60000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 60000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
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
            ddl_statements = (
                """CREATE TABLE IF NOT EXISTS meta (
                       key TEXT PRIMARY KEY,
                       value TEXT NOT NULL
                   )""",
                """CREATE TABLE IF NOT EXISTS tasks (
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
                   )""",
                """CREATE INDEX IF NOT EXISTS idx_tasks_ready
                   ON tasks(state, condition_name, priority, logical_id)""",
                """CREATE TABLE IF NOT EXISTS attempts (
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
                   )""",
                """CREATE TABLE IF NOT EXISTS frozen_results (
                       evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                       attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
                       result_path TEXT NOT NULL,
                       result_sha256 TEXT NOT NULL,
                       frozen_at REAL NOT NULL
                   )""",
                """CREATE TABLE IF NOT EXISTS non_applicable_results (
                       evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                       reason TEXT NOT NULL,
                       evidence_path TEXT NOT NULL,
                       evidence_sha256 TEXT NOT NULL,
                       marked_at REAL NOT NULL
                   )""",
                """CREATE TABLE IF NOT EXISTS task_supersessions (
                       predecessor_evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                       successor_evaluation_key TEXT NOT NULL UNIQUE REFERENCES tasks(evaluation_key),
                       predecessor_logical_id TEXT NOT NULL UNIQUE,
                       successor_logical_id TEXT NOT NULL UNIQUE,
                       identity TEXT NOT NULL UNIQUE,
                       reason TEXT NOT NULL,
                       predecessor_plan_sha256 TEXT NOT NULL,
                       successor_plan_sha256 TEXT NOT NULL,
                       superseded_at REAL NOT NULL
                   )""",
                """CREATE TABLE IF NOT EXISTS task_retirements (
                       exhausted_evaluation_key TEXT PRIMARY KEY REFERENCES tasks(evaluation_key),
                       replacement_evaluation_key TEXT NOT NULL REFERENCES tasks(evaluation_key),
                       exhausted_logical_id TEXT NOT NULL UNIQUE,
                       replacement_logical_id TEXT NOT NULL,
                       identity TEXT NOT NULL UNIQUE,
                       reason TEXT NOT NULL,
                       exhausted_plan_sha256 TEXT NOT NULL,
                       replacement_plan_sha256 TEXT NOT NULL,
                       exhausted_attempt_count INTEGER NOT NULL,
                       exhausted_last_error_class TEXT,
                       retired_at REAL NOT NULL
                   )""",
                """CREATE TABLE IF NOT EXISTS events (
                       event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                       evaluation_key TEXT,
                       attempt_id TEXT,
                       event_type TEXT NOT NULL,
                       event_at REAL NOT NULL,
                       details_json TEXT NOT NULL
                   )""",
            )
            existing_tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "meta" in existing_tables:
                existing = dict(connection.execute("SELECT key, value FROM meta").fetchall())
                existing_schema_version = existing.get("schema_version")
                if existing_schema_version != "state.v3" and not self.allow_schema_upgrade:
                    raise StateContractError(
                        "检测到已有旧版状态库；请显式传入 allow_schema_upgrade=True 后再升级到 state.v3"
                    )
                if existing_schema_version != "state.v3":
                    self._assert_no_running_for_schema_upgrade(connection)
                elif (
                    "task_supersessions" in existing_tables
                    and not self._task_supersession_identity_unique_enforced(connection)
                ):
                    if not self.allow_schema_upgrade:
                        raise StateContractError(
                            "检测到 state.v3 缺少全局唯一 supersession identity；"
                            "请显式传入 allow_schema_upgrade=True 后再修复 schema"
                        )
                    self._assert_no_running_for_schema_upgrade(connection)
                    self._assert_no_duplicate_supersession_identities(connection)
                    connection.execute(
                        """CREATE UNIQUE INDEX IF NOT EXISTS
                               idx_task_supersessions_identity_unique
                           ON task_supersessions(identity)"""
                    )
            else:
                existing = {}
            for statement in ddl_statements:
                connection.execute(statement)
            if "meta" in existing_tables:
                existing = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            existing_attempt_offset = int(existing.get("attempt_offset", "0"))
            legacy_needs_manifest = (
                existing_attempt_offset > 0 and "predecessor_attempt_count" not in existing
            )
            if legacy_needs_manifest and self.predecessor_attempt_manifest is None:
                raise StateContractError(
                    "检测到 legacy attempt_offset 但缺少 predecessor_attempt_manifest；"
                    "请显式提供旧 attempt 清单完成迁移，避免静默重置预算"
                )
            if (
                self.predecessor_attempt_manifest is None
                and int(existing.get("predecessor_attempt_count", "0")) > 0
            ):
                raise StateContractError(
                    "状态库已冻结 predecessor_attempt_manifest 元数据；"
                    "恢复同一 DB 时必须继续传入同一 manifest，避免静默重置预算"
                )
            if self.predecessor_attempt_manifest is None and self.requested_attempt_offset > 0:
                raise StateContractError(
                    "非零 attempt_offset 已禁用；请改用 predecessor_attempt_manifest 提供旧 attempt 清单"
                )
            if self.predecessor_attempt_manifest is not None and self.requested_attempt_offset not in {
                0,
                self.predecessor_attempt_count,
            }:
                raise StateContractError(
                    "attempt_offset 与 predecessor_attempt_manifest.attempt_count 不一致"
                )
            self.attempt_offset = (
                self.predecessor_attempt_count
                if self.predecessor_attempt_manifest is not None
                else existing_attempt_offset
            )
            desired = {
                "attempt_cap": str(self.attempt_cap),
                "logical_task_cap": str(self.logical_task_cap),
                "max_attempts_per_task": str(self.max_attempts_per_task),
                "attempt_offset": str(self.attempt_offset),
                "predecessor_attempt_manifest_path": self.predecessor_attempt_manifest_path,
                "predecessor_attempt_manifest_sha256": self.predecessor_attempt_manifest_sha256,
                "predecessor_attempt_count": str(self.predecessor_attempt_count),
                "schema_version": "state.v3",
            }
            for key, value in desired.items():
                previous = existing.get(key)
                if key == "schema_version" and previous in {"state.v1", "state.v2"}:
                    connection.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
                    continue
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

    @staticmethod
    def _validate_task_spec(spec: TaskSpec) -> None:
        if not spec.evaluation_key or not spec.logical_id:
            raise StateContractError("evaluation_key 和 logical_id 不得为空")
        if spec.condition not in {"clean", "noise001", "noise005"}:
            raise StateContractError(f"未知任务条件: {spec.condition!r}")

    @staticmethod
    def _task_identity_fields(
        *,
        task_type: str,
        condition: str,
        priority: int,
        dependencies_json: str,
    ) -> tuple[str, str, int, str]:
        return (
            str(task_type),
            str(condition),
            int(priority),
            str(dependencies_json),
        )

    @classmethod
    def _task_identity_from_spec(cls, spec: TaskSpec) -> tuple[str, str, int, str]:
        return cls._task_identity_fields(
            task_type=spec.task_type,
            condition=spec.condition,
            priority=spec.priority,
            dependencies_json=json.dumps(list(spec.dependencies), ensure_ascii=False),
        )

    @classmethod
    def _task_identity_from_row(cls, row: sqlite3.Row) -> tuple[str, str, int, str]:
        return cls._task_identity_fields(
            task_type=str(row["task_type"]),
            condition=str(row["condition_name"]),
            priority=int(row["priority"]),
            dependencies_json=str(row["dependencies_json"]),
        )

    @staticmethod
    def _assert_no_running_for_schema_upgrade(connection: sqlite3.Connection) -> None:
        running_tasks = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE state='running'"
            ).fetchone()["count"]
        )
        running_attempts = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM attempts WHERE status='running'"
            ).fetchone()["count"]
        )
        if running_tasks or running_attempts:
            raise StateContractError(
                "状态库仍有 running tasks/attempts，禁止执行 schema upgrade"
            )

    @staticmethod
    def _task_supersession_identity_unique_enforced(connection: sqlite3.Connection) -> bool:
        indexes = connection.execute("PRAGMA index_list('task_supersessions')").fetchall()
        for index in indexes:
            if int(index["unique"]) != 1:
                continue
            columns = connection.execute(
                f"PRAGMA index_info('{str(index['name'])}')"
            ).fetchall()
            if len(columns) != 1:
                continue
            if str(columns[0]["name"]) == "identity":
                return True
        return False

    @staticmethod
    def _assert_no_duplicate_supersession_identities(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """SELECT identity
               FROM task_supersessions
               GROUP BY identity
               HAVING COUNT(*) > 1
               LIMIT 1"""
        ).fetchone()
        if row is not None:
            raise StateContractError(
                f"task_supersessions.identity 存在重复值，禁止升级: {row['identity']!r}"
            )

    @staticmethod
    def _valid_superseded_predecessors(connection: sqlite3.Connection) -> set[str]:
        supersession_rows = connection.execute(
            """SELECT ts.predecessor_evaluation_key
               FROM task_supersessions ts
               JOIN tasks predecessor
                 ON predecessor.evaluation_key = ts.predecessor_evaluation_key
                AND predecessor.logical_id = ts.predecessor_logical_id
               JOIN frozen_results predecessor_frozen
                 ON predecessor_frozen.evaluation_key = ts.predecessor_evaluation_key
               JOIN tasks successor
                 ON successor.evaluation_key = ts.successor_evaluation_key
                AND successor.logical_id = ts.successor_logical_id
               WHERE predecessor.state='superseded'"""
        ).fetchall()
        retirement_rows = connection.execute(
            """SELECT tr.exhausted_evaluation_key AS predecessor_evaluation_key
               FROM task_retirements tr
               JOIN tasks exhausted
                 ON exhausted.evaluation_key = tr.exhausted_evaluation_key
                AND exhausted.logical_id = tr.exhausted_logical_id
                AND exhausted.attempt_count = tr.exhausted_attempt_count
               JOIN tasks replacement
                 ON replacement.evaluation_key = tr.replacement_evaluation_key
                AND replacement.logical_id = tr.replacement_logical_id
               JOIN frozen_results replacement_frozen
                 ON replacement_frozen.evaluation_key = tr.replacement_evaluation_key
               WHERE exhausted.state='superseded'
                 AND replacement.state='frozen'"""
        ).fetchall()
        return {
            str(row["predecessor_evaluation_key"])
            for row in (*supersession_rows, *retirement_rows)
        }

    @classmethod
    def _count_active_logical_tasks(cls, connection: sqlite3.Connection) -> int:
        rows = connection.execute(
            "SELECT evaluation_key, logical_id, state FROM tasks"
        ).fetchall()
        valid_superseded = cls._valid_superseded_predecessors(connection)
        active_revision_bases: set[str] = set()
        for row in rows:
            if str(row["state"]) == "superseded" and str(row["evaluation_key"]) in valid_superseded:
                continue
            active_revision_bases.add(cls._logical_revision_base(str(row["logical_id"])))
        return len(active_revision_bases)

    @classmethod
    def _is_terminal_for_gate(
        cls,
        connection: sqlite3.Connection,
        *,
        evaluation_key: str,
        state: str,
        valid_superseded: set[str] | None = None,
    ) -> bool:
        if state in {"frozen", "non_applicable"}:
            return True
        if state != "superseded":
            return False
        if valid_superseded is None:
            valid_superseded = cls._valid_superseded_predecessors(connection)
        return evaluation_key in valid_superseded

    @staticmethod
    def _load_existing_task_specs(
        connection: sqlite3.Connection,
        *,
        evaluation_keys: Sequence[str],
        logical_ids: Sequence[str],
    ) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
        known_by_evaluation: dict[str, dict[str, str]] = {}
        known_by_logical: dict[str, str] = {}
        for values, column in (
            (tuple(dict.fromkeys(evaluation_keys)), "evaluation_key"),
            (tuple(dict.fromkeys(logical_ids)), "logical_id"),
        ):
            if not values:
                continue
            for start in range(0, len(values), 500):
                chunk = values[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = (
                    "SELECT evaluation_key, logical_id, spec_json FROM tasks "
                    f"WHERE {column} IN ({placeholders})"
                )
                for row in connection.execute(query, chunk).fetchall():
                    evaluation_key = str(row["evaluation_key"])
                    logical_id = str(row["logical_id"])
                    known_by_evaluation[evaluation_key] = {
                        "logical_id": logical_id,
                        "spec_json": str(row["spec_json"]),
                    }
                    known_by_logical[logical_id] = evaluation_key
        return known_by_evaluation, known_by_logical

    def register_task(self, spec: TaskSpec, *, now: float | None = None) -> None:
        self.register_tasks((spec,), now=now)

    def register_tasks(self, specs: Sequence[TaskSpec], *, now: float | None = None) -> None:
        if not specs:
            return
        timestamp = time.time() if now is None else float(now)
        prepared_specs: list[tuple[TaskSpec, str, str]] = []
        for spec in specs:
            self._validate_task_spec(spec)
            prepared_specs.append(
                (
                    spec,
                    spec.canonical_json(),
                    json.dumps(list(spec.dependencies), ensure_ascii=False),
                )
            )
        with self._write_transaction() as connection:
            logical_count = self._active_logical_task_count(connection)
            revision_identities: dict[str, set[tuple[str, str, int, str]]] = {}
            for row in connection.execute(
                """SELECT logical_id, task_type, condition_name, priority,
                          dependencies_json
                   FROM tasks"""
            ).fetchall():
                base = self._logical_revision_base(str(row["logical_id"]))
                revision_identities.setdefault(base, set()).add(
                    self._task_identity_from_row(row)
                )
            known_by_evaluation, known_by_logical = self._load_existing_task_specs(
                connection,
                evaluation_keys=[spec.evaluation_key for spec, _, _ in prepared_specs],
                logical_ids=[spec.logical_id for spec, _, _ in prepared_specs],
            )
            pending_inserts: list[tuple[object, ...]] = []
            pending_events: list[tuple[str, float, dict[str, object]]] = []
            new_logical_count = 0
            new_revision_bases: set[str] = set()
            # 先在内存里完成整批去重、漂移和预算校验，确认无误后再一次性落库。
            for spec, spec_json, dependencies_json in prepared_specs:
                known_entry = known_by_evaluation.get(spec.evaluation_key)
                known_evaluation_key = known_by_logical.get(spec.logical_id)
                if known_entry is None and known_evaluation_key is None:
                    revision_base = self._logical_revision_base(spec.logical_id)
                    spec_identity = self._task_identity_from_spec(spec)
                    existing_identities = revision_identities.get(revision_base)
                    if existing_identities is not None:
                        if existing_identities != {spec_identity}:
                            raise StateContractError(
                                f"逻辑任务版本身份漂移: {revision_base!r}"
                            )
                    else:
                        if revision_base in new_revision_bases:
                            raise StateContractError(
                                f"同批注册了多个新逻辑任务版本: {revision_base!r}"
                            )
                        if logical_count + new_logical_count >= self.logical_task_cap:
                            raise StateContractError(
                                "逻辑任务预算已耗尽: "
                                f"{logical_count + new_logical_count}/{self.logical_task_cap}"
                            )
                        new_logical_count += 1
                        new_revision_bases.add(revision_base)
                        revision_identities[revision_base] = {spec_identity}
                    pending_inserts.append(
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
                        )
                    )
                    pending_events.append(
                        (
                            spec.evaluation_key,
                            timestamp,
                            {"logical_id": spec.logical_id},
                        )
                    )
                    known_by_evaluation[spec.evaluation_key] = {
                        "logical_id": spec.logical_id,
                        "spec_json": spec_json,
                    }
                    known_by_logical[spec.logical_id] = spec.evaluation_key
                    continue
                if known_entry is None or known_evaluation_key != spec.evaluation_key:
                    raise StateContractError(
                        "任务唯一标识冲突: "
                        f"evaluation_key={spec.evaluation_key!r}, logical_id={spec.logical_id!r}"
                    )
                if (
                    known_entry["logical_id"] != spec.logical_id
                    or known_entry["spec_json"] != spec_json
                ):
                    raise StateContractError(
                        f"任务 {spec.logical_id!r} 已存在，但契约内容发生漂移"
                    )
            if pending_inserts:
                connection.executemany(
                    """INSERT INTO tasks(
                           evaluation_key, logical_id, task_type, condition_name,
                           priority, input_hash, prompt_version, schema_version,
                           dependencies_json, spec_json, state, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    pending_inserts,
                )
                for evaluation_key, event_at, details in pending_events:
                    self._event(
                        connection,
                        event_type="task_registered",
                        event_at=event_at,
                        evaluation_key=evaluation_key,
                        details=details,
                    )

    @classmethod
    def _active_logical_task_count(cls, connection: sqlite3.Connection) -> int:
        return cls._count_active_logical_tasks(connection)

    @staticmethod
    def _fetch_tasks_by_evaluation_keys(
        connection: sqlite3.Connection,
        evaluation_keys: Sequence[str],
    ) -> dict[str, sqlite3.Row]:
        rows_by_key: dict[str, sqlite3.Row] = {}
        values = tuple(dict.fromkeys(str(item) for item in evaluation_keys if str(item)))
        for start in range(0, len(values), 500):
            chunk = values[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT * FROM tasks WHERE evaluation_key IN ({placeholders})"
            for row in connection.execute(query, chunk).fetchall():
                rows_by_key[str(row["evaluation_key"])] = row
        return rows_by_key

    @staticmethod
    def _fetch_frozen_bindings(
        connection: sqlite3.Connection,
        evaluation_keys: Sequence[str],
    ) -> set[str]:
        frozen_keys: set[str] = set()
        values = tuple(dict.fromkeys(str(item) for item in evaluation_keys if str(item)))
        for start in range(0, len(values), 500):
            chunk = values[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT evaluation_key FROM frozen_results "
                f"WHERE evaluation_key IN ({placeholders})"
            )
            for row in connection.execute(query, chunk).fetchall():
                frozen_keys.add(str(row["evaluation_key"]))
        return frozen_keys

    @staticmethod
    def _fetch_existing_supersessions(
        connection: sqlite3.Connection,
        *,
        predecessor_keys: Sequence[str],
        successor_keys: Sequence[str],
        identities: Sequence[str],
    ) -> tuple[dict[str, sqlite3.Row], dict[str, sqlite3.Row], dict[str, sqlite3.Row]]:
        by_predecessor: dict[str, sqlite3.Row] = {}
        by_successor: dict[str, sqlite3.Row] = {}
        by_identity: dict[str, sqlite3.Row] = {}
        for values, column in (
            (tuple(dict.fromkeys(str(item) for item in predecessor_keys if str(item))), "predecessor_evaluation_key"),
            (tuple(dict.fromkeys(str(item) for item in successor_keys if str(item))), "successor_evaluation_key"),
            (tuple(dict.fromkeys(str(item) for item in identities if str(item))), "identity"),
        ):
            for start in range(0, len(values), 500):
                chunk = values[start : start + 500]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT * FROM task_supersessions WHERE {column} IN ({placeholders})"
                for row in connection.execute(query, chunk).fetchall():
                    by_predecessor[str(row["predecessor_evaluation_key"])] = row
                    by_successor[str(row["successor_evaluation_key"])] = row
                    by_identity[str(row["identity"])] = row
        return by_predecessor, by_successor, by_identity

    @staticmethod
    def _has_running_attempt(connection: sqlite3.Connection, evaluation_key: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM attempts WHERE evaluation_key = ? AND status = 'running' LIMIT 1",
            (evaluation_key,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _has_active_dependents(connection: sqlite3.Connection, evaluation_key: str) -> bool:
        valid_superseded = TaskStateStore._valid_superseded_predecessors(connection)
        rows = connection.execute(
            """SELECT evaluation_key, state, dependencies_json
               FROM tasks
               WHERE evaluation_key != ?""",
            (evaluation_key,),
        ).fetchall()
        for row in rows:
            dependencies = json.loads(str(row["dependencies_json"]))
            state = str(row["state"])
            dependent_key = str(row["evaluation_key"])
            if evaluation_key not in dependencies:
                continue
            if state in {"frozen", "non_applicable"}:
                continue
            if state == "superseded" and dependent_key in valid_superseded:
                continue
            return True
        return False

    def register_supersession(
        self,
        predecessor_evaluation_key: str,
        successor: TaskSpec,
        *,
        identity: str,
        reason: str,
        predecessor_plan_sha256: str,
        successor_plan_sha256: str,
        now: float | None = None,
    ) -> None:
        self.register_supersession_batch(
            (
                TaskSupersession(
                    predecessor_evaluation_key=predecessor_evaluation_key,
                    successor=successor,
                    identity=identity,
                    reason=reason,
                    predecessor_plan_sha256=predecessor_plan_sha256,
                    successor_plan_sha256=successor_plan_sha256,
                ),
            ),
            now=now,
        )

    def register_supersession_batch(
        self,
        supersessions: Sequence[TaskSupersession],
        *,
        now: float | None = None,
    ) -> None:
        self._register_supersession_batch(
            supersessions,
            now=now,
            allow_proven_dependency_rebinding=False,
        )

    def register_dependency_rebinding_supersession(
        self,
        predecessor_evaluation_key: str,
        successor: TaskSpec,
        *,
        identity: str,
        reason: str,
        predecessor_plan_sha256: str,
        successor_plan_sha256: str,
        now: float | None = None,
    ) -> None:
        """仅允许把依赖按既有 supersession 证明逐位置重绑定。"""

        self.register_dependency_rebinding_supersession_batch(
            (
                TaskSupersession(
                    predecessor_evaluation_key=predecessor_evaluation_key,
                    successor=successor,
                    identity=identity,
                    reason=reason,
                    predecessor_plan_sha256=predecessor_plan_sha256,
                    successor_plan_sha256=successor_plan_sha256,
                ),
            ),
            now=now,
        )

    def register_dependency_rebinding_supersession_batch(
        self,
        supersessions: Sequence[TaskSupersession],
        *,
        now: float | None = None,
    ) -> None:
        self._register_supersession_batch(
            supersessions,
            now=now,
            allow_proven_dependency_rebinding=True,
        )

    @staticmethod
    def _validate_proven_dependency_rebinding(
        connection: sqlite3.Connection,
        *,
        predecessor_key: str,
        predecessor_dependencies_json: str,
        successor_dependencies: Sequence[str],
    ) -> None:
        predecessor_dependencies = tuple(json.loads(predecessor_dependencies_json))
        successor_dependencies = tuple(str(item) for item in successor_dependencies)
        if len(predecessor_dependencies) != len(successor_dependencies):
            raise StateContractError(
                f"任务 {predecessor_key!r} 的 dependency rebinding 长度发生变化"
            )
        for old_dependency, new_dependency in zip(
            predecessor_dependencies, successor_dependencies, strict=True
        ):
            if (
                old_dependency != new_dependency
                and new_dependency in predecessor_dependencies
            ):
                raise StateContractError(
                    f"任务 {predecessor_key!r} 的 dependency rebinding 顺序发生变化"
                )
        valid_superseded = TaskStateStore._valid_superseded_predecessors(connection)
        for index, (old_dependency, new_dependency) in enumerate(
            zip(predecessor_dependencies, successor_dependencies, strict=True)
        ):
            if old_dependency == new_dependency:
                continue
            mapping = connection.execute(
                """SELECT successor_evaluation_key
                   FROM task_supersessions
                   WHERE predecessor_evaluation_key=?""",
                (old_dependency,),
            ).fetchone()
            if (
                mapping is None
                or str(mapping["successor_evaluation_key"]) != new_dependency
                or old_dependency not in valid_superseded
            ):
                raise StateContractError(
                    f"任务 {predecessor_key!r} 的 dependency[{index}] "
                    f"{old_dependency!r}->{new_dependency!r} 未证明"
                )
            replacement = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key=?",
                (new_dependency,),
            ).fetchone()
            replacement_state = None if replacement is None else str(replacement["state"])
            if replacement_state not in {"frozen", "non_applicable"}:
                raise StateContractError(
                    f"dependency replacement {new_dependency!r} 未达到 frozen/non_applicable 终态"
                )
            binding_table = (
                "frozen_results"
                if replacement_state == "frozen"
                else "non_applicable_results"
            )
            binding = connection.execute(
                f"SELECT 1 FROM {binding_table} WHERE evaluation_key=?",
                (new_dependency,),
            ).fetchone()
            if binding is None:
                raise StateContractError(
                    f"dependency replacement {new_dependency!r} 终态缺少审计绑定"
                )

    def _register_supersession_batch(
        self,
        supersessions: Sequence[TaskSupersession],
        *,
        now: float | None,
        allow_proven_dependency_rebinding: bool,
    ) -> None:
        if not supersessions:
            return
        timestamp = time.time() if now is None else float(now)
        prepared: list[tuple[TaskSupersession, str, str]] = []
        seen_predecessors: set[str] = set()
        seen_successor_evaluations: set[str] = set()
        seen_successor_logicals: set[str] = set()
        seen_identities: set[str] = set()
        for item in supersessions:
            predecessor_key = str(item.predecessor_evaluation_key)
            if not predecessor_key:
                raise StateContractError("supersession predecessor 不得为空")
            self._validate_task_spec(item.successor)
            if not item.identity:
                raise StateContractError("supersession identity 不得为空")
            if not item.reason:
                raise StateContractError("supersession reason 不得为空")
            if not item.predecessor_plan_sha256 or not item.successor_plan_sha256:
                raise StateContractError("supersession plan sha256 不得为空")
            if predecessor_key in seen_predecessors:
                raise StateContractError(f"supersession predecessor 重复: {predecessor_key!r}")
            if item.successor.evaluation_key in seen_successor_evaluations:
                raise StateContractError(
                    f"supersession successor evaluation_key 重复: {item.successor.evaluation_key!r}"
                )
            if item.successor.logical_id in seen_successor_logicals:
                raise StateContractError(
                    f"supersession successor logical_id 重复: {item.successor.logical_id!r}"
                )
            if item.identity in seen_identities:
                raise StateContractError(f"supersession identity 重复: {item.identity!r}")
            seen_predecessors.add(predecessor_key)
            seen_successor_evaluations.add(item.successor.evaluation_key)
            seen_successor_logicals.add(item.successor.logical_id)
            seen_identities.add(item.identity)
            prepared.append(
                (
                    item,
                    item.successor.canonical_json(),
                    json.dumps(list(item.successor.dependencies), ensure_ascii=False),
                )
            )
        with self._write_transaction() as connection:
            predecessor_rows = self._fetch_tasks_by_evaluation_keys(
                connection,
                [item.predecessor_evaluation_key for item, _, _ in prepared],
            )
            frozen_bindings = self._fetch_frozen_bindings(
                connection,
                [item.predecessor_evaluation_key for item, _, _ in prepared],
            )
            known_by_evaluation, known_by_logical = self._load_existing_task_specs(
                connection,
                evaluation_keys=[item.successor.evaluation_key for item, _, _ in prepared],
                logical_ids=[item.successor.logical_id for item, _, _ in prepared],
            )
            (
                superseded_by_predecessor,
                superseded_by_successor,
                superseded_by_identity,
            ) = self._fetch_existing_supersessions(
                connection,
                predecessor_keys=[item.predecessor_evaluation_key for item, _, _ in prepared],
                successor_keys=[item.successor.evaluation_key for item, _, _ in prepared],
                identities=[item.identity for item, _, _ in prepared],
            )
            active_logical_count = self._active_logical_task_count(connection)
            pending_inserts: list[tuple[object, ...]] = []
            pending_registration_events: list[tuple[str, float, dict[str, object]]] = []
            pending_supersession_rows: list[tuple[object, ...]] = []
            pending_supersession_events: list[tuple[str, str, str, str, float]] = []
            newly_superseded = 0
            newly_inserted = 0
            for item, successor_spec_json, successor_dependencies_json in prepared:
                predecessor_key = str(item.predecessor_evaluation_key)
                successor = item.successor
                predecessor = predecessor_rows.get(predecessor_key)
                if predecessor is None:
                    raise StateContractError(f"supersession predecessor 不存在: {predecessor_key!r}")
                predecessor_state = str(predecessor["state"])
                existing_mapping = superseded_by_predecessor.get(predecessor_key)
                known_entry = known_by_evaluation.get(successor.evaluation_key)
                known_successor_key = known_by_logical.get(successor.logical_id)
                successor_mapping = superseded_by_successor.get(successor.evaluation_key)
                identity_mapping = superseded_by_identity.get(item.identity)
                predecessor_identity = self._task_identity_from_row(predecessor)
                successor_identity = self._task_identity_fields(
                    task_type=successor.task_type,
                    condition=successor.condition,
                    priority=successor.priority,
                    dependencies_json=successor_dependencies_json,
                )
                if predecessor_identity[:3] != successor_identity[:3]:
                    raise StateContractError(
                        f"supersession identity 不一致: {predecessor_key!r} -> {successor.evaluation_key!r}"
                    )
                if allow_proven_dependency_rebinding:
                    self._validate_proven_dependency_rebinding(
                        connection,
                        predecessor_key=predecessor_key,
                        predecessor_dependencies_json=str(predecessor["dependencies_json"]),
                        successor_dependencies=successor.dependencies,
                    )
                elif predecessor_identity != successor_identity:
                    raise StateContractError(
                        f"supersession identity 不一致: {predecessor_key!r} -> {successor.evaluation_key!r}"
                    )
                if predecessor_state == "superseded":
                    if existing_mapping is None:
                        raise StateContractError(
                            f"任务 {predecessor_key!r} 已是 superseded，但缺少 supersession 绑定"
                        )
                    if (
                        str(existing_mapping["predecessor_logical_id"]) != str(predecessor["logical_id"])
                        or str(existing_mapping["successor_logical_id"]) != successor.logical_id
                    ):
                        raise StateContractError(
                            f"任务 {predecessor_key!r} 的 supersession logical_id 发生漂移"
                        )
                    if str(existing_mapping["successor_evaluation_key"]) != successor.evaluation_key:
                        raise StateContractError(
                            f"任务 {predecessor_key!r} 的 supersession successor 发生漂移"
                        )
                    if (
                        str(existing_mapping["identity"]) != item.identity
                        or str(existing_mapping["reason"]) != item.reason
                        or str(existing_mapping["predecessor_plan_sha256"])
                        != item.predecessor_plan_sha256
                        or str(existing_mapping["successor_plan_sha256"])
                        != item.successor_plan_sha256
                    ):
                        raise StateContractError(
                            f"任务 {predecessor_key!r} 的 supersession manifest 发生漂移"
                        )
                    if (
                        identity_mapping is not None
                        and str(identity_mapping["predecessor_evaluation_key"]) != predecessor_key
                    ):
                        raise StateContractError(f"supersession identity 重复: {item.identity!r}")
                    if (
                        known_entry is None
                        or known_successor_key != successor.evaluation_key
                        or known_entry["spec_json"] != successor_spec_json
                    ):
                        raise StateContractError(
                            f"任务 {successor.logical_id!r} 已存在，但 supersession successor 契约漂移"
                        )
                    continue
                if predecessor_state != "frozen":
                    raise StateContractError(
                        f"任务 {predecessor_key!r} 当前状态 {predecessor_state!r}，不可 supersede"
                    )
                if predecessor_key not in frozen_bindings:
                    raise StateContractError(
                        f"任务 {predecessor_key!r} 缺少 frozen binding，禁止 supersede"
                    )
                if self._has_running_attempt(connection, predecessor_key):
                    raise StateContractError(
                        f"任务 {predecessor_key!r} 存在 running attempt，禁止 supersede"
                    )
                if self._has_active_dependents(connection, predecessor_key):
                    raise StateContractError(
                        f"任务 {predecessor_key!r} 仍被 active dependents 引用，禁止 supersede"
                    )
                if existing_mapping is not None:
                    raise StateContractError(
                        f"任务 {predecessor_key!r} 已绑定 supersession successor，当前请求漂移"
                    )
                if successor_mapping is not None:
                    raise StateContractError(
                        f"任务 {successor.evaluation_key!r} 已绑定其他 supersession predecessor"
                    )
                if identity_mapping is not None:
                    raise StateContractError(f"supersession identity 重复: {item.identity!r}")
                if known_entry is None and known_successor_key is None:
                    pending_inserts.append(
                        (
                            successor.evaluation_key,
                            successor.logical_id,
                            successor.task_type,
                            successor.condition,
                            int(successor.priority),
                            successor.input_hash,
                            successor.prompt_version,
                            successor.schema_version,
                            successor_dependencies_json,
                            successor_spec_json,
                            timestamp,
                            timestamp,
                        )
                    )
                    pending_registration_events.append(
                        (
                            successor.evaluation_key,
                            timestamp,
                            {
                                "logical_id": successor.logical_id,
                                "supersession_predecessor": predecessor_key,
                            },
                        )
                    )
                    known_by_evaluation[successor.evaluation_key] = {
                        "logical_id": successor.logical_id,
                        "spec_json": successor_spec_json,
                    }
                    known_by_logical[successor.logical_id] = successor.evaluation_key
                    newly_inserted += 1
                elif known_entry is None or known_successor_key != successor.evaluation_key:
                    raise StateContractError(
                        "任务唯一标识冲突: "
                        f"evaluation_key={successor.evaluation_key!r}, logical_id={successor.logical_id!r}"
                    )
                elif known_entry["spec_json"] != successor_spec_json:
                    raise StateContractError(
                        f"任务 {successor.logical_id!r} 已存在，但 supersession successor 契约漂移"
                    )
                else:
                    raise StateContractError(
                        f"任务 {successor.logical_id!r} 已存在，不能把现有任务重新声明为新 supersession successor"
                    )
                pending_supersession_rows.append(
                    (
                        predecessor_key,
                        successor.evaluation_key,
                        str(predecessor["logical_id"]),
                        successor.logical_id,
                        item.identity,
                        item.reason,
                        item.predecessor_plan_sha256,
                        item.successor_plan_sha256,
                        timestamp,
                    )
                )
                pending_supersession_events.append(
                    (predecessor_key, successor.evaluation_key, item.identity, item.reason, timestamp)
                )
                newly_superseded += 1
            next_active_logical_count = active_logical_count - newly_superseded + newly_inserted
            if next_active_logical_count > self.logical_task_cap:
                raise StateContractError(
                    f"逻辑任务预算已耗尽: {next_active_logical_count}/{self.logical_task_cap}"
                )
            if pending_inserts:
                connection.executemany(
                    """INSERT INTO tasks(
                           evaluation_key, logical_id, task_type, condition_name,
                           priority, input_hash, prompt_version, schema_version,
                           dependencies_json, spec_json, state, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    pending_inserts,
                )
                for evaluation_key, event_at, details in pending_registration_events:
                    self._event(
                        connection,
                        event_type="task_registered",
                        event_at=event_at,
                        evaluation_key=evaluation_key,
                        details=details,
                    )
            if pending_supersession_rows:
                connection.executemany(
                    """INSERT INTO task_supersessions(
                           predecessor_evaluation_key, successor_evaluation_key,
                           predecessor_logical_id, successor_logical_id, identity, reason,
                           predecessor_plan_sha256, successor_plan_sha256, superseded_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    pending_supersession_rows,
                )
                connection.executemany(
                    """UPDATE tasks
                       SET state='superseded', lease_expires_at=NULL,
                           last_error_class=NULL, updated_at=?
                       WHERE evaluation_key=?""",
                    [(event_at, predecessor_key) for predecessor_key, _, _, _, event_at in pending_supersession_events],
                )
                for predecessor_key, successor_key, identity, reason, event_at in pending_supersession_events:
                    self._event(
                        connection,
                        event_type="task_superseded",
                        event_at=event_at,
                        evaluation_key=predecessor_key,
                        details={
                            "successor_evaluation_key": successor_key,
                            "identity": identity,
                            "reason": reason,
                        },
                    )

    @staticmethod
    def _logical_revision_base(logical_id: str) -> str:
        return re.sub(r"::v\d+$", "", logical_id)

    def retire_exhausted_tasks(
        self,
        retirements: Sequence[TaskRetirement],
        *,
        now: float | None = None,
    ) -> None:
        """把已有同逻辑冻结替代项的旧 exhausted 版本审计化退休。"""

        if not retirements:
            return
        timestamp = time.time() if now is None else float(now)
        seen_exhausted: set[str] = set()
        seen_identities: set[str] = set()
        for item in retirements:
            exhausted_key = str(item.exhausted_evaluation_key)
            if not exhausted_key:
                raise StateContractError("retirement exhausted_evaluation_key 不得为空")
            self._validate_task_spec(item.replacement)
            if not item.identity or not item.reason:
                raise StateContractError("retirement identity 与 reason 不得为空")
            if not item.exhausted_plan_sha256 or not item.replacement_plan_sha256:
                raise StateContractError("retirement plan sha256 不得为空")
            if exhausted_key in seen_exhausted:
                raise StateContractError(f"retirement exhausted task 重复: {exhausted_key!r}")
            if item.identity in seen_identities:
                raise StateContractError(f"retirement identity 重复: {item.identity!r}")
            seen_exhausted.add(exhausted_key)
            seen_identities.add(item.identity)

        pending_rows: list[tuple[object, ...]] = []
        pending_events: list[tuple[str, str, str, str, float]] = []
        with self._write_transaction() as connection:
            for item in retirements:
                exhausted_key = str(item.exhausted_evaluation_key)
                exhausted = connection.execute(
                    "SELECT * FROM tasks WHERE evaluation_key=?",
                    (exhausted_key,),
                ).fetchone()
                if exhausted is None:
                    raise StateContractError(f"retirement exhausted task 不存在: {exhausted_key!r}")
                replacement = connection.execute(
                    "SELECT * FROM tasks WHERE evaluation_key=?",
                    (item.replacement.evaluation_key,),
                ).fetchone()
                if replacement is None:
                    raise StateContractError(
                        f"retirement replacement 不存在: {item.replacement.evaluation_key!r}"
                    )
                if str(replacement["logical_id"]) != item.replacement.logical_id:
                    raise StateContractError("retirement replacement logical_id 漂移")
                if str(replacement["spec_json"]) != item.replacement.canonical_json():
                    raise StateContractError("retirement replacement spec 漂移")

                existing = connection.execute(
                    "SELECT * FROM task_retirements WHERE exhausted_evaluation_key=? OR identity=?",
                    (exhausted_key, item.identity),
                ).fetchall()
                exhausted_state = str(exhausted["state"])
                if exhausted_state == "superseded":
                    if len(existing) != 1:
                        raise StateContractError(
                            f"任务 {exhausted_key!r} 已 superseded，但缺少唯一 retirement 绑定"
                        )
                    row = existing[0]
                    expected = {
                        "replacement_evaluation_key": item.replacement.evaluation_key,
                        "exhausted_logical_id": str(exhausted["logical_id"]),
                        "replacement_logical_id": item.replacement.logical_id,
                        "identity": item.identity,
                        "reason": item.reason,
                        "exhausted_plan_sha256": item.exhausted_plan_sha256,
                        "replacement_plan_sha256": item.replacement_plan_sha256,
                    }
                    if any(str(row[key]) != str(value) for key, value in expected.items()):
                        raise StateContractError(
                            f"任务 {exhausted_key!r} 的 retirement manifest 发生漂移"
                        )
                    continue
                if exhausted_state != "exhausted":
                    raise StateContractError(
                        f"任务 {exhausted_key!r} 当前状态 {exhausted_state!r}，不可退休"
                    )
                if existing:
                    raise StateContractError(
                        f"任务 {exhausted_key!r} 已绑定不同 retirement manifest"
                    )
                if str(replacement["state"]) != "frozen":
                    raise StateContractError("retirement replacement 必须处于 frozen")
                replacement_frozen = connection.execute(
                    "SELECT 1 FROM frozen_results WHERE evaluation_key=?",
                    (item.replacement.evaluation_key,),
                ).fetchone()
                if replacement_frozen is None:
                    raise StateContractError("retirement replacement 缺少 frozen binding")
                exhausted_identity = self._task_identity_from_row(exhausted)
                replacement_identity = self._task_identity_from_row(replacement)
                if exhausted_identity != replacement_identity:
                    raise StateContractError("retirement task identity 不一致")
                exhausted_base = self._logical_revision_base(str(exhausted["logical_id"]))
                replacement_base = self._logical_revision_base(item.replacement.logical_id)
                if exhausted_base != replacement_base:
                    raise StateContractError(
                        f"retirement logical revision 不一致: {exhausted_base!r} != {replacement_base!r}"
                    )
                if self._has_running_attempt(connection, exhausted_key):
                    raise StateContractError("retirement exhausted task 仍有 running attempt")
                if self._has_active_dependents(connection, exhausted_key):
                    raise StateContractError("retirement exhausted task 仍被 active dependents 引用")
                supersession = connection.execute(
                    "SELECT 1 FROM task_supersessions WHERE predecessor_evaluation_key=?",
                    (exhausted_key,),
                ).fetchone()
                if supersession is not None:
                    raise StateContractError("retirement exhausted task 已有 supersession 绑定")

                pending_rows.append(
                    (
                        exhausted_key,
                        item.replacement.evaluation_key,
                        str(exhausted["logical_id"]),
                        item.replacement.logical_id,
                        item.identity,
                        item.reason,
                        item.exhausted_plan_sha256,
                        item.replacement_plan_sha256,
                        int(exhausted["attempt_count"]),
                        exhausted["last_error_class"],
                        timestamp,
                    )
                )
                pending_events.append(
                    (
                        exhausted_key,
                        item.replacement.evaluation_key,
                        item.identity,
                        item.reason,
                        timestamp,
                    )
                )

            if pending_rows:
                connection.executemany(
                    """INSERT INTO task_retirements(
                           exhausted_evaluation_key, replacement_evaluation_key,
                           exhausted_logical_id, replacement_logical_id, identity, reason,
                           exhausted_plan_sha256, replacement_plan_sha256,
                           exhausted_attempt_count, exhausted_last_error_class, retired_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    pending_rows,
                )
                connection.executemany(
                    """UPDATE tasks
                       SET state='superseded', lease_expires_at=NULL,
                           last_error_class=NULL, updated_at=?
                       WHERE evaluation_key=?""",
                    [(event_at, exhausted_key) for exhausted_key, _, _, _, event_at in pending_events],
                )
                for exhausted_key, replacement_key, identity, reason, event_at in pending_events:
                    self._event(
                        connection,
                        event_type="exhausted_task_retired",
                        event_at=event_at,
                        evaluation_key=exhausted_key,
                        details={
                            "replacement_evaluation_key": replacement_key,
                            "identity": identity,
                            "reason": reason,
                        },
                    )

    def attempts_reserved(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()
        return self.attempt_offset + int(row["count"])

    def state_summary(self) -> dict[str, object]:
        with self._connect() as connection:
            state_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM tasks GROUP BY state"
            ).fetchall()
            historical_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]
            )
            superseded_count = int(
                len(self._valid_superseded_predecessors(connection))
            )
            active_count = self._count_active_logical_tasks(connection)
            attempt_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            )
        state_counts = {str(row["state"]): int(row["count"]) for row in state_rows}
        return {
            "logical_tasks": {
                "active": active_count,
                "historical": historical_count,
                "superseded": superseded_count,
                "state_counts": state_counts,
            },
            "attempts": {
                "physical": attempt_count,
                "reserved": self.attempt_offset + attempt_count,
                "offset": self.attempt_offset,
            },
        }

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

    @staticmethod
    def _condition_unlocked(connection: sqlite3.Connection, condition: str) -> bool:
        order = ("clean", "noise001", "noise005")
        try:
            index = order.index(condition)
        except ValueError as exc:
            raise StateContractError(f"未知任务条件: {condition!r}") from exc
        valid_superseded = TaskStateStore._valid_superseded_predecessors(connection)
        for preceding in order[:index]:
            rows = connection.execute(
                """SELECT evaluation_key, state FROM tasks
                   WHERE condition_name = ?""",
                (preceding,),
            ).fetchall()
            if any(
                not TaskStateStore._is_terminal_for_gate(
                    connection,
                    evaluation_key=str(row["evaluation_key"]),
                    state=str(row["state"]),
                    valid_superseded=valid_superseded,
                )
                for row in rows
            ):
                return False
        return True

    @staticmethod
    def _priority_unlocked(
        connection: sqlite3.Connection,
        *,
        condition: str,
        priority: int,
    ) -> bool:
        valid_superseded = TaskStateStore._valid_superseded_predecessors(connection)
        rows = connection.execute(
            """SELECT evaluation_key, state FROM tasks
               WHERE condition_name = ?
                 AND priority < ?""",
            (condition, int(priority)),
        ).fetchall()
        return not any(
            not TaskStateStore._is_terminal_for_gate(
                connection,
                evaluation_key=str(row["evaluation_key"]),
                state=str(row["state"]),
                valid_superseded=valid_superseded,
            )
            for row in rows
        )

    def next_ready_key(self, *, allowed_conditions: Sequence[str]) -> str | None:
        if not allowed_conditions:
            return None
        with self._connect() as connection:
            unlocked_conditions = tuple(
                condition
                for condition in allowed_conditions
                if self._condition_unlocked(connection, condition)
            )
            if not unlocked_conditions:
                return None
            placeholders = ",".join("?" for _ in unlocked_conditions)
            query = f"""SELECT evaluation_key, dependencies_json
                        FROM tasks
                        WHERE state IN ('pending', 'retry_wait')
                          AND condition_name IN ({placeholders})
                        ORDER BY priority ASC, logical_id ASC"""
            rows = connection.execute(query, unlocked_conditions).fetchall()
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
            if not self._condition_unlocked(connection, str(row["condition_name"])):
                raise StateContractError(
                    f"任务 {evaluation_key} 的条件尚未解锁: {row['condition_name']}"
                )
            if not self._priority_unlocked(
                connection,
                condition=str(row["condition_name"]),
                priority=int(row["priority"]),
            ):
                raise StateContractError(
                    f"任务 {evaluation_key} 的低优先级阶段尚未完成"
                )
            if not self._dependencies_frozen(connection, row["dependencies_json"]):
                raise StateContractError(f"任务 {evaluation_key} 的依赖尚未全部冻结")
            count = int(row["attempt_count"])
            if count >= self.max_attempts_per_task:
                raise StateContractError(f"任务 {evaluation_key} 已达到单任务尝试上限")
            global_count = self.attempt_offset + int(
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
            global_count = self.attempt_offset + int(
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

    def promote_failed_attempt(
        self,
        attempt_id: str,
        *,
        result_path: str,
        result_sha256: str,
        allowed_error_classes: Sequence[str],
        audit_reason: str,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        if not result_path or not result_sha256:
            raise StateContractError("补冻结果必须包含路径和 SHA-256")
        if not audit_reason:
            raise StateContractError("补冻结果必须包含审计原因")
        whitelist = tuple(dict.fromkeys(str(item) for item in allowed_error_classes if str(item)))
        if not whitelist:
            raise StateContractError("补冻结果必须提供原 error_class 白名单")
        with self._write_transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise StateContractError(f"未知 attempt: {attempt_id}")
            if attempt["status"] != "failed":
                raise StateContractError(f"attempt {attempt_id!r} 当前状态不是 failed")
            original_error_class = str(attempt["error_class"] or "")
            if original_error_class not in whitelist:
                raise StateContractError(
                    f"attempt {attempt_id!r} 的 error_class={original_error_class!r} 不在补冻白名单内"
                )
            task = connection.execute(
                "SELECT * FROM tasks WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone()
            if task is None:
                raise StateContractError(f"attempt {attempt_id!r} 关联任务不存在")
            if task["state"] not in {"retry_wait", "exhausted"}:
                raise StateContractError(
                    f"任务 {attempt['evaluation_key']} 当前状态 {task['state']!r}，不可补冻 failed attempt"
                )
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
                "UPDATE attempts SET status='accepted' WHERE attempt_id=?",
                (attempt_id,),
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
                event_type="failed_attempt_promoted",
                event_at=timestamp,
                evaluation_key=attempt["evaluation_key"],
                attempt_id=attempt_id,
                details={
                    "result_path": result_path,
                    "result_sha256": result_sha256,
                    "audit_reason": audit_reason,
                    "original_error_class": original_error_class,
                },
            )

    def increase_max_attempts_per_task(
        self,
        new_limit: int,
        *,
        audit_reason: str,
        now: float | None = None,
    ) -> bool:
        """只允许在既有全局物理预算内审计化上调单任务重试上限。"""

        if isinstance(new_limit, bool) or not isinstance(new_limit, int):
            raise StateContractError("单任务尝试上限必须是正整数")
        if new_limit <= 0:
            raise StateContractError("单任务尝试上限必须是正整数")
        if new_limit > self.attempt_cap:
            raise StateContractError("单任务尝试上限不得超过全局物理尝试上限")
        if not audit_reason:
            raise StateContractError("上调单任务尝试上限必须包含审计原因")
        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key='max_attempts_per_task'"
            ).fetchone()
            if row is None:
                raise StateContractError("状态库缺少 max_attempts_per_task 元数据")
            current = int(row["value"])
            if current != self.max_attempts_per_task:
                raise StateContractError(
                    "状态库 max_attempts_per_task 与当前实例不一致"
                )
            if new_limit < current:
                raise StateContractError("禁止下调已冻结的单任务尝试上限")
            if new_limit == current:
                return False
            connection.execute(
                "UPDATE meta SET value=? WHERE key='max_attempts_per_task'",
                (str(new_limit),),
            )
            self._event(
                connection,
                event_type="attempt_limit_increased",
                event_at=timestamp,
                details={
                    "old_limit": current,
                    "new_limit": new_limit,
                    "audit_reason": audit_reason,
                },
            )
        self.max_attempts_per_task = new_limit
        return True

    def reopen_exhausted_failed_attempt(
        self,
        attempt_id: str,
        *,
        allowed_error_classes: Sequence[str],
        reclassified_error_class: str,
        audit_reason: str,
        now: float | None = None,
    ) -> None:
        """审计后重开被旧分类器过早耗尽、且仍有尝试额度的任务。"""

        timestamp = time.time() if now is None else float(now)
        whitelist = tuple(dict.fromkeys(str(item) for item in allowed_error_classes if str(item)))
        if not whitelist:
            raise StateContractError("重开任务必须提供原 error_class 白名单")
        if not reclassified_error_class:
            raise StateContractError("重开任务必须提供新 error_class")
        if not audit_reason:
            raise StateContractError("重开任务必须包含审计原因")
        with self._write_transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise StateContractError(f"未知 attempt: {attempt_id}")
            if attempt["status"] != "failed":
                raise StateContractError(f"attempt {attempt_id!r} 当前状态不是 failed")
            original_error_class = str(attempt["error_class"] or "")
            if original_error_class not in whitelist:
                raise StateContractError(
                    f"attempt {attempt_id!r} 的 error_class={original_error_class!r} 不在重开白名单内"
                )
            task = connection.execute(
                "SELECT * FROM tasks WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone()
            if task is None:
                raise StateContractError(f"attempt {attempt_id!r} 关联任务不存在")
            if task["state"] != "exhausted":
                raise StateContractError(
                    f"任务 {attempt['evaluation_key']} 当前状态 {task['state']!r}，不可重开"
                )
            latest_attempt = connection.execute(
                """SELECT attempt_id FROM attempts
                   WHERE evaluation_key = ?
                   ORDER BY attempt_number DESC LIMIT 1""",
                (attempt["evaluation_key"],),
            ).fetchone()
            if latest_attempt is None or latest_attempt["attempt_id"] != attempt_id:
                raise StateContractError(f"attempt {attempt_id!r} 不是该任务最新尝试")
            if int(task["attempt_count"]) >= self.max_attempts_per_task:
                raise StateContractError(
                    f"任务 {attempt['evaluation_key']} 已耗尽任务级尝试预算"
                )
            global_count = self.attempt_offset + int(
                connection.execute("SELECT COUNT(*) AS count FROM attempts").fetchone()["count"]
            )
            if global_count >= self.attempt_cap:
                raise StateContractError("全局物理尝试预算已耗尽")
            if connection.execute(
                "SELECT 1 FROM frozen_results WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone() is not None:
                raise StateContractError(f"任务 {attempt['evaluation_key']} 已冻结")
            if connection.execute(
                "SELECT 1 FROM non_applicable_results WHERE evaluation_key = ?",
                (attempt["evaluation_key"],),
            ).fetchone() is not None:
                raise StateContractError(f"任务 {attempt['evaluation_key']} 已标记 non_applicable")
            connection.execute(
                """UPDATE tasks
                   SET state='retry_wait', lease_expires_at=NULL,
                       last_error_class=?, updated_at=?
                   WHERE evaluation_key=?""",
                (reclassified_error_class, timestamp, attempt["evaluation_key"]),
            )
            self._event(
                connection,
                event_type="failed_attempt_reopened",
                event_at=timestamp,
                evaluation_key=attempt["evaluation_key"],
                attempt_id=attempt_id,
                details={
                    "audit_reason": audit_reason,
                    "original_error_class": original_error_class,
                    "reclassified_error_class": reclassified_error_class,
                    "attempt_count": int(task["attempt_count"]),
                    "max_attempts_per_task": self.max_attempts_per_task,
                },
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

    def mark_non_applicable(
        self,
        evaluation_key: str,
        *,
        reason: str,
        evidence_path: str,
        evidence_sha256: str,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        if not reason or not evidence_path or not evidence_sha256:
            raise StateContractError("non_applicable 必须包含原因、证据路径和 SHA-256")
        with self._write_transaction() as connection:
            task = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key = ?",
                (evaluation_key,),
            ).fetchone()
            if task is None:
                raise StateContractError(f"未知任务: {evaluation_key}")
            existing = connection.execute(
                """SELECT reason, evidence_path, evidence_sha256
                   FROM non_applicable_results WHERE evaluation_key = ?""",
                (evaluation_key,),
            ).fetchone()
            if existing is not None:
                current = (
                    str(existing["reason"]),
                    str(existing["evidence_path"]),
                    str(existing["evidence_sha256"]),
                )
                requested = (reason, evidence_path, evidence_sha256)
                if current != requested:
                    raise StateContractError(
                        f"任务 {evaluation_key} 的 non_applicable 证据发生漂移"
                    )
                return
            if task["state"] not in {"pending", "retry_wait"}:
                raise StateContractError(
                    f"任务 {evaluation_key} 当前状态 {task['state']!r}，不可标记 non_applicable"
                )
            connection.execute(
                """INSERT INTO non_applicable_results(
                       evaluation_key, reason, evidence_path, evidence_sha256, marked_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (evaluation_key, reason, evidence_path, evidence_sha256, timestamp),
            )
            connection.execute(
                """UPDATE tasks
                   SET state='non_applicable', lease_expires_at=NULL,
                       last_error_class=NULL, updated_at=?
                   WHERE evaluation_key=?""",
                (timestamp, evaluation_key),
            )
            self._event(
                connection,
                event_type="task_non_applicable",
                event_at=timestamp,
                evaluation_key=evaluation_key,
                details={
                    "reason": reason,
                    "evidence_path": evidence_path,
                    "evidence_sha256": evidence_sha256,
                },
            )

    def non_applicable_result(self, evaluation_key: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT reason, evidence_path, evidence_sha256
                   FROM non_applicable_results WHERE evaluation_key = ?""",
                (evaluation_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "reason": str(row["reason"]),
            "evidence_path": str(row["evidence_path"]),
            "evidence_sha256": str(row["evidence_sha256"]),
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
            global_count = self.attempt_offset + int(
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
