"""Run a frozen Stage5 symbolic-judgment plan through the Opus 4.8 API.

The source plan is never modified.  Every result uses a new model-bound key, and
only schema- AND Stage5-semantic-valid outputs enter ``frozen``.  This command
does not run an API request unless ``--execute`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Mapping

import httpx
from jsonschema import Draft7Validator, ValidationError

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    STRICT_EVALUATOR_SYSTEM_PROMPT,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    ContractViolation,
    StructuredOutputViolation,
    canonical_json,
    evaluation_key,
    render_prompt,
    validate_structured_output,
    _parse_single_json_result,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_exhausted_simplifications import (
    run_isolated_simplify_semantic_validator,
)
from check.validate_gplearn_opus48_response import validate_response as validate_gplearn_response


MODEL = "claude-opus-4-8"
KEY_MODEL = "claude-opus-4-8[1m]"
EFFORT = "xhigh"
MAX_TOKENS = 16384
TARIFF_CNY_PER_MILLION = {"input": 3.0, "output": 15.0, "cache_read": 0.3}
ENDPOINT = "https://routify-pub.alibaba-inc.com/protocol/anthropic/v1/messages"
TASK_KINDS = {"pred_simplify": "simplify", "equivalence": "equivalence",
              "stab_structure": "structure"}
_LIMITED_WORKER_BOOTSTRAP = (
    "import resource,sys; "
    "limit=int(sys.argv[1]); "
    "resource.setrlimit(resource.RLIMIT_AS,(limit,limit)); "
    "from AAAI_experiments.stage5_metric_calculation_0831.pipeline.semantic_validation_worker import main; "
    "raise SystemExit(main())"
)


def limited_semantic_worker_command(memory_limit_bytes: int) -> list[str]:
    if memory_limit_bytes <= 0:
        raise ValueError("semantic worker memory limit must be positive")
    # The child sets its own address-space limit before importing SymPy.
    return [sys.executable, "-c", _LIMITED_WORKER_BOOTSTRAP, str(memory_limit_bytes)]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_pair_binding(row: Mapping[str, Any], *, line_number: int) -> None:
    """Reject a downstream task unless its two frozen inputs match the plan evidence."""

    request = row["request"]
    evidence = request.get("deterministic_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"line {line_number}: missing deterministic pair evidence")
    expected_hash = sha256_text(canonical_json(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}))
    if evidence.get("evidence_sha256") != expected_hash or request.get("evidence_hash") != expected_hash:
        raise ValueError(f"line {line_number}: deterministic pair evidence hash drift")
    lhs = evidence.get("lhs_binding")
    rhs = evidence.get("rhs_binding")
    if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping):
        raise ValueError(f"line {line_number}: missing upstream expression bindings")
    dependencies = row.get("dependencies")
    bound_keys = [lhs.get("frozen_evaluation_key"), rhs.get("frozen_evaluation_key")]
    if (not isinstance(dependencies, list) or dependencies != bound_keys
            or any(not isinstance(key, str) or not key for key in bound_keys)):
        raise ValueError(f"line {line_number}: upstream dependency keys differ from pair evidence")
    for side, binding in (("lhs", lhs), ("rhs", rhs)):
        if not isinstance(binding.get("plan_original_expression"), str) or not binding["plan_original_expression"].strip():
            raise ValueError(f"line {line_number}: {side} source expression missing")
        if not isinstance(binding.get("frozen_effective_expression"), str) or not binding["frozen_effective_expression"].strip():
            raise ValueError(f"line {line_number}: {side} effective expression missing")

    if row["task_type"] == "equivalence":
        fields = (("effective_ground_truth_expression", lhs),
                  ("effective_prediction_expression", rhs))
        source_hashes = (rhs.get("source_result_sha256"),)
        if request.get("prediction_result_sha256") != rhs.get("source_result_sha256"):
            raise ValueError(f"line {line_number}: prediction source result drift")
    else:
        fields = (("effective_prediction_a_expression", lhs),
                  ("effective_prediction_b_expression", rhs))
        source_hashes = (lhs.get("source_result_sha256"), rhs.get("source_result_sha256"))
        if request.get("prediction_a_valid_output") is not True or request.get("prediction_b_valid_output") is not True:
            raise ValueError(f"line {line_number}: invalid seed pair must not request structure")
        if (request.get("prediction_a_result_sha256") != lhs.get("source_result_sha256")
                or request.get("prediction_b_result_sha256") != rhs.get("source_result_sha256")):
            raise ValueError(f"line {line_number}: seed-pair source results drift")
        if request.get("deterministic_pair_evidence") != evidence:
            raise ValueError(f"line {line_number}: deterministic pair evidence copy drift")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in source_hashes):
        raise ValueError(f"line {line_number}: prediction source result SHA256 missing")
    for field, binding in fields:
        if request.get(field) != binding.get("frozen_effective_expression"):
            raise ValueError(f"line {line_number}: {field} differs from frozen upstream")


def load_plan(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    plan_type: str | None = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_type") not in TASK_KINDS:
            raise ValueError(f"line {line_number}: unsupported task_type")
        if plan_type is None:
            plan_type = row["task_type"]
        elif row["task_type"] != plan_type:
            raise ValueError(f"line {line_number}: task types cannot share a plan or output state")
        request = row["request"]
        schema = row["schema_content"]
        template = row["prompt_template"]
        if not isinstance(request, dict) or not isinstance(schema, dict):
            raise ValueError(f"line {line_number}: request/schema must be JSON objects")
        if sha256_text(template) != row["prompt_sha256"]:
            raise ValueError(f"line {line_number}: prompt hash drift")
        schema_path = Path(row["schema_path"])
        if not schema_path.is_file() or hashlib.sha256(schema_path.read_bytes()).hexdigest() != row["schema_sha256"]:
            raise ValueError(f"line {line_number}: schema hash drift")
        if render_prompt(template, request, schema) != row["rendered_prompt"]:
            raise ValueError(f"line {line_number}: rendered prompt drift")
        common = dict(
            task_type=row["task_type"], logical_id=row["logical_id"],
            prompt_version=row["prompt_version"], schema_version=row["schema_version"],
            prompt_sha256=row["prompt_sha256"], schema_sha256=row["schema_sha256"],
            normalized_input=row["normalized_input"], evidence_hash=request["evidence_hash"],
        )
        if evaluation_key(**common) != row["evaluation_key"]:
            raise ValueError(f"line {line_number}: source Opus5 key drift")
        if row["task_type"] != "pred_simplify":
            validate_pair_binding(row, line_number=line_number)
        key = evaluation_key(**common, model=KEY_MODEL, effort=EFFORT)
        if key in keys:
            raise ValueError(f"line {line_number}: duplicate Opus4.8 key")
        keys.add(key)
        row["opus48_evaluation_key"] = key
        rows.append(row)
    if not rows:
        raise ValueError("empty plan")
    return rows, hashlib.sha256(raw).hexdigest()


class SlidingWindowLimiter:
    """A process-local sliding window, seeded from persisted request reservations."""

    def __init__(self, limit: int, previous: list[float] | None = None,
                 *, clock=time.time, sleeper=asyncio.sleep) -> None:
        if not 1 <= limit <= 500:
            raise ValueError("RPM must be between 1 and 500")
        self.limit = limit
        self.starts = deque(sorted(previous or []))
        self.last_start = self.starts[-1] if self.starts else None
        self.clock = clock
        self.sleeper = sleeper
        self.lock = asyncio.Lock()

    async def reserve(self, persist) -> float:
        async with self.lock:
            while True:
                now = self.clock()
                while self.starts and self.starts[0] <= now - 60.0:
                    self.starts.popleft()
                spacing = 60.0 / self.limit
                spacing_ready = self.last_start is None or now >= self.last_start + spacing
                if len(self.starts) < self.limit and spacing_ready:
                    # The durable reservation is made before the HTTP request.
                    persist(now)
                    self.starts.append(now)
                    self.last_start = now
                    return now
                wait_until = now
                if len(self.starts) >= self.limit:
                    wait_until = max(wait_until, self.starts[0] + 60.001)
                if not spacing_ready:
                    wait_until = max(wait_until, self.last_start + spacing)
                await self.sleeper(max(0.001, wait_until - now))


def cost_cny(usage: Mapping[str, object]) -> float:
    def count(name: str) -> int:
        value = usage.get(name, 0)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    return (count("input_tokens") * 3 + count("output_tokens") * 15
            + count("cache_read_input_tokens") * 0.3) / 1_000_000


def validate_message(body: object, row: Mapping[str, Any]) -> tuple[dict[str, object], dict[str, object], str | None]:
    if not isinstance(body, Mapping) or body.get("type") != "message" or body.get("role") != "assistant":
        raise StructuredOutputViolation("not an assistant message")
    if body.get("model") != MODEL:
        raise ContractViolation(f"response model drift: {body.get('model')!r}")
    if body.get("stop_reason") != "end_turn":
        raise StructuredOutputViolation(f"stop_reason={body.get('stop_reason')!r}")
    blocks = body.get("content")
    if not isinstance(blocks, list):
        raise StructuredOutputViolation("content is not a list")
    if any(not isinstance(block, Mapping) or block.get("type") not in
           {"text", "thinking", "redacted_thinking"} for block in blocks):
        raise ContractViolation("unexpected API content block")
    parts = [block["text"] for block in blocks if isinstance(block, Mapping)
             and block.get("type") == "text" and isinstance(block.get("text"), str)]
    if not parts:
        raise StructuredOutputViolation("no text output")
    output_text = "\n".join(parts)

    def validate_candidate(text: str) -> dict[str, object]:
        parsed = _parse_single_json_result(text)
        try:
            Draft7Validator(row["schema_content"]).validate(parsed)
        except ValidationError as exc:
            raise StructuredOutputViolation(f"schema: {exc.message}") from exc
        try:
            return validate_structured_output(TASK_KINDS[row["task_type"]], parsed)
        except ContractViolation as exc:
            raise StructuredOutputViolation(str(exc)) from exc

    recovery: str | None = None
    try:
        output = validate_candidate(output_text)
    except StructuredOutputViolation as strict_error:
        fenced: dict[str, dict[str, object]] = {}
        for match in re.finditer(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", output_text):
            try:
                valid = validate_candidate(match.group(1))
            except StructuredOutputViolation:
                continue
            fenced[canonical_json(valid)] = valid
        if len(fenced) == 1:
            output = next(iter(fenced.values()))
            recovery = "single_valid_json_fence"
        else:
            embedded: dict[str, dict[str, object]] = {}
            decoder = json.JSONDecoder()
            for match in re.finditer(r"(?m)^[ \t]*\{", output_text):
                try:
                    parsed, _ = decoder.raw_decode(output_text[match.start():].lstrip())
                    valid = validate_candidate(canonical_json(parsed))
                except (json.JSONDecodeError, StructuredOutputViolation):
                    continue
                embedded[canonical_json(valid)] = valid
            if len(embedded) != 1:
                raise strict_error
            output = next(iter(embedded.values()))
            recovery = "single_valid_embedded_json_object"
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        raise ContractViolation("missing usage")
    return output, dict(usage), recovery


class Ledger:
    def __init__(self, root: Path, plan_sha256: str, count: int) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(root / "state.sqlite")
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS attempts ("
                          "id TEXT PRIMARY KEY, evaluation_key TEXT NOT NULL, started REAL NOT NULL, "
                          "status TEXT NOT NULL, error TEXT, cost REAL DEFAULT 0)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS attempts_key ON attempts(evaluation_key)")
        prior = self.conn.execute("SELECT value FROM meta WHERE name='plan_sha256'").fetchone()
        if prior is not None and prior[0] != plan_sha256:
            raise ValueError("output directory belongs to a different source plan")
        if prior is None:
            self.conn.execute("INSERT INTO meta VALUES ('plan_sha256', ?)", (plan_sha256,))
            self.conn.execute("INSERT INTO meta VALUES ('task_count', ?)", (str(count),))
        self.conn.execute("UPDATE attempts SET status='interrupted' WHERE status='running'")
        self.conn.commit()
        self.cap = math.ceil(1.5 * count)

    def bind_task_type(self, task_type: str) -> None:
        prior = self.conn.execute("SELECT value FROM meta WHERE name='task_type'").fetchone()
        if prior is not None and prior[0] != task_type:
            raise ValueError("output directory belongs to a different task type")
        if prior is None:
            # Older pred_simplify ledgers are still isolated by their frozen plan hash.
            self.conn.execute("INSERT INTO meta VALUES ('task_type', ?)", (task_type,))
            self.conn.commit()

    def recent(self) -> list[float]:
        now = time.time()
        return [float(x[0]) for x in self.conn.execute(
            "SELECT started FROM attempts WHERE started > ?", (now - 60.0,))]

    def count(self, key: str | None = None) -> int:
        if key is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM attempts WHERE evaluation_key=?", (key,)).fetchone()[0])

    def reserve(self, attempt_id: str, key: str, started: float) -> None:
        if self.count() >= self.cap or self.count(key) >= 2:
            raise RuntimeError("physical attempt budget exhausted")
        self.conn.execute("INSERT INTO attempts VALUES (?, ?, ?, 'running', NULL, 0)",
                          (attempt_id, key, started))
        self.conn.commit()

    def finish(self, attempt_id: str, status: str, error: str | None, cost: float) -> None:
        self.conn.execute("UPDATE attempts SET status=?, error=?, cost=? WHERE id=?",
                          (status, error, cost, attempt_id))
        self.conn.commit()

    def summary(self, rows: list[dict[str, Any]]) -> dict[str, object]:
        frozen = self.root / "frozen"
        success = sum((frozen / f"{row['opus48_evaluation_key']}.json").exists() for row in rows)
        counts = dict(self.conn.execute("SELECT status, COUNT(*) FROM attempts GROUP BY status"))
        return {"plan_tasks": len(rows), "frozen": success, "remaining": len(rows)-success,
                "physical_attempts": self.count(), "physical_cap": self.cap,
                "attempt_states": counts,
                "estimated_cost_cny_assumed_tariff": round(float(self.conn.execute(
                    "SELECT COALESCE(SUM(cost),0) FROM attempts").fetchone()[0]), 6)}


async def run(rows: list[dict[str, Any]], ledger: Ledger, token: str, *,
              concurrency: int, rpm: int, timeout: float,
              max_new_tasks: int | None = None,
              semantic_memory_limit_bytes: int = 2 * 1024**3) -> dict[str, object]:
    task_types = {row["task_type"] for row in rows}
    if len(task_types) != 1 or not task_types <= TASK_KINDS.keys():
        raise ValueError("one supported task type is required per plan and output state")
    ledger.bind_task_type(next(iter(task_types)))
    limiter = SlidingWindowLimiter(rpm, ledger.recent())
    semantic_limit = asyncio.Semaphore(4)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    selected = 0
    for row in rows:
        frozen_path = ledger.root / "frozen" / f"{row['opus48_evaluation_key']}.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if frozen.get("evaluation_key") != row["opus48_evaluation_key"] or frozen.get("status") != "frozen":
                raise ValueError(f"corrupt frozen result: {frozen_path}")
            continue
        if ledger.count(row["opus48_evaluation_key"]) >= 2:
            continue
        if max_new_tasks is None or selected < max_new_tasks:
            queue.put_nowait(row)
            selected += 1
    completed = 0
    reported_attempt_milestone = ledger.count() // 100
    stop = asyncio.Event()
    headers = {"Authorization": f"Bearer {token}", "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout,
                                 trust_env=False) as client:
        async def worker() -> None:
            nonlocal completed, reported_attempt_milestone
            while not queue.empty() and not stop.is_set():
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                key = row["opus48_evaluation_key"]
                request = row["request"]
                gplearn_native = request.get("native_semantics_version") == "gplearn_native_protected_prefix.v1"
                prompt = row["rendered_prompt"]
                payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "stream": False,
                           "thinking": {"type": "adaptive"}, "output_config": {"effort": EFFORT},
                           "system": STRICT_EVALUATOR_SYSTEM_PROMPT,
                           "messages": [{"role": "user", "content": prompt}]}
                try:
                    while ledger.count(key) < 2 and ledger.count() < ledger.cap and not stop.is_set():
                        attempt_number = ledger.count(key) + 1
                        attempt_id = f"{key}.a{attempt_number:02d}"
                        try:
                            await limiter.reserve(lambda started: ledger.reserve(attempt_id, key, started))
                        except RuntimeError:
                            break
                        body: object = None
                        status_code: int | None = None
                        response_headers: dict[str, str] = {}
                        output: dict[str, object] | None = None
                        semantic: dict[str, object] | None = None
                        usage: dict[str, object] = {}
                        recovery: str | None = None
                        error: str | None = None
                        retryable = True
                        try:
                            response = await client.post(ENDPOINT, json=payload)
                            status_code = response.status_code
                            response_headers = {k: v for k, v in response.headers.items()
                                                if k.lower() in {"request-id", "anthropic-request-id", "retry-after"}}
                            try:
                                body = response.json()
                            except ValueError:
                                body = {"unparsed_text": response.text[:2000]}
                            if isinstance(body, Mapping) and isinstance(body.get("usage"), Mapping):
                                usage = dict(body["usage"])
                            if status_code != 200:
                                error = f"HTTP {status_code}: {str(body)[:1000]}"
                                retryable = status_code in {408, 409, 429} or status_code >= 500
                                if status_code in {401, 403}:
                                    stop.set()
                            else:
                                output, usage, recovery = validate_message(body, row)
                                if row["task_type"] == "pred_simplify":
                                    async with semantic_limit:
                                        if gplearn_native:
                                            semantic = await asyncio.to_thread(
                                                validate_gplearn_response, request, output)
                                        else:
                                            semantic = await asyncio.to_thread(
                                                run_isolated_simplify_semantic_validator,
                                                evaluation_key=key, request=request,
                                                structured_output=output, timeout_seconds=120.0,
                                                worker_command=limited_semantic_worker_command(
                                                    semantic_memory_limit_bytes))
                                    if semantic.get("status") != "promotable":
                                        error = f"semantic: {semantic.get('status')}: {semantic.get('error')}"
                        except (httpx.HTTPError, ContractViolation, ValueError) as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, ContractViolation) and not isinstance(exc, StructuredOutputViolation):
                                retryable = False
                                stop.set()
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"

                        attempt = {
                            "attempt_id": attempt_id, "evaluation_key": key,
                            "source_evaluation_key": row["evaluation_key"],
                            "logical_id": row["logical_id"], "task_type": row["task_type"],
                            "model": MODEL, "key_model": KEY_MODEL, "effort": EFFORT,
                            "requested_model": MODEL,
                            "response_model": body.get("model") if isinstance(body, Mapping) else None,
                            "prompt_version": row["prompt_version"], "schema_version": row["schema_version"],
                            "prompt_sha256": row["prompt_sha256"], "schema_sha256": row["schema_sha256"],
                            "rendered_prompt_sha256": sha256_text(prompt),
                            "source_input_hash": row["input_hash"], "request": request,
                            "plan_dependencies": row.get("dependencies", []),
                            "terminal_binding_evidence": request.get("terminal_binding_evidence"),
                            "terminal_expression": request.get("expression"),
                            "terminal_source_result_sha256": (
                                request.get("ast_source_evidence", {}).get("result_raw_sha256")
                                if isinstance(request.get("ast_source_evidence"), Mapping) else None),
                            "terminal_source_row_sha256": (
                                request.get("ast_source_evidence", {}).get("source_row_sha256")
                                if isinstance(request.get("ast_source_evidence"), Mapping) else None),
                            "api_request_sha256": sha256_text(canonical_json(payload)),
                            "response": body, "response_headers": response_headers,
                            "http_status": status_code, "structured_output": output,
                            "structured_output_recovery": recovery,
                            "semantic_validation": semantic, "usage": usage,
                            "semantic_worker_rlimit_as_bytes": (
                                semantic_memory_limit_bytes if row["task_type"] == "pred_simplify"
                                and not gplearn_native else None),
                            "estimated_cost_cny_assumed_tariff": cost_cny(usage),
                            "error": error,
                        }
                        atomic_json(ledger.root / "attempts" / f"{attempt_id}.json", attempt)
                        if error is None:
                            frozen = dict(attempt)
                            frozen["status"] = "frozen"
                            atomic_json(ledger.root / "frozen" / f"{key}.json", frozen)
                        ledger.finish(attempt_id, "success" if error is None else "failed",
                                      error, cost_cny(usage))
                        milestone = ledger.count() // 100
                        if milestone > reported_attempt_milestone:
                            reported_attempt_milestone = milestone
                            report = ledger.summary(rows)
                            atomic_json(ledger.root / "progress.json", report)
                            print(json.dumps(report, ensure_ascii=False), flush=True)
                        if error is None or not retryable:
                            break
                        if attempt_number < 2:
                            await asyncio.sleep(min(5.0, attempt_number * 2.0))
                finally:
                    completed += 1
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    summary = ledger.summary(rows)
    summary["scheduled_this_invocation"] = selected
    unresolved = []
    for row in rows:
        key = row["opus48_evaluation_key"]
        if not (ledger.root / "frozen" / f"{key}.json").exists():
            unresolved.append({"evaluation_key": key, "logical_id": row["logical_id"],
                               "source_evaluation_key": row["evaluation_key"],
                               "attempts": ledger.count(key)})
    atomic_json(ledger.root / "summary.json", summary)
    atomic_json(ledger.root / "unresolved.json", {"items": unresolved})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=Path.home() / ".claude/settings.json")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--rpm", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--semantic-memory-gib", type=float, default=2.0)
    parser.add_argument("--max-new-tasks", type=int,
                        help="schedule at most this many currently unfinished plan rows")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 512 or not 1 <= args.rpm <= 500 or args.timeout <= 0:
        parser.error("concurrency must be 1..512, rpm 1..500, timeout positive")
    if args.max_new_tasks is not None and args.max_new_tasks <= 0:
        parser.error("max-new-tasks must be positive")
    if not 0.5 <= args.semantic_memory_gib <= 16:
        parser.error("semantic-memory-gib must be between 0.5 and 16")
    rows, plan_sha = load_plan(args.plan)
    if not args.execute:
        print(json.dumps({"dry_run": True, "tasks": len(rows), "plan_sha256": plan_sha,
                          "new_model_keys": len({r["opus48_evaluation_key"] for r in rows}),
                          "physical_cap": math.ceil(1.5 * len(rows)), "rpm": args.rpm,
                          "concurrency": args.concurrency,
                          "max_new_tasks": args.max_new_tasks}, ensure_ascii=False))
        return 0
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    env = settings.get("env", {})
    if env.get("ANTHROPIC_BASE_URL", "").rstrip("/") != ENDPOINT.removesuffix("/v1/messages"):
        parser.error("settings does not point to the approved Routify Anthropic channel")
    token = env.get("ANTHROPIC_AUTH_TOKEN")
    if not isinstance(token, str) or not token:
        parser.error("ANTHROPIC_AUTH_TOKEN missing")
    ledger = Ledger(args.output, plan_sha, len(rows))
    print(json.dumps(asyncio.run(run(rows, ledger, token, concurrency=args.concurrency,
                                     rpm=args.rpm, timeout=args.timeout,
                                     max_new_tasks=args.max_new_tasks,
                                     semantic_memory_limit_bytes=int(args.semantic_memory_gib * 1024**3))),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
