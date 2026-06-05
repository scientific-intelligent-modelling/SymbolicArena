# Formal 3h Recovery-First Benchmark Design

## Goal

Build an independent formal 3h experiment control plane for:

```text
13 algorithms × SSR50 × seeds(520,521,522) × noise(0,0.01,0.05) × 3h
```

The control plane must first recover usable 3h snapshots from the existing formal 24h batch, then run only the remaining missing tasks.

## Current Evidence

The existing 24h batch is:

```text
formal24h_13alg_ssr50_seed520-522_noise0-001-005_20260531-230535
```

The real full-output root on each remote host is:

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal24h_13alg_ssr50_seed520-522_noise0-001-005_20260531-230535
```

The checked 3h snapshot is:

```text
progress/minute_0180.json
```

Current read-only audit:

```text
direct minute_0180 snapshots = 2767
valid 3h recoverable snapshots = 2755
invalid snapshots = 12
```

Valid sampled payloads include `status="ok"`, `elapsed_seconds≈10800`, `equation`, `valid`, `id_test`, `ood_test`, and `canonical_artifact`. This is sufficient to treat valid `minute_0180.json` files as 3h result payloads.

## Architecture

Use a new `formal3h` batch family rather than mutating `formal24h`.

```text
benchmark-runs/formal3h/
  <batch_id>/
    manifest/
    params/
    params_smoke/
    queues/
    recovery/
    runs/
    audit/
    deploy/
    heartbeat.json
  latest -> <batch_id>
```

The 3h pipeline has five phases:

1. `prepare`: generate 5850-task manifest and 3h parameter files.
2. `recover`: copy valid `minute_0180.json` snapshots into the 3h `runs/` layout.
3. `preflight`: verify remote code, params, data, and queue contracts.
4. `smoke`: run 13 algorithms × 2 datasets × 3 seeds × 3 noise with 600s budget.
5. `full`: run only tasks listed in `recovery/missing_tasks.csv` with 10800s budget.

## Components

### Profile

Add `FORMAL3H_SPEC` with:

```text
name = formal3h_13alg_3seed_3noise
budget.name = formal3h
timeout_in_seconds = 10800
min_runtime_seconds = 10500
progress_snapshot_interval_seconds = 60
```

The algorithm, seed, and noise dimensions match `FULL24H_SPEC`.

### Snapshot Recovery

Add a recovery launcher that reads the formal3h manifest and the old 24h experiment roots. For each task it reconstructs scheduler task id:

```text
<tool_key>_s<seed>_<noise_tag>_g<global_index:04d>
```

It searches for:

```text
<source_root>/<tool_key>/seed<seed>/tasks/<scheduler_task_id>/**/progress/minute_0180.json
```

Valid snapshots are copied into:

```text
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/result.json
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/progress/minute_0180.json
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/recovery_source.json
```

The recovery phase writes:

```text
recovery/recovered_tasks.csv
recovery/missing_tasks.csv
recovery/invalid_snapshot_tasks.csv
recovery/recovery_summary.json
```

### Missing-Only Full Queue

The scheduler must avoid rerunning recovered tasks. The safest minimal interface is a task-id allowlist:

```text
--task-id-allowlist-csv benchmark-runs/formal3h/latest/recovery/missing_tasks.csv
```

The scheduler builds the normal task universe, then filters by `task_id` after task-id construction and before materializing queue state.

### Harvest and Audit

Recovered snapshots already occupy the final `runs/` layout. Fresh 3h runs are harvested into the same layout after full completion.

Merge policy:

1. If a fresh full result exists for a task, harvest it.
2. If no fresh full result exists but a recovered result exists, keep the recovered result.
3. If both are absent, audit as `missing_result`.

Audit must accept `elapsed_seconds` as a runtime source in addition to `runtime_seconds` and `seconds`, because snapshot payloads may use any of these keys.

### Heartbeat

The `formal3h` heartbeat uses independent states:

```text
preparing -> recovering -> preflight -> smoke -> running -> audit -> repair -> rerun -> done
```

The goal is complete only when:

```text
heartbeat.json needs_codex=false
audit/failure_cases.csv is empty
audit/task_audit.csv has 5850 rows
```

## Error Handling

- `status!="ok"` snapshot: write `invalid_snapshot_tasks.csv`, do not recover it.
- Missing `minute_0180.json`: write `missing_tasks.csv`, schedule it for fresh 3h run.
- Missing metrics or artifact: write `invalid_snapshot_tasks.csv`, schedule it for fresh 3h run.
- Runtime under `10500s`: audit as `early_stop`.
- Remote host unreachable during recovery: keep its tasks in `missing_tasks.csv` unless a valid snapshot is found through another source root.
- Existing 24h task processes are not stopped by this design.

## Testing

Add tests for:

- `FORMAL3H_SPEC` values and generated manifest size.
- 3h params generated with `10800`, smoke params generated with `600`.
- Recovery from a synthetic `minute_0180.json`.
- Invalid snapshot rejection.
- Missing-only scheduler filtering.
- `elapsed_seconds` runtime auditing.
- Generated deploy scripts referencing `benchmark-runs/formal3h/latest`, not `formal24h`.

## Scope Boundary

This design does not launch the full experiment by itself. It defines the control-plane implementation required to safely run a 3h formal batch without recomputing already recovered 3h snapshots.
