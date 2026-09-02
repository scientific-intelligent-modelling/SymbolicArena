"""使用 Opus5 Messages API 执行 Stage5 计划。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    AnthropicApiChannel,
    AnthropicApiRunner,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    execute_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore


DEFAULT_CHANNEL_SETTINGS = (
    f"routify={Path.home() / '.claude/settings.json'}",
    f"yapi={Path.home() / '.claude/settings-glm.json'}",
)


class ApiChannelConfigError(ValueError):
    """渠道配置文件缺失或字段不完整。"""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return parsed


def load_api_channels(
    settings_specs: Sequence[str],
    *,
    allow_single_channel: bool = False,
) -> tuple[AnthropicApiChannel, ...]:
    channels: list[AnthropicApiChannel] = []
    seen_names: set[str] = set()
    for spec in settings_specs:
        name, separator, path_text = spec.partition("=")
        if not separator or not name.strip() or not path_text.strip():
            raise ApiChannelConfigError(
                f"channel settings 必须采用 name=/path/to/settings.json: {spec!r}"
            )
        channel_name = name.strip()
        if channel_name in seen_names:
            raise ApiChannelConfigError(f"渠道名称重复: {channel_name!r}")
        path = Path(path_text).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiChannelConfigError(f"无法读取渠道 {channel_name} 配置 {path}: {exc}") from exc
        env = payload.get("env") if isinstance(payload, dict) else None
        if not isinstance(env, dict):
            raise ApiChannelConfigError(f"渠道 {channel_name} 配置缺少 env object")
        base_url = env.get("ANTHROPIC_BASE_URL")
        auth_token = env.get("ANTHROPIC_AUTH_TOKEN")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ApiChannelConfigError(f"渠道 {channel_name} 缺少 ANTHROPIC_BASE_URL")
        if not isinstance(auth_token, str) or not auth_token.strip():
            raise ApiChannelConfigError(f"渠道 {channel_name} 缺少 ANTHROPIC_AUTH_TOKEN")
        channels.append(
            AnthropicApiChannel(
                name=channel_name,
                base_url=base_url.strip(),
                auth_token=auth_token.strip(),
            )
        )
        seen_names.add(channel_name)
    if len(channels) < 2 and not allow_single_channel:
        raise ApiChannelConfigError("正式运行至少需要两个 API 渠道")
    return tuple(channels)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 Stage5 双渠道 Opus5 API plan JSONL")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--logical-id", dest="logical_ids", action="append", default=[])
    parser.add_argument("--workers", type=_positive_int, default=64)
    parser.add_argument("--physical-attempt-offset", type=int, default=0)
    parser.add_argument("--predecessor-attempt-manifest", type=Path)
    parser.add_argument(
        "--channel-settings",
        action="append",
        default=None,
        help="重复提供 name=/path/settings.json；默认使用 routify 与 yapi 配置",
    )
    parser.add_argument(
        "--allow-single-channel",
        action="store_true",
        help="仅用于某一渠道失效后的审计恢复；默认仍强制双渠道",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=300.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    parser.add_argument("--per-channel-concurrency", type=_positive_int, default=32)
    parser.add_argument("--semantic-validation-concurrency", type=_positive_int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        channels = load_api_channels(
            tuple(args.channel_settings or DEFAULT_CHANNEL_SETTINGS),
            allow_single_channel=args.allow_single_channel,
        )
    except ApiChannelConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    def runner_factory(
        store: TaskStateStore,
        *,
        attempts_dir: Path,
        frozen_dir: Path,
    ) -> Any:
        return AnthropicApiRunner(
            store,
            attempts_dir=attempts_dir,
            frozen_dir=frozen_dir,
            channels=channels,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            per_channel_concurrency=args.per_channel_concurrency,
            semantic_validation_concurrency=args.semantic_validation_concurrency,
            allow_single_channel=args.allow_single_channel,
        )

    return execute_plan(
        plan_jsonl=args.plan_jsonl,
        state_db=args.state_db,
        attempts_dir=args.attempts_dir,
        frozen_dir=args.frozen_dir,
        report_json=args.report_json,
        limit=args.limit,
        logical_ids=tuple(args.logical_ids),
        workers=args.workers,
        physical_attempt_offset=args.physical_attempt_offset,
        predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        runner_factory=runner_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
