"""Stage5 正式输入的来源隔离规则。"""

from __future__ import annotations

from collections.abc import Mapping


ABORTED_SOURCE_TOKEN = "all_15alg_fullcpu_v1"
SOURCE_IDENTITY_FIELDS = (
    "batch",
    "path",
    "remote_result_path",
    "selected_path",
    "execution_set",
)


class SourceProvenanceError(ValueError):
    """来源属于已中止批次，不能进入正式冻结或聚合。"""


def reject_aborted_fullcpu_source(
    row: Mapping[str, object], *, context: str
) -> None:
    """拒绝任何直接声明或路径引用已中止 fullcpu 批次的记录。"""
    matched = {
        field: str(row.get(field, ""))
        for field in SOURCE_IDENTITY_FIELDS
        if ABORTED_SOURCE_TOKEN in str(row.get(field, ""))
    }
    if matched:
        evidence = ", ".join(f"{field}={value!r}" for field, value in matched.items())
        raise SourceProvenanceError(
            f"{context}: 来源属于已中止的 {ABORTED_SOURCE_TOKEN} 批次，禁止用于正式输入: "
            f"{evidence}"
        )
