import sys
from pathlib import Path

LLMSR_ROOT = (
    Path(__file__).resolve().parents[1]
    / "scientific_intelligent_modelling"
    / "algorithms"
    / "llmsr_wrapper"
    / "llmsr"
)
sys.path.insert(0, str(LLMSR_ROOT))

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.llmsr.llmsr import (
    code_manipulation,
    evaluator,
)


def test_llmsr_trim_function_body_ignores_null_bytes():
    body = "    return x0 + params[0]\x00\n"

    trimmed = evaluator._trim_function_body(body)

    assert "\x00" not in trimmed
    assert "return x0 + params[0]" in trimmed


def test_llmsr_code_manipulation_ignores_null_bytes_before_ast_parse():
    source = (
        "def equation(x0, params):\n"
        "    return x0 + params[0]\x00\n"
    )

    program = code_manipulation.text_to_program(source)

    assert "\x00" not in str(program)
    assert program.get_function("equation").name == "equation"
