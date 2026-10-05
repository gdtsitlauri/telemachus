import random

import pytest

from telemachus.llm_agent.sandbox import check_program, run_program, UnsafeProgram
from telemachus.llm_agent.generate import parse_number, parse_think, parse_code
from telemachus.llm_agent import cascade


def test_sandbox_runs_arithmetic_and_rejects_everything_else():
    assert run_program("import math\nx = 3 * 4\nanswer = math.sqrt(x * 3)") == 6.0
    for bad in ("import os\nanswer = 1", "answer = open('x').read()", "answer = __import__('os')",
                "x = (1).__class__\nanswer = 1", "answer = eval('1')", "y = 2"):
        with pytest.raises(UnsafeProgram):
            check_program(bad)
        assert run_program(bad) is None
    assert run_program("answer = 1\nwhile True:\n    answer += 1", timeout=1.) is None


def test_answer_parsers():
    assert parse_number("It costs $1,234.50 total") == 1234.5
    assert parse_think("Step... 5 + 6 = 11. The answer is 11.") == 11.
    assert parse_code("Here:\n```python\nanswer = 2\n```") == "answer = 2"


def _rec(i, correct_stage):
    rec = {"id": f"q{i}", "gold": 1.}
    for j, s in enumerate(cascade.STAGES):
        ok = correct_stage is not None and j >= correct_stage
        rec[s] = {"value": 1. if ok else 2., "correct": ok, "tokens": 10 * (j + 1), "mean_logprob": -.1 if ok else -2.,
                  "min_logprob": -.5, "executed": True, "agreement": 1. if ok else .25}
    return rec


def test_cascade_guarantee_and_expert_fallback():
    rng = random.Random(0)
    table = [_rec(i, rng.choice([0, 1, 2, None])) for i in range(400)]
    # Always escalating to the expert is error-free and certifiable.
    always = cascade.run_cascade(table, "confidence", 1.01)
    assert always["error"] == 0. and always["expert_rate"] == 1.
    c = cascade.certify_gate(table, "oracle", .1, .1)
    assert c["selected_theta"] is not None
    r = cascade.run_cascade(table, "oracle", c["selected_theta"])
    assert r["error"] == 0. and r["cost"] < always["cost"]
