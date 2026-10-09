"""Assert every unified JSON case still parses against the schema."""

import json
from pathlib import Path

import pytest

_CASES = sorted(Path(__file__).parent.joinpath("test_cases").glob("*.json"))


@pytest.mark.parametrize("path", _CASES, ids=lambda p: p.name)
def test_unified_case_parses(path: Path) -> None:
    from tests.unified.schema import TestDefinition

    # Explicit encoding: the locale default (GBK on a zh-CN Windows box) cannot
    # decode these UTF-8 case files, and the failure looks like a schema problem.
    TestDefinition(**json.loads(path.read_text(encoding="utf-8")))
