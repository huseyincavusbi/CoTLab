"""JSON logger output format.

The feature branch removed ``indent=2`` from the metrics/summary/result dumps
(format-only change). These tests pin that content round-trips identically and
the output is compact.
"""

import json

from cotlab.logging.json_logger import ExperimentLogger


def test_intermediate_log_compact_and_content_identical(tmp_path):
    logger = ExperimentLogger(tmp_path)
    payload = {"a": [1, 2.5, None], "b": {"nested": True}, "text": "x" * 50}

    logger.log_intermediate("step_1", payload)

    raw = (tmp_path / "intermediate_step_1.json").read_text()
    assert json.loads(raw) == payload, "content must round-trip identically"
    assert raw.count("\n  ") == 0, "must be compact (no pretty indentation)"


def test_summary_log_compact_and_content_identical(tmp_path):
    logger = ExperimentLogger(tmp_path)
    logger.save_summary({"acc": 0.5, "n": 10})

    raw = (tmp_path / "summary.json").read_text()
    assert json.loads(raw)["metrics"]["acc"] == 0.5
    assert raw.count("\n  ") == 0, "summary must also be compact"
