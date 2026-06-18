#!/usr/bin/env python3
"""Parse pipeline benchmark output into a small, stable metrics schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
_SUMMARY_CASE_RE = re.compile(r"^\s{2}(?P<case>\S.*?\))\s{2,}(?P<values>.*)$")
_SUMMARY_HEADER_RE = re.compile(r"^\s{2}Example\s{2,}")
_AVG_LINE_RE = re.compile(
    r"^\s*Host Avg:.*?O4 Cost Avg:.*?O Detail Avg:.*?\(.*?rounds\)\s*$",
    re.MULTILINE,
)
_CASE_RE = re.compile(r"^\s*Case:\s*(?P<case>\S+:\S+)\s*$", re.MULTILINE)
_EXAMPLE_HEADER_RE = re.compile(r"^\s{2}(?P<example>[a-z][a-z0-9_]+)\s*$")
_CASE_DELIM_RE = re.compile(r"^\s*----\s*(?P<case>Case[^-]+?)\s*----\s*$")
_PATTERNS = {
    "e2e_us": [
        re.compile(rf"\bTotal\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
        re.compile(rf"\bTotal\s*\(us\)[^\n]*\n[-\s]+\n[^\n]*?\b{_FLOAT}\b", re.IGNORECASE),
        re.compile(rf"\b(?:elapsed|e2e|end[-_ ]?to[-_ ]?end)\b[^0-9]{{0,32}}{_FLOAT}", re.IGNORECASE),
        re.compile(rf"\bElapsed\b[^0-9]{{0,16}}{_FLOAT}"),
    ],
    "sched_us": [
        re.compile(rf"\bSched\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
        re.compile(rf"\b(?:sched|scheduler)\b[^0-9]{{0,32}}{_FLOAT}", re.IGNORECASE),
    ],
    "orch_us": [
        re.compile(rf"\bOrch\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
        re.compile(rf"\b(?:orch|orchestrator)\b[^0-9]{{0,32}}{_FLOAT}", re.IGNORECASE),
    ],
    "o1_us": [
        re.compile(rf"\bO1\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "o2_us": [
        re.compile(rf"\bO2\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "o1_active_us": [
        re.compile(rf"\bO1\s+Active\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "o1_drain_us": [
        re.compile(rf"\bO1\s+Drain\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "o2_active_us": [
        re.compile(rf"\bO2\s+Active\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "o2_wall_us": [
        re.compile(rf"\bO2\s+Wall\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "pipe_enq_us": [
        re.compile(rf"\bPipe\s+Enq\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "pipe_flush_us": [
        re.compile(rf"\bPipe\s+Flush\s+Avg:\s*{_FLOAT}\s*us\b", re.IGNORECASE),
    ],
    "pipe_task_enq_count": [
        re.compile(
            rf"\bPipe\s+Counts\s+Avg:\s*task={_FLOAT}\s+(?:batch={_FLOAT}\s+)?scope={_FLOAT}\s+flush={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "pipe_task_enq_batch_count": [
        re.compile(
            rf"\bPipe\s+Counts\s+Avg:\s*task={_FLOAT}\s+batch={_FLOAT}\s+scope={_FLOAT}\s+flush={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "pipe_scope_enq_count": [
        re.compile(
            rf"\bPipe\s+Counts\s+Avg:\s*task={_FLOAT}\s+(?:batch={_FLOAT}\s+)?scope={_FLOAT}\s+flush={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "pipe_flush_count": [
        re.compile(
            rf"\bPipe\s+Counts\s+Avg:\s*task={_FLOAT}\s+(?:batch={_FLOAT}\s+)?scope={_FLOAT}\s+flush={_FLOAT}(?:\s+compact={_FLOAT})?\b",
            re.IGNORECASE,
        ),
    ],
    "pipe_compact_deferred_count": [
        re.compile(
            rf"\bPipe\s+Counts\s+Avg:\s*task={_FLOAT}\s+(?:batch={_FLOAT}\s+)?scope={_FLOAT}\s+flush={_FLOAT}\s+compact={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_front_cost_us": [
        re.compile(
            rf"\bO4\s+Cost\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_submitter_cost_us": [
        re.compile(
            rf"\bO4\s+Cost\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_tensormap_cost_us": [
        re.compile(
            rf"\bO4\s+Cost\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_updater_cost_us": [
        re.compile(
            rf"\bO4\s+Cost\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_front_serial_end_us": [
        re.compile(
            rf"\bO4\s+Serial\s+End\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_submitter_serial_end_us": [
        re.compile(
            rf"\bO4\s+Serial\s+End\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_tensormap_serial_end_us": [
        re.compile(
            rf"\bO4\s+Serial\s+End\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o4_updater_serial_end_us": [
        re.compile(
            rf"\bO4\s+Serial\s+End\s+Avg:\s*front={_FLOAT}\s+submitter={_FLOAT}\s+tensormap={_FLOAT}\s+updater={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o_unclassified_active_us": [
        re.compile(
            rf"\bO\s+Detail\s+Avg:\s*unclassified={_FLOAT}\s+alloc={_FLOAT}\s+submit_wrap={_FLOAT}\s+scope_end={_FLOAT}\s+p_func={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o_alloc_tensors_us": [
        re.compile(
            rf"\bO\s+Detail\s+Avg:\s*unclassified={_FLOAT}\s+alloc={_FLOAT}\s+submit_wrap={_FLOAT}\s+scope_end={_FLOAT}\s+p_func={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o_submit_wrapper_us": [
        re.compile(
            rf"\bO\s+Detail\s+Avg:\s*unclassified={_FLOAT}\s+alloc={_FLOAT}\s+submit_wrap={_FLOAT}\s+scope_end={_FLOAT}\s+p_func={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o_scope_end_us": [
        re.compile(
            rf"\bO\s+Detail\s+Avg:\s*unclassified={_FLOAT}\s+alloc={_FLOAT}\s+submit_wrap={_FLOAT}\s+scope_end={_FLOAT}\s+p_func={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
    "o_p_func_us": [
        re.compile(
            rf"\bO\s+Detail\s+Avg:\s*unclassified={_FLOAT}\s+alloc={_FLOAT}\s+submit_wrap={_FLOAT}\s+scope_end={_FLOAT}\s+p_func={_FLOAT}\b",
            re.IGNORECASE,
        ),
    ],
}


def _first_match(text: str, key: str) -> float | None:
    for pattern in _PATTERNS[key]:
        match = pattern.search(text)
        if match:
            if key == "pipe_task_enq_batch_count":
                return float(match.group(2))
            if key == "pipe_scope_enq_count":
                return float(match.group(3))
            if key == "pipe_flush_count":
                return float(match.group(4))
            if key == "pipe_compact_deferred_count":
                return float(match.group(5))
            if key in ("o4_submitter_cost_us", "o4_submitter_serial_end_us"):
                return float(match.group(2))
            if key in ("o4_tensormap_cost_us", "o4_tensormap_serial_end_us"):
                return float(match.group(3))
            if key in ("o4_updater_cost_us", "o4_updater_serial_end_us"):
                return float(match.group(4))
            if key == "o_alloc_tensors_us":
                return float(match.group(2))
            if key == "o_submit_wrapper_us":
                return float(match.group(3))
            if key == "o_scope_end_us":
                return float(match.group(4))
            if key == "o_p_func_us":
                return float(match.group(5))
            return float(match.group(1))
    return None


def parse_stdout(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "e2e_us",
        "sched_us",
        "orch_us",
        "o1_us",
        "o2_us",
        "o1_active_us",
        "o1_drain_us",
        "o2_active_us",
        "o2_wall_us",
        "pipe_enq_us",
        "pipe_flush_us",
        "pipe_task_enq_count",
        "pipe_task_enq_batch_count",
        "pipe_scope_enq_count",
        "pipe_flush_count",
        "pipe_compact_deferred_count",
        "o4_front_cost_us",
        "o4_submitter_cost_us",
        "o4_tensormap_cost_us",
        "o4_updater_cost_us",
        "o4_front_serial_end_us",
        "o4_submitter_serial_end_us",
        "o4_tensormap_serial_end_us",
        "o4_updater_serial_end_us",
        "o_unclassified_active_us",
        "o_alloc_tensors_us",
        "o_submit_wrapper_us",
        "o_scope_end_us",
        "o_p_func_us",
    ):
        value = _first_match(text, key)
        if value is not None:
            metrics[key] = value
    metrics["parse_failed"] = "e2e_us" not in metrics
    return metrics


def parse_case_metrics(text: str) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    current_case = ""
    current_example = ""
    for line in text.splitlines():
        example_match = _EXAMPLE_HEADER_RE.match(line)
        if example_match:
            current_example = example_match.group("example")
            continue
        case_delim_match = _CASE_DELIM_RE.match(line)
        if case_delim_match and current_example:
            current_case = f"{current_example} ({case_delim_match.group('case').strip()})"
            continue
        case_match = _CASE_RE.match(line)
        if case_match:
            current_case = case_match.group("case").replace(":", " (") + ")"
            continue
        if _AVG_LINE_RE.match(line) and current_case:
            cases.setdefault(current_case, {}).update(parse_stdout(line))
            continue
        trimmed_match = re.match(rf"^\s*(?P<name>Host|Device|Total|Sched|Orch|O1 Active|O1 Drain|O2 Active|O2 Wall|Pipe Enq|Pipe Flush)\s+Trimmed\s+Avg:\s*{_FLOAT}\s*us\b", line)
        if trimmed_match and current_case:
            key = trimmed_match.group("name").lower().replace(" ", "_") + "_trimmed_us"
            cases.setdefault(current_case, {})[key] = float(trimmed_match.group(2))

    in_summary = False
    for line in text.splitlines():
        if _SUMMARY_HEADER_RE.match(line):
            in_summary = True
            continue
        if not in_summary:
            continue
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        match = _SUMMARY_CASE_RE.match(line)
        if not match:
            continue
        case_name = match.group("case").strip()
        if not re.search(r"\(Case[^)]*\)$", case_name):
            continue
        values = match.group("values").split()
        if len(values) < 11:
            continue
        metrics = cases.setdefault(case_name, {})
        count_start = next((i for i, value in enumerate(values) if value.startswith("task=")), len(values))
        numeric_values = values[:count_start]
        if len(numeric_values) >= 11:
            keys = (
                "host_us",
                "device_us",
                "e2e_us",
                "sched_us",
                "orch_us",
                "o1_active_us",
                "o1_drain_us",
                "o2_active_us",
                "o2_wall_us",
                "pipe_enq_us",
                "pipe_flush_us",
            )
        elif len(numeric_values) >= 9:
            keys = (
                "host_us",
                "device_us",
                "e2e_us",
                "sched_us",
                "orch_us",
                "o1_active_us",
                "o1_drain_us",
                "pipe_enq_us",
                "pipe_flush_us",
            )
        else:
            continue
        for key, value in zip(keys, numeric_values[: len(keys)]):
            try:
                metrics.setdefault(key, float(value))
            except ValueError:
                pass
        if count_start < len(values):
            metrics.setdefault("pipe_counts", " ".join(values[count_start:]))
            for item in values[count_start:]:
                if item.startswith("compact="):
                    try:
                        metrics.setdefault("pipe_compact_deferred_count", float(item.split("=", 1)[1]))
                    except ValueError:
                        pass
    return cases


def parse_run(run_dir: Path, strategy_id: int | None, source: str, run_index: int, returncode: int) -> dict[str, Any]:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""

    metrics = parse_stdout(stdout)
    metrics.update(
        {
            "strategy_id": strategy_id,
            "source": source,
            "run_index": run_index,
            "returncode": returncode,
            "success": returncode == 0 and not metrics.get("parse_failed", False),
            "stderr_bytes": len(stderr.encode("utf-8")),
        }
    )
    return metrics


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
