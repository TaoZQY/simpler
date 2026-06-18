#!/usr/bin/env python3
"""Run base plus selected pipeline strategies and rank the results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline_metrics import parse_case_metrics, parse_run, write_metrics


STRATEGY_NAMES = {
    None: "4T_O3S_baseline",
    0: "2O2S_cross_cluster_debug_strategy0",
    1: "2S2O_split_ctrl_strategy1",
    2: "2O2S_cross_cluster_strategy2",
    3: "2S4O_split_orch",
    4: "2S4O_split_mixed",
    5: "2S2O_pipeline_orch",
}


def _parse_strategies(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _run_once(cmd: str, cwd: Path, env: dict[str, str], run_dir: Path, stream_output: bool) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    if stream_output:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            proc = subprocess.Popen(
                shlex.split(cmd),
                cwd=str(cwd),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_thread = threading.Thread(
                target=_copy_stream, args=(proc.stdout, sys.stdout, stdout_file), daemon=True
            )
            stderr_thread = threading.Thread(
                target=_copy_stream, args=(proc.stderr, sys.stderr, stderr_file), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            rc = proc.wait()
            stdout_thread.join()
            stderr_thread.join()
        return rc

    proc = subprocess.run(
        shlex.split(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def _copy_stream(source: Any, console: Any, log_file: Any) -> None:
    for line in source:
        console.write(line)
        console.flush()
        log_file.write(line)
        log_file.flush()


def _aggregate(metrics: list[dict[str, Any]], strategy_id: int | None, source: str) -> dict[str, Any]:
    e2e = [m["e2e_us"] for m in metrics if m.get("success") and "e2e_us" in m]
    row: dict[str, Any] = {
        "strategy_id": "" if strategy_id is None else strategy_id,
        "strategy_name": STRATEGY_NAMES.get(strategy_id, str(strategy_id)),
        "source": source,
        "repeat": len(metrics),
        "success_count": sum(1 for m in metrics if m.get("success")),
        "fail_count": sum(1 for m in metrics if not m.get("success")),
        "mean_e2e_us": statistics.fmean(e2e) if e2e else None,
        "median_e2e_us": statistics.median(e2e) if e2e else None,
        "p90_e2e_us": _percentile(e2e, 90),
        "p99_e2e_us": _percentile(e2e, 99),
        "stddev_e2e_us": statistics.pstdev(e2e) if len(e2e) > 1 else 0.0 if e2e else None,
    }
    return row


def _score(row: dict[str, Any], mode: str) -> float | None:
    mean = row.get("mean_e2e_us")
    p90 = row.get("p90_e2e_us")
    stddev = row.get("stddev_e2e_us")
    if mean is None:
        return None
    if mode == "mean":
        return float(mean)
    if mode == "p90":
        return float(p90 if p90 is not None else mean)
    return float(mean) + 0.2 * float(p90 if p90 is not None else mean) + 0.1 * float(stddev or 0.0)


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], score_mode: str) -> dict[str, Any]:
    base_mean = next((r["mean_e2e_us"] for r in rows if r["source"] == "base"), None)

    for row in rows:
        row["speedup_vs_base"] = base_mean / row["mean_e2e_us"] if base_mean and row["mean_e2e_us"] else None
        row["score"] = _score(row, score_mode)

    ranked = sorted(
        [r for r in rows if r["source"] == "pipeline" and r["fail_count"] == 0 and r["score"] is not None],
        key=lambda r: r["score"],
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", "")

    fieldnames = [
        "strategy_id",
        "strategy_name",
        "source",
        "repeat",
        "success_count",
        "fail_count",
        "mean_e2e_us",
        "median_e2e_us",
        "p90_e2e_us",
        "p99_e2e_us",
        "stddev_e2e_us",
        "speedup_vs_base",
        "score",
        "rank",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    best = ranked[0] if ranked else None
    best_payload = {
        "best_strategy_id": best["strategy_id"] if best else None,
        "best_strategy_name": best["strategy_name"] if best else None,
        "score_mode": score_mode,
        "speedup_vs_base": best.get("speedup_vs_base") if best else None,
        "reason": "lowest score among successful pipeline runs" if best else "no successful pipeline runs",
    }
    (output_dir / "best_strategy.json").write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Pipeline Benchmark Summary", ""]
    lines.append(f"- best_strategy: {best_payload['best_strategy_name']}")
    lines.append("")
    lines.append("| strategy | source | success | mean_e2e_us | speedup_vs_base | score | rank |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['strategy_name']} | {row['source']} | {row['success_count']}/{row['repeat']} | "
            f"{row.get('mean_e2e_us')} | {row.get('speedup_vs_base')} | {row.get('score')} | {row.get('rank')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_payload


def _read_case_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    stdout_path = run_dir / "stdout.log"
    if not stdout_path.exists():
        return {}
    return parse_case_metrics(stdout_path.read_text(encoding="utf-8", errors="replace"))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.2f}%"


def _delta(strategy: dict[str, Any], base: dict[str, Any], key: str) -> float | None:
    left = strategy.get(key)
    right = base.get(key)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def _first_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _o_complete_us(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("o2_wall_us", "o2_active_trimmed_us", "o2_active_us", "o1_active_trimmed_us", "o1_active_us"))


def _base_o_complete_us(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("o1_active_trimmed_us", "o1_active_us", "orch_trimmed_us", "orch_us"))


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _classify_delta(delta: float | None, threshold_us: float = 1.0) -> str:
    if delta is None:
        return "missing"
    if delta < -threshold_us:
        return "improved"
    if delta > threshold_us:
        return "regressed"
    return "flat"


def _collect_case_comparison(output_dir: Path, strategies: list[int], repeat: int) -> dict[str, Any]:
    base_by_run: dict[int, dict[str, dict[str, Any]]] = {}
    strategy_by_run: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}

    for run_index in range(1, repeat + 1):
        base_by_run[run_index] = _read_case_metrics(output_dir / "base" / f"run_{run_index:03d}")
        for strategy in strategies:
            strategy_by_run.setdefault(strategy, {})[run_index] = _read_case_metrics(
                output_dir / f"strategy_{strategy}" / f"run_{run_index:03d}"
            )

    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for run_index in range(1, repeat + 1):
            base_cases = base_by_run.get(run_index, {})
            strategy_cases = strategy_by_run.get(strategy, {}).get(run_index, {})
            case_names = sorted(set(base_cases) | set(strategy_cases))
            for case_name in case_names:
                base = base_cases.get(case_name, {})
                current = strategy_cases.get(case_name, {})
                base_o_complete = _base_o_complete_us(base)
                strategy_o_complete = _o_complete_us(current)
                rows.append(
                    {
                        "strategy_id": strategy,
                        "strategy_name": STRATEGY_NAMES.get(strategy, str(strategy)),
                        "run_index": run_index,
                        "case": case_name,
                        "base": base,
                        "strategy": current,
                        "derived": {
                            "base_o_complete_us": base_o_complete,
                            "strategy_o_complete_us": strategy_o_complete,
                            "delta_o_complete_us": _subtract(strategy_o_complete, base_o_complete),
                        },
                        "delta": {
                            key: _delta(current, base, key)
                            for key in (
                                "e2e_us",
                                "total_trimmed_us",
                                "sched_us",
                                "sched_trimmed_us",
                                "orch_us",
                                "orch_trimmed_us",
                                "o1_active_us",
                                "o1_active_trimmed_us",
                                "o2_active_us",
                                "o2_active_trimmed_us",
                                "o2_wall_us",
                                "pipe_enq_us",
                                "pipe_flush_us",
                                "o4_front_cost_us",
                                "o4_submitter_cost_us",
                                "o4_tensormap_cost_us",
                                "o4_updater_cost_us",
                                "o_unclassified_active_us",
                                "o_alloc_tensors_us",
                                "o_submit_wrapper_us",
                                "o_scope_end_us",
                                "o_p_func_us",
                            )
                        },
                    }
                )
    return {"rows": rows}


def _write_case_comparison(output_dir: Path, strategies: list[int], repeat: int) -> dict[str, Any]:
    comparison = _collect_case_comparison(output_dir, strategies, repeat)
    rows = comparison["rows"]
    (output_dir / "case_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    csv_fields = [
        "strategy_id",
        "run_index",
        "case",
        "base_total_trim",
        "strategy_total_trim",
        "delta_total_trim",
        "base_o_complete",
        "strategy_o_complete",
        "delta_o_complete",
        "o_complete_status",
        "total_status",
        "base_sched_trim",
        "strategy_sched_trim",
        "delta_sched_trim",
        "base_o1_trim",
        "strategy_o1_trim",
        "delta_o1_trim",
        "base_o2_trim",
        "strategy_o2_trim",
        "delta_o2_trim",
        "base_o2_wall",
        "strategy_o2_wall",
        "delta_o2_wall",
        "base_pipe_enq",
        "strategy_pipe_enq",
        "delta_pipe_enq",
        "base_pipe_flush",
        "strategy_pipe_flush",
        "delta_pipe_flush",
        "base_pipe_counts",
        "strategy_pipe_counts",
    ]
    with (output_dir / "case_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            base = row["base"]
            strategy = row["strategy"]
            delta = row["delta"]
            derived = row["derived"]
            writer.writerow(
                {
                    "strategy_id": row["strategy_id"],
                    "run_index": row["run_index"],
                    "case": row["case"],
                    "base_total_trim": base.get("total_trimmed_us", ""),
                    "strategy_total_trim": strategy.get("total_trimmed_us", ""),
                    "delta_total_trim": delta.get("total_trimmed_us", ""),
                    "base_o_complete": derived.get("base_o_complete_us", ""),
                    "strategy_o_complete": derived.get("strategy_o_complete_us", ""),
                    "delta_o_complete": derived.get("delta_o_complete_us", ""),
                    "o_complete_status": _classify_delta(derived.get("delta_o_complete_us")),
                    "total_status": _classify_delta(delta.get("total_trimmed_us")),
                    "base_sched_trim": base.get("sched_trimmed_us", ""),
                    "strategy_sched_trim": strategy.get("sched_trimmed_us", ""),
                    "delta_sched_trim": delta.get("sched_trimmed_us", ""),
                    "base_o1_trim": base.get("o1_active_trimmed_us", ""),
                    "strategy_o1_trim": strategy.get("o1_active_trimmed_us", ""),
                    "delta_o1_trim": delta.get("o1_active_trimmed_us", ""),
                    "base_o2_trim": base.get("o2_active_trimmed_us", ""),
                    "strategy_o2_trim": strategy.get("o2_active_trimmed_us", ""),
                    "delta_o2_trim": delta.get("o2_active_trimmed_us", ""),
                    "base_o2_wall": base.get("o2_wall_us", ""),
                    "strategy_o2_wall": strategy.get("o2_wall_us", ""),
                    "delta_o2_wall": delta.get("o2_wall_us", ""),
                    "base_pipe_enq": base.get("pipe_enq_us", ""),
                    "strategy_pipe_enq": strategy.get("pipe_enq_us", ""),
                    "delta_pipe_enq": delta.get("pipe_enq_us", ""),
                    "base_pipe_flush": base.get("pipe_flush_us", ""),
                    "strategy_pipe_flush": strategy.get("pipe_flush_us", ""),
                    "delta_pipe_flush": delta.get("pipe_flush_us", ""),
                    "base_pipe_counts": base.get("pipe_counts", ""),
                    "strategy_pipe_counts": strategy.get("pipe_counts", ""),
                }
            )

    lines = [
        "",
        "================================================================",
        "  Unified Case Comparison",
        "================================================================",
        "",
        "Fixed review order:",
        "1. Full sweep: base versus selected strategy for every case; flag any non-target Total regressions.",
        "2. Target case: compare O completion first (base O versus strategy O1+O2 completion).",
        "3. Then inspect Total to separate O-side wins from scheduler/device tail.",
        "",
        "O completion and Total comparison:",
    ]
    header = (
        f"{'Strategy':>8}  {'Case':<44}  {'BaseO':>8}  {'StratO':>8}  {'dO':>8}  {'OStat':>9}  "
        f"{'BaseTot':>8}  {'StratTot':>8}  {'dTot':>8}  {'TotStat':>9}  {'PipeFlush':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        base = row["base"]
        strategy = row["strategy"]
        delta = row["delta"]
        derived = row["derived"]
        lines.append(
            f"{row['strategy_id']:>8}  {row['case']:<44}  "
            f"{_fmt(derived.get('base_o_complete_us')):>8}  {_fmt(derived.get('strategy_o_complete_us')):>8}  "
            f"{_fmt(derived.get('delta_o_complete_us')):>8}  "
            f"{_classify_delta(derived.get('delta_o_complete_us')):>9}  "
            f"{_fmt(base.get('total_trimmed_us')):>8}  {_fmt(strategy.get('total_trimmed_us')):>8}  {_fmt(delta.get('total_trimmed_us')):>8}  "
            f"{_classify_delta(delta.get('total_trimmed_us')):>9}  "
            f"{_fmt(strategy.get('pipe_flush_us')):>10}"
        )

    lines.extend(["", "Raw O split comparison:"])
    header = (
        f"{'Strategy':>8}  {'Case':<44}  {'BaseO1':>8}  {'StratO1':>8}  {'dO1':>8}  "
        f"{'StratO2A':>9}  {'StratO2W':>9}  {'PipeEnq':>8}  {'PipeFlush':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        base = row["base"]
        strategy = row["strategy"]
        delta = row["delta"]
        lines.append(
            f"{row['strategy_id']:>8}  {row['case']:<44}  "
            f"{_fmt(base.get('o1_active_trimmed_us')):>8}  {_fmt(strategy.get('o1_active_trimmed_us')):>8}  "
            f"{_fmt(delta.get('o1_active_trimmed_us')):>8}  "
            f"{_fmt(strategy.get('o2_active_trimmed_us')):>9}  {_fmt(strategy.get('o2_wall_us')):>9}  "
            f"{_fmt(strategy.get('pipe_enq_us')):>8}  {_fmt(strategy.get('pipe_flush_us')):>10}"
        )

    lines.extend(["", "O4 cost comparison:"])
    header = (
        f"{'Strategy':>8}  {'Case':<44}  {'dFront':>8}  {'dSubmit':>8}  {'dTM':>8}  {'dUpdate':>8}  "
        f"{'BaseFront':>9}  {'StratFront':>10}  {'BaseSubmit':>10}  {'StratSubmit':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        base = row["base"]
        strategy = row["strategy"]
        delta = row["delta"]
        lines.append(
            f"{row['strategy_id']:>8}  {row['case']:<44}  "
            f"{_fmt(delta.get('o4_front_cost_us')):>8}  {_fmt(delta.get('o4_submitter_cost_us')):>8}  "
            f"{_fmt(delta.get('o4_tensormap_cost_us')):>8}  {_fmt(delta.get('o4_updater_cost_us')):>8}  "
            f"{_fmt(base.get('o4_front_cost_us')):>9}  {_fmt(strategy.get('o4_front_cost_us')):>10}  "
            f"{_fmt(base.get('o4_submitter_cost_us')):>10}  {_fmt(strategy.get('o4_submitter_cost_us')):>11}"
        )

    lines.extend(["", "O detail comparison:"])
    header = (
        f"{'Strategy':>8}  {'Case':<44}  {'dUnclass':>8}  {'dAlloc':>8}  {'dWrap':>8}  "
        f"{'dScope':>8}  {'dPFunc':>8}  {'BasePFunc':>9}  {'StratPFunc':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        base = row["base"]
        strategy = row["strategy"]
        delta = row["delta"]
        lines.append(
            f"{row['strategy_id']:>8}  {row['case']:<44}  "
            f"{_fmt(delta.get('o_unclassified_active_us')):>8}  {_fmt(delta.get('o_alloc_tensors_us')):>8}  "
            f"{_fmt(delta.get('o_submit_wrapper_us')):>8}  {_fmt(delta.get('o_scope_end_us')):>8}  "
            f"{_fmt(delta.get('o_p_func_us')):>8}  {_fmt(base.get('o_p_func_us')):>9}  {_fmt(strategy.get('o_p_func_us')):>10}"
        )
    lines.append("")
    text = "\n".join(lines)
    (output_dir / "case_comparison.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return comparison


def _summary_metric_value(row: dict[str, Any], side: str, metric_key: str) -> float | None:
    if metric_key == "o_complete":
        derived = row["derived"]
        key = "base_o_complete_us" if side == "base" else "strategy_o_complete_us"
        value = derived.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    source = row[side]
    if metric_key == "total":
        return _first_number(source, ("total_trimmed_us", "total_us", "e2e_us"))
    if metric_key == "sched":
        return _first_number(source, ("sched_trimmed_us", "sched_us"))
    return None


def _summary_metric_stats(case_rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    base_values: list[float] = []
    strategy_values: list[float] = []
    skipped_rows = 0
    for row in case_rows:
        base_value = _summary_metric_value(row, "base", metric_key)
        strategy_value = _summary_metric_value(row, "strategy", metric_key)
        if base_value is None or strategy_value is None:
            skipped_rows += 1
            continue
        base_values.append(base_value)
        strategy_values.append(strategy_value)

    base_avg = statistics.fmean(base_values) if base_values else None
    strategy_avg = statistics.fmean(strategy_values) if strategy_values else None
    diff = strategy_avg - base_avg if base_avg is not None and strategy_avg is not None else None
    ratio = (base_avg - strategy_avg) / base_avg if diff is not None and base_avg else None
    return {
        "base_us": base_avg,
        "strategy_us": strategy_avg,
        "strategy_minus_base_us": diff,
        "improvement_ratio": ratio,
        "matched_rows": len(base_values),
        "skipped_rows": skipped_rows,
    }


def _write_single_strategy_final_summary(
    output_dir: Path, strategies: list[int], comparison: dict[str, Any]
) -> None:
    if len(strategies) != 1:
        return

    strategy_id = strategies[0]
    rows = [row for row in comparison["rows"] if row["strategy_id"] == strategy_id]
    if not rows:
        return

    metric_specs = [
        ("o_complete", "OComplete(us)"),
        ("sched", "Sched/S(us)"),
        ("total", "Total(us)"),
    ]

    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(row["case"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for case_name in sorted(rows_by_case):
        case_rows = rows_by_case[case_name]
        summary_rows.append(
            {
                "case": case_name,
                "metrics": {
                    metric_key: _summary_metric_stats(case_rows, metric_key)
                    for metric_key, _ in metric_specs
                },
            }
        )

    payload = {
        "strategy_id": strategy_id,
        "strategy_name": STRATEGY_NAMES.get(strategy_id, str(strategy_id)),
        "definition": "Each case is reported separately. Repeat rows for the same case are averaged. "
        "Total/S use trimmed values when available. "
        "Diff is Strategy-Base. Ratio is (Base-Strategy)/Base. "
        "A negative Diff and positive Ratio mean the strategy is lower/faster.",
        "rows": summary_rows,
    }
    (output_dir / "final_base_strategy_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "",
        "================================================================",
        "  Final Base vs Strategy Summary",
        "================================================================",
        f"Strategy: {strategy_id} ({payload['strategy_name']})",
        "Scope: each case is reported separately; repeat rows for the same case are averaged.",
        "Total/S use trimmed values when available.",
        "Definition: Diff = Strategy-Base; Ratio = (Base-Strategy)/Base.",
        "Interpretation: negative Diff and positive Ratio mean the strategy is lower/faster.",
        "",
    ]
    group_width = 45
    group_header = (
        f"{'Case':<44}  "
        f"{'OComplete(us)':^{group_width}}  "
        f"{'Sched/S(us)':^{group_width}}  "
        f"{'Total(us)':^{group_width}}"
    )
    header = (
        f"{'Case':<44}  "
        f"{'Base':>10}  {'Strategy':>10}  {'Diff':>10}  {'Ratio':>9}  "
        f"{'Base':>10}  {'Strategy':>10}  {'Diff':>10}  {'Ratio':>9}  "
        f"{'Base':>10}  {'Strategy':>10}  {'Diff':>10}  {'Ratio':>9}"
    )
    lines.append(group_header)
    lines.append(header)
    lines.append("-" * max(len(group_header), len(header)))
    for row in summary_rows:
        metrics = row["metrics"]
        o_metric = metrics["o_complete"]
        s_metric = metrics["sched"]
        t_metric = metrics["total"]
        lines.append(
            f"{row['case']:<44}  "
            f"{_fmt(o_metric['base_us']):>10}  {_fmt(o_metric['strategy_us']):>10}  "
            f"{_fmt(o_metric['strategy_minus_base_us']):>10}  {_fmt_pct(o_metric['improvement_ratio']):>9}  "
            f"{_fmt(s_metric['base_us']):>10}  {_fmt(s_metric['strategy_us']):>10}  "
            f"{_fmt(s_metric['strategy_minus_base_us']):>10}  {_fmt_pct(s_metric['improvement_ratio']):>9}  "
            f"{_fmt(t_metric['base_us']):>10}  {_fmt(t_metric['strategy_us']):>10}  "
            f"{_fmt(t_metric['strategy_minus_base_us']):>10}  {_fmt_pct(t_metric['improvement_ratio']):>9}"
        )
    lines.append("")
    text = "\n".join(lines)
    (output_dir / "final_base_strategy_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--pipeline-dir", default=".")
    parser.add_argument("--strategies", default="0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score", choices=["mean", "p90", "balanced"], default="balanced")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="stream child command stdout/stderr to the console while still saving logs",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    pipeline_dir = Path(args.pipeline_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else pipeline_dir / "results" / "pipeline" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "base_dir": str(base_dir),
        "pipeline_dir": str(pipeline_dir),
        "strategies": args.strategies,
        "repeat": args.repeat,
        "cmd": args.cmd,
        "score": args.score,
        "env": {
            "SIMPLER_BLOCK_DIM": os.environ.get("SIMPLER_BLOCK_DIM", ""),
            "SIMPLER_AICPU_THREAD_NUM": os.environ.get("SIMPLER_AICPU_THREAD_NUM", ""),
            "SIMPLER_PIPELINE_DEFER_SUBMIT": os.environ.get("SIMPLER_PIPELINE_DEFER_SUBMIT", ""),
            "BENCH_BASELINE_CASES": os.environ.get("BENCH_BASELINE_CASES", ""),
            "PTO2_ORCH_TO_SCHED": os.environ.get("PTO2_ORCH_TO_SCHED", ""),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"pipeline benchmark output_dir={output_dir}", flush=True)
    print(f"command={args.cmd}", flush=True)
    print(f"env={json.dumps(run_config['env'], sort_keys=True)}", flush=True)

    all_rows: list[dict[str, Any]] = []
    base_metrics: list[dict[str, Any]] = []
    for run_index in range(1, args.repeat + 1):
        run_dir = output_dir / "base" / f"run_{run_index:03d}"
        base_env = os.environ.copy()
        base_env.pop("SIMPLER_PIPELINE_STRATEGY", None)
        print(f"start base run={run_index}/{args.repeat} log_dir={run_dir}", flush=True)
        rc = _run_once(args.cmd, base_dir, base_env, run_dir, args.stream_output)
        metrics = parse_run(run_dir, None, "base", run_index, rc)
        write_metrics(run_dir / "metrics.json", metrics)
        base_metrics.append(metrics)
        print(f"done base run={run_index} rc={rc} success={metrics['success']}", flush=True)
    all_rows.append(_aggregate(base_metrics, None, "base"))

    strategies = _parse_strategies(args.strategies)
    for strategy in strategies:
        strategy_metrics: list[dict[str, Any]] = []
        for run_index in range(1, args.repeat + 1):
            run_dir = output_dir / f"strategy_{strategy}" / f"run_{run_index:03d}"
            env = os.environ.copy()
            env["SIMPLER_PIPELINE_STRATEGY"] = str(strategy)
            print(
                f"start strategy={strategy} run={run_index}/{args.repeat} "
                f"log_dir={run_dir}",
                flush=True,
            )
            rc = _run_once(args.cmd, pipeline_dir, env, run_dir, args.stream_output)
            metrics = parse_run(run_dir, strategy, "pipeline", run_index, rc)
            write_metrics(run_dir / "metrics.json", metrics)
            strategy_metrics.append(metrics)
            print(f"done strategy={strategy} run={run_index} rc={rc} success={metrics['success']}", flush=True)
        all_rows.append(_aggregate(strategy_metrics, strategy, "pipeline"))

    best = _write_summary(output_dir, all_rows, args.score)
    comparison = _write_case_comparison(output_dir, strategies, args.repeat)
    print(json.dumps(best, indent=2))
    _write_single_strategy_final_summary(output_dir, strategies, comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
