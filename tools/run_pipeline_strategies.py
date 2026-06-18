#!/usr/bin/env python3
"""Run one command across pipeline strategies and write per-run metrics."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from pipeline_metrics import parse_run, write_metrics


def _parse_strategies(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _run_once(cmd: str, cwd: Path, env: dict[str, str], run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        shlex.split(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default="0,5")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--output-dir", default="results/pipeline/manual")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    output_dir = Path(args.output_dir).resolve()
    strategies = _parse_strategies(args.strategies)

    for strategy in strategies:
        for run_index in range(1, args.repeat + 1):
            run_dir = output_dir / f"strategy_{strategy}" / f"run_{run_index:03d}"
            env = os.environ.copy()
            env["SIMPLER_PIPELINE_STRATEGY"] = str(strategy)
            rc = _run_once(args.cmd, workdir, env, run_dir)
            metrics = parse_run(run_dir, strategy, "pipeline", run_index, rc)
            write_metrics(run_dir / "metrics.json", metrics)
            print(f"strategy={strategy} run={run_index} rc={rc} success={metrics['success']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
