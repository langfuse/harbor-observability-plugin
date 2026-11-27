"""Export finished Harbor jobs to Langfuse: harbor-langfuse export jobs/<job> [--dataset NAME] [--run-name NAME]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import LangfuseClient
from .export import export_trial, job_name_of, run_scores, trial_dirs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harbor-langfuse", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="send a finished job's trials to Langfuse as an experiment run")
    e.add_argument("job_dirs", nargs="+", type=Path)
    e.add_argument("--dataset", help="Langfuse dataset name (default: the trials' Harbor dataset)")
    e.add_argument("--run-name", help="experiment run name (default: <job> (<agent>, <model>))")
    e.add_argument("--tags", default="", help="comma-separated trace tags")
    args = p.parse_args(argv)
    client = LangfuseClient()
    client.check()
    datasets: dict[str, str] = {}

    def ensure(name: str) -> str:
        if name not in datasets:
            datasets[name] = client.ensure_dataset(name, description="Harbor tasks, one item per task (created by harbor-langfuse)")
        return datasets[name]

    exports = []
    for job_dir in args.job_dirs:
        name = job_name_of(job_dir)
        for trial in trial_dirs(job_dir):
            ex = export_trial(client, trial, job_name=name, dataset=args.dataset, run_name=args.run_name,
                              tags=[t for t in args.tags.split(",") if t], ensure_dataset=ensure)
            exports.append(ex)
            print(f"{ex.trial}: trace {ex.trace_id} -> run '{ex.experiment_name}' {ex.rewards or ex.error or ''}")
    run_scores(client, exports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
