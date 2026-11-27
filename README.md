# Harbor observability plugin for Langfuse

A [Harbor](https://github.com/harbor-framework/harbor) plugin that sends agent evaluation jobs to [Langfuse](https://langfuse.com) as experiment runs, with trajectories as traces and verifier rewards as scores.

| Harbor          | Langfuse                                                                         |
| --------------- | -------------------------------------------------------------------------------- |
| dataset         | dataset, named after the Harbor dataset (for example `terminal-bench/terminal-bench-2-1`) |
| task            | dataset item, the same in every job                                              |
| job             | one experiment run per agent and model (per dataset if the job has several)     |
| trial           | trace: one generation per model call, one tool observation per tool call         |
| verifier reward | score on the trace, plus `mean_<reward>` and `error_rate` on the run             |

Because a task maps to the same dataset item in every job, you can compare runs task by task in Langfuse.

## Install

Install the plugin into the same environment as Harbor:

```bash
uv tool install harbor --with "harbor-langfuse @ git+https://github.com/langfuse/harbor-observability-plugin"
# or, in an existing environment with Harbor:
pip install "git+https://github.com/langfuse/harbor-observability-plugin"
```

`harbor plugins list` now shows `langfuse`.

## Use

Set your Langfuse API keys and add `--plugin langfuse` to `harbor run`:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com" # EU region; US: https://us.cloud.langfuse.com

harbor run -d terminal-bench/terminal-bench-2-1 \
  -a claude-code -m anthropic/claude-sonnet-5-5 \
  --plugin langfuse
```

Each trial is sent when it ends; run-level scores are sent when the job ends. A failed export logs a warning and never fails the job. If you resume a job, only its new trials are sent, and the run-level scores are recalculated over all trials of the job.

Options, passed with `--pk key=value`:

| Option     | Default                                   |
| ---------- | ----------------------------------------- |
| `dataset`  | the Harbor dataset of each trial          |
| `run_name` | `<job name> (<agent>, <model>)`           |
| `tags`     | none; comma-separated, added to each trace |
| `base_url` | `LANGFUSE_BASE_URL`                       |

To send a job that already ran:

```bash
harbor-langfuse export jobs/<job-name>
```

The command exits with a non-zero status if a trial could not be sent; the other trials are still sent.

## What you see per agent

Traces are built from the trajectory the agent writes (`agent/trajectory.json`), converted with [`harbor-atif2otel`](https://pypi.org/project/harbor-atif2otel/). What a trace shows depends on what the agent records:

| Agent                                    | Tool calls  | Tokens and cost |
| ---------------------------------------- | ----------- | --------------- |
| Claude Code, Codex, OpenCode, Terminus-2 | yes         | yes             |
| mini-SWE-agent                           | inputs only | yes             |
| OpenHands SDK                            | yes         | no              |
| Goose                                    | no          | no              |

Trials without a trajectory (for example Aider, or a trial that failed before the agent ran) appear as a single observation, so they still count in the run. The trace of a failed trial is marked as an error, with or without a trajectory. Trajectories don't record tool call durations.

## Details

- Traces use the `sdk-experiment` environment, like experiments run with the Langfuse SDK.
- Agent spans repeat the token usage and cost of their model calls. The plugin removes usage from all non-model spans so Langfuse doesn't count it twice.
- IDs are deterministic: re-sending a job updates the same traces, dataset items and scores.
- A dataset item is only written when it is new or its task changed, so a trial without a trajectory doesn't remove the instruction from the item.
- The dataset name leaves out the dataset version, so all versions of a Harbor dataset share one Langfuse dataset.
- Spans are sent with `x-langfuse-ingestion-version: 4`, which Langfuse needs to create experiment runs from OpenTelemetry spans.

## Development

```bash
uv run --python 3.12 --with pytest pytest -q
```

## License

MIT
