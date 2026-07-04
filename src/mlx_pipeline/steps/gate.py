"""Node 5 — gate: decide whether the model is good enough to publish.

Rules come from the `gate:` section of the config:
  must_beat_baseline -> eval_loss must be below the base model's zero-shot
                        loss on the same split (minus optional baseline_margin).
                        Self-calibrating: "did tuning actually help?"
  min_<metric>       -> metrics["<metric>"] must be >= this (e.g. min_f1: 0.93)
  max_<metric>       -> metrics["<metric>"] must be <= this (e.g. max_eval_loss: 2.5)

The verdict (and each failure reason) is recorded in MLflow so runs are
auditable even after they're pruned locally.
"""

import mlflow
from zenml import step

RESERVED = {"must_beat_baseline", "baseline_margin"}


def check_gate(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    failures = []
    if thresholds.get("must_beat_baseline"):
        eval_loss = metrics.get("eval_loss", float("inf"))
        baseline = metrics.get("baseline_loss")
        margin = thresholds.get("baseline_margin", 0.0)
        if baseline is None:
            failures.append("must_beat_baseline set but no baseline_loss metric recorded")
        elif eval_loss > baseline - margin:
            failures.append(
                f"eval_loss {eval_loss} did not beat baseline {baseline}"
                + (f" by margin {margin}" if margin else "")
            )
    for key, threshold in thresholds.items():
        if key in RESERVED:
            continue
        if key.startswith("min_"):
            metric = key[4:]
            value = metrics.get(metric, 0.0)
            if value < threshold:
                failures.append(f"{metric} {value} < min {threshold}")
        elif key.startswith("max_"):
            metric = key[4:]
            value = metrics.get(metric, float("inf"))
            if value > threshold:
                failures.append(f"{metric} {value} > max {threshold}")
        else:
            failures.append(f"unknown gate rule: {key}")
    return (not failures, failures)


# no cache: the verdict must be logged to MLflow on every run
@step(enable_cache=False)
def gate(metrics: dict, thresholds: dict) -> bool:
    passed, failures = check_gate(metrics, thresholds)
    mlflow.log_metric("gate_passed", int(passed))
    if failures:
        mlflow.set_tag("gate_failures", "; ".join(failures))
        raise RuntimeError("quality gate failed: " + "; ".join(failures))
    return True
