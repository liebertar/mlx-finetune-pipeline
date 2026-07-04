"""Node 5 — gate: decide whether the model is good enough to publish.

Rules come from the `gate:` section of the config:
  must_beat_baseline -> eval_loss must be below the base model's zero-shot
                        loss on the same split (minus optional baseline_margin).
                        Self-calibrating: "did tuning actually help?"
  max_eval_loss      -> metrics["eval_loss"] must be <= this absolute value
  min_eval_accuracy  -> metrics["eval_accuracy"] must be >= this

The verdict (and each failure reason) is recorded in MLflow so runs are
auditable even after they're pruned locally.
"""

import mlflow
from zenml import step


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
    if "max_eval_loss" in thresholds:
        if metrics.get("eval_loss", float("inf")) > thresholds["max_eval_loss"]:
            failures.append(
                f"eval_loss {metrics.get('eval_loss')} > max {thresholds['max_eval_loss']}"
            )
    if "min_eval_accuracy" in thresholds:
        if metrics.get("eval_accuracy", 0.0) < thresholds["min_eval_accuracy"]:
            failures.append(
                f"eval_accuracy {metrics.get('eval_accuracy')} < min {thresholds['min_eval_accuracy']}"
            )
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
