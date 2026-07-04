"""Node 5 — gate: decide whether the model is good enough to publish.

Thresholds come from the `gate:` section of the config:
  max_eval_loss     -> metrics["eval_loss"] must be <= this
  min_eval_accuracy -> metrics["eval_accuracy"] must be >= this

The verdict (and each failure reason) is recorded in MLflow so runs are
auditable even after they're pruned locally.
"""

import mlflow
from zenml import step


def check_gate(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    failures = []
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


@step
def gate(metrics: dict, thresholds: dict) -> bool:
    passed, failures = check_gate(metrics, thresholds)
    mlflow.log_metric("gate_passed", int(passed))
    if failures:
        mlflow.set_tag("gate_failures", "; ".join(failures))
        raise RuntimeError("quality gate failed: " + "; ".join(failures))
    return True
