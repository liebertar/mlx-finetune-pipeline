from mlx_pipeline.steps.gate import check_gate


def test_passes_when_loss_below_max():
    passed, failures = check_gate({"eval_loss": 1.8}, {"max_eval_loss": 2.5})
    assert passed and not failures


def test_fails_when_loss_above_max():
    passed, failures = check_gate({"eval_loss": 3.0}, {"max_eval_loss": 2.5})
    assert not passed
    assert "eval_loss" in failures[0]


def test_fails_when_metric_missing():
    passed, _ = check_gate({}, {"max_eval_loss": 2.5})
    assert not passed


def test_accuracy_threshold():
    assert check_gate({"eval_accuracy": 0.9}, {"min_eval_accuracy": 0.7})[0]
    assert not check_gate({"eval_accuracy": 0.5}, {"min_eval_accuracy": 0.7})[0]


def test_no_thresholds_always_passes():
    assert check_gate({"eval_loss": 99.0}, {})[0]


def test_baseline_gate_passes_when_tuning_helped():
    metrics = {"eval_loss": 4.3, "baseline_loss": 5.5}
    assert check_gate(metrics, {"must_beat_baseline": True})[0]


def test_baseline_gate_fails_when_tuning_hurt():
    metrics = {"eval_loss": 6.6, "baseline_loss": 5.5}
    passed, failures = check_gate(metrics, {"must_beat_baseline": True})
    assert not passed
    assert "did not beat baseline" in failures[0]


def test_baseline_gate_respects_margin():
    metrics = {"eval_loss": 5.45, "baseline_loss": 5.5}
    assert check_gate(metrics, {"must_beat_baseline": True})[0]
    assert not check_gate(metrics, {"must_beat_baseline": True, "baseline_margin": 0.2})[0]


def test_baseline_gate_fails_without_baseline_metric():
    passed, failures = check_gate({"eval_loss": 4.0}, {"must_beat_baseline": True})
    assert not passed
    assert "no baseline_loss" in failures[0]
