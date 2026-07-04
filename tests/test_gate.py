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
