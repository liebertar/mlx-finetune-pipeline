"""Wires the seven nodes into the DAG:

ingest -> prepare -> train -> evaluate -> gate -> fuse -> publish
"""

from zenml import pipeline

from mlx_pipeline.config import PipelineConfig
from mlx_pipeline.steps import evaluate, fuse, gate, ingest, prepare, publish, train


@pipeline
def finetune_pipeline(cfg: PipelineConfig, work_dir: str):
    records = ingest(source=cfg.data["source"], fmt=cfg.data["format"])
    dataset_dir = prepare(
        records=records,
        valid_split=cfg.data.get("valid_split", 0.1),
        test_split=cfg.data.get("test_split", 0.1),
        work_dir=work_dir,
    )
    adapter_path = train(
        flavor=cfg.flavor,
        model=cfg.model,
        dataset_dir=dataset_dir,
        train_cfg=cfg.train,
        work_dir=work_dir,
    )
    metrics = evaluate(
        flavor=cfg.flavor, model=cfg.model, dataset_dir=dataset_dir, adapter_path=adapter_path
    )
    passed = gate(metrics=metrics, thresholds=cfg.gate)
    model_dir = fuse(
        flavor=cfg.flavor, model=cfg.model, adapter_path=adapter_path, work_dir=work_dir,
        after=[passed],
    )
    publish(model_dir=model_dir, base_model=cfg.model, metrics=metrics, publish_cfg=cfg.publish)
