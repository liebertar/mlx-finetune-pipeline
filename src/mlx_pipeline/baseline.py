"""Zero-shot test loss of the raw base model (no adapter) — the reference the
quality gate compares tuned models against.

Run as a subprocess by the evaluate node:
  python -m mlx_pipeline.baseline --model <hf-id> --data <dataset-dir> [--mask-prompt]
Prints exactly one parseable line:  Base loss X.XXX
"""

import argparse
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mask-prompt", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # imports deferred so --help works without loading MLX
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import evaluate as tuner_evaluate
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    dataset_args = types.SimpleNamespace(
        data=args.data, train=False, test=True, hf_dataset=None,
        mask_prompt=args.mask_prompt, max_seq_length=args.max_seq_length,
    )
    _, _, test_set = load_dataset(dataset_args, tokenizer)
    loss = tuner_evaluate(
        model=model, dataset=CacheDataset(test_set), batch_size=args.batch_size,
        num_batches=-1, max_seq_length=args.max_seq_length,
    )
    print(f"Base loss {float(loss):.3f}")


if __name__ == "__main__":
    main()
