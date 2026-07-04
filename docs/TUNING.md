# Tuning recipes — combinations that actually work

Research-backed hyperparameter combinations for MLX fine-tuning, mapped to this
pipeline's config keys. Sources: the [Unsloth LoRA hyperparameter guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide),
[2026 LoRA/QLoRA guides](https://effloow.com/articles/llm-fine-tuning-lora-qlora-guide-2026),
[MLX-specific writeups](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/), and
the [mlx-lm LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

## TL;DR — recipe per intent

| Intent | method | rank | learning_rate | schedule | iters* | mask_prompt | notes |
|---|---|---|---|---|---|---|---|
| **Instruction / chat SFT** | lora | 16 | 1e-4 | cosine + 10% warmup | ~2–3 epochs | **true** | masking the prompt (train on completions only) gives 1–2% accuracy for free |
| **Domain adaptation** (new vocabulary/knowledge) | lora | 32–64 | 1e-4 | cosine + warmup | 2–3 epochs | false | domain shift needs capacity → higher rank |
| **Style / persona** | lora | 8–16 | 1e-4 | cosine | 1–2 epochs | true | low rank is enough; overfits fast, watch val loss |
| **Preference tuning (DPO/GRPO)** | — | 16 | **5e-6** | linear | 1 epoch | — | RL-style objectives need ~40× lower LR |
| **Embedding models** | lora | 8–16 | 2e-5 | linear | 2–4 epochs | — | contrastive objectives; smaller LR than SFT |
| **Vision-language (mlx-vlm)** | lora/qlora | 8–16 | 1e-5 | constant | 1–2 epochs | — | VLM adapters are more fragile; start low |

\* iters ≈ (dataset_size × epochs) / batch_size. 2,000–5,000 high-quality
examples is the LoRA sweet spot ([InsiderLLM](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/)).

## Parameter-by-parameter

**Learning rate** — the single highest-impact knob. Modern consensus for
LoRA/QLoRA SFT is **2e-4 as the aggressive default, 1e-4 conservative**; the
usable range is 2e-4 → 5e-6 ([Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)).
mlx-lm's built-in default (1e-5) is very conservative — expect slow convergence.

**Rank (r) and scale** — r=16 is the reliable starting point; r=32–64 when the
task involves real domain shift; 2025 research finds intermediate ranks (32–64)
the best capacity/stability balance ([Meta Intelligence](https://www.meta-intelligence.tech/en/insight-lora-finetuning)).
Keep alpha/rank ≥ 1; alpha = 2×r is the common heuristic. **MLX note:** mlx-lm
expresses this as `scale` (≈ alpha/rank), set via `train.rank` + `train.scale`
in our configs (default scale 20.0 is mlx-lm's own default).

**Epochs / iters** — 1–3 epochs; more mostly memorizes. Signs of overfitting:
train loss under ~0.2 while val loss stalls or rises.

**Batch size + gradient accumulation** — target an *effective* batch of 4–16.
On unified memory prefer `batch_size: 2–4` + `grad_accumulation_steps: 4–8`
rather than a big raw batch — same math, ~¼ the memory.

**mask_prompt** — for instruction data, train on completions only. Free
accuracy on multi-turn/chat datasets. Our configs expose `train.mask_prompt`.

**Schedule + warmup** — cosine decay with warmup over 5–10% of steps is the
default recipe; linear is fine too. `train.lr_schedule: cosine` wires this up.

**Dropout / weight decay** — LoRA dropout is "not that useful" for typical runs
(keep 0–0.05); weight decay 0.01–0.1 if you see overfitting.

**Gradient checkpointing** — `train.grad_checkpoint: true` trades ~20–30%
speed for large activation-memory savings; turn it on when a bigger model or
longer `max_seq_length` doesn't fit.

**Quantized base (QLoRA)** — fine-tuning a 4-bit `mlx-community/*-4bit` base
is QLoRA in practice: ~4× less memory, slightly lower ceiling than fp16 LoRA.
On a Mac Studio, prefer fp16 base + LoRA for ≤8B models, 4-bit base above that.

**DoRA** — decomposes updates into magnitude + direction; often converges
better at low rank. `train.method: dora` (mlx-lm supports it natively). The
2026 favorite starting combo is r=16 + DoRA ([n1n.ai](https://explore.n1n.ai/blog/fine-tune-llm-lora-qlora-guide-2026-2026-04-17)).

**Seed** — always pin (`train.seed: 42`) so sweep comparisons mean something.

## Overfitting / underfitting playbook

| Symptom | Do this |
|---|---|
| Train loss < 0.2, val loss flat/rising | lower LR, fewer epochs, weight decay 0.01–0.1, dropout 0.1, halve scale |
| Val loss barely moves | raise LR (short runs), more epochs, raise rank *and* scale together |
| Loss spikes / NaN | lower LR first; check for garbage records (the `ingest` node's schema check helps) |
| Out of memory | grad_checkpoint on, batch 1 + more grad-accum, shorter max_seq_length, 4-bit base |

## Config key → mlx-lm mapping

| our config (`train.*`) | mlx-lm |
|---|---|
| method | `--fine-tune-type {lora,dora,full}` |
| rank, scale, dropout | `-c config.yaml` → `lora_parameters` |
| learning_rate | `--learning-rate` |
| lr_schedule: cosine | `-c config.yaml` → `lr_schedule: cosine_decay` + warmup |
| iters, batch_size | `--iters`, `--batch-size` |
| grad_accumulation_steps | `--grad-accumulation-steps` |
| mask_prompt | `--mask-prompt` |
| grad_checkpoint | `--grad-checkpoint` |
| max_seq_length, num_layers | `--max-seq-length`, `--num-layers` |
| seed | `--seed` |

## Empirical sweeps

Literature gives starting points; your data decides. Run the bundled sweep to
compare combinations on *your* dataset (sequential — MLX saturates the GPU):

```bash
make sweep-llm          # trains the same model over several rank/LR combos,
                        # compares val loss, emits sweep.json + comparison chart
```

The next `make run-llm` picks up the sweep artifacts and publishes them into
the model card (comparison table + chart), so the Hub page records *why* this
combination was chosen.
