# Looped OPSD

SFT and depth-based on-policy self-distillation (OPSD), using recurrent TinyLlama from the model family introduced in [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384).

[Article: Boosting On-Policy Self-Distillation](https://anantg.xyz/writing/boosting-on-policy-self-distillation/)

| Config (`configs/`) | Runner (`python -m …`) |
| --- | --- |
| `sft.json` | `train` |
| `fixed-d1-forward-kl.json` | `experiments.opsd_preservation_s42.runner` |
| `fixed-d12-forward-kl.json` | `experiments.opsd_preservation_d12.runner` |
| `sampled123-forward-kl.json` | `experiments.opsd_sampled_preservation.runner` |
| `shared-d1-history-forward-kl.json` | `experiments.opsd_shared_d1_d2.runner` |
| `sampled123-reverse-kl.json` | `experiments.opsd_reverse_kl.runner` |

Student and frozen teacher start from the same SFT64 checkpoint. Each OPSD recipe adds a unit-weight forward-KL D4 anchor on a separate student-D4 rollout. Shared history means D1/D2 share a D1 rollout. Reverse KL changes only the shallow loss. All KL losses use the full vocabulary and average response tokens per sequence, including genuine EOS. Hyperparameters are held fixed; no method-specific sweeps.

## Run

Install `requirements.txt` on Linux/CUDA and set `PYTHONPATH=src`. Download [the model](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-4) at revision `80ae03166c4a3783ae74c8c17226974790a1dd19` into `models/tinyllama`; loading uses its custom Hugging Face code. Place original [GSM8K](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data) `train.jsonl`/`test.jsonl` in `data/gsm8k`. The supplied hashes and split manifest are checked when loading.

```sh
export PYTHONPATH=src WANDB_MODE=disabled
python -m train --config configs/sft.json --model-dir models/tinyllama --tokenizer-dir models/tinyllama --data-dir data/gsm8k --output outputs/sft
```

For OPSD, use the runner/config above and supply `--model-dir`, `--data-dir`, `--output`, plus both `--student-checkpoint` and `--teacher-checkpoint` pointing to `outputs/sft/checkpoint.pt`, with its SHA256 in `--student-sha` and `--teacher-sha`. First run `--gate-only` in a separate output directory; then train with `--gate PATH/memory-gate.json`. Reverse KL also requires `--start-predictions outputs/sft/predictions.jsonl`. See each runner's `--help`.

Resume with `--resume PATH/recovery.pt` and matching source/config/logs. W&B is optional (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`). This source extraction passed CPU objective/gradient and import/CLI checks; no full GPU reproduction was run for the release.
