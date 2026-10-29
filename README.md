# Depth OPD recipes

Minimal research code for native recurrent TinyLlama SFT and depth on-policy distillation (OPD) with a frozen maximum-depth anchor. No privileged-information teacher, infrastructure, checkpoints, results, or experiment notes.

| Config in `configs/` | Python module | Shallow objective / history |
| --- | --- | --- |
| `sft.json` | `train` | D4 supervised cross-entropy |
| `fixed-d1-forward-kl.json` | `experiments.opd_preservation_s42.runner` | D1 on its own sampled history |
| `fixed-d12-forward-kl.json` | `experiments.opd_preservation_d12.runner` | Equal D1/D2 mean, independent sampled histories |
| `sampled123-forward-kl.json` | `experiments.opd_sampled_preservation.runner` | Uniform D1/D2/D3, one depth per optimizer update |
| `shared-d1-history-forward-kl.json` | `experiments.opd_shared_d1_d2.runner` | Equal D1/D2 mean on one D1 sampled history |
| `sampled123-reverse-kl.json` | `experiments.opd_reverse_kl.runner` | Sampled D1/D2/D3, reverse shallow KL |

All OPD recipes start student and frozen teacher from the same final SFT64 checkpoint. Forward KL means KL(teacher || student); reverse means KL(student || teacher), summed over the full vocabulary. Every OPD recipe adds a unit-weight **forward-KL D4 anchor on a separately sampled student-D4 history**, including the shared-shallow-history recipe. Response tokens are averaged per sequence, then across examples; genuine EOS is included and prompts/padding are masked. Training has no reference CE or gold-solution teacher context. SFT uses the training solutions.

The common settings are seed 42, 1,024 training questions, batch 16/microbatch 1, 64 updates per stage, core plus input-adapter training, AdamW8bit, LR 5e-6, eight-update warmup and cosine decay, T=1 categorical rollouts capped at 512 tokens, and greedy evaluation at D1/D2/D4 on 256 validation questions. Method-specific rollout costs differ; equal updates do not imply equal compute.

## Use

Use Linux with CUDA and sufficient GPU/CPU memory for a student, frozen teacher, and dense vocabulary targets. Install `requirements.txt` in a fresh environment. Set `PYTHONPATH=src` and run commands from the repository root.

Download `smcleish/Recurrent-TinyLlama-3T-train-recurrence-4` at revision `80ae03166c4a3783ae74c8c17226974790a1dd19` into `models/tinyllama`. The model uses upstream custom code (`trust_remote_code=True`); use the pinned revision. Obtain the original `train.jsonl` and `test.jsonl` from [GSM8K](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data) in `data/gsm8k`. Loading verifies their SHA256 hashes and the supplied index/hash split manifest. Test rows are encoded for the historical data audit; these GSM recipes evaluate validation, with no test generation or test-based selection.

```sh
export PYTHONPATH=src
export WANDB_MODE=disabled
python -m train --config configs/sft.json \
  --model-dir models/tinyllama --tokenizer-dir models/tinyllama \
  --data-dir data/gsm8k --output outputs/sft
```

For OPD, choose the module/config from the table. Set `SFT_SHA` to the SHA256 of `outputs/sft/checkpoint.pt`. Run its native parity/gradient/memory gate in a separate output directory, then train from the original SFT checkpoint; gate weights are discarded.

```sh
MODULE=experiments.opd_sampled_preservation.runner
CONFIG=configs/sampled123-forward-kl.json
python -m "$MODULE" --config "$CONFIG" --model-dir models/tinyllama \
  --data-dir data/gsm8k --output outputs/opd-gate --gate-only \
  --student-checkpoint outputs/sft/checkpoint.pt --student-sha "$SFT_SHA" \
  --teacher-checkpoint outputs/sft/checkpoint.pt --teacher-sha "$SFT_SHA"
python -m "$MODULE" --config "$CONFIG" --model-dir models/tinyllama \
  --data-dir data/gsm8k --output outputs/opd --gate outputs/opd-gate/memory-gate.json \
  --student-checkpoint outputs/sft/checkpoint.pt --student-sha "$SFT_SHA" \
  --teacher-checkpoint outputs/sft/checkpoint.pt --teacher-sha "$SFT_SHA"
```

For reverse KL, also pass `--start-predictions outputs/sft/predictions.jsonl` to training for the same-start evaluation check. Resume interrupted runs with their original output directory and `--resume PATH/recovery.pt`, preserving matching logs/config/source. This curated source release is for new reproductions; its source identities differ from historical operational packages.

W&B is optional. OPD enables it when `WANDB_API_KEY` is present; set `WANDB_ENTITY` and optionally `WANDB_PROJECT`. SFT defaults to offline logging unless disabled as above.

This is a source extraction of the recorded recipes, with common modules deduplicated and account-specific paths removed. CPU objective/gradient, RNG, EOS/mask, import, and CLI checks were performed; no full GPU reproduction was run for this release. Hyperparameters were held fixed across these recipes rather than swept independently.
