# Qwen3-0.6B-Base Post-Training

Compare **base → SFT**, **SFT → GRPO (RLVR)** and **SFT → response distillation** on the same verifiable reasoning benchmark.

### Fixed splits

- `train`: 90% of the official GSM8K training set, shuffled with `seed=42`. Used for SFT, RLVR, and teacher generation.
- `validation`: remaining 10% of the official training set. Used for debugging, hyperparameters, and early stopping comparisons.
- `test`: official GSM8K test set. Compare every selected checkpoint only after tuning on validation.

## Quickstart

```bash
uv run python -m scripts.prepare_data

# Greedy validation baseline on the untouched base model (200 prompts)
uv run python -m scripts.evaluate --split validation \
  --out results/base_val.json

# Stage 1: Supervised fine-tuning (LoRA adapter)
uv run python scripts/train_sft.py --steps 300
uv run python scripts/evaluate.py --split validation --adapter outputs/sft \
  --out results/sft_val.json

# Stage 2A: Group Relative Policy Optimization with exact-match verifiable reward
uv run python scripts/train_grpo.py --sft outputs/sft --steps 150
uv run python scripts/evaluate.py --split validation --adapter outputs/grpo \
  --out results/grpo_val.json

# Stage 2B: Generate verified, teacher-written solutions from TRAIN prompts only
uv run python scripts/generate_teacher_data.py \
  --teacher Qwen/Qwen3-4B --max-items 500
uv run python scripts/train_distill.py \
  --teacher-jsonl data/teacher_train.jsonl --sft outputs/sft --steps 150
uv run python scripts/evaluate.py --split validation --adapter outputs/distilled \
  --out results/distilled_val.json
```

**After selecting hyperparameters on validation**, evaluate the finalists against the untouched official test set by changing `--split test --max-samples 0` and giving each a unique `--out` path. Default is 200 validation examples as a quick smoke evaluation, _not_ a publishable accuracy score. Always run the full final test set and include confidence intervals if making strong conclusions.

## Algorithms

- **SFT**: LoRA fine-tuning with gold reasoning and final-answer target. Prompt-completion formatting ensures loss is applied to the answer, not the question. Text formatting is deliberately identical across algorithms (no chat template changes).
- **GRPO / RLVR**: 4 student completions per prompt, binary reward on the last `<final>number</final>` tag. Trainer estimates advantages relative to each group of completions. With `beta=0` there is no reference-KL penalty; inspect KL/drift in a follow-up run with nonzero `beta`.
- **Response distillation**: Stronger model generates answer trajectories on **training prompts only**; numerical verifier retains correct ones; the SFT checkpoint is trained on these _teacher responses_. This is **offline hard-target / response distillation**, NOT on-policy KL. Compare against a matched additional SFT control to distinguish benefits from merely more gradient steps.
- **On-policy distillation (future extension)**: Sample rollouts from the student, score teacher token log-probabilities on _those student rollouts_, and minimize teacher/student KL under student-distribution states. See Tinker's on-policy-distillation recipes or the RLHF Book distillation implementation. Requires an additional training loop and compatible teacher scorer; not implemented in this starter.

## Results

| Model         | Val accuracy | Test accuracy | Invalid format | Mean output tokens | Training tokens/cost |
| ------------- | ------------ | ------------- | -------------- | ------------------ | -------------------- |
| Base          | —            | —             | —              | —                  | 0                    |
| SFT           | —            | —             | —              | —                  | —                    |
| SFT + GRPO    | —            | —             | —              | —                  | —                    |
| SFT + distill | —            | —             | —              | —                  | —                    |

training loss, correctness reward, proportion of **zero-variance groups** in GRPO (all candidates right/wrong), sampled output distributions, and at least 10 hand-inspected failure cases. Track output length and format validity so reward hacking cannot masquerade as reasoning improvement.

## References

- [Nathan Lambert's RLHF Book](https://github.com/natolambert/rlhf-book/tree/main/code)
- [Hugging Face TRL SFTTrainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Hugging Face TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer)
- [Tinker recipes for on-policy distillation](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)

## Experiment log

256 vs 512 length for base model eval

256: Accuracy 2%, invalid format: 98%, reached token cap 10/50
512: Accuracy 2%, invalid format: 98%, reached token cap 3/50

So, we need better instruction following with SFT.
