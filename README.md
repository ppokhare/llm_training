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

## Experiment log and evaluation protocol

Keep a unique JSON result for every evaluation; do not overwrite earlier runs.
The evaluator saves per-question outputs, generated token counts (including EOS),
EOS presence, token-limit hits, stopping reason, tag counts, and generation time.
Run metadata includes the selected data hash, greedy decoding flag, explicit token
budget, model revision, device/dtype, package versions, Git state, and evaluator /
verifier source hashes. `generation_config` records the model defaults; the
explicit `max_new_tokens`, `do_sample`, and `pad_token_id` fields override them.
A token-limit hit can coincide with EOS; `stop_reason` gives EOS precedence.

The original smoke baseline is `results/base_val_smoke.json`: untouched base,
first 50 validation questions, greedy, 256 new tokens. It predates diagnostic
logging. Retokenization suggests 10/50 reached the cap; raw EOS information is
unavailable. Its strict accuracy is 2% and invalid-format rate is 98%.

The controlled length diagnostic uses the same base model, first 50 validation
questions, prompt, verifier, and greedy decoding, changing only the budget:

```bash
uv run python -m scripts.evaluate --split validation --max-samples 50 \
  --max-new-tokens 512 --out results/base_val_smoke_512.json
```

This does not change the evaluator's 256-token default or GRPO's 256-token rollout
budget. If 512 is selected for subsequent evaluation, pass it explicitly for all
checkpoints and label those comparisons separately from the original baseline.
Use validation to settle the protocol; reserve the official test set for finalists.

For each experiment, append a dated entry below with its hypothesis, exact command,
parent checkpoint, changed settings, result path, key metrics, interpretation, and
next decision. Training scripts now save final trainer state after training. Preserve training
output directories, checkpoint trainer state /
log history, training arguments, and a terminal log. Record loss, learning rate,
reward, output length, and zero-variance reward groups where available; evaluating
checkpoints with this script provides accuracy and format diagnostics separately.
Record training steps and processed tokens when comparing training methods.

W&B is optional. Local JSON and this log are the source of record for this small
comparison. A dashboard becomes useful for comparing multiple training curves or
hyperparameter runs. Training currently uses `report_to="none"`; W&B has not been
installed, enabled, or connected by this change.

### 2026-11-20 — Base output-length diagnostic

- Hypothesis: 256 tokens truncates some solutions, while missing numeric final tags
  account for many other failures.
- Control: original 50-question, greedy base evaluation at 256 tokens, preserved.
- Change: run the command above at 512 tokens with diagnostic logging.
- Result: strict accuracy 2% (1/50), invalid format 98% (49/50), mean
  generated length 227.96 tokens including EOS, 3/50 at the 512-token cap,
  47/50 stopped with EOS; evaluation loop took 154.9 seconds.
- Paired check: all questions/golds match the original. Exactly the ten previously
  capped outputs changed; the other forty outputs are identical. Seven of those
  ten now stop with EOS. Newly finished correct prose answers still lack final tags.
- Interpretation: more budget reduces truncation (estimated 20% to observed 6%)
  without improving strict accuracy. Format compliance remains the main bottleneck;
  no general EOS-stopping failure is evident.
- Decision: recommend explicitly using 512 for subsequent checkpoint comparisons
  to reduce the length confound. Keep the 256 result as the historical baseline;
  the 512 result is the matching baseline for those comparisons. No training or
  GRPO rollout settings were changed.
- Verification: Python compilation passed; checked all 50 records for paired
  question/gold identity, verifier agreement, token budget, and stopping metadata.
