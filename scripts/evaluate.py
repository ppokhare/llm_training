"""Generate greedy answers and save numerical accuracy, format, and length metrics.

Supports base models or saved LoRA adapters. Dataset hashes make comparisons
checkable, and per-question records preserve outputs for failure analysis.
"""

import argparse
import json
import hashlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version

from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.data import (
    FINAL_RE,
    MODEL_ID,
    PROMPT_STYLES,
    canonical_number,
    extract_final,
    has_think_format,
    make_prompt,
    with_prompt_style,
)


def make_record(row, token_ids, tokenizer, eos_ids, max_new_tokens, prompt_tokens, elapsed):
    """Score one unpadded completion; elapsed time covers its whole generation batch."""
    eos_seen = any(token in eos_ids for token in token_ids)
    hit_limit = len(token_ids) >= max_new_tokens
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    think_ids = tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    result = tokenizer.decode(token_ids, skip_special_tokens=True)
    prediction = extract_final(result)
    gold = canonical_number(row["answer"])
    return {
        "id": row.get("id"), "question": row["question"],
        "prompt_tokens": prompt_tokens, "generated_tokens": len(token_ids),
        "last_token_id": token_ids[-1] if token_ids else None,
        "eos_seen": eos_seen, "hit_token_limit": hit_limit,
        "stop_reason": "eos" if eos_seen else ("length" if hit_limit else "other"),
        "final_tag_count": len(FINAL_RE.findall(result)),
        "think_format_valid": has_think_format(raw),
        "generated_token_ids": token_ids,
        "think_token_ids": think_ids,
        "generated_think_token_ids": [token for token in token_ids if token in think_ids],
        "generation_seconds": elapsed,
        "output_with_special_tokens": raw,
        "gold": gold, "prediction": prediction, "correct": prediction == gold,
        "output": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--adapter", help="SFT / GRPO / distillation adapter dir; omit for base"
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--out", default="results/eval.json")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="plain")
    parser.add_argument("--samples-per-question", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--question-offset", type=int, default=0)
    args = parser.parse_args()
    sampled = args.samples_per_question > 1
    if args.question_offset < 0:
        parser.error("question offset must be nonnegative")
    if args.samples_per_question < 1 or args.batch_size < args.samples_per_question:
        parser.error("samples per question must fit in the generation batch")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation must end each padded prompt on its real last token.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    if sampled:
        torch.manual_seed(args.sample_seed)
    data = with_prompt_style(load_from_disk(str(Path(args.data) / args.split)), args.prompt_style)

    if args.question_offset >= len(data):
        parser.error("question offset exceeds the evaluation split")
    if args.max_samples > 0 or args.question_offset:
        end = len(data)
        if args.max_samples > 0:
            end = min(args.question_offset + args.max_samples, end)
        data = data.select(range(args.question_offset, end))
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    # Hash the ordered prompts and answers, not framework-specific cache metadata.
    dataset_hash = hashlib.sha256(json.dumps(
        [{k: row[k] for k in ("question", "prompt", "answer")} for row in data],
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    if args.expected_dataset_sha256 and dataset_hash != args.expected_dataset_sha256:
        raise RuntimeError(f"Evaluation dataset differs from baseline: {dataset_hash}")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    records = []
    questions_per_batch = args.batch_size // args.samples_per_question
    for start in tqdm(range(0, len(data), questions_per_batch)):
        rows = [data[i] for i in range(start, min(start + questions_per_batch, len(data)))]
        ids = tokenizer([row["prompt"] for row in rows], padding=True, return_tensors="pt").to(model.device)
        example_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=sampled,
                **({"num_return_sequences": args.samples_per_question, "temperature": 1.0,
                    "top_k": 0, "top_p": 1.0} if sampled else {}),
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - example_started
        for index, completion in enumerate(generated):
            row_index, sample_index = divmod(index, args.samples_per_question)
            row = rows[row_index]
            token_ids = completion[ids["input_ids"].shape[-1] :].tolist()
            # Batched generation pads finished rows while other rows continue.
            first_eos = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
            if first_eos is not None:
                token_ids = token_ids[:first_eos + 1]
            records.append(make_record(row, token_ids, tokenizer, eos_ids,
                                       args.max_new_tokens, int(ids['attention_mask'][row_index].sum()), elapsed))
            if sampled:
                records[-1]["sample_index"] = sample_index
    score = sum(r["correct"] for r in records) / len(records) if records else 0.0
    invalid = sum(r["prediction"] is None for r in records) / len(records) if records else 0.0
    # Preserve both aggregate scores and enough provenance to reproduce the run.
    payload = {
        "prompt_style": args.prompt_style,
        "prompt_template": make_prompt("{question}", args.prompt_style),
        "think_format_rate": sum(r["think_format_valid"] for r in records)/len(records) if records else 0.0,
        "batch_size": args.batch_size,
        "generation_timing_scope": "batch" if args.batch_size > 1 else "example",
        "schema_version": 2,
        "started_at_utc": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "do_sample": sampled,
        "samples_per_question": args.samples_per_question,
        "sampling": {"seed": args.sample_seed, "temperature": 1.0, "top_k": 0, "top_p": 1.0} if sampled else None,
        "eos_token_ids": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "generation_config": model.generation_config.to_dict(),
        "dataset_sha256": dataset_hash,
        "dataset_fingerprint": data._fingerprint,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(__file__).with_name("data.py"))},
        "environment": {"python": platform.python_version(),
                        "packages": {name: version(name) for name in
                                     ("torch", "transformers", "datasets", "peft", "accelerate")},
                        "device": str(model.device), "dtype": str(model.dtype),
                        "model_revision": getattr(model.config, "_commit_hash", None)},
        "mean_output_tokens": sum(r["generated_tokens"] for r in records) / len(records) if records else 0.0,
        "token_limit_rate": sum(r["hit_token_limit"] for r in records) / len(records) if records else 0.0,
        "eos_rate": sum(r["eos_seen"] for r in records) / len(records) if records else 0.0,
        "model": MODEL_ID,
        "adapter": args.adapter,
        "split": args.split,
        "n": len(records),
        "accuracy_greedy": score if not sampled else None,
        "sample_accuracy": score if sampled else None,
        "n_questions": len(data),
        "question_offset": args.question_offset,
        "invalid_format_rate": invalid,
        "max_new_tokens": args.max_new_tokens,
        "samples": records,
    }
    if sampled:
        from collections import Counter

        groups = [records[i:i + args.samples_per_question]
                  for i in range(0, len(records), args.samples_per_question)]
        counts = [sum(row["correct"] for row in group) for group in groups]
        histogram = Counter(counts)
        k = args.samples_per_question
        payload["group_metrics"] = {
            "pass_at_1": score,
            f"pass_at_{k}": sum(count > 0 for count in counts) / len(counts),
            "mixed_reward_fraction": sum(0 < count < k for count in counts) / len(counts),
            "all_correct_fraction": histogram[k] / len(counts),
            "all_wrong_fraction": histogram[0] / len(counts),
            "correct_count_histogram": dict(sorted(histogram.items())),
            "mean_within_group_reward_std": sum(((count / k) * (1 - count / k)) ** 0.5
                                                for count in counts) / len(counts),
        }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("split", "n", "accuracy_greedy", "sample_accuracy", "invalid_format_rate")
            },
            indent=2,
        )
    )
    print(f"Saved per-example predictions to {dest}")


if __name__ == "__main__":
    main()
