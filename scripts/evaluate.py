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

from scripts.data import MODEL_ID, FINAL_RE, canonical_number, extract_final


def make_record(row, token_ids, tokenizer, eos_ids, max_new_tokens, prompt_tokens, elapsed):
    """Score one unpadded completion; elapsed time covers its whole generation batch."""
    eos_seen = any(token in eos_ids for token in token_ids)
    hit_limit = len(token_ids) >= max_new_tokens
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
        "generation_seconds": elapsed,
        "output_with_special_tokens": tokenizer.decode(token_ids, skip_special_tokens=False),
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
    args = parser.parse_args()
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
    data = load_from_disk(str(Path(args.data) / args.split))

    if args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))
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
    for start in tqdm(range(0, len(data), args.batch_size)):
        rows = [data[i] for i in range(start, min(start + args.batch_size, len(data)))]
        ids = tokenizer([row["prompt"] for row in rows], padding=True, return_tensors="pt").to(model.device)
        example_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - example_started
        for index, row in enumerate(rows):
            token_ids = generated[index][ids["input_ids"].shape[-1] :].tolist()
            # Batched generation pads finished rows while other rows continue.
            first_eos = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
            if first_eos is not None:
                token_ids = token_ids[:first_eos + 1]
            records.append(make_record(row, token_ids, tokenizer, eos_ids,
                                       args.max_new_tokens, int(ids['attention_mask'][index].sum()), elapsed))
    score = sum(r["correct"] for r in records) / len(records) if records else 0.0
    invalid = sum(r["prediction"] is None for r in records) / len(records) if records else 0.0
    # Preserve both aggregate scores and enough provenance to reproduce the run.
    payload = {
        "batch_size": args.batch_size,
        "generation_timing_scope": "batch" if args.batch_size > 1 else "example",
        "schema_version": 2,
        "started_at_utc": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "do_sample": False,
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
        "accuracy_greedy": score,
        "invalid_format_rate": invalid,
        "max_new_tokens": args.max_new_tokens,
        "samples": records,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("split", "n", "accuracy_greedy", "invalid_format_rate")
            },
            indent=2,
        )
    )
    print(f"Saved per-example predictions to {dest}")


if __name__ == "__main__":
    main()
