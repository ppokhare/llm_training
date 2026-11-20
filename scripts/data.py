"""Canonical GSM8K preprocessing shared by SFT, GRPO, and evaluation.

Only the official *training* split is used to produce training and validation data.
The official test split is never used to build supervision or select hyperparameters.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
PROMPT_PREFIX = (
    "Solve the math problem. Show your reasoning, then finish with exactly one "
    "<final>number</final> tag.\n\nQuestion: "
)
PROMPT_SUFFIX = "\nSolution:\n"
FINAL_RE = re.compile(r"<final>\s*([^<>]+?)\s*</final>", re.IGNORECASE)


def canonical_number(value: str | None) -> str | None:
    """Convert common numeric answer spellings to stable decimal strings."""
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "")
    if value.endswith("."):
        value = value[:-1]
    try:
        d = Decimal(value)
        if not d.is_finite():
            return None
        if d == 0:
            return "0"
        return format(d.normalize(), "f")
    except InvalidOperation:
        return None


def extract_final(output: str) -> str | None:
    matches = FINAL_RE.findall(output)
    return canonical_number(matches[-1]) if matches else None


def unpack_gsm_answer(answer: str) -> tuple[str, str]:
    if "####" not in answer:
        raise ValueError("GSM8K record missing #### final answer marker")
    reasoning, number = answer.rsplit("####", 1)
    number = canonical_number(number)
    if number is None:
        raise ValueError(f"Unparseable GSM8K answer: {answer!r}")
    # Strip arithmetic annotation tokens specific to GSM8K; keep human-written reasoning.
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
    return reasoning, number


def make_prompt(question: str) -> str:
    return PROMPT_PREFIX + question.strip() + PROMPT_SUFFIX


def format_record(record: dict, idx: int) -> dict:
    reasoning, answer = unpack_gsm_answer(record["answer"])
    prompt = make_prompt(record["question"])
    return {
        "id": idx,
        "question": record["question"],
        "prompt": prompt,
        "completion": f"{reasoning}\n<final>{answer}</final>",
        "answer": answer,
    }


def make_splits(seed: int = 42, val_fraction: float = 0.1) -> dict[str, Dataset]:
    from datasets import load_dataset

    source = load_dataset("openai/gsm8k", "main")
    raw_train = source["train"].map(lambda row, idx: format_record(row, idx), with_indices=True)
    raw_test = source["test"].map(lambda row, idx: format_record(row, 100000 + idx), with_indices=True)
    train_val = raw_train.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)

    # Check exact-question overlap: protects against accidental leakage.
    train_questions = set(train_val["train"]["question"])
    val_questions = set(train_val["test"]["question"])
    test_questions = set(raw_test["question"])
    if train_questions & val_questions or (train_questions | val_questions) & test_questions:
        raise RuntimeError("Found exact duplicate questions between splits; deduplicate first")

    return {"train": train_val["train"], "validation": train_val["test"], "test": raw_test}


def as_rl_dataset(dataset: Dataset) -> Dataset:
    """GRPO receives prompts and expected answers; no reference reasoning is exposed."""
    return dataset.remove_columns(["completion", "question", "id"])


def as_sft_dataset(dataset: Dataset) -> Dataset:
    """Completion-only loss keeps the mathematical question out of the SFT objective."""
    return dataset.select_columns(["prompt", "completion"])


def reward_exact_match(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    return [float(extract_final(c) == canonical_number(a)) for c, a in zip(completions, answer)]
