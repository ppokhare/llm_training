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
THINK_PROMPT_PREFIX = (
    "First write your reasoning inside <think> and </think> tags. Then finish with exactly one "
    "<final>number</final>.\n\nQuestion: "
)
SFT_THINK_PROMPT_PREFIX = (
    "Solve the math problem. Write your reasoning inside "
    "<think>...</think>, then finish with exactly one "
    "<final>number</final> tag.\n\nQuestion: "
)
PROMPT_STYLES = ("plain", "think", "sft-think")
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
    """Score the last final-answer tag; missing or nonnumeric answers are invalid."""
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


def make_prompt(question: str, prompt_style: str = "plain") -> str:
    """Version prompts explicitly; preserve the completed SFT/plain-RLVR baseline."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    prefix = {"plain": PROMPT_PREFIX, "think": THINK_PROMPT_PREFIX,
              "sft-think": SFT_THINK_PROMPT_PREFIX}[prompt_style]
    return prefix + question.strip() + PROMPT_SUFFIX


def with_prompt_style(dataset: Dataset, prompt_style: str) -> Dataset:
    """Rebuild new prompts from questions in memory, without rewriting cached splits."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    if prompt_style == "plain":
        return dataset
    return dataset.map(lambda row: {"prompt": make_prompt(row["question"], prompt_style)}, keep_in_memory=True)


def add_thinking_format(row: dict) -> dict:
    """Wrap the existing gold reasoning; never manufacture new reasoning or answers."""
    reasoning, final = row["completion"].rsplit("\n<final>", 1)
    if "<think>" in reasoning or "</think>" in reasoning:
        raise ValueError("Expected original plain SFT targets, not already-wrapped reasoning")
    return {
        "prompt": make_prompt(row["question"], "sft-think"),
        "completion": f"<think>\n{reasoning}\n</think>\n<final>{final}",
    }


def thinking_sft_dataset(dataset: Dataset) -> Dataset:
    """Apply the format-only adaptation in RAM; retain the fixed split and keys."""
    return dataset.map(add_thinking_format, keep_in_memory=True)


def has_think_format(output: str) -> bool:
    """Diagnostic only: one nonempty think block before one final-answer tag."""
    if output.count("<think>") != 1 or output.count("</think>") != 1:
        return False
    match = re.search(r"<think>(.*?)</think>", output, flags=re.DOTALL)
    finals = list(FINAL_RE.finditer(output))
    return bool(match and match.group(1).strip() and len(finals) == 1
                and finals[0].start() >= match.end())


def format_record(record: dict, idx: int) -> dict:
    """Keep gold reasoning separate from the prompt so SFT can mask question tokens."""
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
    # Stable IDs let evaluation results be paired across model checkpoints.
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
    """Give GRPO one binary numerical-correctness reward per sampled completion."""
    if len(completions) != len(answer):
        raise ValueError("Completion and answer counts differ")
    gold = [canonical_number(a) for a in answer]
    if any(a is None for a in gold):
        raise ValueError("Reward received an invalid gold answer")
    return [float(extract_final(c) == a) for c, a in zip(completions, gold)]
