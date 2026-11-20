"""Generate verified teacher solutions for OFF-POLICY response distillation.

Only uses data/gsm8k/train. Incorrect or unparseable teacher solutions are rejected.
Never generate teacher supervision from the official test or validation split.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.data import canonical_number, extract_final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--output", default="data/teacher_train.jsonl")
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use an NVIDIA GPU for efficient teacher generation")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    train = load_from_disk(str(Path(args.data) / "train"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output.open("w") as writer:
        for row in tqdm(train.select(range(min(args.max_items, len(train))))):
            gold = canonical_number(row["answer"])
            # Teacher sees the question but NEVER the answer. The answer is used only by the verifier.
            messages = [{"role": "user", "content": row["prompt"]}]
            try:
                ids = teacher_tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(model.device)
            except TypeError:
                ids = teacher_tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
            for _ in range(args.attempts):
                with torch.inference_mode():
                    generated = model.generate(
                        ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=teacher_tok.eos_token_id,
                    )
                completion = teacher_tok.decode(
                    generated[0][ids.shape[-1] :], skip_special_tokens=True
                )
                if extract_final(completion) == gold:
                    writer.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "teacher": args.teacher,
                                "prompt": row["prompt"],
                                "completion": completion,
                            }
                        )
                        + "\n"
                    )
                    accepted += 1
                    break
    print(
        f"Accepted {accepted}/{min(args.max_items, len(train))} prompts with verified teacher solutions"
    )
    print(f"Saved to {output}. Inspect samples before starting distillation.")
    if accepted == 0:
        raise RuntimeError(
            "No correct teacher completions: inspect prompt, output formatting, and model"
        )


if __name__ == "__main__":
    main()
