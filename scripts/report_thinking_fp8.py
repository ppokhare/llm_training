"""Summarize paired validation results and plot the fresh SFT training curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    adapter = run / "adapter"
    metadata = json.loads((adapter / "training_precision.json").read_text())
    token_checks = json.loads((adapter / "thinking_token_checks.json").read_text())
    lines = [
        "# Fresh FP8 thinking-format SFT", "",
        f"Training used {metadata['steps']} updates and an effective batch of "
        f"{metadata['effective_batch_size']} at learning rate {metadata['learning_rate']:g}. "
        f"Seed {metadata['seed']}. Maximum training sequence length {metadata['max_length']}.", "",
        f"Reloaded adapter produced a {token_checks['logit_gap']:.5g} logit gap between the two thinking tags. "
        "The validation evaluations use greedy decoding with a 512-token generation cap.", "",
        "| Prompt | Adapter | Questions | Accuracy | Think compliance | Valid final | Mean tokens | Hit cap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for style in ("plain", "sft-think"):
        reports = [json.loads((run / f"{label}-{style}.json").read_text())
                   for label in ("original", "fresh")]
        if reports[0]["dataset_sha256"] != reports[1]["dataset_sha256"]:
            raise ValueError("Cannot compare different validation prompts")
        for label, report in zip(("Original", "Fresh FP8"), reports):
            lines.append(
                f"| {style} | {label} | {report['n']} | {report['accuracy_greedy']:.2%} | "
                f"{report['think_format_rate']:.2%} | {1 - report['invalid_format_rate']:.2%} | "
                f"{report['mean_output_tokens']:.1f} | {report['token_limit_rate']:.2%} |"
            )
    lines.extend(["", "See commands.jsonl for the recorded invocations, run.log for training and evaluation logs, "
                  "and adapter/training_precision.json for data, environment and truncation details."])
    (run / "report.md").write_text("\n".join(lines) + "\n")

    history = json.loads((adapter / "trainer_state.json").read_text())["log_history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for key, label in [("loss", "Train"), ("eval_loss", "Validation")]:
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            axes[0].plot(*zip(*points), label=label)
    axes[0].set(xlabel="Optimizer update", ylabel="Completion loss")
    axes[0].legend()
    generation = json.loads((adapter / "generated_history.json").read_text())
    for key, label in [("accuracy", "Accuracy"), ("think_format_rate", "Think compliance")]:
        axes[1].plot([row["step"] for row in generation], [row[key] for row in generation],
                     marker="o", label=label)
    axes[1].set(xlabel="Optimizer update", ylabel="Fixed-panel fraction", ylim=(0, 1.02))
    axes[1].legend()
    fig.savefig(run / "training.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
