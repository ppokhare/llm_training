"""Build a matched full-validation report from the three saved adapters."""
import argparse
import gzip
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

MODELS = {"original": "Original SFT", "step50": "Thinking step 50", "step100": "Thinking step 100"}


def paired_interval(left, right, indices):
    differences = right - left
    lower, upper = np.quantile(differences[indices].mean(axis=1), [.025, .975])
    return {"change": float(differences.mean()), "ci95": [float(lower), float(upper)],
            "improved_questions": int((differences > 0).sum()),
            "regressed_questions": int((differences < 0).sum())}


def generation_metrics(rows):
    return {
        "numerical_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "think_compliance": sum(row["think_format_valid"] for row in rows) / len(rows),
        "parseable_final": sum(row["prediction"] is not None for row in rows) / len(rows),
        "single_valid_final": sum(row["prediction"] is not None and row["final_tag_count"] == 1
                                  for row in rows) / len(rows),
        "mean_new_tokens": sum(row["generated_tokens"] for row in rows) / len(rows),
        "cap_hits": sum(row["hit_token_limit"] for row in rows),
        "length_stops": sum(row["stop_reason"] == "length" for row in rows),
        "cap_rate": sum(row["hit_token_limit"] for row in rows) / len(rows),
    }


def archive_prediction(path, destination):
    with destination.open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", filename="", mtime=0) as archive:
            archive.write(path.read_bytes())


def load_condition(source, output, style, count):
    condition = {}
    panel = None
    settings = None
    for model in MODELS:
        directory = source / model
        paths = ([directory / f"sample-shard-{shard}" / f"{style}-{count}.json" for shard in range(4)]
                 if count > 1 else [directory / f"{style}-{count}.json"])
        rows = []
        current_settings = []
        for shard, path in enumerate(paths):
            report = json.loads(path.read_text())
            offset = shard * 192
            n = min(192, 748 - offset) if count > 1 else 748
            expected = {"split": "validation", "prompt_style": style, "n_questions": n,
                        "max_new_tokens": 512, "do_sample": count > 1, "samples_per_question": count}
            if (any(report[key] != value for key, value in expected.items())
                    or report.get("question_offset", 0) != offset or len(report["samples"]) != n * count):
                raise RuntimeError(f"Unexpected evaluation coverage or settings in {path}")
            rows.extend(report["samples"])
            current_settings.append({key: report[key] for key in (
                "dataset_sha256", "prompt_template", "batch_size", "sampling",
                "eos_token_ids", "pad_token_id", "environment")})
            suffix = f"-shard{shard}" if count > 1 else ""
            archive_prediction(path, output / f"{model}-{style}-{count}{suffix}.json.gz")
        question_rows = [(row["id"], row["question"], row["gold"]) for row in rows]
        current_panel = question_rows[::count]
        if len({row[0] for row in current_panel}) != 748 or any(
            row != current_panel[i // count] for i, row in enumerate(question_rows)
        ):
            raise RuntimeError(f"Invalid question grouping for {model}, {style}, {count}")
        if count > 1 and any(row["sample_index"] != i % count for i, row in enumerate(rows)):
            raise RuntimeError(f"Invalid sample ordering for {model}, {style}")
        if panel is not None and (current_panel != panel or current_settings != settings):
            raise RuntimeError(f"Mismatched evaluation condition for {model}, {style}, {count}")
        panel, settings = current_panel, current_settings
        scores = np.asarray([row["correct"] for row in rows], dtype=float).reshape(748, count)
        group_correct = scores.sum(axis=1)
        predictions = [row["prediction"] for row in rows]
        summary = generation_metrics(rows)
        if count > 1:
            group_rates = group_correct / count
            summary.update(
                pass_at_1=float(group_rates.mean()),
                pass_at_8=float((group_correct > 0).mean()),
                mixed_reward_fraction=float(((group_correct > 0) & (group_correct < count)).mean()),
                all_correct_fraction=float((group_correct == count).mean()),
                all_wrong_fraction=float((group_correct == 0).mean()),
                correct_count_histogram=np.bincount(group_correct.astype(int), minlength=count + 1).tolist(),
                mean_within_group_reward_std=float(np.sqrt(group_rates * (1 - group_rates)).mean()),
                mean_distinct_valid_answers=sum(len(set(predictions[start:start + count]) - {None})
                                                for start in range(0, len(predictions), count)) / 748,
            )
        condition[model] = (summary, scores.mean(axis=1), (group_correct > 0).astype(float))
    return condition, panel, settings


def plot(output, summaries, history, extension):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    training = [(row["step"], row["loss"]) for row in history if "loss" in row]
    validation = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
    axes[0].plot(*zip(*training), label="Training", alpha=.75)
    axes[0].plot(*zip(*validation), "o-", label="Validation")
    axes[0].set(title="Completion loss", xlabel="Optimizer update", ylabel="Cross-entropy")
    axes[0].legend(frameon=False)
    labels = ["Original", "Step 50", "Step 100"]
    x = np.arange(len(MODELS) + len(extension))
    plain = [row["numerical_accuracy"] for row in summaries["plain-1"].values()]
    thinking = [row["numerical_accuracy"] for row in summaries["sft-think-1"].values()]
    plain.extend(stage["metrics"]["plain"]["numerical_accuracy"] for stage in extension)
    thinking.extend(stage["metrics"]["sft-think"]["numerical_accuracy"] for stage in extension)
    axes[1].set_xticks(x, labels + [f"Step {stage['step']}" for stage in extension])
    axes[1].bar(x - .19, plain,
                width=.38, label="Plain")
    axes[1].bar(x + .19, thinking,
                width=.38, label="Thinking")
    axes[1].set(title="Greedy numerical accuracy")
    axes[1].legend(frameon=False)
    x = np.arange(len(MODELS))
    axes[2].set_xticks(x, labels)
    sampled = summaries["sft-think-8"].values()
    axes[2].bar(x - .19, [row["pass_at_1"] for row in sampled], width=.38, label="Pass@1")
    axes[2].bar(x + .19, [row["pass_at_8"] for row in sampled], width=.38, label="Pass@8")
    axes[2].set(title="Sampled thinking prompt")
    axes[2].legend(frameon=False)
    for ax in axes[1:]:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)


def load_extension(args, indices):
    if args.extension is None:
        return []
    results, checkpoints, decisions = args.extension
    previous = {style: json.loads((args.results / "step100" / f"{style}-1.json").read_text())
                for style in ("plain", "sft-think")}
    extension = []
    for decision in map(json.loads, decisions.read_text().splitlines()):
        step = decision["step"]
        stage = checkpoints / f"stage-{step}"
        metrics, paired = {}, {}
        for style, prior in previous.items():
            path = results / f"step{step}-{style}.json"
            current = json.loads(path.read_text())
            fields = ("dataset_sha256", "batch_size", "n", "max_new_tokens", "do_sample")
            if any(current[key] != prior[key] for key in fields):
                raise RuntimeError(f"Unmatched extension evaluation: {path}")
            records = current["samples"]
            if [row["id"] for row in records] != [row["id"] for row in prior["samples"]]:
                raise RuntimeError(f"Question order changed: {path}")
            metrics[style] = generation_metrics(records)
            paired[style] = paired_interval(
                np.asarray([row["correct"] for row in prior["samples"]], dtype=float),
                np.asarray([row["correct"] for row in records], dtype=float), indices)
            previous[style] = current
            archive_prediction(path, args.output / f"step{step}-{style}-1.json.gz")
        extension.append({"step": step, "decision": decision, "metrics": metrics, "paired": paired,
                          "training": json.loads((stage / "training_precision.json").read_text()),
                          "optimizer": json.loads((stage / "resume_optimizer.json").read_text()),
                          "fp8": json.loads((stage / "fp8_profile.json").read_text()),
                          "reload": json.loads((stage / "thinking_token_checks.json").read_text())})
    return extension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extension", type=Path, nargs=3,
                        metavar=("RESULTS", "STAGES", "DECISIONS"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    indices = np.random.default_rng(42).integers(0, 748, size=(20_000, 748), dtype=np.int32)
    summaries, comparisons, hashes = {}, {}, {}
    comparison_panel = None
    for style, count in (("plain", 1), ("sft-think", 1), ("sft-think", 8)):
        label = f"{style}-{count}"
        data, panel, settings = load_condition(args.results, args.output, style, count)
        if comparison_panel is not None and panel != comparison_panel:
            raise RuntimeError("Prompts must use the same ordered questions and answers")
        comparison_panel = panel
        hashes[label] = settings
        summaries[label] = {model: data[model][0] for model in MODELS}
        comparisons[label] = {}
        for left, right in combinations(MODELS, 2):
            pair = {"accuracy": paired_interval(data[left][1], data[right][1], indices)}
            if count > 1:
                pair["pass_at_8"] = paired_interval(data[left][2], data[right][2], indices)
            comparisons[label][f"{right}_minus_{left}"] = pair
    history = json.loads((args.adapter / "trainer_state.json").read_text())["log_history"]
    training = json.loads((args.adapter / "training_precision.json").read_text())
    extension = load_extension(args, indices)
    if extension:
        last_stage = args.extension[1] / f"stage-{extension[-1]['step']}"
        history = json.loads((last_stage / "trainer_state.json").read_text())["log_history"]
    summary = {"questions": 748, "bootstrap": {"resamples": 20_000, "seed": 42, "unit": "question"},
               "evaluation_conditions": hashes, "metrics": summaries, "paired": comparisons,
               "training": training, "training_history": history, "extension": extension}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(args.output, summaries, history, extension)

    lines = ["# Full GSM8K validation of thinking SFT", "",
             "All cells use the same 748 validation questions on H100 with a 512-token generation cap. "
             "Greedy results are matched separately within each prompt. The sampled thinking prompt uses temperature "
             "1.0, unrestricted sampling and eight completions per question, with seeds 42–45 across four ordered shards.", "",
             "## Greedy generation", "",
             "| Prompt | Adapter | Accuracy | Think format | Parseable final | One valid final | Mean tokens | Cap hits |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in ("plain-1", "sft-think-1"):
        style = "Plain" if label == "plain-1" else "Thinking"
        for model, row in summaries[label].items():
            lines.append(f"| {style} | {MODELS[model]} | {row['numerical_accuracy']:.2%} | "
                         f"{row['think_compliance']:.2%} | {row['parseable_final']:.2%} | "
                         f"{row['single_valid_final']:.2%} | {row['mean_new_tokens']:.1f} | {row['cap_hits']} |")
    lines.extend(["", "## Sampled generation", "",
                  "Pass@1 is the estimated success probability of one sampled completion, equal to accuracy across "
                  "the 5,984 samples per adapter. Pass@8 is the fraction of questions with at least one correct "
                  "sample. Reward diversity uses binary numerical correctness.", "",
                  "| Adapter | Pass@1 / sample accuracy | Pass@8 | Mixed | All correct | All wrong | Reward SD |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model, row in summaries["sft-think-8"].items():
        lines.append(f"| {MODELS[model]} | {row['pass_at_1']:.2%} | {row['pass_at_8']:.2%} | "
                     f"{row['mixed_reward_fraction']:.2%} | {row['all_correct_fraction']:.2%} | "
                     f"{row['all_wrong_fraction']:.2%} | {row['mean_within_group_reward_std']:.3f} |")
    lines.extend(["", "## Paired comparisons", "",
                  "Intervals use 20,000 percentile bootstrap resamples of questions with seed 42. All eight "
                  "completions stay together when resampling a sampled result.", "",
                  "| Condition | Comparison | Accuracy Δ (95% CI) | Pass@8 Δ (95% CI) |",
                  "| --- | --- | ---: | ---: |"])
    for condition, pairs in comparisons.items():
        for label, pair in pairs.items():
            right, left = label.split("_minus_")
            cells = []
            for outcome in ("accuracy", "pass_at_8"):
                if outcome not in pair:
                    cells.append("—")
                    continue
                values = pair[outcome]
                low, high = values["ci95"]
                cells.append(f"{100 * values['change']:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]")
            lines.append(f"| {condition} | {MODELS[right]} vs {MODELS[left]} | {cells[0]} | {cells[1]} |")
    if extension:
        lines.extend(["", "## SFT extension", "",
                      "| Step | Prompt | Accuracy | Think format | One valid final | Mean tokens | Cap hits | Accuracy Δ vs previous (95% CI) |",
                      "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for stage in extension:
            for style, row in stage["metrics"].items():
                pair = stage["paired"][style]
                low, high = pair["ci95"]
                lines.append(f"| {stage['step']} | {style} | {row['numerical_accuracy']:.2%} | "
                             f"{row['think_compliance']:.2%} | {row['single_valid_final']:.2%} | "
                             f"{row['mean_new_tokens']:.1f} | {row['cap_hits']} | "
                             f"{100 * pair['change']:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}] |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
