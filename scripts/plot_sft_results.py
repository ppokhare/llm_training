"""Render training curves and matched base/SFT metrics from saved JSON artifacts.

Produces PNGs for the README and SVGs for export; no model or GPU is needed.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render files without requiring a desktop display.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base")
    parser.add_argument("--sft")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = json.loads(Path(args.history).read_text())["log_history"]
    # Training and validation are logged at different optimizer-step intervals.
    training = [row for row in history if "loss" in row]
    validation = [row for row in history if "eval_loss" in row]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 160})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
    axes[0, 0].plot([r['step'] for r in training], [r['loss'] for r in training],
                     color="#2563eb", alpha=.65, label="Training (5-step logs)")
    axes[0, 0].plot([r['step'] for r in validation], [r['eval_loss'] for r in validation],
                     "o-", color="#c2410c", markersize=4, label="Validation (748 examples)")
    axes[0, 0].set(title="Completion-token loss", ylabel="Cross-entropy loss")
    axes[0, 0].legend(frameon=False)
    # Teacher-forced token prediction is distinct from generated-answer accuracy.
    axes[0, 1].plot([r['step'] for r in validation],
                     [r['eval_mean_token_accuracy'] for r in validation], "o-", color="#c2410c")
    axes[0, 1].set(title="Validation token prediction", ylabel="Teacher-forced token accuracy")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].plot([r['step'] for r in training], [r['learning_rate'] for r in training], color="#2563eb")
    axes[1, 0].set(title="Learning-rate schedule", ylabel="Learning rate")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 1].plot([r['step'] for r in training], [r['grad_norm'] for r in training], color="#2563eb")
    axes[1, 1].set(title="Adapter gradient norms", ylabel="Gradient norm")
    fig.suptitle("Qwen3-0.6B · FP8 LoRA SFT · 300 steps · effective batch 32", fontsize=14)
    fig.savefig(out / "sft_training.png")
    fig.savefig(out / "sft_training.svg")
    plt.close(fig)
    if args.base and args.sft:
        base, sft = [json.loads(Path(p).read_text()) for p in (args.base, args.sft)]
        # Refuse to chart scores side by side when evaluation conditions differ.
        for key in ("dataset_sha256", "max_new_tokens", "do_sample", "n"):
            if base[key] != sft[key]:
                raise ValueError(f"Evaluation mismatch: {key}")
        if [r['id'] for r in base['samples']] != [r['id'] for r in sft['samples']]:
            raise ValueError("Evaluation sample order differs")
        fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
        for ax, key, title in zip(axes, ['accuracy_greedy', 'invalid_format_rate', 'token_limit_rate'],
                                  ['Numerical accuracy ↑', 'Unparseable final answer ↓', 'Output token-limit hits ↓']):
            values = [base[key], sft[key]]
            bars = ax.bar(['Base', 'SFT'], values, color=['#64748b', '#2563eb'], width=.55)
            ax.set(title=title, ylim=(0, 1.12))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.025,
                        f"{value:.1%}\n{round(value*base['n'])}/{base['n']}", ha="center", va="bottom")
        fig.suptitle(f"Matched H100 evaluation · {base['n']} validation questions · greedy · 512-token cap", fontsize=13)
        fig.savefig(out / "base_vs_sft.png")
        fig.savefig(out / "base_vs_sft.svg")
        plt.close(fig)
    print(f"Plots saved in {out}")


if __name__ == "__main__":
    main()
