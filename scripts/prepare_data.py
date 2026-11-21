"""Download GSM8K and create one immutable train / validation / test partition."""

import argparse
from pathlib import Path

from scripts.data import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output)
    # Persist the split once so every training and evaluation stage reuses it.
    for name, dataset in make_splits(seed=args.seed).items():
        dataset.save_to_disk(str(root / name))
        print(f"{name}: {len(dataset)} examples → {root / name}")
    print(
        "Freeze these files. Tune on validation; report official test once at the end."
    )


if __name__ == "__main__":
    main()
