from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot daily power load data from a CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/power_daily_raw.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/power_daily_raw.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--title",
        default="Daily Power Load",
        help="Chart title.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"date", "OT"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )

    df = df.loc[:, ["date", "OT"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["OT"] = pd.to_numeric(df["OT"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_series(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["OT"], color="#1f77b4", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    plot_series(df=df, output_path=args.output, title=args.title)
    print(f"Saved plot to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
