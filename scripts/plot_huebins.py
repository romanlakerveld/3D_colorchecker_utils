#!/usr/bin/env python3
"""
plot_huebins.py

Read a CSV where each column is a hue bin (1..N) and each row is a measurement.
Sum all bins across rows and plot a histogram (bar plot) of summed counts per hue bin.

Usage example:
  python plot_huebins.py -i ../../huebins.csv -o hue_hist.png

Options:
  --normalize    Plot fractions instead of raw counts
  --log          Use log scale for y-axis
  --show         Show the plot interactively (default: False)

This script depends on pandas and matplotlib.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_huebins(path: str) -> pd.DataFrame:
    """Read CSV containing hue-bin columns. Returns DataFrame (rows x bins).

    Assumes no header. If the file contains a header row, pandas will still
    read it as row 0; the script will still sum numeric columns and ignore
    non-numeric values.
    """
    # Detect simple delimiter (semicolon is used in this project)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    sep = ";" if ";" in first else ","

    # Read with header row (the file contains 'timepoint;sample;bin0;bin1;...')
    df = pd.read_csv(path, sep=sep, header=0, engine="python")

    # Select only columns that look like bins: e.g. 'bin0', 'bin1', ...
    bin_cols = [c for c in df.columns if str(c).lower().startswith("bin")]
    if not bin_cols:
        # Fall back: if there are no named bin columns, try all numeric columns
        numeric = df.select_dtypes(include=["number"])
        return numeric

    # Sort bin columns by integer index after the 'bin' prefix
    def bin_key(name: str) -> int:
        s = str(name).lower()
        try:
            return int(s.replace("bin", ""))
        except Exception:
            return 10 ** 9

    bin_cols_sorted = sorted(bin_cols, key=bin_key)
    return df[bin_cols_sorted]


def sum_bins(df: pd.DataFrame) -> np.ndarray:
    """Sum each column (bin) across rows and return 1-D numpy array of sums."""
    # Convert to numeric; coerce errors to NaN then treat them as zero
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    col_sums = numeric.sum(axis=0).to_numpy(dtype=float)
    return col_sums


def plot_bins(sums: np.ndarray, out_path: str, normalize: bool = False, use_log: bool = False, show: bool = False) -> None:
    bins = len(sums)
    x = np.arange(1, bins + 1)
    heights = sums.astype(float)
    if normalize:
        total = heights.sum()
        if total > 0:
            heights = heights / total

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, heights, width=1.0, align="center", color="tab:blue")
    ax.set_xlim(0.5, bins + 0.5)
    ax.set_xlabel("Hue bin")
    ax.set_ylabel("Fraction" if normalize else "Count")
    ax.set_title("Hue-bin histogram (summed across rows)")
    if use_log:
        ax.set_yscale("log")
    plt.tight_layout()
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot summed hue bins from CSV file")
    p.add_argument("-i", "--input", required=True, help="Input CSV file path (columns = hue bins)")
    p.add_argument("-o", "--output", required=True, help="Output image path (PNG, PDF, etc.)")
    p.add_argument("--normalize", action="store_true", help="Normalize to fractions (sum to 1)")
    p.add_argument("--log", dest="log", action="store_true", help="Use logarithmic y-axis")
    p.add_argument("--show", action="store_true", help="Show plot interactively after saving")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    df = read_huebins(args.input)
    if df.shape[1] == 0:
        print(f"ERROR: input file appears empty or has no columns: {args.input}", file=sys.stderr)
        return 3

    sums = sum_bins(df)
    plot_bins(sums, args.output, normalize=args.normalize, use_log=args.log, show=args.show)
    print(f"Saved histogram to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
