"""Merge RGB CSV files from the Data folder.

Each CSV is expected to contain RGB values (three columns). Filenames are
like "A1-T061230.csv" where the first part (A1) is the patch id and the
second part is a time code (T061230). This script reads all CSVs in the
specified data directory, adds columns parsed from the filename, and
concatenates them into a single output CSV.

Usage:
    python scripts/merge_rgb_csvs.py --data-dir ../Data --out combined.csv

The resulting CSV will include columns: R,G,B,patch,time_hms,source_file
"""
from pathlib import Path
import pandas as pd
import argparse
import sys


def parse_filename(path: Path):
    """Extract patch and time info from filename.

    Examples:
      A1-T061230.csv -> patch='A1', time_hms='06:12:30'
      c1-T125115.csv -> patch='C1', time_hms='12:51:15'
    """
    stem = path.stem
    parts = stem.split("-", 1)
    patch = parts[0].upper()
    time_hms = ""
    if len(parts) > 1:
        time_part = parts[1]
        # remove leading T if present
        if time_part.upper().startswith("T"):
            time_part = time_part[1:]
        # expect HHMMSS
        if len(time_part) == 6 and time_part.isdigit():
            hh = time_part[0:2]
            mm = time_part[2:4]
            ss = time_part[4:6]
            time_hms = f"{hh}:{mm}:{ss}"
        else:
            time_hms = time_part
    return patch, time_hms


def read_and_label(file_path: Path):
    """Read CSV and add metadata columns parsed from filename."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipping {file_path} — read error: {e}")
        return None

    # Ensure at least three columns for R,G,B
    if df.shape[1] < 3:
        print(f"Skipping {file_path} — not enough columns ({df.shape[1]})")
        return None

    # If columns aren't named R,G,B, try to coerce first three columns
    cols = list(df.columns)
    if set([c.upper() for c in cols[:3]]) >= {"R", "G", "B"}:
        # already has R,G,B names (or similar)
        df = df.rename(columns={cols[0]: "R", cols[1]: "G", cols[2]: "B"})
    else:
        # take first three columns and name them R,G,B
        df = df.copy()
        df.columns = ["R", "G", "B"] + list(df.columns[3:])

    patch, time_hms = parse_filename(file_path)
    df["patch"] = patch
    df["time_hms"] = time_hms
    df["source_file"] = file_path.name
    return df


def merge_all(data_dir: Path, out_file: Path):
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {data_dir}")
        return 1

    parts = []
    for f in files:
        df = read_and_label(f)
        if df is not None:
            parts.append(df)

    if not parts:
        print("No valid CSVs were read.")
        return 1

    combined = pd.concat(parts, ignore_index=True)

    # Keep only the first three color columns plus metadata
    keep_cols = [c for c in combined.columns if c in ("R", "G", "B")] + ["patch", "time_hms", "source_file"]
    combined = combined[keep_cols]

    combined.to_csv(out_file, index=False)
    print(f"Wrote combined CSV to {out_file} ({len(combined)} rows)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=False, default="./Data",
                        help="Path to folder containing the CSV files")
    parser.add_argument("--out", required=False, default="combined.csv",
                        help="Output CSV file path")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()

    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return 2

    return merge_all(data_dir, out_file)


if __name__ == "__main__":
    raise SystemExit(main())
