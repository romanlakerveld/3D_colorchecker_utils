#!/usr/bin/env python3
"""rgb_hue_hist.py

Read RGB values from all text files in a folder, combine into one DataFrame,
convert RGB (0-255) to hue (degrees), and plot/save a hue histogram.

Each input file may be CSV-like or whitespace-separated. The script will try
to detect three numeric columns per file and name them R,G,B. Non-numeric
rows are skipped.

Usage:
    python rgb_hue_hist.py -i path/to/sticks_folder -o out.csv --bins 36 --save_plot hue.png
"""
import argparse
import os
import glob
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def read_rgb_file(path: str, cols: list = None) -> pd.DataFrame:
    """Attempt to read a text file containing RGB triplets.

    Tries flexible separators. Returns a DataFrame with columns ['R','G','B']
    or an empty DataFrame if nothing suitable is found.
    """
    # try pandas auto-detect sep
    try:
        df = pd.read_csv(path, sep=None, engine='python', header=None, comment='#')
    except Exception:
        try:
            df = pd.read_csv(path, delim_whitespace=True, header=None, comment='#')
        except Exception:
            return pd.DataFrame(columns=['R', 'G', 'B'])

    # If user provided column positions (1-based), pick those
    if cols is not None:
        # ensure we have enough columns
        max_col = max(cols)
        if df.shape[1] < max_col:
            return pd.DataFrame(columns=['R', 'G', 'B'])
        # convert to numeric then select positions
        df = df.apply(pd.to_numeric, errors='coerce')
        try:
            sel = df.iloc[:, [c - 1 for c in cols]]
        except Exception:
            return pd.DataFrame(columns=['R', 'G', 'B'])
        sel.columns = ['R', 'G', 'B']
        sel = sel.dropna()
        return sel.astype(float)

    # keep only numeric columns
    df = df.apply(pd.to_numeric, errors='coerce')
    # drop columns with all NaN
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < 3:
        return pd.DataFrame(columns=['R', 'G', 'B'])

    # take first three numeric columns
    df = df.iloc[:, :3]
    df.columns = ['R', 'G', 'B']
    # drop rows with NaN
    df = df.dropna()
    # convert to numeric
    df = df.astype(float)
    return df


def rgb_to_hue_degrees(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert Nx3 RGB array in 0-255 to hue in degrees [0,360).
    Uses matplotlib.colors.rgb_to_hsv which expects 0..1 floats.
    Returns array of hue degrees (float).
    """
    if rgb_arr.size == 0:
        return np.array([], dtype=float)
    rgb_norm = np.clip(rgb_arr / 255.0, 0.0, 1.0)
    # matplotlib.colors.rgb_to_hsv expects shape (...,3)
    hsv = mcolors.rgb_to_hsv(rgb_norm)
    # hsv[...,0] is hue in 0..1
    hue = hsv[..., 0] * 360.0
    return hue


def combine_folder(input_dir: str, pattern: str = '*.txt') -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f'No files matching {pattern} in {input_dir}')

    frames: List[pd.DataFrame] = []
    for f in files:
        df = read_rgb_file(f)
        if df.empty:
            continue
        # ensure values in 0-255; if they look like 0..1 scale, scale up
        if df[['R','G','B']].max().max() <= 1.0:
            df[['R','G','B']] = df[['R','G','B']] * 255.0
        frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame(columns=['R','G','B'])

    big = pd.concat(frames, ignore_index=True)
    # clamp to 0..255
    big[['R','G','B']] = big[['R','G','B']].clip(0, 255)
    return big


def plot_hue_hist(hues: np.ndarray, bins: int = 36, out_path: str = None, show: bool = False):
    plt.figure(figsize=(8, 4))
    # wrap-around histogram: hue 0..360
    counts, edges = np.histogram(hues, bins=bins, range=(0, 360))
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (edges[1] - edges[0])

    # Map each bin center hue (degrees) -> RGB color. Use full saturation/value for vivid colors.
    # matplotlib expects HSV in 0..1 for h, s, v
    if centers.size > 0:
        hsv = np.stack([centers / 360.0, np.ones_like(centers), np.ones_like(centers)], axis=1)
        try:
            bin_colors = mcolors.hsv_to_rgb(hsv)
        except Exception:
            # fallback: make them orange if hsv conversion fails for any reason
            bin_colors = ['tab:orange'] * len(centers)
    else:
        bin_colors = ['tab:orange'] * bins

    # Plot bars with the computed colors per bin
    plt.bar(centers, counts, width=width, align='center', color=bin_colors, edgecolor='k')

    plt.xlabel('Hue (degrees)')
    plt.ylabel('Count')
    plt.title('Hue histogram')
    plt.xlim(0, 360)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved hue histogram to {out_path}')
    if show:
        plt.show()
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Combine RGB txt files and plot hue histogram')
    p.add_argument('-i', '--input_dir', required=True, help='Folder containing txt files with RGB values')
    p.add_argument('-o', '--out_csv', default='combined_rgb.csv', help='Output CSV path for combined RGB values')
    p.add_argument('--pattern', default='*.txt', help='Glob pattern for files (default *.txt)')
    p.add_argument('--bins', type=int, default=36, help='Number of histogram bins across 0..360 degrees')
    p.add_argument('--save_plot', default='hue_hist.png', help='Path to save hue histogram PNG (optional)')
    p.add_argument('--show', action='store_true', help='Show the histogram interactively')
    p.add_argument('--cols', nargs=3, type=int, help='One-based column indices for R G B in the files (e.g. --cols 4 5 6)')
    return p.parse_args()


def main():
    args = parse_args()
    # pass cols through to reader if provided
    # combine_folder currently calls read_rgb_file without cols; implement local loop here
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    frames: List[pd.DataFrame] = []
    for f in files:
        df = read_rgb_file(f, cols=args.cols if hasattr(args, 'cols') else None)
        if df.empty:
            continue
        # ensure values in 0-255; if they look like 0..1 scale, scale up
        if df[['R','G','B']].max().max() <= 1.0:
            df[['R','G','B']] = df[['R','G','B']] * 255.0
        frames.append(df)
    if len(frames) == 0:
        print('No RGB data found in folder. Exiting.')
        sys.exit(1)
    big = pd.concat(frames, ignore_index=True)
    big[['R','G','B']] = big[['R','G','B']].clip(0, 255)
    big.to_csv(args.out_csv, index=False)
    print(f'Saved combined RGB to {args.out_csv} ({len(big)} rows)')

    rgb_arr = big[['R','G','B']].to_numpy(dtype=float)
    hues = rgb_to_hue_degrees(rgb_arr)

    if args.save_plot:
        plot_hue_hist(hues, bins=args.bins, out_path=args.save_plot, show=args.show)
    else:
        plot_hue_hist(hues, bins=args.bins, out_path=None, show=args.show)


if __name__ == '__main__':
    main()
