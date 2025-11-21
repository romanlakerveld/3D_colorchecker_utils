"""Compare sample RGB measurements to a color-checker reference and plot results.

This script implements a small analysis pipeline for per-patch color samples
and a reference color-checker. Main features:

- Read a CSV of measured samples (must contain columns R, G, B and a `patch`
    column and optional `time_hms` strings). RGB may be 0-255 or 0-1; the code
    normalizes to 0-1 automatically.
- Read a checker/reference CSV (should contain `row` and `col` or a `patch`
    column together with R,G,B reference values).
- Compute HSV for samples and checker references (H in degrees, S and V in 0-1),
    then compute per-sample differences: signed hue difference (deg, circular),
    signed/absolute S and V differences, and a simple combined HSV deviation metric.
- Write a comparison CSV (input filename + "_compared" suffix by default)
    containing sample HSV and reference HSV plus difference columns.
- Produce per-patch violin plots for H, S and V over time. Time is parsed from
    the `time_hms` column (accepts "HH:MM:SS" or "HHMMSS") and violins are
    positioned proportionally on a 0-24 hour numeric x-axis.
- Color each patch's violins using the reference RGB for that patch and overlay
    individual sample points colored by their measured RGBs.
- Produce three combined histograms (H, S, V differences) across all samples.
- Produce a per-patch scatter plot: mean Hue vs mean absolute Hue error (patch
    points can be colored by the reference RGB). A CSV with scaled display
    values (H scaled 0-255, S/V scaled 0-255) is also written.

Usage (example):
        python scripts/colorchecker_plotter.py --infile combined.csv \
                --checkerfile 3D_colorchecker/color_checker/color_checker_values.csv

CLI arguments:
- --infile       Input CSV of samples (required)
- --checkerfile  Reference/checker CSV (required)
- --out          Optional path for the comparison CSV (defaults to infile_compared.csv)
- --overwrite    (not used currently for outputs)

Outputs (by default, saved next to the comparison CSV):
- <infile>_compared.csv      Comparison table with HSV and difference columns
- <infile>_disp.csv          Per-patch display summary (H scaled 0-255, S/V *255)
- *_violin_<PATCH>_H.png     Per-patch hue violin plots (one PNG per patch)
- *_violin_<PATCH>_S.png     Per-patch saturation violin plots
- *_violin_<PATCH>_V.png     Per-patch value violin plots
- *_histogram_H.png          Combined histogram of signed hue differences
- *_histogram_S.png          Combined histogram of saturation differences
- *_histogram_V.png          Combined histogram of value differences
- *_scatter_H_vs_Habs.png    Per-patch scatter: mean H vs mean |H error|

Dependencies: pandas, numpy, matplotlib, seaborn

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


def add_hsv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with HSV columns added (works on columns R,G,B)."""
    if not set(["R", "G", "B"]).issubset(df.columns):
        raise ValueError("Input DataFrame must contain columns named R, G and B")

    rgb = df[["R", "G", "B"]].to_numpy(dtype=float)
    # Normalize if values look like 0-255
    if rgb.max() > 1.0:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)

    # Convert to HSV
    hsv = mcolors.rgb_to_hsv(rgb)

    # Extract H, S, V
    h_norm = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    # Add to output DataFrame
    out = df.copy()
    out["H_norm"] = h_norm
    out["H"] = (h_norm * 360.0) % 360.0
    out["S"] = s
    out["V"] = v
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add HSV columns to CSVs with R,G,B columns")
    parser.add_argument("--infile", required=True, help="Input CSV file path")
    parser.add_argument("--out", required=False, help="Output CSV path (default: add _hsv before extension)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input file with the augmented CSV")
    parser.add_argument("--checkerfile", required=True, help="Path to checker file")
    args = parser.parse_args(argv)

    infile = Path(args.infile).expanduser().resolve()
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    checkerfile = Path(args.checkerfile).expanduser().resolve()
    if not checkerfile.exists():
        raise SystemExit(f"Checker file not found: {checkerfile}")
    
    checkerdf = pd.read_csv(checkerfile)

    # Combine the row and col columns to create a 'patch' column
    if {"row", "col"}.issubset(checkerdf.columns):
        row_s = checkerdf["row"].astype(str).str.strip().str.upper()
        col_s = checkerdf["col"].astype(str).str.strip()
        checkerdf["patch"] = row_s + col_s
        print(f"Added 'patch' column to checker dataframe ({len(checkerdf)} rows)")
    else:
        print("Checker file does not contain both 'row' and 'col' columns; skipping patch creation")

    # Load the dataframe containing sample measurements and add HSV columns
    df = pd.read_csv(infile)
    try:
        df_with_hsv = add_hsv_columns(df)
    except Exception as e:
        raise SystemExit(f"Failed to compute HSV: {e}")
    
    # Add HSV columns to checker dataframe
    try:
        checkerdf_with_hsv = add_hsv_columns(checkerdf)
    except Exception as e:
        raise SystemExit(f"Failed to compute HSV: {e}")
    # Merge sample data with reference checker values on 'patch'
    if "patch" not in df_with_hsv.columns:
        raise SystemExit("Input samples do not contain a 'patch' column — cannot match to checker references")

    # select reference HSV columns
    ref_cols = ["patch", "H", "S", "V"]
    if not set(ref_cols).issubset(checkerdf_with_hsv.columns):
        raise SystemExit("Checker dataframe does not contain required HSV columns after processing")

    # Add the reference measurement to the sample dataframe, so that it can be used to calculate differences
    merged = df_with_hsv.merge(
        checkerdf_with_hsv[ref_cols].rename(columns={"H": "H_ref", "S": "S_ref", "V": "V_ref"}),
        on="patch",
        how="left",
    )

    # Warn if any samples lack a matching reference
    missing_ref = merged[merged["H_ref"].isna()]
    if len(missing_ref) > 0:
        print(f"Warning: {len(missing_ref)} sample rows have no matching reference patch in checker file")

    # Compute circular signed hue difference in degrees in range [-180, 180]
    h_sample = merged["H"].to_numpy()
    h_ref = merged["H_ref"].to_numpy()
    h_diff_signed = (h_sample - h_ref + 180.0) % 360.0 - 180.0
    h_diff_abs = np.abs(h_diff_signed)

    # Saturation and Value differences (signed and absolute)
    s_diff = merged["S"].to_numpy() - merged["S_ref"].to_numpy()
    v_diff = merged["V"].to_numpy() - merged["V_ref"].to_numpy()
    s_diff_abs = np.abs(s_diff)
    v_diff_abs = np.abs(v_diff)

    # Combined HSV deviation metric (unitless): normalize hue by 180 degrees to [-1,1]
    hsv_euclid = np.sqrt((h_diff_signed / 180.0) ** 2 + (s_diff) ** 2 + (v_diff) ** 2)

    # Attach columns to dataframe
    merged["H_diff_signed_deg"] = h_diff_signed
    merged["H_diff_abs_deg"] = h_diff_abs
    merged["S_diff"] = s_diff
    merged["S_diff_abs"] = s_diff_abs
    merged["V_diff"] = v_diff
    merged["V_diff_abs"] = v_diff_abs
    merged["HSV_deviation_metric"] = hsv_euclid

    # Determine output path
    out_path = None
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = infile.with_name(infile.stem + "_compared" + infile.suffix)

    merged.to_csv(out_path, index=False)
    print(f"Wrote comparison CSV to {out_path} ({len(merged)} rows)")

    # Print simple per-patch summary (mean values and mean absolute deviations)
    # Compute raw means for S and V but compute a circular mean for H
    # because hue is an angle (0-360) and simple averaging can produce
    # incorrect results near the wraparound (e.g. 357 and 2 -> ~180).
    def circular_mean_deg(angles_deg: np.ndarray) -> float:
        """Compute circular mean of angles in degrees, returning value in [0,360).

        angles_deg may contain NaNs; they will be ignored. Returns np.nan if
        all values are NaN.
        """
        if angles_deg is None:
            return np.nan
        arr = np.asarray(angles_deg, dtype=float)
        if arr.size == 0:
            return np.nan
        # mask NaNs
        mask = ~np.isnan(arr)
        if not np.any(mask):
            return np.nan
        vals = arr[mask]
        # convert to radians
        rads = np.deg2rad(vals)
        c = np.nanmean(np.cos(rads))
        s = np.nanmean(np.sin(rads))
        mean_rad = np.arctan2(s, c)
        mean_deg = np.degrees(mean_rad) % 360.0
        return float(mean_deg)

    # Compute S/V means and absolute-deviation stats with a normal agg
    summary = merged.groupby("patch").agg(
        S_mean=("S", "mean"),
        V_mean=("V", "mean"),
        H_abs_mean=("H_diff_abs_deg", "mean"),
        S_abs_mean=("S_diff_abs", "mean"),
        V_abs_mean=("V_diff_abs", "mean"),
        count=("H", "count"),
    )

    # Compute circular mean for H separately and insert into the summary
    h_mean_series = merged.groupby("patch")["H"].apply(lambda x: circular_mean_deg(x.to_numpy()))
    summary.insert(0, "H_mean_deg", h_mean_series)

    # Create a display copy where H is scaled from 0-360 -> 0-255 and S,V multiplied by 255
    disp = summary.copy()
    # H: degrees (0-360)
    disp["H"] = disp["H_mean_deg"]
    # S and V: 0-1 -> 0-255
    disp["S"] = disp["S_mean"] * 255.0
    disp["V"] = disp["V_mean"] * 255.0

    # Keep ordering: show scaled H, S, V plus absolute deviation stats and count
    disp = disp[["H", "S", "V", "H_abs_mean", "S_abs_mean", "V_abs_mean", "count"]]

    print("Per-patch mean values (H scaled 0-255, S/V scaled 0-255) and mean absolute deviations:")
    print(disp.sort_index())

    # Save the display summary (scaled H, S, V) to CSV next to the comparison CSV
    try:
        disp_out = out_path.with_name(out_path.stem + "_disp.csv")
        disp.to_csv(disp_out, index=True)
        print(f"Wrote display summary CSV to {disp_out}")
    except Exception as e:
        print(f"Warning: failed to write display summary CSV: {e}")

    # --- Compute an affine RGB transform mapping measured -> reference and save it ---
    try:
        # Compute per-patch mean measured RGB from the merged dataframe
        measured_means = merged.groupby("patch")[['R', 'G', 'B']].mean()

        # Reference RGBs from checker dataframe (ensure normalized to 0-1)
        ref_rgb_df = checkerdf_with_hsv.set_index('patch')[['R', 'G', 'B']].copy()
        # Normalize reference if in 0-255 range
        if ref_rgb_df.values.size > 0 and np.nanmax(ref_rgb_df.values) > 1.0:
            ref_rgb_df = np.clip(ref_rgb_df / 255.0, 0.0, 1.0)

        # Align patches present in both measured and reference
        common = measured_means.index.intersection(ref_rgb_df.index)
        if len(common) >= 3:
            src = measured_means.loc[common][['R', 'G', 'B']].to_numpy(dtype=float)
            dst = ref_rgb_df.loc[common][['R', 'G', 'B']].to_numpy(dtype=float)

            # Remove any rows with NaNs
            mask = ~np.isnan(src).any(axis=1) & ~np.isnan(dst).any(axis=1)
            src = src[mask]
            dst = dst[mask]

            if src.shape[0] >= 3:
                # Solve for affine transform: dst ≈ [src, 1] @ T  where T shape (4,3)
                M = np.hstack([src, np.ones((src.shape[0], 1), dtype=float)])  # Nx4
                coeffs, *_ = np.linalg.lstsq(M, dst, rcond=None)  # coeffs shape (4,3)
                T = coeffs.T  # shape (3,4)
                A = T[:, :3]
                b = T[:, 3]

                # Save transform next to out_path
                transform_out = out_path.with_name(out_path.stem + "_rgb_transform.npz")
                np.savez(transform_out, A=A, b=b)
                print(f"Saved RGB affine transform to {transform_out} — matrix A shape {A.shape}, bias b shape {b.shape}")
            else:
                print("Not enough valid patch rows after filtering to compute RGB transform; need at least 3")
        else:
            print("Not enough matching patches between measurements and reference to compute RGB transform; need at least 3 patches in common")
    except Exception as e:
        print(f"Warning: failed to compute or save RGB transform: {e}")

    # --- Scatter: mean hue (deg) vs mean absolute hue error (deg) per patch ---
    try:
        scatter_df = summary.dropna(subset=["H_mean_deg", "H_abs_mean"]).copy()
        # Exclude patched in D row, since these are greyscale patches with undefined hue
        exclude_patches = {"D1", "D2", "D3"}
        before_count = len(scatter_df)
        scatter_df = scatter_df.loc[~scatter_df.index.isin(exclude_patches)].copy()
        after_count = len(scatter_df)
        if before_count != after_count:
            print(f"Excluded {before_count - after_count} patches from H vs H_abs scatter: {sorted(exclude_patches)}")
        
        
        if len(scatter_df) > 0:
            fig_sc, ax_sc = plt.subplots(figsize=(6, 6))
            # prepare colors per patch from checker references (fallback grey)
            # this is used to color the scatter points by their reference RGB
            colors = []
            for patch in scatter_df.index:
                try:
                    rrow = checkerdf_with_hsv.loc[checkerdf_with_hsv["patch"] == patch]
                    if len(rrow) > 0:
                        rgbvals = rrow.iloc[0][["R", "G", "B"]].to_numpy(dtype=float)
                        if np.nanmax(rgbvals) > 1.0:
                            rgbvals = np.clip(rgbvals / 255.0, 0.0, 1.0)
                        colors.append(tuple(rgbvals.tolist()))
                    else:
                        colors.append((0.6, 0.6, 0.6))
                except Exception:
                    colors.append((0.6, 0.6, 0.6))

            ax_sc.scatter(scatter_df["H_mean_deg"].to_numpy(), scatter_df["H_abs_mean"].to_numpy(), c=colors, s=40, edgecolors="k", alpha=0.9)
            # annotate patch labels
            for xi, yi, lab in zip(scatter_df["H_mean_deg"].to_numpy(), scatter_df["H_abs_mean"].to_numpy(), scatter_df.index):
                ax_sc.annotate(str(lab), (xi, yi), fontsize=8, xytext=(4, 2), textcoords="offset points")

            ax_sc.set_xlabel("Mean Hue (deg)")
            ax_sc.set_ylabel("Mean absolute Hue error (deg)")
            ax_sc.set_title("Per-patch: Mean Hue vs Mean absolute Hue error")
            ax_sc.set_xlim(0, 360)
            plt.tight_layout()
            out_scatter = out_path.with_name(out_path.stem + "_scatter_H_vs_Habs.png")
            fig_sc.savefig(str(out_scatter), dpi=200)
            plt.close(fig_sc)
            print(f"Saved per-patch H vs H_abs scatter to {out_scatter}")
    except Exception as e:
        print(f"Warning: failed to create H vs H_abs scatter: {e}")

    # --- Violin plots of hue difference over time for all patches ---
    if "time_hms" not in merged.columns:
        print("No 'time_hms' column found in merged data; skipping violin plots")
        return

    sns.set(style="whitegrid")

    # helper to normalize time tokens to HH:MM:SS strings
    def normalize_time(t):
        try:
            s = str(t).strip()
            if ":" in s:
                return s
            if len(s) == 6 and s.isdigit():
                return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
            return s
        except Exception:
            return str(t)

    patches = sorted(merged["patch"].dropna().unique())
    if len(patches) == 0:
        print("No patches found in merged data; skipping violin plots")
        return

    # Save plots next to the comparison CSV
    plots_dir = out_path.parent

    for plot_patch in patches:
        sub = merged[merged["patch"] == plot_patch].copy()
        sub = sub.dropna(subset=["time_hms", "H_diff_signed_deg"]).copy()
        if sub.empty:
            print(f"No valid data for patch {plot_patch}; skipping")
            continue

        # Parse times into numeric hours (0-24) for proportional placement
        def time_to_seconds(t):
            s = str(t).strip()
            if ":" in s:
                parts = s.split(":")
                if len(parts) >= 3:
                    try:
                        h = int(parts[0]) % 24
                        m = int(parts[1])
                        sec = int(parts[2])
                        return h * 3600 + m * 60 + sec
                    except Exception:
                        return None
            if len(s) == 6 and s.isdigit():
                try:
                    h = int(s[0:2]) % 24
                    m = int(s[2:4])
                    sec = int(s[4:6])
                    return h * 3600 + m * 60 + sec
                except Exception:
                    return None
            return None

        sub["time_seconds"] = sub["time_hms"].apply(time_to_seconds)
        sub = sub.dropna(subset=["time_seconds"]).copy()
        if sub.empty:
            print(f"No parsable times for patch {plot_patch}; skipping")
            continue

        sub["time_hours"] = (sub["time_seconds"].astype(float) / 3600.0) % 24.0

        # Group by exact hour value (float). Use sorted unique hour positions for plotting
        hour_positions = sorted(sub["time_hours"].unique())
        if len(hour_positions) == 0:
            print(f"No valid numeric times for patch {plot_patch}; skipping")
            continue

        data_list = [sub.loc[np.isclose(sub["time_hours"], h), "H_diff_signed_deg"].to_numpy() for h in hour_positions]

        fig, ax = plt.subplots(figsize=(6.5, 3.75))

        # Determine width based on minimal spacing between hour positions
        if len(hour_positions) > 1:
            dists = np.diff(sorted(hour_positions))
            min_dist = max(0.05, float(np.min(dists)))
            width = min_dist * 0.8
        else:
            width = 0.5

        # Use matplotlib violinplot so we can set facecolors reliably; positions are numeric hours
        parts = ax.violinplot(data_list, positions=hour_positions, widths=width, showmeans=False, showmedians=True)

        # Determine patch reference RGB (from checkerdf_with_hsv)
        ref_rgb = None
        try:
            ref_row = checkerdf_with_hsv.loc[checkerdf_with_hsv["patch"] == plot_patch]
            if len(ref_row) > 0:
                rgbvals = ref_row.iloc[0][["R", "G", "B"]].to_numpy(dtype=float)
                if np.nanmax(rgbvals) > 1.0:
                    rgbvals = np.clip(rgbvals / 255.0, 0.0, 1.0)
                ref_rgb = tuple(rgbvals.tolist())
        except Exception:
            ref_rgb = None

        if ref_rgb is None:
            facecolor = (0.6, 0.6, 0.6)
        else:
            facecolor = ref_rgb

        # Color each violin body with the patch reference color
        for pc in parts["bodies"]:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        # Overlay sample points colored by their sample RGB
        # jitter x positions for visibility
        sample_rgbs = sub[["R", "G", "B"]].to_numpy(dtype=float)
        if sample_rgbs.size > 0 and np.nanmax(sample_rgbs) > 1.0:
            sample_rgbs = np.clip(sample_rgbs / 255.0, 0.0, 1.0)

        for i, h in enumerate(hour_positions):
            grp = sub.loc[np.isclose(sub["time_hours"], h)]
            if grp.empty:
                continue
            xs = np.random.normal(loc=h, scale=width * 0.12, size=len(grp))
            rgbs = grp[["R", "G", "B"]].to_numpy(dtype=float)
            if rgbs.size > 0 and np.nanmax(rgbs) > 1.0:
                rgbs = np.clip(rgbs / 255.0, 0.0, 1.0)
            ax.scatter(xs, grp["H_diff_signed_deg"].to_numpy(), c=rgbs, s=10, edgecolors="none", alpha=0.5)

        # X-axis numeric 0-24 hours
        ax.set_xlim(0, 24)
        ax.set_ylim(-50, 50)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([f"{int(x):02d}:00" for x in np.arange(0, 25, 2)], rotation=45)
        ax.set_ylabel("Signed Hue difference (deg)")
        ax.set_title(f"Hue difference over time — patch {plot_patch}")
        plt.tight_layout()

        # Save hue violin (use _H suffix)
        violin_out_h = plots_dir / f"{out_path.stem}_violin_{plot_patch}_H.png"
        fig.savefig(str(violin_out_h), dpi=200)
        plt.close(fig)
        print(f"Saved hue violin plot to {violin_out_h}")

        # --- Saturation violin (S_diff) ---
        data_list_s = [sub.loc[np.isclose(sub["time_hours"], h), "S_diff"].to_numpy() for h in hour_positions]
        fig_s, ax_s = plt.subplots(figsize=(6.5, 3.75))
        parts_s = ax_s.violinplot(data_list_s, positions=hour_positions, widths=width, showmeans=False, showmedians=True)
        for pc in parts_s["bodies"]:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        for i, h in enumerate(hour_positions):
            grp = sub.loc[np.isclose(sub["time_hours"], h)]
            if grp.empty:
                continue
            xs = np.random.normal(loc=h, scale=width * 0.12, size=len(grp))
            rgbs = grp[["R", "G", "B"]].to_numpy(dtype=float)
            if rgbs.size > 0 and np.nanmax(rgbs) > 1.0:
                rgbs = np.clip(rgbs / 255.0, 0.0, 1.0)
            ax_s.scatter(xs, grp["S_diff"].to_numpy(), c=rgbs, s=10, edgecolors="none", alpha=0.5)
        ax_s.set_xlim(0, 24)
        ax_s.set_ylim(-0.7, 0.7)
        ax_s.set_xticks(np.arange(0, 25, 2))
        ax_s.set_xticklabels([f"{int(x):02d}:00" for x in np.arange(0, 25, 2)], rotation=45)
        ax_s.set_ylabel("Signed Saturation difference")
        ax_s.set_title(f"Saturation difference over 24h — patch {plot_patch}")
        plt.tight_layout()
        violin_out_s = plots_dir / f"{out_path.stem}_violin_{plot_patch}_S.png"
        fig_s.savefig(str(violin_out_s), dpi=200)
        plt.close(fig_s)
        print(f"Saved saturation violin plot to {violin_out_s}")

        # --- Value violin (V_diff) ---
        data_list_v = [sub.loc[np.isclose(sub["time_hours"], h), "V_diff"].to_numpy() for h in hour_positions]
        fig_v, ax_v = plt.subplots(figsize=(6.5, 3.75))
        parts_v = ax_v.violinplot(data_list_v, positions=hour_positions, widths=width, showmeans=False, showmedians=True)
        for pc in parts_v["bodies"]:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        for i, h in enumerate(hour_positions):
            grp = sub.loc[np.isclose(sub["time_hours"], h)]
            if grp.empty:
                continue
            xs = np.random.normal(loc=h, scale=width * 0.12, size=len(grp))
            rgbs = grp[["R", "G", "B"]].to_numpy(dtype=float)
            if rgbs.size > 0 and np.nanmax(rgbs) > 1.0:
                rgbs = np.clip(rgbs / 255.0, 0.0, 1.0)
            ax_v.scatter(xs, grp["V_diff"].to_numpy(), c=rgbs, s=10, edgecolors="none", alpha=0.5)
        ax_v.set_xlim(0, 24)
        ax_v.set_ylim(-0.5, 0)
        ax_v.set_xticks(np.arange(0, 25, 2))
        ax_v.set_xticklabels([f"{int(x):02d}:00" for x in np.arange(0, 25, 2)], rotation=45)
        ax_v.set_ylabel("Signed Value difference")
        ax_v.set_title(f"Value difference over 24h — patch {plot_patch}")
        plt.tight_layout()
        violin_out_v = plots_dir / f"{out_path.stem}_violin_{plot_patch}_V.png"
        fig_v.savefig(str(violin_out_v), dpi=200)
        plt.close(fig_v)
        print(f"Saved value violin plot to {violin_out_v}")

    # --- Combined histograms across all patches/timepoints ---
    # Hue differences (signed degrees)
    h_all = merged["H_diff_signed_deg"].dropna()
    if len(h_all) > 0:
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        bins_h = np.linspace(-180, 180, 37)  # 10-degree bins
        ax_h.hist(h_all.to_numpy(), bins=bins_h, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax_h.set_xlabel("Signed Hue difference (deg)")
        ax_h.set_ylabel("Count")
        ax_h.set_title("Histogram of signed hue differences (all samples)")
        plt.tight_layout()
        out_hist_h = plots_dir / f"{out_path.stem}_histogram_H.png"
        fig_h.savefig(str(out_hist_h), dpi=200)
        plt.close(fig_h)
        print(f"Saved combined hue histogram to {out_hist_h}")

    # Saturation differences
    s_all = merged["S_diff"].dropna()
    if len(s_all) > 0:
        # choose symmetric range around zero for meaningful visualization
        smin, smax = float(s_all.min()), float(s_all.max())
        s_bound = max(abs(smin), abs(smax), 0.01)
        bins_s = np.linspace(-s_bound, s_bound, 50)
        fig_s_all, ax_s_all = plt.subplots(figsize=(8, 4))
        ax_s_all.hist(s_all.to_numpy(), bins=bins_s, color="#55A868", edgecolor="black", alpha=0.8)
        ax_s_all.set_xlabel("Signed Saturation difference")
        ax_s_all.set_ylabel("Count")
        ax_s_all.set_title("Histogram of saturation differences (all samples)")
        plt.tight_layout()
        out_hist_s = plots_dir / f"{out_path.stem}_histogram_S.png"
        fig_s_all.savefig(str(out_hist_s), dpi=200)
        plt.close(fig_s_all)
        print(f"Saved combined saturation histogram to {out_hist_s}")

    # Value differences
    v_all = merged["V_diff"].dropna()
    if len(v_all) > 0:
        vmin, vmax = float(v_all.min()), float(v_all.max())
        v_bound = max(abs(vmin), abs(vmax), 0.01)
        bins_v = np.linspace(-v_bound, v_bound, 50)
        fig_v_all, ax_v_all = plt.subplots(figsize=(8, 4))
        ax_v_all.hist(v_all.to_numpy(), bins=bins_v, color="#C44E52", edgecolor="black", alpha=0.8)
        ax_v_all.set_xlabel("Signed Value difference")
        ax_v_all.set_ylabel("Count")
        ax_v_all.set_title("Histogram of value differences (all samples)")
        plt.tight_layout()
        out_hist_v = plots_dir / f"{out_path.stem}_histogram_V.png"
        fig_v_all.savefig(str(out_hist_v), dpi=200)
        plt.close(fig_v_all)
        print(f"Saved combined value histogram to {out_hist_v}")



if __name__ == "__main__":
    main()
