#!/usr/bin/env python3
"""Convert CIELab input file to sRGB output.

Reads a tab-separated file with rows like:
Label<TAB>L,a<TAB>a,b<TAB>b (decimal comma or dot allowed)

Example input lines (decimal commas are accepted):
A1	37,54	14,37	14,92

Output is a TSV with columns: Label\tR\tG\tB\tHEX

Usage: python scripts/lab_to_rgb.py input.tsv -o output.tsv
"""
import argparse
import re
import sys
from typing import Tuple, Optional


def parse_lab_line(line: str) -> Optional[Tuple[str, float, float, float]]:
    # Expect label then three numeric values; numbers may use comma as decimal sep.
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 2:
        # try whitespace separation fallback
        parts = re.split(r"\s+", line)
    label = parts[0]
    # gather numeric tokens from the rest of the line (handles stray text)
    rest = "\t".join(parts[1:])
    # match floats with comma or dot decimal separators, optional sign
    nums = re.findall(r"[+-]?\d+[\.,]?\d*", rest)
    if len(nums) < 3:
        return None
    def norm(n: str) -> float:
        return float(n.replace(',', '.'))
    L = norm(nums[0])
    a = norm(nums[1])
    b = norm(nums[2])
    return label, L, a, b


def lab_to_xyz(L: float, a: float, b: float) -> Tuple[float, float, float]:
    # Using D65 reference white (Xn, Yn, Zn)
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_to_xyz_component(f: float, ref: float, L_val: float) -> float:
        if f ** 3 > 0.008856:
            return ref * (f ** 3)
        else:
            return ref * ((116.0 * f - 16.0) / 903.3)

    # For Y we can also check L threshold, but the above function handles it correctly
    X = f_to_xyz_component(fx, Xn, L)
    Y = f_to_xyz_component(fy, Yn, L)
    Z = f_to_xyz_component(fz, Zn, L)
    return X, Y, Z


def xyz_to_srgb(X: float, Y: float, Z: float) -> Tuple[int, int, int]:
    # Convert XYZ (with Y=100 reference) to linear sRGB (D65), then gamma correct
    # First scale to 0..1
    x = X / 100.0
    y = Y / 100.0
    z = Z / 100.0

    # sRGB conversion matrix (D65)
    r_lin = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g_lin = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_lin = 0.0557 * x - 0.2040 * y + 1.0570 * z

    def compand(c: float) -> float:
        # clamp tiny negatives before companding to avoid math domain issues
        if c <= 0.0:
            c = 0.0
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    r = compand(r_lin)
    g = compand(g_lin)
    b = compand(b_lin)

    def to_8bit(v: float) -> int:
        v = max(0.0, min(1.0, v))
        return int(round(v * 255.0))

    return to_8bit(r), to_8bit(g), to_8bit(b)


def lab_to_rgb_tuple(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """Convert a single Lab triplet to 8-bit sRGB using scikit-image when available.

    Falls back to the previous XYZ/matrix method if scikit-image is not installed.
    """
    try:
        # Import locally to avoid a hard dependency at module import time
        import numpy as _np
        from skimage import color

        lab = _np.array([[[L, a, b]]], dtype=float)
        # skimage.color.lab2rgb returns float image in [0, 1]
        rgb = color.lab2rgb(lab,illuminant="D50")[0, 0]
        # Clamp and convert to 8-bit
        def to_8bit(v: float) -> int:
            v = float(v)
            if v != v:  # NaN guard
                v = 0.0
            v = max(0.0, min(1.0, v))
            return int(round(v * 255.0))

        return to_8bit(rgb[0]), to_8bit(rgb[1]), to_8bit(rgb[2])
    except Exception:
        print("Falling back on pure-Python Lab to sRGB conversion.")
        # Fall back to the pure-Python implementation if scikit-image or numpy
        # are not available, or if conversion fails for any reason.
        X, Y, Z = lab_to_xyz(L, a, b)
        return xyz_to_srgb(X, Y, Z)


def process_file(infile: str, outfile: Optional[str] = None) -> None:
    out_lines = []
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_lab_line(line)
            if parsed is None:
                continue
            label, L, a, b = parsed
            r, g, bb = lab_to_rgb_tuple(L, a, b)
            hexv = f"#{r:02X}{g:02X}{bb:02X}"
            out_lines.append(f"{label}\t{r}\t{g}\t{bb}\t{hexv}\n")

    if outfile:
        with open(outfile, 'w', encoding='utf-8') as out:
            out.write("Label\tR\tG\tB\tHEX\n")
            out.writelines(out_lines)
    else:
        sys.stdout.write("Label\tR\tG\tB\tHEX\n")
        sys.stdout.writelines(out_lines)


def main():
    parser = argparse.ArgumentParser(description='Convert CIELab TSV to sRGB TSV')
    parser.add_argument('input', help='Input TSV file (label and three Lab values)')
    parser.add_argument('-o', '--output', help='Output TSV file (defaults to stdout)')
    args = parser.parse_args()
    process_file(args.input, args.output)


if __name__ == '__main__':
    main()
