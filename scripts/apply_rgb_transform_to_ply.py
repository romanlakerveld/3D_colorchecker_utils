"""Apply an affine RGB transform to a .ply point cloud's vertex colors.

Example:
    python scripts/apply_rgb_transform_to_ply.py --ply input.ply --transform combined_compared_rgb_transform.npz

The transform .npz is expected to contain arrays 'A' (3x3) and 'b' (3,) such that
    rgb_out = A @ rgb_in + b
where rgb_in are in 0..1. Output colors are clipped to 0..1 before saving.
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def apply_transform_to_colors(colors: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply A (3x3) and b (3,) to an (N,3) color array. Colors assumed in 0..1."""
    if colors.size == 0:
        return colors
    c = colors.astype(float)
    # If colors are in 0-255, normalize
    if np.nanmax(c) > 1.0:
        c = np.clip(c / 255.0, 0.0, 1.0)
    corrected = (A @ c.T).T + b.reshape((1, 3))
    corrected = np.clip(corrected, 0.0, 1.0)
    return corrected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply saved RGB affine transform to PLY colors")
    parser.add_argument("--ply", required=True, help="Input PLY file path")
    parser.add_argument("--transform", required=True, help="NPZ file with A and b arrays")
    parser.add_argument("--out", required=False, help="Optional output PLY path")
    args = parser.parse_args(argv)

    ply_path = Path(args.ply).expanduser().resolve()
    if not ply_path.exists():
        raise SystemExit(f"PLY file not found: {ply_path}")

    transform_path = Path(args.transform).expanduser().resolve()
    if not transform_path.exists():
        raise SystemExit(f"Transform file not found: {transform_path}")

    data = np.load(str(transform_path))
    if "A" not in data or "b" not in data:
        raise SystemExit("Transform file must contain arrays 'A' and 'b'")
    A = data["A"]
    b = data["b"]
    if A.shape != (3, 3) or b.shape not in [(3,), (3, 1)]:
        raise SystemExit(f"Unexpected transform shapes: A {A.shape}, b {b.shape}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        raise SystemExit("Input PLY contains no vertex colors to transform")

    corrected = apply_transform_to_colors(colors, A, b)
    pcd.colors = o3d.utility.Vector3dVector(corrected)

    if args.out:
        out_ply = Path(args.out).expanduser().resolve()
    else:
        out_ply = ply_path.with_name(ply_path.stem + "_corrected" + ply_path.suffix)

    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"Wrote corrected PLY to {out_ply}")


if __name__ == "__main__":
    main()
