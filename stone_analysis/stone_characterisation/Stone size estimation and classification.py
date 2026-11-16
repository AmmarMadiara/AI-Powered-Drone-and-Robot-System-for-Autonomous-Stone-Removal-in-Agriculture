# -*- coding: utf-8 -*-
"""
Converts pixel-based detection areas into real-world measurements
using ground sampling distance (GSD), estimates effective stone
footprints and equivalent circular diameters, and assigns
size classes based on configurable diameter thresholds.

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(
        description="Compute real-world stone size (meters) per detection and assign size classes."
    )
    ap.add_argument("-i", "--input", required=True, help="DBSCAN-ready detections CSV")
    ap.add_argument(
        "-o", "--output",
        default="detections_with_real_size_and_class.csv",
        help="Output CSV with size + size_class_real"
    )
    ap.add_argument("--area_col", default="area_px", help="BBox area in px^2")
    ap.add_argument("--gsd_col", default="gsd_m_per_px", help="GSD in m/px")
    ap.add_argument(
        "--fill_factor",
        type=float,
        default=0.60,
        help="Fraction of bbox area that is actual stone footprint (0.3–0.9 typical)"
    )

    # Real-world size thresholds (meters)
    ap.add_argument("--t_small", type=float, default=0.10, help="small < t_small (m)")
    ap.add_argument("--t_medium", type=float, default=0.20, help="medium < t_medium (m)")
    ap.add_argument("--t_large", type=float, default=0.30, help="large < t_large (m)")

    args = ap.parse_args()

    if not (0.05 <= args.fill_factor <= 1.0):
        raise ValueError("--fill_factor must be in (0.05..1.0].")

    df = pd.read_csv(args.input, sep=None, engine="python")

    # Validate required columns
    for c in [args.area_col, args.gsd_col]:
        if c not in df.columns:
            raise ValueError(
                f"Missing required column '{c}'. "
                f"Found columns: {list(df.columns)}"
            )

    # Ensure numeric
    df[args.area_col] = pd.to_numeric(df[args.area_col], errors="coerce")
    df[args.gsd_col] = pd.to_numeric(df[args.gsd_col], errors="coerce")

    # Drop rows that cannot be sized
    df = df.dropna(subset=[args.area_col, args.gsd_col]).copy()

    # --- Size computation (per detection) ---

    # BBox area on ground
    df["area_m2_bbox"] = df[args.area_col] * (df[args.gsd_col] ** 2)

    # Effective stone footprint
    df["area_m2_eff"] = args.fill_factor * df["area_m2_bbox"]

    # Equivalent circular diameter (meters)
    df["eq_diameter_m_eff"] = 2.0 * np.sqrt(df["area_m2_eff"] / np.pi)

    # --- Size class (real-world, meters) ---
    d = df["eq_diameter_m_eff"].astype(float)

    df["size_class_real"] = pd.cut(
        d,
        bins=[-1, args.t_small, args.t_medium, args.t_large, 1e9],
        labels=["small", "medium", "large", "very_large"],
    )

    # Record assumptions
    df["fill_factor_used"] = args.fill_factor

    df.to_csv(args.output, index=False)

    print(f"[OK] Wrote: {args.output}")
    print("[INFO] Added columns:")
    print("  - area_m2_bbox")
    print("  - area_m2_eff")
    print("  - eq_diameter_m_eff (meters)")
    print("  - size_class_real")
    print("\n[INFO] Size class distribution:")
    print(
        df["size_class_real"]
        .value_counts()
        .reindex(["small", "medium", "large", "very_large"], fill_value=0)
    )

if __name__ == "__main__":
    main()
