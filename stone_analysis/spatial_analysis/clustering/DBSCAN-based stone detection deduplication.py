# -*- coding: utf-8 -*-
"""
Groups spatially close detections using DBSCAN clustering
in UTM coordinates and aggregates each cluster into a
single representative stone record, including spatial
coordinates and available stone measurements.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate detection-level objects using DBSCAN in UTM space."
    )
    ap.add_argument("-i", "--input", required=True, help="Detection-level CSV (e.g. detections_large_plus.csv)")
    ap.add_argument(
        "-o", "--output",
        default="stones_large_dedup.csv",
        help="Output stone-level CSV after DBSCAN"
    )
    ap.add_argument("--x_col", default="x_utm", help="UTM X column")
    ap.add_argument("--y_col", default="y_utm", help="UTM Y column")
    ap.add_argument("--eps", type=float, default=0.8,
                    help="DBSCAN eps in meters (max distance for same stone)")
    ap.add_argument("--min_samples", type=int, default=2,
                    help="Minimum detections to accept a stone")

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Validate required columns
    for c in [args.x_col, args.y_col]:
        if c not in df.columns:
            raise ValueError(
                f"Missing required column '{c}'. "
                f"Found columns: {list(df.columns)}"
            )

    # Extract coordinates
    coords = df[[args.x_col, args.y_col]].to_numpy(dtype=float)

    # Run DBSCAN (UTM meters)
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels = db.fit_predict(coords)

    df["cluster_id"] = labels

    # Remove noise
    dfc = df[df["cluster_id"] >= 0].copy()

    if dfc.empty:
        print("[WARN] No clusters found. Try increasing eps.")
        return

    # Aggregate one row per cluster → one physical stone
    agg_dict = {
        "stone_x_utm": (args.x_col, "median"),
        "stone_y_utm": (args.y_col, "median"),
        "n_detections": ("cluster_id", "size"),
    }

    if "eq_diameter_m_eff" in dfc.columns:
        agg_dict["eq_diameter_m_eff_median"] = ("eq_diameter_m_eff", "median")
        agg_dict["eq_diameter_m_eff_p25"] = (
            "eq_diameter_m_eff",
            lambda x: np.nanpercentile(x, 25),
        )
        agg_dict["eq_diameter_m_eff_p75"] = (
            "eq_diameter_m_eff",
            lambda x: np.nanpercentile(x, 75),
        )

    if "size_class_real" in dfc.columns:
        agg_dict["size_class_real"] = ("size_class_real", "first")

    stones = (
        dfc.groupby("cluster_id", as_index=False)
           .agg(**agg_dict)
    )

    stones.to_csv(args.output, index=False)

    print(f"[OK] Wrote: {args.output}")
    print(f"[INFO] eps = {args.eps:.2f} m | min_samples = {args.min_samples}")
    print(f"[INFO] Input detections: {len(df)}")
    print(f"[INFO] Output stones: {len(stones)}")

if __name__ == "__main__":
    main()
