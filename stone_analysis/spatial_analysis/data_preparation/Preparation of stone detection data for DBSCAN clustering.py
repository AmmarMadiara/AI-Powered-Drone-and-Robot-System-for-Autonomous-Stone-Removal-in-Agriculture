# -*- coding: utf-8 -*-
"""
Extracts spatial coordinates and relevant detection attributes
from the original detection dataset and generates a simplified
CSV file for subsequent spatial clustering and validation.
"""

import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(
        description="Prepare a DBSCAN-ready CSV by keeping only useful columns."
    )
    ap.add_argument("-i", "--input", required=True, help="Raw detections CSV")
    ap.add_argument(
        "-o", "--output",
        default="detections_dbscan_ready.csv",
        help="Output CSV for DBSCAN"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=None, engine="python")

    # Columns that are useful for DBSCAN + validation
    keep_cols = [
        "det_id",
        "image_id",
        "x_utm",
        "y_utm",
        "cam_x",
        "cam_y",
        "yaw_deg",
        "area_px",
        "w_px",
        "h_px",
        "gsd_m_per_px",
    ]

    # Keep only columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    if "x_utm" not in keep_cols or "y_utm" not in keep_cols:
        raise ValueError(
            "x_utm and y_utm are required for DBSCAN but were not found."
        )

    df_out = df[keep_cols].copy()

    df_out.to_csv(args.output, index=False)

    print(f"[OK] Wrote DBSCAN-ready file: {args.output}")
    print("[INFO] Columns kept:")
    for c in keep_cols:
        print(f"  - {c}")

if __name__ == "__main__":
    main()
