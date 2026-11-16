# -*- coding: utf-8 -*-
"""
Generates spatial scatter plots of the stone locations
identified using different DBSCAN distance thresholds
(eps), allowing visual comparison of the resulting
stone distributions.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_one(stones_file, out_file, title_extra=""):
    df = pd.read_csv(stones_file)

    if "stone_x_utm" not in df.columns or "stone_y_utm" not in df.columns:
        print(f"[WARN] Missing required columns in {stones_file}")
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(
        df["stone_x_utm"],
        df["stone_y_utm"],
        s=10,
    )

    plt.xlabel("UTM X (m)")
    plt.ylabel("UTM Y (m)")
    plt.title(f"DBSCAN Stones {title_extra}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"[OK] Saved plot: {out_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot all DBSCAN sweep results from a folder."
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing stones_eps*.csv files",
    )
    ap.add_argument(
        "--out_dir",
        default="dbscan_plots",
        help="Output folder for plots",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "stones_eps*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("[ERROR] No stones_eps*.csv files found.")
        return

    for f in files:
        filename = os.path.basename(f)
        tag = filename.replace("stones_", "").replace(".csv", "")

        out_plot = os.path.join(args.out_dir, f"plot_{tag}.png")

        plot_one(
            stones_file=f,
            out_file=out_plot,
            title_extra=f"({tag})",
        )


if __name__ == "__main__":
    main()