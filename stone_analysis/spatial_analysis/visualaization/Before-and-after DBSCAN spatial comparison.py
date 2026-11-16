# -*- coding: utf-8 -*-
"""
Generates overlay scatter plots comparing original stone
detections with deduplicated stone locations obtained
through DBSCAN clustering. Supports single-result and
multiple-parameter comparison modes.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay plot(s): BEFORE (detections) vs AFTER (deduplicated stones).

Supports:
- single AFTER file (old behavior): --after path.csv -o out.png
- sweep folder with many AFTER files: --after_dir results_folder --out_dir plots_folder

AFTER sweep expects files like: stones_eps0p80.csv (or stones_*.csv).
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def pick_xy(df):
    """Detect which columns contain UTM coordinates."""
    if "stone_x_utm" in df.columns and "stone_y_utm" in df.columns:
        return "stone_x_utm", "stone_y_utm"
    if "x_utm" in df.columns and "y_utm" in df.columns:
        return "x_utm", "y_utm"
    raise ValueError(
        "No UTM columns found (need stone_x_utm/stone_y_utm or x_utm/y_utm)."
    )


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    xcol, ycol = pick_xy(df)
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = x.notna() & y.notna()
    return x[m], y[m]


def _tag_from_filename(path: str) -> str:
    """stones_eps0p80.csv -> eps0p80 ; stones_anything.csv -> anything"""
    base = os.path.basename(path)
    if base.startswith("stones_") and base.lower().endswith(".csv"):
        return base[len("stones_") : -len(".csv")]
    if base.lower().endswith(".csv"):
        return base[:-len(".csv")]
    return base


def plot_overlay(
    before_csv: str,
    after_csv: str,
    output_png: str,
    title: str,
    before_label: str,
    after_label: str,
    before_size: float,
    after_size: float,
    before_alpha: float,
    after_alpha: float,
):
    xb, yb = _load_xy(before_csv)
    xa, ya = _load_xy(after_csv)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        xb,
        yb,
        s=before_size,
        alpha=before_alpha,
        label=f"{before_label} (n={len(xb)})",
    )
    plt.scatter(
        xa,
        ya,
        s=after_size,
        alpha=after_alpha,
        label=f"{after_label} (n={len(xa)})",
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("UTM Easting (m)")
    plt.ylabel("UTM Northing (m)")
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Saved: {output_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Overlay plot: before vs after DBSCAN (single file or sweep folder)."
    )
    ap.add_argument("--before", required=True, help="CSV before DBSCAN (detections)")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--after", help="CSV after DBSCAN (deduplicated stones)")
    group.add_argument(
        "--after_dir",
        help="Folder containing DBSCAN sweep results (stones_*.csv)",
    )

    # Single-output mode
    ap.add_argument(
        "-o",
        "--output",
        default="before_after_dbscan.png",
        help="Output PNG filename (only used with --after)",
    )

    # Sweep-output mode
    ap.add_argument(
        "--out_dir",
        default="before_after_dbscan_plots",
        help="Output folder for plots (only used with --after_dir)",
    )
    ap.add_argument(
        "--pattern",
        default="stones_*.csv",
        help="Glob pattern inside --after_dir (default: stones_*.csv)",
    )

    ap.add_argument("--title", default="Before vs After DBSCAN", help="Base plot title")
    ap.add_argument("--before_label", default="Before", help="Legend label for BEFORE")
    ap.add_argument("--after_label", default="After", help="Legend label for AFTER")
    ap.add_argument(
        "--before_size", type=float, default=6, help="Marker size for BEFORE points"
    )
    ap.add_argument(
        "--after_size", type=float, default=12, help="Marker size for AFTER points"
    )
    ap.add_argument(
        "--before_alpha", type=float, default=0.30, help="Alpha for BEFORE points"
    )
    ap.add_argument(
        "--after_alpha", type=float, default=0.90, help="Alpha for AFTER points"
    )

    args = ap.parse_args()

    if args.after:
        # Old behavior: one plot
        plot_overlay(
            before_csv=args.before,
            after_csv=args.after,
            output_png=args.output,
            title=args.title,
            before_label=args.before_label,
            after_label=args.after_label,
            before_size=args.before_size,
            after_size=args.after_size,
            before_alpha=args.before_alpha,
            after_alpha=args.after_alpha,
        )
        return

    # New behavior: sweep folder
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.after_dir, args.pattern)))
    if not files:
        raise SystemExit(
            f"[ERROR] No files matched {args.pattern!r} in folder: {args.after_dir}"
        )

    for after_csv in files:
        tag = _tag_from_filename(after_csv)
        out_png = os.path.join(args.out_dir, f"before_after_{tag}.png")
        title = f"{args.title} ({tag})"

        plot_overlay(
            before_csv=args.before,
            after_csv=after_csv,
            output_png=out_png,
            title=title,
            before_label=args.before_label,
            after_label=args.after_label,
            before_size=args.before_size,
            after_size=args.after_size,
            before_alpha=args.before_alpha,
            after_alpha=args.after_alpha,
        )


if __name__ == "__main__":
    main()