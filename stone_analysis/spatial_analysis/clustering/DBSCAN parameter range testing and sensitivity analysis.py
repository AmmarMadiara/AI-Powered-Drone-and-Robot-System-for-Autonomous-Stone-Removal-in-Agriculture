# -*- coding: utf-8 -*-
"""

Runs DBSCAN clustering using multiple spatial distance
thresholds (eps) to evaluate their effects on detection
grouping, noise identification and the resulting number
of individual stone clusters. Saves stone-level results
and a summary for each tested parameter value.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def fmt_eps(eps: float) -> str:
    """Format eps for filenames: 0.8 -> eps0p80"""
    return f"eps{eps:.2f}".replace(".", "p")


def run_one_eps(df: pd.DataFrame, x_col: str, y_col: str, eps: float, min_samples: int):
    # Extract coordinates
    coords = df[[x_col, y_col]].to_numpy(dtype=float)

    # Run DBSCAN (UTM meters)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    df_labeled = df.copy()
    df_labeled["cluster_id"] = labels

    # Split noise vs clustered
    n_total = len(df_labeled)
    n_noise = int((df_labeled["cluster_id"] < 0).sum())
    dfc = df_labeled[df_labeled["cluster_id"] >= 0].copy()

    if dfc.empty:
        # No clusters produced
        summary = {
            "eps": eps,
            "min_samples": min_samples,
            "input_detections": n_total,
            "noise_detections": n_noise,
            "clustered_detections": 0,
            "output_stones": 0,
        }
        return df_labeled, None, summary

    # Aggregate one row per cluster → one physical stone
    agg_dict = {
        "stone_x_utm": (x_col, "median"),
        "stone_y_utm": (y_col, "median"),
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

    summary = {
        "eps": eps,
        "min_samples": min_samples,
        "input_detections": n_total,
        "noise_detections": n_noise,
        "clustered_detections": int(len(dfc)),
        "output_stones": int(len(stones)),
    }

    return df_labeled, stones, summary


def float_range(start: float, stop: float, step: float):
    """Inclusive range with float step."""
    if step <= 0:
        raise ValueError("--eps_step must be > 0")
    # include stop (with small tolerance)
    vals = []
    x = start
    tol = step / 10.0
    while x <= stop + tol:
        vals.append(round(x, 10))
        x += step
    return vals


def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate detection-level objects using DBSCAN in UTM space. "
                    "Can sweep eps over a range and save results per eps."
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Detection-level CSV (e.g. detections_large_plus.csv)")
    ap.add_argument("--x_col", default="x_utm", help="UTM X column")
    ap.add_argument("--y_col", default="y_utm", help="UTM Y column")

    # Single eps OR range sweep
    ap.add_argument("--eps", type=float, default=None,
                    help="Run a single DBSCAN eps (meters). If omitted, uses --eps_range.")
    ap.add_argument("--eps_range", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help="Sweep eps from START to END (meters), inclusive. Example: --eps_range 0.6 1.4")
    ap.add_argument("--eps_step", type=float, default=0.1,
                    help="Step for eps sweep (meters). Default 0.1")

    ap.add_argument("--min_samples", type=int, default=2,
                    help="Minimum detections to accept a stone")

    ap.add_argument("--out_dir", default="dbscan_sweep_results",
                    help="Output directory for results (created if missing)")

    ap.add_argument("--save_labeled", action="store_true",
                    help="Also save detections with cluster_id for each eps")

    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.input)

    # Validate required columns
    for c in [args.x_col, args.y_col]:
        if c not in df.columns:
            raise ValueError(
                f"Missing required column '{c}'. Found columns: {list(df.columns)}"
            )

    # Ensure numeric and drop NaNs for coords (DBSCAN can't handle NaNs)
    df = df.copy()
    df[args.x_col] = pd.to_numeric(df[args.x_col], errors="coerce")
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[args.x_col, args.y_col]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with NaN coords in {args.x_col}/{args.y_col}")

    # Decide eps list
    if args.eps is not None:
        eps_list = [float(args.eps)]
    else:
        if args.eps_range is None:
            # Keep old behavior default if nothing provided
            eps_list = [0.8]
            print("[INFO] No --eps or --eps_range provided. Using default eps=0.8")
        else:
            start, end = args.eps_range
            if end < start:
                start, end = end, start
            eps_list = float_range(start, end, args.eps_step)

    # Prepare output folder
    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []

    for eps in eps_list:
        tag = fmt_eps(eps)

        df_labeled, stones, summary = run_one_eps(
            df=df,
            x_col=args.x_col,
            y_col=args.y_col,
            eps=eps,
            min_samples=args.min_samples,
        )
        summaries.append(summary)

        # Save outputs
        if stones is not None and len(stones) > 0:
            out_stones = os.path.join(args.out_dir, f"stones_{tag}.csv")
            stones.to_csv(out_stones, index=False)
            print(f"[OK] Wrote stones: {out_stones}  (n={len(stones)})")
        else:
            print(f"[WARN] eps={eps:.2f}: No clusters found. (Try larger eps)")

        if args.save_labeled:
            out_lab = os.path.join(args.out_dir, f"detections_labeled_{tag}.csv")
            df_labeled.to_csv(out_lab, index=False)
            print(f"[OK] Wrote labeled detections: {out_lab}")

        print(f"[INFO] eps = {eps:.2f} m | min_samples = {args.min_samples} | "
              f"input = {summary['input_detections']} | "
              f"noise = {summary['noise_detections']} | "
              f"stones = {summary['output_stones']}")

    # Save summary table
    summary_df = pd.DataFrame(summaries).sort_values("eps")
    out_summary = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"\n[OK] Wrote sweep summary: {out_summary}")


if __name__ == "__main__":
    main()