# -*- coding: utf-8 -*-
"""

Impact-summary calculations and plots.

Optimization model impact analysis.

Evaluates the computational impact of symmetry breaking
and valid inequalities by comparing different model variants
against the baseline formulation. Calculates performance
statistics, relative improvements, and generates impact
summary tables and comparison plots.


"""

import csv
import os
from statistics import median
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from experiment_tracking import (
    IMPACT_CSV,
    PLOTS_IMPACT,
    _float_or_nan,
    _group_by_variant,
    _isnan,
    _safe_name,
    ensure_dirs,
    read_runs,
    sort_variants,
    variant_label,
)


# -----------------------------
# IMPACT OF EXTRAS (valid inequalities / symmetry breaking / etc.)
# -----------------------------
def _summary_stats(vals: List[float]) -> Dict[str, float]:
    vals = [v for v in vals if not _isnan(v)]
    if not vals:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": len(vals),
        "mean": sum(vals) / float(len(vals)),
        "median": median(vals),
    }


def write_impact_summary(baseline_variant: str = "baseline") -> None:
    """
    Compares each variant against a baseline using the logged runs.csv.
    Produces:
      - results/impact_summary.csv
      - results/impact_summary.txt
    """
    ensure_dirs()
    runs = read_runs()
    if not runs:
        return

    by = _group_by_variant(runs)
    if baseline_variant not in by:
        baseline_variant = sort_variants(by.keys())[0]

    metrics = [
        ("time_to_first_feasible", False),
        ("time_to_best_incumbent", False),
        ("final_gap", True),          # store as % in output
        ("best_incumbent", False),
        ("runtime_sec", False),
    ]

    base_rows = by.get(baseline_variant, [])
    out_rows: List[Dict[str, Any]] = []

    for var in sort_variants(by.keys()):
        rows = by[var]
        for metric, as_percent in metrics:
            base_vals = [_float_or_nan(r.get(metric, "nan")) for r in base_rows]
            vals = [_float_or_nan(r.get(metric, "nan")) for r in rows]

            if as_percent:
                base_vals = [v * 100.0 if not _isnan(v) else v for v in base_vals]
                vals = [v * 100.0 if not _isnan(v) else v for v in vals]

            s_base = _summary_stats(base_vals)
            s_var = _summary_stats(vals)

            rel = float("nan")
            b = s_base["median"]
            vmed = s_var["median"]
            if not _isnan(b) and abs(b) > 1e-12 and not _isnan(vmed):
                rel = (b - vmed) / b

            out_rows.append({
                "baseline_variant": baseline_variant,
                "variant": var,
                "metric": metric + ("_pct" if as_percent else ""),
                "n_baseline": s_base["n"],
                "median_baseline": s_base["median"],
                "mean_baseline": s_base["mean"],
                "n_variant": s_var["n"],
                "median_variant": s_var["median"],
                "mean_variant": s_var["mean"],
                "relative_improvement_median": rel,
            })

    with open(IMPACT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Short readable text summary (journal-friendly)
    focus = {"time_to_first_feasible", "time_to_best_incumbent", "final_gap_pct"}
    lines: List[str] = []
    lines.append(f"IMPACT SUMMARY (baseline = {variant_label(baseline_variant)})")
    lines.append("Positive relative_improvement_median => better than baseline (lower median).")
    lines.append("")
    for r in out_rows:
        if r["metric"] not in focus:
            continue
        lines.append(
            f"{r['metric']}: {variant_label(r['variant'])} vs {variant_label(baseline_variant)} | "
            f"median={r['median_variant']:.4g} (base {r['median_baseline']:.4g}) | "
            f"rel_impr={r['relative_improvement_median']:.3g}"
        )

    with open(IMPACT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_impact_bars(baseline_variant: str = "baseline") -> None:
    """
    Bar chart: median relative improvement vs baseline for key metrics.
    """
    ensure_dirs()
    if not os.path.exists(IMPACT_CSV):
        write_impact_summary(baseline_variant=baseline_variant)

    with open(IMPACT_CSV, "r", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    focus = ["time_to_first_feasible", "time_to_best_incumbent", "final_gap_pct"]
    variants = sort_variants({r["variant"] for r in rows if r["variant"] != baseline_variant})

    for metric in focus:
        xs: List[str] = []
        ys: List[float] = []

        for v in variants:
            match = [r for r in rows if r["variant"] == v and r["metric"] == metric]
            if not match:
                continue
            rel = _float_or_nan(match[0].get("relative_improvement_median", "nan"))
            if _isnan(rel):
                continue
            xs.append(variant_label(v))
            ys.append(rel * 100.0)  # %

        if not xs:
            continue

        metric_title = {
            "time_to_first_feasible": "Time to first feasible solution",
            "time_to_best_incumbent": "Time to best incumbent",
            "final_gap_pct": "Final optimality gap",
        }.get(metric, metric)

        plt.figure()
        plt.bar(xs, ys)
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("Median improvement vs baseline (%)")
        plt.title(f"Impact on {metric_title}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_IMPACT, f"impact_{_safe_name(metric)}.png"), dpi=300)
        plt.close()


