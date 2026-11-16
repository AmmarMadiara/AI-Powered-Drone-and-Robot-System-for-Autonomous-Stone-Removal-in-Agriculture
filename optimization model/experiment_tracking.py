# -*- coding: utf-8 -*-
"""
Optimization experiment tracking and convergence monitoring.

Records Gurobi solver progress, including incumbent solutions,
best bounds, runtime and optimality gaps. Calculates solver
performance metrics, maintains experiment logs, and coordinates
the generation of performance and impact analysis results.
"""

# Key goals:
#   - clear, short titles
#   - descriptive legend labels (Baseline + additions)
#   - consistent ordering of variants across figures
#   - readable axes and units
#


from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


# -----------------------------
# Paths / folders
# -----------------------------
RESULTS_DIR = "results"
CONV_DIR = os.path.join(RESULTS_DIR, "convergence")
RUNS_CSV = os.path.join(RESULTS_DIR, "runs.csv")
IMPACT_CSV = os.path.join(RESULTS_DIR, "impact_summary.csv")
IMPACT_TXT = os.path.join(RESULTS_DIR, "impact_summary.txt")

PLOTS_DIR = "plots"
PLOTS_PERF = os.path.join(PLOTS_DIR, "performance")
PLOTS_SENS = os.path.join(PLOTS_DIR, "sensitivity")
PLOTS_CONV = os.path.join(PLOTS_DIR, "convergence")
PLOTS_IMPACT = os.path.join(PLOTS_DIR, "impact")


# -----------------------------
# Variant naming for plots (reader-friendly)
# -----------------------------
VARIANT_LABELS: Dict[str, str] = {
    "baseline": "Baseline",
    "symbreak": "Baseline + Symmetry Breaking",
    "validineq": "Baseline + Valid Inequalities",
    "validineq+symbreak": "Baseline + Valid Inequalities + Symmetry Breaking",
}

# Consistent ordering across all figures (journal-friendly)
VARIANT_ORDER: List[str] = [
    "baseline",
    "symbreak",
    "validineq",
    "validineq+symbreak",
]


def variant_label(v: str) -> str:
    v0 = str(v).strip()
    return VARIANT_LABELS.get(v0, v0)


def sort_variants(keys: Iterable[str]) -> List[str]:
    keys_list = [str(k) for k in keys]
    order_index = {v: i for i, v in enumerate(VARIANT_ORDER)}
    return sorted(keys_list, key=lambda k: (order_index.get(k, 10_000), k))


# -----------------------------
# EJOR-ish Matplotlib defaults (light-touch)
# (No forced colors/styles; keep clean and consistent.)
# -----------------------------
def _set_plot_defaults() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (6.2, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": False,
    })


_set_plot_defaults()


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONV_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_PERF, exist_ok=True)
    os.makedirs(PLOTS_SENS, exist_ok=True)
    os.makedirs(PLOTS_CONV, exist_ok=True)
    os.makedirs(PLOTS_IMPACT, exist_ok=True)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# -----------------------------
# Convergence recording callback
# -----------------------------
@dataclass
class ConvergenceRecorder:
    run_id: str
    rows: List[Tuple[float, float, float, float]]  # (time_sec, incumbent, bound, gap)

    def add(self, t: float, incumbent: float, bound: float, gap: float) -> None:
        self.rows.append((t, incumbent, bound, gap))

    def save_csv(self) -> str:
        ensure_dirs()
        path = os.path.join(CONV_DIR, f"{self.run_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "incumbent_obj", "best_bound", "mip_gap"])
            for row in self.rows:
                w.writerow(row)
        return path


def make_convergence_callback(rec: ConvergenceRecorder, min_delta_t: float = 0.1):
    """
    Records incumbent/bound/gap over time.
    - time_to_first_feasible: first time incumbent becomes finite.
    """
    last_t = {"v": -1e18}

    def cb(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIP:
            return

        t = float(model.cbGet(GRB.Callback.RUNTIME))
        if t - last_t["v"] < float(min_delta_t):
            return

        best_obj = float(model.cbGet(GRB.Callback.MIP_OBJBST))
        best_bnd = float(model.cbGet(GRB.Callback.MIP_OBJBND))

        gap = float("nan")
        if best_obj < GRB.INFINITY and abs(best_obj) > 1e-12:
            gap = abs(best_obj - best_bnd) / abs(best_obj)

        rec.add(t, best_obj, best_bnd, gap)
        last_t["v"] = t

    return cb


# -----------------------------
# Read convergence CSV + compute metrics
# -----------------------------
def _read_convergence(run_id: str) -> List[Dict[str, float]]:
    path = os.path.join(CONV_DIR, f"{run_id}.csv")
    if not os.path.exists(path):
        return []

    out: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["time_sec"])
            inc = float(row["incumbent_obj"])
            bnd = float(row["best_bound"])
            g_raw = row["mip_gap"]
            gap = float(g_raw) if g_raw != "nan" else float("nan")
            out.append({"t": t, "inc": inc, "bnd": bnd, "gap": gap})
    return out


def compute_performance_metrics(run_id: str) -> Dict[str, float]:
    """
    Returns:
      time_to_first_feasible: seconds (nan if never)
      time_to_best_incumbent: seconds (nan if never)
      final_gap: fraction (nan if unknown)
      best_incumbent: value (nan if none)
    """
    data = _read_convergence(run_id)
    if not data:
        return {
            "time_to_first_feasible": float("nan"),
            "time_to_best_incumbent": float("nan"),
            "final_gap": float("nan"),
            "best_incumbent": float("nan"),
        }

    t_first = float("nan")
    best_inc = float("inf")
    t_best = float("nan")

    for row in data:
        inc = row["inc"]
        t = row["t"]
        if inc < GRB.INFINITY:
            if t_first != t_first:  # nan check
                t_first = t
            if inc < best_inc - 1e-12:
                best_inc = inc
                t_best = t

    final_gap = float("nan")
    for row in reversed(data):
        g = row["gap"]
        if g == g:  # not nan
            final_gap = g
            break

    if best_inc == float("inf"):
        best_inc = float("nan")

    return {
        "time_to_first_feasible": t_first,
        "time_to_best_incumbent": t_best,
        "final_gap": final_gap,
        "best_incumbent": best_inc,
    }


# -----------------------------
# runs.csv logging
# -----------------------------
def _ensure_runs_header(fieldnames: List[str]) -> None:
    ensure_dirs()
    if os.path.exists(RUNS_CSV):
        return
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_run_summary(row: Dict[str, Any]) -> None:
    """
    Append one run row into results/runs.csv.
    Required keys:
      run_id, variant, N, time_limit_sec
      time_to_first_feasible, time_to_best_incumbent, final_gap
    """
    fieldnames = [
        "run_id", "variant", "N", "time_limit_sec",
        "time_to_first_feasible", "time_to_best_incumbent", "final_gap",
        "best_incumbent",
        "runtime_sec", "status",
        "param_name", "param_value",
    ]
    _ensure_runs_header(fieldnames)

    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})


def read_runs() -> List[Dict[str, str]]:
    if not os.path.exists(RUNS_CSV):
        return []
    with open(RUNS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -----------------------------
# Plot helpers
# -----------------------------
def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(s))


def _float_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _isnan(x: float) -> bool:
    return x != x


def _group_by_variant(runs: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in runs:
        var = str(r.get("variant", "default") or "default")
        by.setdefault(var, []).append(r)
    return by




# -----------------------------
# One-call finalizer per run
# -----------------------------
def finalize_and_plot(
    model: gp.Model,
    *,
    run_id: str,
    variant: str,
    N: int,
    time_limit_sec: float,
    param_name: str = "",
    param_value: Any = "",
    make_plots_now: bool = True,
    sensitivity_param_to_plot: str = "",
    baseline_variant_for_impact: str = "baseline",
) -> Dict[str, float]:
    """
    Call after solve (and after rec.save_csv()).

    It will:
      - compute performance metrics from results/convergence/<run_id>.csv
      - append them to results/runs.csv
      - generate plots:
          * performance boxplots + cactus plots
          * convergence plots (this run + aggregated)
          * sensitivity plot (optional)
          * impact summary + impact bars
          * incumbent overlay by variant

    Returns computed metrics dict.
    """
    ensure_dirs()

    perf = compute_performance_metrics(run_id)

    append_run_summary({
        "run_id": run_id,
        "variant": variant,
        "N": int(N),
        "time_limit_sec": float(time_limit_sec),
        "time_to_first_feasible": perf["time_to_first_feasible"],
        "time_to_best_incumbent": perf["time_to_best_incumbent"],
        "final_gap": perf["final_gap"],
        "best_incumbent": perf["best_incumbent"],
        "runtime_sec": float(getattr(model, "Runtime", float("nan"))),
        "status": int(getattr(model, "Status", -1)),
        "param_name": str(param_name),
        "param_value": param_value,
    })

    if make_plots_now:
        from performance_plots import (
            plot_aggregated_gap_by_variant,
            plot_cactus,
            plot_convergence_run,
            plot_incumbent_by_variant,
            plot_performance_by_variant,
            plot_sensitivity,
        )
        from impact_analysis import plot_impact_bars, write_impact_summary

        plot_performance_by_variant()

        plot_cactus("time_to_first_feasible")
        plot_cactus("time_to_best_incumbent")
        plot_cactus("final_gap", as_percent=True)

        plot_convergence_run(run_id)
        plot_aggregated_gap_by_variant(max_time_sec=time_limit_sec, step_sec=5.0)

        if sensitivity_param_to_plot:
            plot_sensitivity(sensitivity_param_to_plot)

        write_impact_summary(baseline_variant=baseline_variant_for_impact)
        plot_impact_bars(baseline_variant=baseline_variant_for_impact)

        plot_incumbent_by_variant()

    return perf