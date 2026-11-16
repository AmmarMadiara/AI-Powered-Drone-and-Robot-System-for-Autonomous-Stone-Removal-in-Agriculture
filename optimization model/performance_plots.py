# -*- coding: utf-8 -*-
"""'
Performance, convergence, and sensitivity plotting functions.

Generates boxplots, cactus plots, convergence curves,
aggregated MIP gap plots, and sensitivity figures to
evaluate and compare the computational performance
of different optimization experiments and model variants.


"""

import os
from statistics import median
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from experiment_tracking import (
    PLOTS_CONV,
    PLOTS_PERF,
    PLOTS_SENS,
    _float_or_nan,
    _group_by_variant,
    _isnan,
    _read_convergence,
    _safe_name,
    ensure_dirs,
    read_runs,
    sort_variants,
    variant_label,
)


def _boxplot(
    data_dict: Dict[str, List[float]],
    title: str,
    ylabel: str,
    outfile: str,
) -> None:
    ensure_dirs()

    variants = sort_variants(data_dict.keys())
    filt_vars: List[str] = []
    filt_data: List[List[float]] = []

    for v in variants:
        d = [x for x in data_dict.get(v, []) if not _isnan(x)]
        if len(d) > 0:
            filt_vars.append(v)
            filt_data.append(d)

    if not filt_vars:
        return

    labels = [variant_label(v) for v in filt_vars]

    plt.figure()
    plt.boxplot(filt_data, labels=labels)
    plt.title(title)
    plt.xlabel("Model variant")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PERF, outfile), dpi=300)
    plt.close()


# -----------------------------
# PERFORMANCE PLOTS (across variants)
# -----------------------------
def plot_performance_by_variant() -> None:
    """
    Boxplots comparing variants:
      - time_to_first_feasible
      - time_to_best_incumbent
      - final_gap (%)
    """
    runs = read_runs()
    if not runs:
        return

    by_var_first: Dict[str, List[float]] = {}
    by_var_best: Dict[str, List[float]] = {}
    by_var_gap: Dict[str, List[float]] = {}

    for r in runs:
        var = str(r.get("variant", "default") or "default")
        t_first = _float_or_nan(r.get("time_to_first_feasible", "nan"))
        t_best = _float_or_nan(r.get("time_to_best_incumbent", "nan"))
        gap = _float_or_nan(r.get("final_gap", "nan"))
        if not _isnan(t_first):
            by_var_first.setdefault(var, []).append(t_first)
        if not _isnan(t_best):
            by_var_best.setdefault(var, []).append(t_best)
        if not _isnan(gap):
            by_var_gap.setdefault(var, []).append(gap * 100.0)

    _boxplot(
        by_var_first,
        "Time to first feasible solution",
        "Time (s)",
        "time_to_first_feasible_box.png",
    )
    _boxplot(
        by_var_best,
        "Time to best incumbent (within time limit)",
        "Time (s)",
        "time_to_best_incumbent_box.png",
    )
    _boxplot(
        by_var_gap,
        "Final optimality gap at termination",
        "MIP gap (%)",
        "final_gap_box.png",
    )


def plot_cactus(metric: str, *, as_percent: bool = False) -> None:
    """
    Cactus plot (EJOR-friendly): for each variant, sort metric values and plot rank vs value.

    metric ∈ {"time_to_first_feasible","time_to_best_incumbent","final_gap","runtime_sec"}.
    """
    runs = read_runs()
    if not runs:
        return

    by = _group_by_variant(runs)

    plt.figure()
    for var in sort_variants(by.keys()):
        rows = by[var]
        vals: List[float] = []
        for r in rows:
            v = _float_or_nan(r.get(metric, "nan"))
            if _isnan(v):
                continue
            if as_percent:
                v = v * 100.0
            vals.append(v)

        if not vals:
            continue

        vals.sort()
        xs = list(range(1, len(vals) + 1))
        plt.plot(xs, vals, marker="o", linewidth=1, markersize=3, label=variant_label(var))

    # Titles/labels: short + readable
    metric_title = {
        "time_to_first_feasible": "Time to first feasible solution",
        "time_to_best_incumbent": "Time to best incumbent",
        "final_gap": "Final optimality gap",
        "runtime_sec": "Runtime",
        "best_incumbent": "Best incumbent objective",
    }.get(metric, metric)

    plt.title(f"Cactus plot: {metric_title}")
    plt.xlabel("Runs (sorted)")
    plt.ylabel(metric_title + (" (%)" if as_percent else (" (s)" if "time" in metric or metric == "runtime_sec" else "")))
    plt.legend()
    plt.tight_layout()

    out = f"cactus_{_safe_name(metric)}.png"
    plt.savefig(os.path.join(PLOTS_PERF, out), dpi=300)
    plt.close()


# -----------------------------
# CONVERGENCE PLOTS (per run + aggregated)
# -----------------------------
def plot_convergence_run(run_id: str) -> None:
    """
    For one run_id:
      - incumbent & bound vs time
      - gap (%) vs time
    """
    ensure_dirs()
    data = _read_convergence(run_id)
    if not data:
        return

    ts = [d["t"] for d in data]
    inc = [d["inc"] for d in data]
    bnd = [d["bnd"] for d in data]
    gap = [d["gap"] * 100.0 if d["gap"] == d["gap"] else float("nan") for d in data]

    # incumbent & bound
    plt.figure()
    plt.plot(ts, inc, label="Incumbent")
    plt.plot(ts, bnd, label="Best bound")
    plt.xlabel("Time (s)")
    plt.ylabel("Objective value")
    plt.title("Convergence: incumbent and bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_CONV, f"{run_id}_inc_bnd.png"), dpi=300)
    plt.close()

    # gap
    plt.figure()
    plt.plot(ts, gap)
    plt.xlabel("Time (s)")
    plt.ylabel("MIP gap (%)")
    plt.title("Convergence: optimality gap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_CONV, f"{run_id}_gap.png"), dpi=300)
    plt.close()


def _grid_times(data: List[Dict[str, float]], grid: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Sample (inc,bnd,gap) on a fixed time grid using "last observation carried forward".
    Returns lists aligned with grid. Missing remains nan until first observation.
    """
    inc_out: List[float] = []
    bnd_out: List[float] = []
    gap_out: List[float] = []

    j = 0
    last_inc = float("nan")
    last_bnd = float("nan")
    last_gap = float("nan")

    for t in grid:
        while j < len(data) and data[j]["t"] <= t:
            last_inc = data[j]["inc"]
            last_bnd = data[j]["bnd"]
            last_gap = data[j]["gap"]
            j += 1
        inc_out.append(last_inc)
        bnd_out.append(last_bnd)
        gap_out.append(last_gap)
    return inc_out, bnd_out, gap_out


def plot_aggregated_gap_by_variant(max_time_sec: Optional[float] = None, step_sec: float = 5.0) -> None:
    """
    Aggregated convergence: median MIP gap (%) vs time for each variant.
    """
    runs = read_runs()
    if not runs:
        return

    # choose horizon
    if max_time_sec is None:
        max_time_sec = 0.0
        for r in runs:
            tl = _float_or_nan(r.get("time_limit_sec", "nan"))
            if not _isnan(tl):
                max_time_sec = max(max_time_sec, tl)
        if max_time_sec <= 0:
            max_time_sec = 600.0

    grid = [i * float(step_sec) for i in range(int(max_time_sec // step_sec) + 1)]

    by = _group_by_variant(runs)
    plt.figure()

    for var in sort_variants(by.keys()):
        rows = by[var]
        sampled_gaps: List[List[float]] = []
        for r in rows:
            rid = str(r.get("run_id", ""))
            data = _read_convergence(rid)
            if not data:
                continue
            _, _, g = _grid_times(data, grid)
            sampled_gaps.append(g)

        if not sampled_gaps:
            continue

        # median at each time (ignoring nans)
        med_g: List[float] = []
        for k in range(len(grid)):
            vals = [row[k] for row in sampled_gaps if row[k] == row[k]]
            if not vals:
                med_g.append(float("nan"))
            else:
                med_g.append(median(vals) * 100.0)

        plt.plot(grid, med_g, label=variant_label(var))

    plt.xlabel("Time (s)")
    plt.ylabel("Median MIP gap (%)")
    plt.title("Median optimality gap over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_CONV, "median_gap_by_variant.png"), dpi=300)
    plt.close()


# -----------------------------
# SENSITIVITY PLOTS
# -----------------------------
def plot_sensitivity(param_name: str) -> None:
    """
    Scatter plots of the three metrics vs param_value (supports multiple variants).
    Saved into plots/sensitivity/.
    """
    ensure_dirs()
    runs = read_runs()
    if not runs:
        return

    pts_first: List[Tuple[float, float, str]] = []
    pts_best: List[Tuple[float, float, str]] = []
    pts_gap: List[Tuple[float, float, str]] = []

    for r in runs:
        if str(r.get("param_name", "")) != str(param_name):
            continue
        var = str(r.get("variant", "default") or "default")
        pv = _float_or_nan(r.get("param_value", "nan"))
        t_first = _float_or_nan(r.get("time_to_first_feasible", "nan"))
        t_best = _float_or_nan(r.get("time_to_best_incumbent", "nan"))
        gap = _float_or_nan(r.get("final_gap", "nan"))

        if not _isnan(pv) and not _isnan(t_first):
            pts_first.append((pv, t_first, var))
        if not _isnan(pv) and not _isnan(t_best):
            pts_best.append((pv, t_best, var))
        if not _isnan(pv) and not _isnan(gap):
            pts_gap.append((pv, gap * 100.0, var))

    def _scatter(pts: List[Tuple[float, float, str]], title: str, ylabel: str, outfile: str) -> None:
        if not pts:
            return

        plt.figure()

        for var in sort_variants({p[2] for p in pts}):
            xs = [p[0] for p in pts if p[2] == var]
            ys = [p[1] for p in pts if p[2] == var]
            plt.scatter(xs, ys, label=variant_label(var))

            # simple trend: connect sorted points (readable, journal-friendly)
            if len(xs) >= 2:
                ord_idx = sorted(range(len(xs)), key=lambda k: xs[k])
                plt.plot([xs[i] for i in ord_idx], [ys[i] for i in ord_idx], linewidth=1)

        plt.title(title)
        plt.xlabel(str(param_name))
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_SENS, outfile), dpi=300)
        plt.close()

    safe = _safe_name(param_name)
    _scatter(pts_first, "Sensitivity: time to first feasible solution", "Time (s)", f"{safe}_first_feasible.png")
    _scatter(pts_best, "Sensitivity: time to best incumbent", "Time (s)", f"{safe}_best_time.png")
    _scatter(pts_gap, "Sensitivity: final optimality gap", "MIP gap (%)", f"{safe}_final_gap.png")




def plot_incumbent_by_variant() -> None:
    """
    Overlay incumbent trajectories for all runs, grouped by variant.
    Legend labels are shown once per variant (no duplicates).
    """
    runs = read_runs()
    if not runs:
        return

    by = _group_by_variant(runs)

    plt.figure()
    for var in sort_variants(by.keys()):
        rows = by[var]
        label_used = False
        for r in rows:
            rid = r.get("run_id", "")
            data = _read_convergence(str(rid))
            if not data:
                continue

            ts = [d["t"] for d in data]
            inc = [d["inc"] for d in data]

            if not label_used:
                plt.plot(ts, inc, linewidth=1, label=variant_label(var))
                label_used = True
            else:
                plt.plot(ts, inc, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Incumbent objective")
    plt.title("Incumbent objective over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_CONV, "incumbent_vs_time_by_variant.png"), dpi=300)
    plt.close()


