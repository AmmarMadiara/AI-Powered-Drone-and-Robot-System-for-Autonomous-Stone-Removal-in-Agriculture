# -*- coding: utf-8 -*-
"""
Main execution script for the optimization model.

Loads the configuration and input data, constructs
optimization instances, builds and solves the multi-trip
CVRP with energy constraints, and coordinates reporting,
experiment tracking, and performance evaluation across
different instance sizes and model variants.
"""

# Entry point: load data, select subset, build instance, build model, solve, report.
# Printing: redirect full report output 

from typing import Any, Dict, List, Tuple
from copy import deepcopy
from contextlib import redirect_stdout

import pandas as pd

from config import get_config
from data import (
    build_instance_from_df,
    compute_Rmax,
    fingerprint_df,
    compute_depot_outside_field,
)
from model import build_model_B_plus_E
from reporting import print_debug_setup, print_results

from experiment_tracking import (
    new_run_id,
    ConvergenceRecorder,
    make_convergence_callback,
    finalize_and_plot,
)


def select_subset(df: pd.DataFrame, N_take: int) -> pd.DataFrame:
    if N_take is not None and int(N_take) > 0:
        return df.head(int(N_take)).copy()
    return df.copy()


def choose_depot_xy(
    df_use: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    x_col: str,
    y_col: str,
) -> Tuple[float, float]:
    """
    Depot placement logic (keeps backward compatibility).

    Priority:
      1) depot.MODE == "outside_corner" => bbox corner with margin 
      2) depot.MODE == "centroid" or USE_CENTROID_DEPOT=True => centroid
      3) depot.MODE == "fixed_xy" or USE_CENTROID_DEPOT=False => DEPOT_XY
    """
    depot_cfg = cfg.get("depot", {})
    mode = str(depot_cfg.get("MODE", "")).strip().lower()

    if mode == "outside_corner":
        corner = str(depot_cfg.get("CORNER", "SW"))
        margin_frac = float(depot_cfg.get("MARGIN_FRAC", 0.05))
        margin_m = depot_cfg.get("MARGIN_M", None)
        return compute_depot_outside_field(
            df_use, x_col, y_col, corner=corner, margin_frac=margin_frac, margin_m=margin_m
        )

    use_centroid = bool(depot_cfg.get("USE_CENTROID_DEPOT", False))
    if mode == "centroid" or use_centroid:
        return (float(df_use[x_col].mean()), float(df_use[y_col].mean()))

    # fixed xy
    return tuple(depot_cfg.get("DEPOT_XY", (0.0, 0.0)))


def _apply_variant_flags(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """
    Make variants actually control model extras.
    - baseline: no symmetry break, no valid inequalities 
    - symbreak: symmetry break only
    - validineq: valid inequalities/tightenings only
    - validineq+symbreak: both
    """
    cfg_v = deepcopy(cfg)

    v = str(variant).strip().lower()

    # ensure solver dict exists
    cfg_v["solver"] = dict(cfg_v.get("solver", {}))

    # set flags based on variant name
    cfg_v["solver"]["SYMMETRY_BREAK"] = ("symbreak" in v)
    cfg_v["solver"]["VALID_INEQ"] = ("validineq" in v)

    # keep a record
    cfg_v["experiment"] = dict(cfg_v.get("experiment", {}))
    cfg_v["experiment"]["VARIANT"] = variant

    return cfg_v


def main() -> None:
    cfg: Dict[str, Any] = get_config()

    # --- pull config ---
    id_col = cfg["data"]["id_col"]
    x_col = cfg["data"]["x_col"]
    y_col = cfg["data"]["y_col"]
    mass_col = cfg["data"]["mass_col"]

    extra_trips = int(cfg["sizing"]["EXTRA_TRIPS"])
    M_max = float(cfg["model"]["M_MAX"])
    verbose = bool(cfg["solver"]["VERBOSE"])

    # Sensitivity label (we will set value per-run)
    sens_param_name = str(cfg.get("experiment", {}).get("SENS_PARAM_NAME", "N_TAKE"))

    # Variants
    variants = cfg.get("experiment", {}).get("VARIANTS", ["baseline"])

    # N sweep: take from experiment.N_VALUES; fallback to sizing.N_TAKE if empty
    N_values = cfg.get("experiment", {}).get("N_VALUES", [])
    if not N_values:
        N_values = [int(cfg["sizing"]["N_TAKE"])]

    # --- load data once ---
    df = pd.read_excel(cfg["data"]["CSV_PATH"])
    if len(df) == 0:
        raise ValueError("The input data file is empty.")

    # --- run all N values ---
    for N_take in N_values:
        N_take = int(N_take)
        cfg["sizing"]["N_TAKE"] = N_take  # keep cfg consistent for downstream code

        # select subset for this N
        df_use = select_subset(df, N_take)
        if len(df_use) == 0:
            raise ValueError(f"No rows selected for N_TAKE={N_take}. Check N_VALUES or the input file.")

        # --- depot handling ---
        depot_xy = choose_depot_xy(df_use, cfg, x_col=x_col, y_col=y_col)
        use_centroid_depot = bool(cfg.get("depot", {}).get("USE_CENTROID_DEPOT", False))

        # fingerprint for reproducibility/debug (depends on df_use)
        cols_fp: List[str] = [x_col, y_col, mass_col]
        if id_col is not None and id_col in df_use.columns:
            cols_fp = [id_col] + cols_fp
        fp = fingerprint_df(df_use, cols_fp)

        # --- sizing trips ---
        total_mass = float(df_use[mass_col].sum())
        lb_trips, R_max = compute_Rmax(total_mass, M_max, extra_trips, n_take=len(df_use))

        # --- build instance ---
        inst = build_instance_from_df(df_use, depot_xy, x_col, y_col, mass_col, id_col=id_col)

        # basic feasibility check
        max_single = max(inst.mass[i] for i in inst.I)
        if max_single > M_max + 1e-9:
            raise ValueError(
                f"Infeasible by construction for N_TAKE={N_take}: "
                f"a stone mass {max_single} exceeds M_max={M_max}."
            )

        print("\n" + "#" * 100)
        print(f"INSTANCE SIZE: N_TAKE = {N_take} | R_max = {R_max} | total_mass = {total_mass:.6g}")
        print("#" * 100)

        # Sensitivity value for this run (very important for plots/runs.csv)
        sens_param_value = N_take if sens_param_name == "N_TAKE" else cfg.get(sens_param_name, "")

        # --- run all variants for this N ---
        for variant in variants:
            print("\n" + "=" * 80)
            print(f"RUNNING: N_TAKE={N_take} | VARIANT={variant}")
            print("=" * 80)

            run_id = new_run_id("cvrp")
            cfg_v = _apply_variant_flags(cfg, variant)

            # --- build model ---
            model, vars_dict = build_model_B_plus_E(inst, cfg_v, R_max, n_total=len(df_use))

            # --- debug prints (pre-solve) ---
            if verbose:
                E_usable_val = float(vars_dict["E_usable"])
                print_debug_setup(
                    cfg_v,
                    inst,
                    total_mass=total_mass,
                    lb_trips=lb_trips,
                    R_max=R_max,
                    E_usable=E_usable_val,
                    battery_enforced=(E_usable_val >= 0),
                    time_unit_label=str(vars_dict["time_unit_label"]),
                    p_idle=float(vars_dict["p_idle"]),
                    alpha=float(vars_dict["alpha"]),
                    beta=float(vars_dict["beta"]),
                    E_pick=float(vars_dict["E_pick"]),
                    E_unload=float(vars_dict["E_unload"]),
                    fp=fp,
                    df_rows_total=len(df),
                    df_rows_used=len(df_use),
                    use_centroid_depot=use_centroid_depot,
                    depot_xy=depot_xy,
                    cols_fp=cols_fp,
                    df_head_str=str(df_use[cols_fp].head(5)),
                )

            # --- solve (WITH convergence logging) ---
            rec = ConvergenceRecorder(run_id=run_id, rows=[])
            cb = make_convergence_callback(rec, min_delta_t=0.5)
            model.optimize(cb)

            # Guarantee logging has at least one point and always ends with the FINAL solver state
            runtime = float(getattr(model, "Runtime", 0.0))

            if model.SolCount > 0:
                incumbent = float(model.ObjVal)
            else:
                incumbent = float("inf")

            bound = float(getattr(model, "ObjBound", float("nan")))
            gap = float(getattr(model, "MIPGap", float("nan")))

            # If the callback never recorded anything (fast solve), add a row.
            # Even if it did record, append a final snapshot so "final_gap" is truly final.
            rec.add(runtime, incumbent, bound, gap)

            rec.save_csv()

            # save solution
            model.write(f"solution_N{N_take}_{variant}.sol")

            # --- report ---
            if model.SolCount > 0:
                with open(f"report_N{N_take}_{variant}.txt", "w", encoding="utf-8") as f:
                    with redirect_stdout(f):
                        _ = print_results(inst, model, vars_dict, verbose=verbose)
            else:
                if verbose:
                    print(f"No incumbent solution found for N_TAKE={N_take}, variant={variant}.")

            # --- finalize results + plots ---
            time_limit = float(cfg_v["solver"]["TIME_LIMIT"]) if cfg_v["solver"]["TIME_LIMIT"] is not None else float(model.Runtime)

            perf = finalize_and_plot(
                model,
                run_id=run_id,
                variant=variant,
                N=len(inst.I),
                time_limit_sec=time_limit,
                param_name=sens_param_name,
                param_value=sens_param_value,
                sensitivity_param_to_plot=sens_param_name,
                baseline_variant_for_impact="baseline",
            )

            print(f"Done | N_TAKE={N_take} | Variant={variant} | Run id: {run_id}")
            if model.SolCount > 0:
                print(
                    f"  Performance: first_feasible={perf['time_to_first_feasible']:.3g}s, "
                    f"best_incumbent_time={perf['time_to_best_incumbent']:.3g}s, "
                    f"final_gap={perf['final_gap']*100.0:.3g}%"
                )


if __name__ == "__main__":
    main()