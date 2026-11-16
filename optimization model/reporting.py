# -*- coding: utf-8 -*-
"""
Optimization solution reporting and validation.

Extracts the optimized routes and reports trip assignments,
distances, travel times, payloads, and energy consumption.
Also provides detailed energy breakdowns and consistency
checks for the optimization solutions.
"""


from typing import Any, Dict, List, Tuple

import gurobipy as gp
from data import Instance


def _node_label(inst: Instance, node: int) -> str:
    if node == 0:
        return "DEPOT"
    if inst.stone_ids is not None and node in inst.stone_ids:
        return f"id={inst.stone_ids[node]}"
    return f"node={node}"


def extract_route_from_x(
    inst: Instance,
    x_var: Dict[Tuple[int, int, int], gp.Var],
    r: int,
    eps: float = 0.5
) -> List[int]:
    succ: Dict[int, int] = {}
    for i in inst.V:
        for j in inst.V:
            if i == j:
                continue
            if x_var[(i, j, r)].X > eps:
                if i in succ and succ[i] != j:
                    print(f"WARNING: multiple successors for node {i} in trip r={r}: {succ[i]} and {j}")
                else:
                    succ[i] = j

    if 0 not in succ:
        return []

    route = [0]
    cur = 0
    guard = 0
    seen = {0}

    while True:
        guard += 1
        if guard > len(inst.V) + 5:
            print(f"WARNING: route extraction guard triggered for trip r={r}. Partial route: {route}")
            break
        nxt = succ.get(cur, None)
        if nxt is None:
            print(f"WARNING: route extraction failed at node {cur} for trip r={r}. Partial route: {route}")
            break
        route.append(nxt)
        cur = nxt
        if cur == 0:
            break
        if cur in seen:
            print(f"WARNING: route extraction detected a cycle not returning to depot for trip r={r}. Route: {route}")
            break
        seen.add(cur)

    return route


def print_debug_setup(
    cfg: Dict[str, Any],
    inst: Instance,
    *,
    total_mass: float,
    lb_trips: int,
    R_max: int,
    E_usable: float,
    battery_enforced: bool,
    time_unit_label: str,
    p_idle: float,
    alpha: float,
    beta: float,
    E_pick: float,
    E_unload: float,
    fp: str,
    df_rows_total: int,
    df_rows_used: int,
    use_centroid_depot: bool,
    depot_xy: Tuple[float, float],
    cols_fp: List[str],
    df_head_str: str,
) -> None:
    CSV = cfg
    M_max = float(CSV["model"]["M_MAX"])
    v = float(CSV["time"]["V_SPEED"])
    t_pick = float(CSV["time"]["T_PICK"])
    t_unload = float(CSV["time"]["T_UNLOAD"])
    extra_trips = int(CSV["sizing"]["EXTRA_TRIPS"])

    print("\n" + "=" * 110)
    print("INSTANCE FINGERPRINT (must match to compare runs)")
    print("=" * 110)
    print(f"Rows provided: {df_rows_total} | Rows used: {df_rows_used}")
    print(f"use_centroid_depot: {use_centroid_depot}")
    print(f"Depot_xy: {depot_xy}")
    print(f"Total mass (selected): {float(total_mass):.6g}")
    print(f"Fingerprint (SHA256-16 over {cols_fp}): {fp}")
    print("First 5 rows snapshot:")
    print(df_head_str)
    print("=" * 110 + "\n")

    print("\n" + "=" * 110)
    print("DEBUG SETUP (single-source parameters from CONFIG)")
    print("=" * 110)
    print(f"Depot: {depot_xy} (use_centroid_depot={use_centroid_depot})")
    print(f"Params: M_max={M_max}, v={v}, t_pick={t_pick}, t_unload={t_unload}")
    print(f"Total mass={total_mass:.6g} -> LB trips={lb_trips}, extra_trips={extra_trips} -> R_max={R_max}")
    print(f"TIME_UNIT='{time_unit_label}' -> p_idle={p_idle:.6g} Wh/{time_unit_label[:-1]} (from P_idle={CSV['energy']['P_IDLE']} W)")
    print(f"Energy: E_max={CSV['energy']['E_MAX']}, E_res={CSV['energy']['E_RES']} => E_usable={E_usable} Wh | battery_enforced={battery_enforced}")
    print(f"Energy coefs: alpha={alpha} Wh/m, beta={beta} Wh/(m*kg), E_pick={E_pick} Wh/stone, E_unload={E_unload} Wh/trip")
    print("=" * 110 + "\n")


def print_results(
    inst: Instance,
    model: gp.Model,
    vars_dict: Dict[str, Any],
    *,
    verbose: bool = True
) -> Dict[str, Any]:
    if not verbose:
        return {}

    x = vars_dict["x"]
    y = vars_dict["y"]
    z = vars_dict["z"]
    f = vars_dict["f"]
    n_r = vars_dict["n_r"]
    m_r = vars_dict["m_r"]
    d_r = vars_dict["d_r"]
    t_r = vars_dict["t_r"]
    E_r = vars_dict["E_r"]

    R = vars_dict["R"]
    R_max = vars_dict["R_max"]

    E_usable = float(vars_dict["E_usable"])
    battery_enforced = (E_usable >= 0)
    time_unit_label = vars_dict["time_unit_label"]
    p_idle = float(vars_dict["p_idle"])
    alpha = float(vars_dict["alpha"])
    beta = float(vars_dict["beta"])
    E_pick = float(vars_dict["E_pick"])
    E_unload = float(vars_dict["E_unload"])
    M_max = float(vars_dict["M_max"])

    status = int(model.Status)
    used_trips = [r for r in R if z[r].X > 0.5]

    assign_sums = [sum(y[i, r].X for r in R) for i in inst.I]
    min_assign = min(assign_sums) if assign_sums else float("nan")
    max_assign = max(assign_sums) if assign_sums else float("nan")

    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)
    print(f"Gurobi status code: {status} | SolCount={int(model.SolCount)}")
    print(f"Objective value (sum t_r): {float(model.ObjVal):.6g}")
    print(f"Best bound: {model.ObjBound:.6g}")
    print(f"MIPGap: {model.MIPGap:.6%}")
    print(f"Runtime: {model.Runtime:.2f} sec | Nodes: {int(model.NodeCount)}")
    print(f"Trips used: {len(used_trips)} / R_max={R_max}")
    print(f"Assignment check: min Σ_r y[i,r]={min_assign:.6g}, max={max_assign:.6g} (should be 1.0)")
    print(f"Battery enforced: {battery_enforced} | E_usable={E_usable:.6g} Wh")
    print("=" * 110)

    def arcs_used_for_trip(r: int, eps: float = 0.5):
        arcs = []
        for i in inst.V:
            for j in inst.V:
                if i == j:
                    continue
                if x[(i, j, r)].X > eps:
                    arcs.append((i, j))
        return arcs

    for r in used_trips:
        assigned = [i for i in inst.I if y[i, r].X > 0.5]
        assigned_labels = [_node_label(inst, i) for i in assigned]

        route = extract_route_from_x(inst, x, r)
        lbl = [_node_label(inst, node) for node in route]

        dist_val = float(d_r[r].X)
        time_val = float(t_r[r].X)
        n_val = float(n_r[r].X)
        payload_val = float(m_r[r].X)
        E_val = float(E_r[r].X)

        load_dist_val = 0.0
        for i in inst.V:
            for j in inst.V:
                if i == j:
                    continue
                load_dist_val += inst.c[(i, j)] * float(f[(i, j, r)].X)

        E_motion = float(alpha) * dist_val
        E_load = float(beta) * load_dist_val
        E_pick_term = float(E_pick) * n_val
        E_idle_term = float(p_idle) * time_val
        E_unload_term = float(E_unload) * float(z[r].X)

        cap_ok = payload_val <= M_max + 1e-6
        bat_ok = (not battery_enforced) or (E_val <= E_usable + 1e-6)

        print("\n" + "-" * 110)
        print(f"Trip r={r}")
        print(f"  Stones: {assigned_labels}")
        print(f"  Payload m_r[{r}] = {payload_val:.6g}   (<= M_max={M_max} ? {cap_ok})")
        print(f"  Number of stones n_r[{r}] = {int(round(n_val))} (raw {n_val:.6g})")
        print(f"  Route distance d_r[{r}] = {dist_val:.6g}")
        print(f"  Trip time t_r[{r}] = {time_val:.6g} ({time_unit_label})")
        print(f"  Trip energy E_r[{r}] = {E_val:.6g} Wh   (<= E_usable={E_usable:.6g} ? {bat_ok})")
        print("  Energy breakdown (Wh):")
        print(f"    motion  = alpha*d                 = {E_motion:.6g}")
        print(f"    load    = beta*sum(c_ij*f_ij)     = {E_load:.6g}   (sum c_ij*f_ij = {load_dist_val:.6g} kg*m)")
        print(f"    pickup  = E_pick*n                = {E_pick_term:.6g}")
        print(f"    idle    = p_idle*t                = {E_idle_term:.6g}")
        print(f"    unload  = E_unload*z              = {E_unload_term:.6g}")
        print(f"    total   =                          {E_motion+E_load+E_pick_term+E_idle_term+E_unload_term:.6g} (model E={E_val:.6g})")
        print(f"  Route: {' -> '.join(lbl) if lbl else '(not extracted)'}")

        arcs = arcs_used_for_trip(r)

        print("  Cumulative distance + load evolution (load increases at pickup nodes):")
        print("    (Note: travel_load = f[a,b] is the load carried DURING arc a->b,")
        print("           i.e., the load upon ARRIVAL at b BEFORE picking up at b.")
        print("           Pickup at b increases the load and affects the outgoing arc from b.)")

        cum_dist = 0.0

        if route and len(route) >= 2:
            for a, b in zip(route[:-1], route[1:]):
                if a == b:
                    continue

                dist_ab = inst.c[(a, b)]
                load_on_arc = float(f[(a, b, r)].X)

                a_lbl = _node_label(inst, a)
                b_lbl = _node_label(inst, b)

                new_cum_dist = cum_dist + dist_ab

                if b != 0:
                    pickup_mass = float(inst.mass[b])
                    after_pickup_load = load_on_arc + pickup_mass
                    pickup_str = f" | pickup at {b_lbl}: {load_on_arc:.6g} + {pickup_mass:.6g} = {after_pickup_load:.6g} kg (for outgoing arc)"
                else:
                    pickup_str = " | arrive DEPOT -> unload"

                print(
                    f"    {a_lbl:>8} -> {b_lbl:<12} "
                    f"leg_dist={dist_ab:10.6g} m   "
                    f"cum_dist: {cum_dist:.6g} + {dist_ab:.6g} = {new_cum_dist:.6g} m   "
                    f"travel_load={load_on_arc:10.6g} kg"
                    f"{pickup_str}"
                )

                cum_dist = new_cum_dist
        else:
            print("    (Route not extracted; showing arc loads without cumulative sequence)")
            for (i, j) in arcs:
                dist_ij = inst.c[(i, j)]
                load_ij = float(f[(i, j, r)].X)
                print(f"    {_node_label(inst, i):>8} -> {_node_label(inst, j):<12} dist={dist_ij:10.6g} m  load={load_ij:10.6g} kg")

        inv_v = vars_dict.get("inv_v", None)
        t_pick_cfg = float(vars_dict.get("t_pick", 0.0))
        t_unload_cfg = float(vars_dict.get("t_unload", 0.0))

        print("  Cumulative ENERGY evolution:")
        print("    Travel energy per arc = alpha*dist + beta*dist*load + p_idle*(dist/v)")
        print("    Pickup at node b adds: E_pick + p_idle*t_pick   |   Unload at depot adds: E_unload + p_idle*t_unload")

        cum_E = 0.0
        cum_E_motion = 0.0
        cum_E_load = 0.0
        cum_E_idle = 0.0
        cum_E_pick = 0.0
        cum_E_unload = 0.0

        if route and len(route) >= 2:
            for a, b in zip(route[:-1], route[1:]):
                if a == b:
                    continue

                dist_ab = float(inst.c[(a, b)])
                load_on_arc = float(f[(a, b, r)].X)

                a_lbl = _node_label(inst, a)
                b_lbl = _node_label(inst, b)

                leg_E_motion = float(alpha) * dist_ab
                leg_E_load = float(beta) * dist_ab * load_on_arc

                if inv_v is not None:
                    leg_travel_time = float(inv_v) * dist_ab
                    leg_E_idle = float(p_idle) * leg_travel_time
                else:
                    leg_travel_time = None
                    leg_E_idle = 0.0

                cum_E_motion += leg_E_motion
                cum_E_load += leg_E_load
                cum_E_idle += leg_E_idle
                cum_E += (leg_E_motion + leg_E_load + leg_E_idle)

                time_str = f"{leg_travel_time:.6g}" if leg_travel_time is not None else "N/A"
                print(
                    f"    {a_lbl:>8} -> {b_lbl:<12} "
                    f"dist={dist_ab:10.6g} m, load={load_on_arc:10.6g} kg | "
                    f"E_travel = motion({leg_E_motion:.6g}) + load({leg_E_load:.6g}) + idle({leg_E_idle:.6g}, t={time_str}) "
                    f"=> cum_E = {cum_E:.6g} Wh  (ARRIVE {b_lbl} before pickup)"
                )

                if b != 0:
                    pickup_mass = float(inst.mass[b])
                    # After pickup at b, load increases and affects the NEXT arc leaving b.
                    leg_E_pick = float(E_pick)
                    leg_E_idle_pick = float(p_idle) * t_pick_cfg

                    cum_E_pick += leg_E_pick
                    cum_E_idle += leg_E_idle_pick
                    cum_E += (leg_E_pick + leg_E_idle_pick)

                    print(
                        f"             pickup at {b_lbl}: +mass {pickup_mass:.6g} | "
                        f"+E_pick({leg_E_pick:.6g}) + idle_pick({leg_E_idle_pick:.6g}) "
                        f"=> cum_E = {cum_E:.6g} Wh"
                    )
                else:
                    leg_E_unload = float(E_unload) * float(z[r].X)
                    leg_E_idle_unload = float(p_idle) * t_unload_cfg * float(z[r].X)

                    cum_E_unload += leg_E_unload
                    cum_E_idle += leg_E_idle_unload
                    cum_E += (leg_E_unload + leg_E_idle_unload)

                    print(
                        f"             arrive DEPOT -> unload | "
                        f"+E_unload({leg_E_unload:.6g}) + idle_unload({leg_E_idle_unload:.6g}) "
                        f"=> cum_E = {cum_E:.6g} Wh"
                    )

            print("  Energy totals built from steps (Wh):")
            print(f"    motion total = {cum_E_motion:.6g}")
            print(f"    load   total = {cum_E_load:.6g}")
            print(f"    pickup total = {cum_E_pick:.6g}")
            print(f"    idle   total = {cum_E_idle:.6g}")
            print(f"    unload total = {cum_E_unload:.6g}")
            print(f"    cum_E  total = {cum_E:.6g}  (model E_r[{r}] = {float(E_r[r].X):.6g})")
            print(f"    difference   = {cum_E - float(E_r[r].X):.6g}")
        else:
            print("    (Route not extracted; cannot compute sequential energy evolution.)")

        depot_out_load = sum(float(f[(0, j, r)].X) for j in inst.I)
        depot_in_load = sum(float(f[(i, 0, r)].X) for i in inst.I)

        print("  Load-flow checks (should be near 0 residual):")
        print(f"    depot outgoing load sum_j f[0,j] = {depot_out_load:.6g}   (should be 0)")
        print(f"    depot incoming  load sum_i f[i,0] = {depot_in_load:.6g}   (should equal payload {payload_val:.6g})")
        print(f"    depot_in - payload = {depot_in_load - payload_val:.6g}")

        max_abs_res = 0.0
        worst = None
        for i in assigned:
            out_load = sum(float(f[(i, j, r)].X) for j in inst.V if j != i)
            in_load = sum(float(f[(j, i, r)].X) for j in inst.V if j != i)
            res = (out_load - in_load) - inst.mass[i]
            if abs(res) > max_abs_res:
                max_abs_res = abs(res)
                worst = (i, res)
        if worst is not None:
            wi, wres = worst
            print(f"    max |(out-in) - m_i| among visited stones = {max_abs_res:.6g} (worst at { _node_label(inst, wi) }, res={wres:.6g})")

        if route:
            expected_len = len(assigned) + 2
            if len(route) != expected_len:
                print(f"  WARNING: extracted route length {len(route)} != expected {expected_len}. Potential subtour/route extraction issue.")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)

    return {
        "status": status,
        "objective": float(model.ObjVal),
        "used_trips": used_trips,
        "R_max": R_max,
        "E_usable": E_usable,
        "battery_enforced": battery_enforced,
        "TIME_UNIT": time_unit_label,
    }