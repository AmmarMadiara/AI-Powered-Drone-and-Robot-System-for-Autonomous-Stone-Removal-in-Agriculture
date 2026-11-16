# -*- coding: utf-8 -*-
"""
Multi-Trip Capacitated Vehicle Routing Optimization
with Energy Constraints (CVRP + Energy).

Builds the mathematical optimization model using Gurobi,
including routing decisions, stone assignments, vehicle
payload capacity, trip time, load flow, and battery energy
constraints. Supports optional symmetry breaking and
valid inequalities.
"""



from typing import Any, Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from data import Instance


def build_model_B_plus_E(
    inst: Instance,
    cfg: Dict[str, Any],
    R_max: int,
    *,
    n_total: int,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build Model B (Multi-Trip CVRP) + energy (B+E) with arc-based load f_ijr.

    Returns:
        (model, vars_dict)
    where vars_dict contains x,y,z,n_r,m_r,d_r,t_r,f,E_r plus metadata.
    """

    CSV = cfg

    # -------------------------
    # Pull config
    # -------------------------
    M_max = float(CSV["model"]["M_MAX"])

    # Energy
    E_max = float(CSV["energy"]["E_MAX"])
    E_res = float(CSV["energy"]["E_RES"])
    alpha = float(CSV["energy"]["ALPHA"])
    beta = float(CSV["energy"]["BETA"])
    E_pick = float(CSV["energy"]["E_PICK"])
    P_idle = float(CSV["energy"]["P_IDLE"])  # Watts
    E_unload = float(CSV["energy"]["E_UNLOAD"])

    # Solver toggles
    symmetry_break = bool(CSV["solver"].get("SYMMETRY_BREAK", False))
    valid_ineq = bool(CSV["solver"].get("VALID_INEQ", True))  # controls tightenings/extras

    time_limit = CSV["solver"].get("TIME_LIMIT", None)
    mip_gap = CSV["solver"].get("MIP_GAP", None)
    gurobi_output = bool(CSV["solver"].get("GUROBI_OUTPUT", True))

    sp = CSV.get("solver_params", {})

    # -------------------------
    # Energy setup (prevent negative usable energy: allow 0)
    # -------------------------
    E_usable = float(E_max) - float(E_res)
    if E_usable < 0:
        raise ValueError(
            f"Infeasible battery parameters: "
            f"E_MAX ({E_max}) must be >= E_RES ({E_res}). "
            f"Computed E_usable = {E_usable}."
        )

    # -------------------------
    # TIME UNIT NORMALIZATION
    # Goal:
    #   - t_r measured in TIME_UNIT (seconds or minutes)
    #   - p_idle in Wh per TIME_UNIT
    #   - inv_v in TIME_UNIT per meter
    # Assumption:
    #   V_SPEED is in m/s 
    #   T_PICK, T_UNLOAD are provided in TIME_UNIT
    # -------------------------
    TIME_UNIT = str(CSV["time"]["TIME_UNIT"]).strip().lower()

    v_speed_m_per_s = float(CSV["time"]["V_SPEED"])
    if v_speed_m_per_s <= 0:
        raise ValueError("V_SPEED must be > 0.")

    t_pick = float(CSV["time"]["T_PICK"])
    t_unload = float(CSV["time"]["T_UNLOAD"])

    if TIME_UNIT.startswith("sec"):
        time_unit_label = "seconds"
        p_idle = P_idle / 3600.0                 # Wh per second
        inv_v = 1.0 / v_speed_m_per_s            # seconds per meter
    elif TIME_UNIT.startswith("min"):
        time_unit_label = "minutes"
        p_idle = P_idle / 60.0                   # Wh per minute
        inv_v = (1.0 / v_speed_m_per_s) / 60.0   # minutes per meter
    else:
        raise ValueError("TIME_UNIT must be 'seconds' or 'minutes'.")

    # -------------------------
    # Sets
    # -------------------------
    R = list(range(1, int(R_max) + 1))
    n = len(inst.I)

    # -------------------------
    # Build model
    # -------------------------
    model = gp.Model("ModelB_MultiTrip_CVRP_TRUE_DISTANCE_ENERGY")
    model.Params.OutputFlag = 1 if gurobi_output else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    # solver knobs 
    for k, vparam in sp.items():
        setattr(model.Params, k, vparam)

    # -------------------------
    # Variables
    # -------------------------
    y = model.addVars(inst.I, R, vtype=GRB.BINARY, name="y")
    z = model.addVars(R, vtype=GRB.BINARY, name="z")

    x = {}
    for r in R:
        for i in inst.V:
            for j in inst.V:
                if i == j:
                    continue
                x[(i, j, r)] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j},{r}]")

    # Auxiliary trip quantities
    n_r = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="n_r")
    m_r = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="m_r")
    d_r = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="d_r")
    t_r = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="t_r")

    # Load flow
    f = {}
    for r in R:
        for i in inst.V:
            for j in inst.V:
                if i == j:
                    continue
                f[(i, j, r)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f[{i},{j},{r}]")

    # Energy
    E_r = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, name="E_r")

    model.update()

    # Objective: minimize sum_r t_r
    model.setObjective(gp.quicksum(t_r[r] for r in R), GRB.MINIMIZE)

    # -------------------------
    # Constraints (Model B)
    # -------------------------

    # (C1) Assign each stone exactly once
    for i in inst.I:
        model.addConstr(gp.quicksum(y[i, r] for r in R) == 1, name=f"AssignOnce[{i}]")

    # (C2) Activation
    for i in inst.I:
        for r in R:
            model.addConstr(y[i, r] <= z[r], name=f"Activate[{i},{r}]")

    # (C3) Payload definition + capacity
    for r in R:
        model.addConstr(m_r[r] == gp.quicksum(inst.mass[i] * y[i, r] for i in inst.I), name=f"PayloadDef[{r}]")
        model.addConstr(m_r[r] <= M_max * z[r], name=f"Capacity[{r}]")

    # (C4) Degree constraints at stones
    for r in R:
        for i in inst.I:
            model.addConstr(
                gp.quicksum(x[(i, j, r)] for j in inst.V if j != i) == y[i, r],
                name=f"OutDeg[{i},{r}]"
            )
            model.addConstr(
                gp.quicksum(x[(j, i, r)] for j in inst.V if j != i) == y[i, r],
                name=f"InDeg[{i},{r}]"
            )

    # Depot out/in
    for r in R:
        model.addConstr(gp.quicksum(x[(0, j, r)] for j in inst.I) == z[r], name=f"DepotOut[{r}]")
        model.addConstr(gp.quicksum(x[(j, 0, r)] for j in inst.I) == z[r], name=f"DepotIn[{r}]")

    # (C5) Trip metrics
    for r in R:
        model.addConstr(n_r[r] == gp.quicksum(y[i, r] for i in inst.I), name=f"CountDef[{r}]")

    for r in R:
        model.addConstr(
            d_r[r] == gp.quicksum(
                inst.c[(i, j)] * x[(i, j, r)]
                for i in inst.V for j in inst.V if j != i
            ),
            name=f"DistDef[{r}]"
        )

    # (C6) No empty active trips (keep always)
    for r in R:
        model.addConstr(n_r[r] >= z[r], name=f"TripStoneLB[{r}]")

    # (C7) Trip time definition 
    for r in R:
        model.addConstr(
            t_r[r] == inv_v * d_r[r] + t_unload * z[r] + t_pick * n_r[r],
            name=f"TimeDef[{r}]"
        )

    # (C10) Trip ordering symmetry breaking (optional)
    if symmetry_break and len(R) >= 2:
        for k in range(1, len(R)):
            model.addConstr(z[R[k - 1]] >= z[R[k]], name=f"SymBreak[{R[k - 1]}]")

    # (C11) Payload ordering symmetry breaking (optional)
    if symmetry_break and len(R) >= 2:
        for k in range(1, len(R)):
            model.addConstr(m_r[R[k - 1]] >= m_r[R[k]], name=f"SymBreakPayload[{R[k - 1]}]")

    # -------------------------
    # Optional tightenings (VALID_INEQ)
    # -------------------------
    max_c = max(inst.c.values()) if inst.c else 0.0
    d_ub = (n + 1) * max_c
    t_ub = inv_v * d_ub + t_unload + t_pick * n

    if valid_ineq:
        for r in R:
            model.addConstr(n_r[r] <= n * z[r], name=f"TripStoneUB[{r}]")
            model.addConstr(d_r[r] <= d_ub * z[r], name=f"DistUB[{r}]")
            model.addConstr(t_r[r] <= t_ub * z[r], name=f"TimeUB[{r}]")

    # -------------------------
    # Energy constraints (Model B+E)
    # -------------------------

    # (C8) Arc linking / capacity: f <= M_max * x
    for r in R:
        for i in inst.V:
            for j in inst.V:
                if i == j:
                    continue
                model.addConstr(f[(i, j, r)] <= M_max * x[(i, j, r)], name=f"ArcLoadCap[{i},{j},{r}]")

    # (C8) Load conservation at stones: outgoing - incoming = m_i*y
    for r in R:
        for i in inst.I:
            out_load = gp.quicksum(f[(i, j, r)] for j in inst.V if j != i)
            in_load = gp.quicksum(f[(j, i, r)] for j in inst.V if j != i)
            model.addConstr(out_load - in_load == inst.mass[i] * y[i, r], name=f"LoadBal[{i},{r}]")

    # (C8) Depot load balance: start empty, return with payload
    for r in R:
        model.addConstr(gp.quicksum(f[(0, j, r)] for j in inst.I) == 0.0, name=f"DepotLoadOut0[{r}]")
        model.addConstr(gp.quicksum(f[(i, 0, r)] for i in inst.I) == m_r[r], name=f"DepotLoadInPayload[{r}]")

    # Optional tightening: arc load <= trip payload
    if valid_ineq:
        for r in R:
            for i in inst.V:
                for j in inst.V:
                    if i == j:
                        continue
                    model.addConstr(f[(i, j, r)] <= m_r[r], name=f"ArcLoadLePayload[{i},{j},{r}]")

    # (C9) Energy definition
    for r in R:
        load_distance = gp.quicksum(
            inst.c[(i, j)] * f[(i, j, r)]
            for i in inst.V for j in inst.V if j != i
        )
        model.addConstr(
            E_r[r] == alpha * d_r[r]
                    + beta * load_distance
                    + E_pick * n_r[r]
                    + p_idle * t_r[r]
                    + E_unload * z[r],
            name=f"EnergyDef[{r}]"
        )

    # Optional tightening: E_r <= E_ub * z
    if valid_ineq:
        load_dist_ub = d_ub * M_max
        E_ub = alpha * d_ub + beta * load_dist_ub + E_pick * n + p_idle * t_ub + E_unload
        for r in R:
            model.addConstr(E_r[r] <= E_ub * z[r], name=f"EnergyUB[{r}]")

    # (C9) Battery feasibility (always enforced)
    for r in R:
        model.addConstr(E_r[r] <= E_usable * z[r], name=f"Battery[{r}]")

    vars_dict: Dict[str, Any] = {
        "x": x,
        "y": y,
        "z": z,
        "n_r": n_r,
        "m_r": m_r,
        "d_r": d_r,
        "t_r": t_r,
        "f": f,
        "E_r": E_r,
        "R": R,
        "R_max": R_max,
        "n": n,
        "n_total": n_total,
        "E_usable": E_usable,
        "time_unit_label": time_unit_label,
        "p_idle": p_idle,
        "alpha": alpha,
        "beta": beta,
        "E_pick": E_pick,
        "E_unload": E_unload,
        "M_max": M_max,
        "inv_v": inv_v,
        "t_pick": t_pick,
        "t_unload": t_unload,
        "symmetry_break": symmetry_break,
        "valid_ineq": valid_ineq,
    }

    return model, vars_dict