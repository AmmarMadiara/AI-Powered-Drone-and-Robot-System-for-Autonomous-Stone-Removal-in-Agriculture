# -*- coding: utf-8 -*-
"""

Central configuration for the optimization model.

Defines input data settings, depot placement, vehicle capacity,
time and energy parameters, Gurobi solver controls, and
experimental configurations for model evaluation.

"""

# Central configuration dictionary for the project.


from typing import Any, Dict


def get_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {

        # -------------------------
        # Data / columns
        # -------------------------
        "data": {
            # Path to your Excel file
            "CSV_PATH": "data.xlsx",

            # Column names in the Excel file
            "id_col": "stone_id",            
            "x_col": "x",              # x coordinate column
            "y_col": "y",              # y coordinate column
            "mass_col": "mass",        # mass/weight column
        },

        # -------------------------
        # Depot placement
        # -------------------------
        "depot": {
            # MODE options: "outside_corner", "centroid", "fixed_xy"
            "MODE": "outside_corner",

            # For MODE="outside_corner"
            "CORNER": "SW",        # "SW","SE","NW","NE"
            "MARGIN_FRAC": 0.05,   # 5% margin beyond bounding box
            "MARGIN_M": None,      # optional absolute margin in meters

            # For MODE="fixed_xy"
            "DEPOT_XY": (0.0, 0.0),

            # Backward compatibility
            "USE_CENTROID_DEPOT": False,
        },

        # -------------------------
        # Instance sizing
        # -------------------------
        "sizing": {
            # Default N used when no experiment loop is active
            "N_TAKE": 10,

            # Additional slack trips
            "EXTRA_TRIPS": 1,
        },

        # -------------------------
        # Model parameters
        # -------------------------
        "model": {
            "M_MAX": 25.0,         # max payload capacity (kg)
        },

        # -------------------------
        # Time parameters
        # -------------------------
        "time": {
            "TIME_UNIT": "seconds",

            # Vehicle speed (m/s)
            "V_SPEED": 1.0,

            # Service times
            "T_PICK": 10.0,
            "T_UNLOAD": 30.0,
        },

        # -------------------------
        # Energy parameters
        # -------------------------
        "energy": {
            "E_MAX": 1000.0,       # battery capacity (Wh)
            "E_RES": 50.0,         # reserve energy (Wh)

            "ALPHA": 0.001,        # Wh per meter
            "BETA": 0.00001,       # Wh per (meter*kg)

            "E_PICK": 1.0,         # Wh per pickup
            "E_UNLOAD": 2.0,       # Wh per unload

            "P_IDLE": 50.0,        # idle power (W)
        },

        # -------------------------
        # Solver controls
        # -------------------------
        "solver": {
            "TIME_LIMIT": 3600.0,   # seconds
            "MIP_GAP": 0.0,
            "GUROBI_OUTPUT": True,
            "VERBOSE": True,

            # Defaults (overridden by variants)
            "SYMMETRY_BREAK": False,
            "VALID_INEQ": True,
        },

        # -------------------------
        # Extra Gurobi parameters
        # -------------------------
        "solver_params": {

            # Presolve settings
            # "Presolve": 2,

            #"Aggregate": 1,

            # Parallelism
            #"Threads": 2,

            # Heuristics
            #"Heuristics": 0.30,

            # Additional heuristics
            #"PumpPasses": 20,
            #"RINS": 15,
            #"NoRelHeurTime": 30,
            #"Seed": 1,
        },

        # -------------------------
        # Experiment setup
        # -------------------------
        "experiment": {

            # Parameter used for sensitivity plots
            "SENS_PARAM_NAME": "N_TAKE",

            # Instance sizes to test automatically
            # If empty [] the model will fall back to sizing["N_TAKE"]
            "N_VALUES": [10, 15, 20, 30, 40, 50],
            #"N_VALUES": [10],

            # Model variants
            "VARIANTS": [
                "baseline",
                "symbreak",
                "validineq",
                "validineq+symbreak",
            ],
        },
    }

    return cfg