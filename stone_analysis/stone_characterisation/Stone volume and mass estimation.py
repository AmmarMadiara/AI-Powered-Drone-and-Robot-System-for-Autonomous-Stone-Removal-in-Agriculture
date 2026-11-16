# -*- coding: utf-8 -*-
"""
Estimates stone volume using a simplified flat-ellipsoid model,
assuming thickness is proportional to the equivalent stone
diameter. Calculates approximate stone mass using an assumed
material density.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(
        description="Add mass estimate (kg) using flat-stone model: thickness = k*D, density=rho."
    )
    ap.add_argument("-i", "--input", required=True, help="Input CSV (detections or stones)")
    ap.add_argument("-o", "--output", default="with_mass.csv", help="Output CSV with mass columns added")

    ap.add_argument(
        "--diam_col",
        default=None,
        help="Diameter column in meters. If omitted, auto-detect eq_diameter_m_eff_median then eq_diameter_m_eff."
    )
    ap.add_argument("--k_thickness", type=float, default=0.20, help="Thickness ratio k (h = k*D). Default 0.20")
    ap.add_argument("--rho", type=float, default=2600.0, help="Density kg/m^3. Default 2600")

    args = ap.parse_args()

    if not (0.05 <= args.k_thickness <= 1.0):
        raise ValueError("--k_thickness should be between 0.05 and 1.0")
    if not (500 <= args.rho <= 4000):
        raise ValueError("--rho looks unrealistic. Use something like 1800..3000 for rocks.")

    df = pd.read_csv(args.input, sep=None, engine="python")

    # Auto-detect diameter column if not provided
    if args.diam_col is None:
        if "eq_diameter_m_eff_median" in df.columns:
            diam_col = "eq_diameter_m_eff_median"
        elif "eq_diameter_m_eff" in df.columns:
            diam_col = "eq_diameter_m_eff"
        else:
            raise ValueError(
                "Could not auto-detect diameter column. "
                "Provide --diam_col, or ensure CSV has eq_diameter_m_eff_median or eq_diameter_m_eff."
            )
    else:
        diam_col = args.diam_col
        if diam_col not in df.columns:
            raise ValueError(f"Column '{diam_col}' not found. Columns: {list(df.columns)}")

    d = pd.to_numeric(df[diam_col], errors="coerce")
    df = df.copy()
    df["_D_m"] = d

    # Volume and mass (flat ellipsoid): V = (pi/6) * k * D^3
    df["volume_m3_flat"] = (np.pi / 6.0) * args.k_thickness * (df["_D_m"] ** 3)
    df["mass_kg_flat"] = args.rho * df["volume_m3_flat"]

    # Record assumptions
    df["k_thickness_used"] = args.k_thickness
    df["rho_used_kg_m3"] = args.rho
    df["diameter_col_used"] = diam_col

    # Clean helper
    df.drop(columns=["_D_m"], inplace=True)

    df.to_csv(args.output, index=False)

    print(f"[OK] Wrote: {args.output}")
    print(f"[INFO] Diameter column used: {diam_col}")
    print(f"[INFO] k_thickness = {args.k_thickness} | rho = {args.rho} kg/m^3")
    print("[INFO] Added columns: volume_m3_flat, mass_kg_flat")

if __name__ == "__main__":
    main()
