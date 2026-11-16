# -*- coding: utf-8 -*-
"""
Optimization data preparation and instance construction.

Defines the optimization instance structure, calculates
trip bounds, determines depot coordinates, computes
Euclidean distances between nodes, and generates data
fingerprints for reproducibility.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class Instance:
    I: List[int]
    V: List[int]
    x: Dict[int, float]
    y: Dict[int, float]
    mass: Dict[int, float]
    c: Dict[Tuple[int, int], float]
    stone_ids: Optional[Dict[int, str]] = None


def fingerprint_df(df: pd.DataFrame, cols: Sequence[str]) -> str:
    s = df[list(cols)].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def compute_Rmax(total_mass: float, M_max: float, extra_trips: int, n_take: int) -> Tuple[int, int]:
    """
    Lower bound trips: ceil(total_mass / M_max)
    Upper bound R_max: lb + extra_trips, capped by n_take (can't have more non-empty trips than stones).
    """
    if M_max <= 0:
        raise ValueError("M_max must be positive.")
    if n_take <= 0:
        raise ValueError("n_take must be positive.")

    # FIX: correct ceiling
    lb = int(math.ceil(float(total_mass) / float(M_max))) if total_mass > 0 else 1

    R_max = lb + max(0, int(extra_trips))
    R_max = min(R_max, n_take)
    R_max = max(1, R_max)
    return lb, R_max


def compute_depot_outside_field(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    corner: str = "SW",
    margin_frac: float = 0.05,
    margin_m: Optional[float] = None,
) -> Tuple[float, float]:
    xs = df[x_col].astype(float).to_list()
    ys = df[y_col].astype(float).to_list()
    if len(xs) == 0:
        raise ValueError("Empty dataframe: cannot place depot.")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    m = float(margin_m) if margin_m is not None else float(margin_frac) * float(max(w, h))

    c = corner.upper().strip()
    if c not in {"SW", "SE", "NW", "NE"}:
        raise ValueError("corner must be one of {'SW','SE','NW','NE'}")

    if c == "SW":
        return (x_min - m, y_min - m)
    if c == "SE":
        return (x_max + m, y_min - m)
    if c == "NW":
        return (x_min - m, y_max + m)
    return (x_max + m, y_max + m)


def build_instance_from_df(
    df: pd.DataFrame,
    depot_xy: Tuple[float, float],
    x_col: str,
    y_col: str,
    mass_col: str,
    *,
    id_col: Optional[str] = None,
) -> Instance:
    n = len(df)
    if n <= 0:
        raise ValueError("Empty dataframe: cannot build instance.")

    I = list(range(1, n + 1))
    V = [0] + I

    x: Dict[int, float] = {0: float(depot_xy[0])}
    y: Dict[int, float] = {0: float(depot_xy[1])}
    mass: Dict[int, float] = {}

    stone_ids: Optional[Dict[int, str]] = None
    if id_col is not None and id_col in df.columns:
        stone_ids = {}

    df2 = df.reset_index(drop=True)
    for idx in range(1, n + 1):
        row = df2.iloc[idx - 1]
        x[idx] = float(row[x_col])
        y[idx] = float(row[y_col])
        mass[idx] = float(row[mass_col])
        if stone_ids is not None:
            stone_ids[idx] = str(row[id_col])

    c: Dict[Tuple[int, int], float] = {}
    for i in V:
        xi, yi = x[i], y[i]
        for j in V:
            if i == j:
                continue
            c[(i, j)] = math.hypot(xi - x[j], yi - y[j])

    return Instance(I=I, V=V, x=x, y=y, mass=mass, c=c, stone_ids=stone_ids)