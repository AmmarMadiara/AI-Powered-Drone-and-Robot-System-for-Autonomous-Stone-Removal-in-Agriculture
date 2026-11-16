# -*- coding: utf-8 -*-
"""
Filters detection records according to their assigned size
categories, allowing specific stone size groups to be selected
for subsequent spatial analysis.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser(
        description="Filter a CSV by real-world size class (small, medium, large, very_large)."
    )
    ap.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV (detections or stones)"
    )
    ap.add_argument(
        "-o", "--output",
        default="filtered_by_size.csv",
        help="Output CSV with selected size class(es)"
    )
    ap.add_argument(
        "--class_col",
        default="size_class_real",
        help="Column containing size classes"
    )
    ap.add_argument(
        "--keep",
        nargs="+",
        required=True,
        help="Size class(es) to keep (e.g. small medium OR very_large)"
    )

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.class_col not in df.columns:
        raise ValueError(
            f"Column '{args.class_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df_filtered = df[df[args.class_col].isin(args.keep)].copy()

    df_filtered.to_csv(args.output, index=False)

    print(f"[OK] Kept {len(df_filtered)} rows where {args.class_col} ∈ {args.keep}")
    print(f"[OK] Wrote: {args.output}")

if __name__ == "__main__":
    main()
