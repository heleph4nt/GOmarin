#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters
N_THRESHOLDS = 100
PROB_THR_MIN = 0.0
PROB_THR_MAX = 1.0


def read_predictions_matrix(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    first_col = df.columns[0]
    if not str(first_col).startswith("GO:"):
        df = df.set_index(first_col)

    go_cols = [c for c in df.columns if str(c).startswith("GO:")]
    df = df[go_cols]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def read_ia_table(ia_path: str) -> dict:
    """
    Returns: {GO: IA_value}
    """
    ia_df = pd.read_csv(ia_path, sep="\t", header=None, names=["go", "ia"])
    ia_df["ia"] = pd.to_numeric(ia_df["ia"], errors="coerce")
    return dict(zip(ia_df["go"], ia_df["ia"]))


def filter_go_by_ia(df: pd.DataFrame, ia_map: dict, ia_threshold: float) -> pd.DataFrame:
    keep = [
        c for c in df.columns
        if ia_map.get(c) is not None and ia_map.get(c) >= ia_threshold
    ]
    return df[keep]


def compute_hits(df: pd.DataFrame, prob_thresholds: np.ndarray) -> np.ndarray:
    row_max = df.max(axis=1, skipna=True).to_numpy()
    return np.array([(row_max >= p).sum() for p in prob_thresholds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_tsv", required=True)
    ap.add_argument("--shuffled_tsv", required=True)
    ap.add_argument("--ia_tsv", required=True)
    ap.add_argument("--ia_threshold", type=float, required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    orig = read_predictions_matrix(args.original_tsv)
    shuf = read_predictions_matrix(args.shuffled_tsv)

    common_cols = sorted(set(orig.columns).intersection(set(shuf.columns)))
    orig = orig[common_cols]
    shuf = shuf[common_cols]

    ia_map = read_ia_table(args.ia_tsv)

    # Filter by IA
    orig_f = filter_go_by_ia(orig, ia_map, args.ia_threshold)
    shuf_f = filter_go_by_ia(shuf, ia_map, args.ia_threshold)

    if orig_f.shape[1] == 0:
        raise RuntimeError("No GO terms remain after IA filtering.")

    total_queries = orig_f.shape[0]

    prob_thresholds = np.linspace(PROB_THR_MIN, PROB_THR_MAX, N_THRESHOLDS)

    orig_hits = compute_hits(orig_f, prob_thresholds)
    shuf_hits = compute_hits(shuf_f, prob_thresholds)
    fdr = shuf_hits / float(total_queries)

    # Save table
    out_df = pd.DataFrame({
        "prob_threshold": prob_thresholds,
        "original_hits": orig_hits,
        "shuffled_hits": shuf_hits,
        "fdr": fdr,
        "ia_threshold": args.ia_threshold,
        "n_go_after_filter": orig_f.shape[1],
        "total_queries": total_queries
    })
    out_df.to_csv(str(out_prefix) + ".tsv", sep="\t", index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(fdr, orig_hits, marker="o", markersize=3)
    plt.xlabel("FDR = shuffled_hits / total_queries")
    plt.ylabel("Original hits")
    plt.title(f"FDR curve (IA ≥ {args.ia_threshold})")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + ".png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()


'''
How to use? (in terminal)

python plot/fdr_plot_ia.py --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv --ia_tsv bacteria_sequences/function_only/IA_all.tsv --ia_threshold 3 --out_prefix plot/fdr_plot_ia/fdr_ia_thres
'''