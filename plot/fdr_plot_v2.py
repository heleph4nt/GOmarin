#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_GO = "GO:0003674"

# Fixed parameters
MIN_DISTANCE = 4
N_THRESHOLDS = 101
THR_MIN = 0.0
THR_MAX = 1.0


def load_go_distance_map(json_path: str, root_go: str = ROOT_GO) -> dict:
    """
    JSON schema:
    "GO:xxxx": {"ancestors": [{"id":"GO:....","distance": 5}, ...], ...}

    Returns: {go_term: distance_to_root_go} for terms where distance exists.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dist_map = {}
    for go, entry in data.items():
        ancestors = entry.get("ancestors", None)
        if not isinstance(ancestors, list):
            continue

        d = None
        for a in ancestors:
            if isinstance(a, dict) and a.get("id") == root_go:
                try:
                    d = int(a.get("distance"))
                except Exception:
                    d = None
                break

        if d is not None:
            dist_map[go] = d

    return dist_map


def read_predictions_matrix(tsv_path: str) -> pd.DataFrame:
    """
    Reads wide tsv:
      - rows = query (first column is query id)
      - cols = GO terms (headers are GO:xxxx)
      - values = probabilities (0..1)

    Handles possible 'Unnamed: 0' header for the first column.
    """
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # If the first column is not a GO term, treat it as query id and set as index
    first_col = df.columns[0]
    if not str(first_col).startswith("GO:"):
        df = df.set_index(first_col)

    # Keep only GO columns
    go_cols = [c for c in df.columns if str(c).startswith("GO:")]
    df = df[go_cols]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def filter_go_columns_by_distance(df: pd.DataFrame, dist_map: dict, min_distance: int) -> pd.DataFrame:
    keep = []
    for c in df.columns:
        d = dist_map.get(c)
        if d is not None and d >= min_distance:
            keep.append(c)
    return df[keep]


def compute_hits_over_thresholds(df: pd.DataFrame, thresholds: np.ndarray) -> np.ndarray:
    """
    hit(query, thr)=1 if max_prob(query) >= thr
    hits(thr)= number of queries that are hits at threshold thr
    """
    row_max = df.max(axis=1, skipna=True).to_numpy()
    hits = np.array([(row_max >= t).sum() for t in thresholds], dtype=int)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_tsv", required=True, help="cyanobact.original.predictions.tsv")
    ap.add_argument("--shuffled_tsv", required=True, help="cyanobact.shuffled.predictions.tsv")
    ap.add_argument("--json_dict", required=True, help="function_data_json_dict.json")
    ap.add_argument("--out_prefix", required=True, help="output prefix, e.g. results/fdr_prob_distance4")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) distances
    dist_map = load_go_distance_map(args.json_dict)

    # 2) read matrices
    orig = read_predictions_matrix(args.original_tsv)
    shuf = read_predictions_matrix(args.shuffled_tsv)

    # Align GO columns intersection (safer)
    common_cols = sorted(set(orig.columns).intersection(set(shuf.columns)))
    orig = orig[common_cols]
    shuf = shuf[common_cols]

    # 3) filter by GO distance >= MIN_DISTANCE
    orig_f = filter_go_columns_by_distance(orig, dist_map, MIN_DISTANCE)
    shuf_f = filter_go_columns_by_distance(shuf, dist_map, MIN_DISTANCE)

    if orig_f.shape[1] == 0:
        raise RuntimeError(
            "No GO columns remain after distance filtering. "
            "Check json coverage or MIN_DISTANCE."
        )

    total_queries = orig_f.shape[0]
    if shuf_f.shape[0] != total_queries:
        print("[WARN] original/shuffled query counts differ. Using original total_queries for FDR denominator.")

    # 4) probability thresholds (fixed)
    thresholds = np.linspace(THR_MIN, THR_MAX, N_THRESHOLDS)

    # 5) hits
    orig_hits = compute_hits_over_thresholds(orig_f, thresholds)
    shuf_hits = compute_hits_over_thresholds(shuf_f, thresholds)

    # FDR definition (fixed): shuffled_hits / total_queries
    fdr = shuf_hits / float(total_queries)

    # 6) save table
    out_tsv = str(out_prefix) + ".tsv"
    out_df = pd.DataFrame({
        "prob_threshold": thresholds,
        "original_hits": orig_hits,
        "shuffled_hits": shuf_hits,
        "fdr_shuf_over_total": fdr,
        "total_queries": total_queries,
        "min_go_distance": MIN_DISTANCE,
        "prob_threshold_min": THR_MIN,
        "prob_threshold_max": THR_MAX,
        "n_thresholds": N_THRESHOLDS,
        "n_go_cols_after_filter": orig_f.shape[1],
    })
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Saved curve table: {out_tsv}")
    print(f"[INFO] total_queries={total_queries}, GO_cols_after_filter={orig_f.shape[1]}")

    # 7) plot: x=FDR, y=original_hits
    out_png = str(out_prefix) + ".png"
    plt.figure(figsize=(8, 5))
    plt.plot(fdr, orig_hits, marker="o", markersize=3, linewidth=1)
    plt.xlabel("FDR = shuffled_hits / total_queries")
    plt.ylabel("Original hits")
    plt.title(f"FDR curve (GO distance ≥ {MIN_DISTANCE})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out_png}")


if __name__ == "__main__":
    main()


'''
How to use? (in terminal)

python plot/fdr_plot_v2.py --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv --json_dict bacteria_sequences/function_only/function_data_json_dict.json --out_prefix plot/fdr_plot_v2/fdr_dist_4
'''