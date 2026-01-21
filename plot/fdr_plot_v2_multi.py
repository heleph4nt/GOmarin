#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_GO = "GO:0003674"

# Fixed analysis parameters
DISTANCE_LIST = [3, 4, 5]   # <- compare these thresholds
N_THRESHOLDS = 100
THR_MIN = 0.0
THR_MAX = 1.0


def load_go_distance_map(json_path: str, root_go: str = ROOT_GO) -> dict:
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
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    first_col = df.columns[0]
    if not str(first_col).startswith("GO:"):
        df = df.set_index(first_col)

    go_cols = [c for c in df.columns if str(c).startswith("GO:")]
    df = df[go_cols]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def filter_go_columns_by_distance(df: pd.DataFrame, dist_map: dict, min_distance: int) -> pd.DataFrame:
    keep = [c for c in df.columns if (dist_map.get(c) is not None and dist_map.get(c) >= min_distance)]
    return df[keep]


def compute_hits_over_thresholds(df: pd.DataFrame, thresholds: np.ndarray) -> np.ndarray:
    row_max = df.max(axis=1, skipna=True).to_numpy()
    return np.array([(row_max >= t).sum() for t in thresholds], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_tsv", required=True)
    ap.add_argument("--shuffled_tsv", required=True)
    ap.add_argument("--json_dict", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load GO distances
    dist_map = load_go_distance_map(args.json_dict)

    # Read matrices
    orig = read_predictions_matrix(args.original_tsv)
    shuf = read_predictions_matrix(args.shuffled_tsv)

    # Align GO columns
    common_cols = sorted(set(orig.columns).intersection(set(shuf.columns)))
    orig = orig[common_cols]
    shuf = shuf[common_cols]

    total_queries = orig.shape[0]

    thresholds = np.linspace(THR_MIN, THR_MAX, N_THRESHOLDS)

    # Store per-distance results
    curves = []

    for dmin in DISTANCE_LIST:
        orig_f = filter_go_columns_by_distance(orig, dist_map, dmin)
        shuf_f = filter_go_columns_by_distance(shuf, dist_map, dmin)

        if orig_f.shape[1] == 0:
            print(f"[WARN] distance≥{dmin}: no GO columns remain; skip.")
            continue

        orig_hits = compute_hits_over_thresholds(orig_f, thresholds)
        shuf_hits = compute_hits_over_thresholds(shuf_f, thresholds)
        fdr = shuf_hits / float(total_queries)

        curves.append((dmin, fdr, orig_hits, shuf_hits, orig_f.shape[1]))

        # Save each curve table (optional but useful)
        df_out = pd.DataFrame({
            "prob_threshold": thresholds,
            "original_hits": orig_hits,
            "shuffled_hits": shuf_hits,
            "fdr_shuf_over_total": fdr,
            "total_queries": total_queries,
            "min_go_distance": dmin,
            "n_go_cols_after_filter": orig_f.shape[1],
            "prob_threshold_min": THR_MIN,
            "prob_threshold_max": THR_MAX,
            "n_thresholds": N_THRESHOLDS,
        })
        df_out.to_csv(str(out_prefix) + f".distance_ge_{dmin}.tsv", sep="\t", index=False)
        print(f"[OK] Saved table for distance≥{dmin}: {str(out_prefix)}.distance_ge_{dmin}.tsv")

    if not curves:
        raise RuntimeError("No curves generated. Check DISTANCE_LIST and json coverage.")

    # Plot all curves in one figure
    out_png = str(out_prefix) + ".png"
    plt.figure(figsize=(8, 5))

    for dmin, fdr, orig_hits, shuf_hits, ncols in curves:
        plt.plot(fdr, orig_hits, marker="o", markersize=3, linewidth=1,
                 label=f"distance ≥ {dmin} (GO cols={ncols})")

    plt.xlabel("FDR = shuffled_hits / total_queries")
    plt.ylabel("Original hits")
    plt.title("FDR curves for different GO distance thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved multi-curve plot: {out_png}")


if __name__ == "__main__":
    main()


'''
How to use? (in terminal)

python plot/fdr_plot_v2_multi.py --original_tsv bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv --shuffled_tsv bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv --json_dict bacteria_sequences/function_only/function_data_json_dict.json --out_prefix plot/fdr_plot_v2/fdr_dist_345
'''