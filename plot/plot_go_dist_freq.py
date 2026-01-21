#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_GO = "GO:0003674"  # molecular_function root


def load_go_frequencies(terms_tsv: str, aspect_keep: str = "MFO") -> pd.DataFrame:
    """
    terms_for_IA_calculation.tsv columns:
    EntryID    term    aspect
    Frequency = count(term) / number_of_unique_EntryID
    """
    df = pd.read_csv(terms_tsv, sep="\t", dtype=str)

    # flexible column handling
    cols = [c.lower() for c in df.columns]
    # expected: EntryID, term, aspect
    # try to locate them robustly
    entry_col = df.columns[cols.index("entryid")] if "entryid" in cols else df.columns[0]
    term_col = df.columns[cols.index("term")] if "term" in cols else df.columns[1]
    aspect_col = df.columns[cols.index("aspect")] if "aspect" in cols and len(df.columns) >= 3 else None

    if aspect_col is not None and aspect_keep is not None:
        df = df[df[aspect_col] == aspect_keep].copy()

    total_proteins = df[entry_col].nunique(dropna=True)

    freq = (
        df.groupby(term_col, dropna=True)
          .size()
          .rename("count")
          .reset_index()
          .rename(columns={term_col: "go"})
    )
    freq["frequency"] = freq["count"] / float(total_proteins)
    freq["total_proteins"] = total_proteins
    return freq


def distance_to_root_from_json_entry(entry: dict, root_go: str = ROOT_GO):
    """
    Your JSON schema:
    {
      "GO:xxxx": {
        "id": "GO:xxxx",
        "is_obsolete": false,
        "ancestors": [
          {"id": "GO:....", "distance": 5},
          ...
        ]
      }
    }
    We want the distance value where ancestor.id == ROOT_GO.
    """
    if not isinstance(entry, dict):
        return None
    ancestors = entry.get("ancestors")
    if not isinstance(ancestors, list):
        return None

    for a in ancestors:
        if isinstance(a, dict) and a.get("id") == root_go:
            d = a.get("distance")
            if d is None:
                return None
            try:
                return int(d)
            except Exception:
                return None
    return None


def load_go_distances_from_json(json_path: str, go_list) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    missing = 0
    for go in go_list:
        entry = data.get(go)
        d = distance_to_root_from_json_entry(entry, ROOT_GO)
        if d is None:
            missing += 1
        rows.append({"go": go, "distance": d})

    print(f"[INFO] JSON distances missing for {missing}/{len(go_list)} GO terms.")
    return pd.DataFrame(rows)


def make_scatter(df: pd.DataFrame, out_png: str, threshold: int = 4, logy: bool = True):
    plot_df = df.dropna(subset=["distance", "frequency"]).copy()
    plot_df["distance"] = plot_df["distance"].astype(int)

    if plot_df.empty:
        raise RuntimeError(
            "No points to plot (all distances missing). "
            "Check JSON coverage or provide OBO fallback."
        )

    y = plot_df["frequency"].astype(float).values
    if logy:
        y = [math.log10(v) if v > 0 else float("nan") for v in y]
        ylabel = "log10(GO frequency)"
    else:
        ylabel = "GO frequency"

    plt.figure(figsize=(9, 5))
    plt.scatter(plot_df["distance"].values, y, alpha=0.35, s=10)
    plt.axvline(threshold, linestyle="--", color="red", alpha=0.6)
    plt.xlabel(f"Distance to {ROOT_GO} (molecular_function root)")
    plt.ylabel(ylabel)
    plt.title("GO term distance vs frequency (for threshold selection)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms_tsv", required=True,
                    help="terms_for_IA_calculation.tsv path")
    ap.add_argument("--json_dict", required=True,
                    help="function_data_json_dict.json path")
    ap.add_argument("--out_prefix", required=True,
                    help="output prefix, e.g. results/go_distance_freq")
    ap.add_argument("--threshold", type=int, default=4,
                    help="vertical line threshold to display (default 4)")
    ap.add_argument("--aspect", default="MFO",
                    help="keep only this aspect (default MFO); set to '' to keep all")
    ap.add_argument("--no_logy", action="store_true",
                    help="disable log10 on y-axis")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    aspect_keep = None if args.aspect.strip() == "" else args.aspect.strip()

    # 1) frequency
    freq_df = load_go_frequencies(args.terms_tsv, aspect_keep=aspect_keep)

    # 2) distance from JSON (your schema)
    go_list = freq_df["go"].tolist()
    dist_df = load_go_distances_from_json(args.json_dict, go_list)

    # 3) merge + save
    merged = freq_df.merge(dist_df, on="go", how="left")
    out_tsv = str(out_prefix) + ".tsv"
    merged.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Saved merged table: {out_tsv}")

    # quick sanity print
    non_missing = merged["distance"].notna().sum()
    print(f"[INFO] Non-missing distances: {non_missing}/{len(merged)}")

    # 4) plot
    out_png = str(out_prefix) + ".png"
    make_scatter(merged, out_png, threshold=args.threshold, logy=(not args.no_logy))
    print(f"[OK] Saved plot: {out_png}")


if __name__ == "__main__":
    main()

'''
How to use? (in terminal)

python plot/plot_go_dist_freq.py --terms_tsv bacteria_sequences/function_only/terms_for_IA_calculation.tsv --json_dict bacteria_sequences/function_only/function_data_json_dict.json --out_prefix plot/go_dist_vs_freq/test_1 --threshold 4
'''