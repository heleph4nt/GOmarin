import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================
# 0) 配置你的文件路径
# ========================
# SHUF_PATH = "/shared/projects/deepmar/data/cyanobacteriota/marine_unlabeled_data/shuffled_search_in_bacteria_test/shuffled_results.tsv"
# ORIG_PATH = "/shared/projects/deepmar/data/cyanobacteriota/marine_unlabeled_data/original_search_in_bacteria_test/original_results.tsv"

SHUF_PATH = "/shared/projects/deepmar/data/bacteria_sequences/function_only/model_training/cyanobact.shuffled.predictions.tsv"
ORIG_PATH = "/shared/projects/deepmar/data/bacteria_sequences/function_only/model_training/cyanobact.original.predictions.tsv"

GO_MAP_PATHS = [
    "/shared/projects/deepmar/data/bacteria_sequences/uniprot_taxonomy_bacteria_20_11_2025.tsv"
]

OUT_DIR   = "/shared/projects/deepmar/data/plot/test_3"
TOPK      = 5
N_STEPS   = 101
PLOT_DPI  = 180

os.makedirs(OUT_DIR, exist_ok=True)


# ========= 基础函数 =========
def read_predictions(path: str) -> pd.DataFrame:
    """
    读 tmvec search 输出 tsv，兼容列名差异，输出统一列：
    query_id, tm-score, rank
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep="\t")

    # 容错列名
    ren = {}
    for c in df.columns:
        l = c.lower().replace("_", "-")
        if l in ("tm-score", "tm score"):
            ren[c] = "tm-score"
        if l in ("query-id", "query id"):
            ren[c] = "query_id"
        if l == "rank":
            ren[c] = "rank"

    df = df.rename(columns=ren)

    need = {"query_id", "tm-score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 缺列: {need - set(df.columns)}; got {list(df.columns)}")

    df["tm-score"] = pd.to_numeric(df["tm-score"], errors="coerce")
    df = df.dropna(subset=["tm-score"])

    # 若无 rank，则以 tm-score 降序构造 rank（query 内）
    if "rank" not in df.columns:
        df["rank"] = df.groupby("query_id")["tm-score"].rank(
            method="first", ascending=False
        ).astype(int)

    return df[["query_id", "tm-score", "rank"]]


def harmonize_and_topk(orig: pd.DataFrame, shuf: pd.DataFrame, topk=10):
    """
    1) 只保留 orig & shuf 共同的 query（保证分母一致）
    2) 每个 query 取 topK（按 tm-score 降序）
    """
    common_q = np.intersect1d(orig["query_id"].unique(), shuf["query_id"].unique())
    orig = orig[orig["query_id"].isin(common_q)].copy()
    shuf = shuf[shuf["query_id"].isin(common_q)].copy()

    def topk_per_query(df):
        df = df.sort_values(["query_id", "tm-score"], ascending=[True, False])
        return df.groupby("query_id", group_keys=False).head(topk)

    orig = topk_per_query(orig)
    shuf = topk_per_query(shuf)

    return orig, shuf, common_q


# ========= 主流程 =========
orig = read_predictions(ORIG_PATH)
shuf = read_predictions(SHUF_PATH)
orig, shuf, common_q = harmonize_and_topk(orig, shuf, topk=TOPK)

# 总 sequence 数量（按共同 query 数量；与你例子中的 4212 对应）
TOTAL_SEQS = int(len(common_q))
if TOTAL_SEQS == 0:
    raise ValueError("orig 与 shuf 没有共同的 query_id，无法计算 FDR。")

# ========= 阈值扫描（新逻辑） =========
thresholds = np.linspace(0, 1, N_STEPS)
rows = []

for thr in thresholds:
    o = orig[orig["tm-score"] >= thr]
    s = shuf[shuf["tm-score"] >= thr]

    # prediction 数量：在该阈值下至少有一条命中的 query 数量
    orig_pred = int(o["query_id"].nunique()) if len(o) else 0
    shuf_pred = int(s["query_id"].nunique()) if len(s) else 0

    # ✅ 新 FDR：shuffled prediction 数量 / 总 sequence 数量
    FDR = shuf_pred / TOTAL_SEQS

    rows.append({
        "tm_threshold": thr,
        "orig_pred": orig_pred,
        "shuf_pred": shuf_pred,
        "FDR": FDR
    })

curve = pd.DataFrame(rows)

curve_path = os.path.join(OUT_DIR, f"tmvec_fdr_curve_new_top{TOPK}.tsv")
curve.to_csv(curve_path, sep="\t", index=False)

# ========= 画图：x=FDR, y=orig_pred（不裁剪 FDR） =========
plt.figure()
plt.plot(curve["FDR"], curve["orig_pred"], marker="o", linewidth=1)
plt.xlabel("FDR = (#shuffled predictions) / (total #sequences)")
plt.ylabel("#original predictions")
plt.title(f"TM-Vec Curve (top{TOPK}; total_seqs={TOTAL_SEQS})")
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, f"tmvec_fdr_curve_new_top{TOPK}.png")
plt.savefig(plot_path, dpi=PLOT_DPI)
plt.close()

print("Saved:")
print(f"- {curve_path}")
print(f"- {plot_path}")
print(f"- total sequences used in denominator (common queries): {TOTAL_SEQS}")
