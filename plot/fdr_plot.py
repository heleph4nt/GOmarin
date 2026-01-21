import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========================
# 0) 配置你的文件路径
# ========================
SHUF_PATH = "/shared/projects/deepmar/data/cyanobacteriota/marine_unlabeled_data/shuffled_search_in_bacteria_test/shuffled_results.tsv"
ORIG_PATH = "/shared/projects/deepmar/data/cyanobacteriota/marine_unlabeled_data/original_search_in_bacteria_test/original_results.tsv"
# 你的 GO 映射文件名（放在 /mnt/data/ 下即可；扩展名 tsv/txt/csv 都可）
GO_MAP_PATHS = [
    "/shared/projects/deepmar/data/bacteria_sequences/uniprot_taxonomy_bacteria_20_11_2025.tsv"
]

#OUT_TABLE = "/shared/projects/deepmar/data/plot/tmvec_fdr_curve.tsv"
#OUT_PLOT  = "/shared/projects/deepmar/data/plot/tmvec_fdr_curve.png"

OUT_DIR   = "/shared/projects/deepmar/data/plot/test_2"
TOPK      = 1         
N_STEPS   = 101          
PLOT_DPI  = 180

os.makedirs(OUT_DIR, exist_ok=True)

# ========= 基础函数 =========
def extract_accession(db_id: str) -> str:
    if not isinstance(db_id, str): return ""
    parts = db_id.split("|")
    return parts[1].strip() if len(parts) >= 3 else db_id.strip()

def read_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    # 容错列名
    ren = {}
    for c in df.columns:
        l = c.lower().replace("_","-")
        if l in ("tm-score","tm score"): ren[c] = "tm-score"
        if l in ("query-id","query id"): ren[c] = "query_id"
        if l in ("database-id","subject-id","subject id","db_id","db-id"): ren[c] = "database_id"
        if l == "rank": ren[c] = "rank"
    df = df.rename(columns=ren)
    need = {"query_id","database_id","tm-score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 缺列: {need - set(df.columns)}; got {list(df.columns)}")
    df["accession"] = df["database_id"].map(extract_accession)
    df["tm-score"] = pd.to_numeric(df["tm-score"], errors="coerce")
    df = df.dropna(subset=["tm-score"])
    # 若无 rank，则以 tm-score 降序构造 rank
    if "rank" not in df.columns:
        df["rank"] = df.groupby("query_id")["tm-score"].rank(method="first", ascending=False).astype(int)
    return df[["query_id","accession","tm-score","rank"]]

# ---- GO 解析：支持三列本体，正则抓 GO:ID ----
_GO_RE = re.compile(r"GO:\d{7}")

def load_go_map() -> dict:
    path = next((p for p in GO_MAP_PATHS if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("GO 映射表未找到")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # accession 列
    if "Entry" in df.columns:
        acc_col = "Entry"
    else:
        for c in df.columns:
            if c.lower() in {"accession","entry_ac","uniprot","uniprot_id","uniprotkb-ac"}:
                acc_col = c; break
        else:
            raise ValueError(f"找不到 accession 列；表头: {list(df.columns)}")

    # 收集 GO 文本列
    go_cols = [c for c in df.columns if c.lower().startswith("gene ontology")]
    if not go_cols:
        go_cols = [c for c in df.columns if "go" in c.lower()]
    if not go_cols:
        raise ValueError(f"找不到 GO 列；表头: {list(df.columns)}")

    acc = df[acc_col].astype(str).str.strip().values
    go_counts = []
    for _, row in tqdm(df[go_cols].iterrows(), total=len(df), desc="Parsing GO map"):
        txt = " ; ".join(str(row[c]) for c in go_cols if isinstance(row[c], str))
        ids = set(_GO_RE.findall(txt))
        go_counts.append(len(ids))
    m = pd.DataFrame({"accession": acc, "go_count": go_counts})
    m = m[m["go_count"] > 0]
    # accession 可能重复：取最大（或和；一般差别不大）
    m = m.groupby("accession", as_index=False)["go_count"].max()
    return dict(zip(m["accession"], m["go_count"]))

# ---- 统一公平性：交集 query + per-query top-K + 去重 (query, accession) ----
def harmonize_and_annotate(orig: pd.DataFrame, shuf: pd.DataFrame, acc2go: dict, topk=50):
    # 只保留共同的 query
    common_q = np.intersect1d(orig["query_id"].unique(), shuf["query_id"].unique())
    orig = orig[orig["query_id"].isin(common_q)].copy()
    shuf = shuf[shuf["query_id"].isin(common_q)].copy()

    # 每个 query 内取 top-K（tm-score 降序；若有 rank 用 rank）
    def topk_per_query(df):
        df = df.sort_values(["query_id","tm-score"], ascending=[True, False])
        return df.groupby("query_id", group_keys=False).head(topk)

    orig = topk_per_query(orig)
    shuf = topk_per_query(shuf)

    # 去重到 (query_id, accession)
    orig = orig.drop_duplicates(["query_id","accession"])
    shuf = shuf.drop_duplicates(["query_id","accession"])

    # 注入两种命中口径
    orig["go_sum"]   = orig["accession"].map(acc2go).fillna(0).astype(int)     # subject 的 GO 数目
    orig["go_bin"]   = (orig["go_sum"] > 0).astype(int)                        # subject 是否有 ≥1 个 GO
    shuf["go_sum"]   = shuf["accession"].map(acc2go).fillna(0).astype(int)
    shuf["go_bin"]   = (shuf["go_sum"] > 0).astype(int)
    return orig, shuf, common_q

# ========= 主流程 =========
orig = read_predictions(ORIG_PATH)
shuf = read_predictions(SHUF_PATH)
acc2go = load_go_map()
orig, shuf, common_q = harmonize_and_annotate(orig, shuf, acc2go, topk=TOPK)

# 诊断：tm-score 分布 & 每 query 候选数（理论上都应 ~TOPK）
diag_dir = os.path.join(OUT_DIR, "diag"); os.makedirs(diag_dir, exist_ok=True)
plt.figure(); plt.hist(orig["tm-score"], bins=50, alpha=0.7, label="original"); plt.hist(shuf["tm-score"], bins=50, alpha=0.5, label="shuffled"); plt.legend(); plt.title("tm-score distributions"); plt.tight_layout(); plt.savefig(os.path.join(diag_dir,"tm_score_hist.png"), dpi=PLOT_DPI); plt.close()
orig_n = orig.groupby("query_id").size(); shuf_n = shuf.groupby("query_id").size()
pd.DataFrame({"orig_n": orig_n, "shuf_n": shuf_n}).to_csv(os.path.join(diag_dir,"per_query_counts.tsv"), sep="\t")

# 阈值扫描
thresholds = np.linspace(0, 1, N_STEPS)
rows = []

for thr in thresholds:
    o = orig[orig["tm-score"] >= thr]
    s = shuf[shuf["tm-score"] >= thr]

    # 两种口径的“总命中”：
    orig_hits_sumGO = int(o["go_sum"].sum())
    shuf_hits_sumGO = int(s["go_sum"].sum())
    orig_hits_bin   = int(o["go_bin"].sum())
    shuf_hits_bin   = int(s["go_bin"].sum())

    # y 轴（保持你原来的定义）：original 里 “有 ≥1 个 GO 的 query 数”
    q_go = o.groupby("query_id")["go_sum"].sum() if len(o) else pd.Series(dtype=int)
    queries_with_GO = int((q_go > 0).sum())

    # 两条 FDR 曲线
    FDR_sumGO   = np.nan if orig_hits_sumGO == 0 else (shuf_hits_sumGO / orig_hits_sumGO)
    FDR_binSubj = np.nan if orig_hits_bin   == 0 else (shuf_hits_bin   / orig_hits_bin)

    rows.append({
        "tm_threshold": thr,
        "orig_hits_sumGO": orig_hits_sumGO,
        "shuf_hits_sumGO": shuf_hits_sumGO,
        "FDR_sumGO": FDR_sumGO,
        "orig_hits_binSubj": orig_hits_bin,
        "shuf_hits_binSubj": shuf_hits_bin,
        "FDR_binarySubj": FDR_binSubj,
        "queries_with_GO": queries_with_GO
    })

curve = pd.DataFrame(rows).dropna(subset=["FDR_sumGO","FDR_binarySubj"], how="all")
curve_path = os.path.join(OUT_DIR, f"tmvec_fdr_curve_top{TOPK}.tsv")
curve.to_csv(curve_path, sep="\t", index=False)

# 画两条曲线
'''
plt.figure()
if curve["FDR_sumGO"].notna().any():
    plt.plot(curve["FDR_sumGO"], curve["queries_with_GO"], marker="o", linewidth=1, label="Sum-GO counting")
if curve["FDR_binarySubj"].notna().any():
    plt.plot(curve["FDR_binarySubj"], curve["queries_with_GO"], marker="s", linewidth=1, label="Binary subject counting")
plt.xlabel("FDR (shuffled / original)")
plt.ylabel("#Queries with ≥1 GO (original)")
plt.title(f"TM-Vec FDR Curve (per-query top{TOPK}, dedup)")
plt.grid(True); plt.legend()
plot_path = os.path.join(OUT_DIR, f"tmvec_fdr_curve_top{TOPK}.png")
plt.tight_layout(); plt.savefig(plot_path, dpi=PLOT_DPI); plt.close()

print(f"Saved:\n- {curve_path}\n- {plot_path}\n- Diagnostics in {diag_dir}/")'''

# =====================
# Plot: only keep points with FDR <= 2
# =====================
FDR_MAX = 1.1  # 改成 <2.0 则用严格小于

mask_sum = curve["FDR_sumGO"].notna() & (curve["FDR_sumGO"] <= FDR_MAX)
mask_bin = curve["FDR_binarySubj"].notna() & (curve["FDR_binarySubj"] <= FDR_MAX)

plt.figure()

if mask_sum.any():
    plt.plot(
        curve.loc[mask_sum, "FDR_sumGO"],
        curve.loc[mask_sum, "queries_with_GO"],
        marker="o", linewidth=1, label="Sum-GO counting"
    )

if mask_bin.any():
    plt.plot(
        curve.loc[mask_bin, "FDR_binarySubj"],
        curve.loc[mask_bin, "queries_with_GO"],
        marker="s", linewidth=1, label="Binary subject counting"
    )

plt.xlabel("FDR (shuffled / original)")
plt.ylabel("#Queries with ≥1 GO (original)")
plt.title(f"TM-Vec FDR Curve (per-query top{TOPK}, dedup; FDR≤{FDR_MAX})")
plt.grid(True); plt.legend()

plot_path = os.path.join(OUT_DIR, f"tmvec_fdr_curve_top{TOPK}_FDRle{int(FDR_MAX)}.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=PLOT_DPI)
plt.close()

print(f"Saved:\n- {curve_path}\n- {plot_path}\n- Diagnostics in {diag_dir}/")
