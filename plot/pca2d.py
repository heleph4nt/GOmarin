import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = d["embeddings"]
    headers = d["headers"] if "headers" in d.files else None
    return X, headers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_npz", required=True, help="database embeddings .npz")
    ap.add_argument("--orig_npz", required=True, help="original query embeddings .npz")
    ap.add_argument("--shuf_npz", required=True, help="shuffled query embeddings .npz")
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--db_max_points", type=int, default=330000)
    ap.add_argument("--pca_fit_max_points", type=int, default=300000)

    ap.add_argument("--query_max_points", type=int, default=5000)
    ap.add_argument("--query_point_size", type=float, default=2.0)
    ap.add_argument("--query_alpha", type=float, default=0.15)

    ap.add_argument("--hex_gridsize", type=int, default=450)
    args = ap.parse_args()

    # 1) load db/orig/shuf
    Xdb, Hdb = load_npz(args.db_npz)
    Xo, Ho = load_npz(args.orig_npz)
    Xs, Hs = load_npz(args.shuf_npz)

    # 2) fit PCA on (optionally sampled) DB
    n_fit = min(args.pca_fit_max_points, Xdb.shape[0])
    Xdb_fit = Xdb[:n_fit]
    pca = PCA(n_components=2, random_state=0)
    pca.fit(Xdb_fit)

    # 3) transform DB for background (optionally sampled)
    n_bg = min(args.db_max_points, Xdb.shape[0])
    Ydb = pca.transform(Xdb[:n_bg])

    # 4) transform queries (optionally downsample for plotting)
    no = min(args.query_max_points, Xo.shape[0])
    ns = min(args.query_max_points, Xs.shape[0])
    Yo = pca.transform(Xo[:no])
    Ys = pca.transform(Xs[:ns])

    # 5) save coordinates 
    np.save(args.out_prefix + ".db.pca2d.npy", Ydb)
    np.save(args.out_prefix + ".orig.pca2d.npy", Yo)
    np.save(args.out_prefix + ".shuf.pca2d.npy", Ys)

    # 6) plot
    plt.figure(figsize=(10, 8))

    hb = plt.hexbin(
        Ydb[:, 0], Ydb[:, 1],
        gridsize=args.hex_gridsize,
        #bins="log",                # color changes with density
        color="lightgrey",
        mincnt=1
    )

    # original overlay
    plt.scatter(
        Yo[:, 0], Yo[:, 1],
        s=args.query_point_size,
        alpha=args.query_alpha,
        label=f"original (n={no})"
    )

    # shuffled overlay
    plt.scatter(
        Ys[:, 0], Ys[:, 1],
        s=args.query_point_size,
        alpha=args.query_alpha,
        label=f"shuffled (n={ns})"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        "DB (hexbin log density) + Original/Shuffled overlays (PCA-2D from DB fit)\n"
        f"DB_bg={n_bg}, PCA_fit={n_fit}"
    )
    # cb = plt.colorbar(hb)         # color changes with density
    # cb.set_label("log10(count)")  # color changes with density

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out_prefix + ".pca2d.overlay.png", dpi=300)

if __name__ == "__main__":
    main()
