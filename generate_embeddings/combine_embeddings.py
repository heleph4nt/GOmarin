import numpy as np

a = np.load("embeddings/big_chunk_0_1_combined.npz")
b = np.load("embeddings/big_chunk_2.npz")

combined = {
    "headers": np.concatenate([a["headers"], b["headers"]], axis=0),
    "embeddings": np.concatenate([a["embeddings"], b["embeddings"]], axis=0),
    "input_fasta": a["input_fasta"],
    "tm_vec_weights": a["tm_vec_weights"],
    "protrans_model_path": a["protrans_model_path"],
}

np.savez("big_chunk_0_1_2_combined.npz", **combined)
