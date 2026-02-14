"""
task2.py - SVD-based word embeddings using a term-document matrix.
Decomposes the matrix with Truncated SVD at multiple dimensions.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
import os
import time
import matplotlib.pyplot as plt

from utils import load_and_build_term_doc_matrix, get_nearest_neighbors


if __name__ == "__main__":
    # paths and settings
    JSON_PATH = "updated_vocab_document_dict.json"
    DIMENSIONS = [50, 100, 200, 300]
    TEST_WORDS = ["government", "said", "India", "year"]

    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # build the sparse term-document matrix from the CC-News JSON
    Z, vocab, vocab2id = load_and_build_term_doc_matrix(JSON_PATH)

    all_latencies = {}

    # run SVD for each embedding dimension
    for d in DIMENSIONS:
        print(f"\n--- Running Truncated SVD for Dimension: {d} ---")

        # fixed random state so results are reproducible
        svd = TruncatedSVD(n_components=d, random_state=42)

        # time the SVD computation
        start_time = time.time()

        # fit_transform returns U_d * Sigma_d, which are our word embeddings
        W_d = svd.fit_transform(Z)

        end_time = time.time()
        latency = end_time - start_time
        all_latencies[d] = latency
        print(f"-> SVD (d={d}) completed in {latency:.2f} seconds.")

        # save these embeddings so Task 4 can load them
        np.save(f"embeddings/svd_{d}d.npy", W_d)

        # quick check: print nearest neighbors for our test words
        print(f"Top-5 Nearest Neighbors (SVD d={d}):")
        for word in TEST_WORDS:
            neighbors = get_nearest_neighbors(W_d, vocab, vocab2id, word)
            print(f"  {word}: {[n[0] for n in neighbors]}")

    # --- generate latency bar chart for the report ---
    print("\nGenerating SVD Latency Plot...")
    plt.figure(figsize=(8, 5))
    plt.bar([str(d) for d in DIMENSIONS], [all_latencies[d] for d in DIMENSIONS], color=['#c2c2f0','#ffb3e6','#c2f0c2','#ffb3b3'])
    plt.title("Truncated SVD Latency vs. Embedding Dimension")
    plt.xlabel("Embedding Dimension (d)")
    plt.ylabel("Execution Time (Seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, d in enumerate(DIMENSIONS):
        plt.text(i, all_latencies[d] + (max(all_latencies.values()) * 0.01), f"{all_latencies[d]:.1f}s", ha='center', va='bottom', fontweight='bold')
    plt.savefig("plots/svd_latency.png")

    print("\nTask 2 Complete! SVD embeddings and latency plot saved.")