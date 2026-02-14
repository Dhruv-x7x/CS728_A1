"""
task5.py - Extra Credit: Apply TF-IDF to the term-document matrix before SVD.
Compares the resulting embeddings against raw SVD in two ways:
  1) Nearest neighbor quality
  2) NER performance via MLP
"""

import os
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils import load_and_build_term_doc_matrix, get_nearest_neighbors
from task4 import NERDataset, train_and_evaluate_mlp


if __name__ == "__main__":
    JSON_PATH = "updated_vocab_document_dict.json"
    BEST_D = 300  # best SVD dimension from Task 4 results

    # PDF asks for 5 diverse words for Quality Check 1
    TEST_WORDS = ["government", "said", "India", "year", "economic"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the raw term-document matrix (same as Task 2)
    Z_raw, vocab, vocab2id = load_and_build_term_doc_matrix(JSON_PATH)

    # apply TF-IDF weighting to downweight common words
    print("\nApplying TF-IDF Transformation...")
    tfidf_transformer = TfidfTransformer()
    Z_tfidf = tfidf_transformer.fit_transform(Z_raw)

    # run SVD on the TF-IDF weighted matrix
    print(f"\nRunning Truncated SVD for Dimension: {BEST_D}...")
    svd = TruncatedSVD(n_components=BEST_D, random_state=42)
    W_tfidf = svd.fit_transform(Z_tfidf)

    # --- Quality Check 1: compare nearest neighbors ---
    print("\n" + "="*50)
    print(" QUALITY CHECK 1: NEAREST NEIGHBORS (TF-IDF vs RAW)")
    print("="*50)

    # load the raw SVD embeddings from Task 2 for comparison
    W_raw = np.load(f"embeddings/svd_{BEST_D}d.npy")

    for word in TEST_WORDS:
        raw_nn = get_nearest_neighbors(W_raw, vocab, vocab2id, word)
        tfidf_nn = get_nearest_neighbors(W_tfidf, vocab, vocab2id, word)

        print(f"\nWord: '{word}'")
        print(f"  Raw SVD : {[n[0] for n in raw_nn]}")
        print(f"  TF-IDF SVD: {[n[0] for n in tfidf_nn]}")

    # --- Quality Check 2: train an MLP and compare NER performance ---
    print("\n" + "="*50)
    print(" QUALITY CHECK 2: NER MLP EVALUATION")
    print("="*50)

    # load CoNLL-2003 and build datasets using the TF-IDF SVD embeddings
    conll = load_dataset("conll2003")
    train_dataset = NERDataset(conll['train'], vocab2id, W_tfidf)
    test_dataset = NERDataset(conll['test'], vocab2id, W_tfidf)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # train and evaluate the MLP
    print(f"Training MLP using TF-IDF SVD embeddings (d={BEST_D})...")
    acc, f1 = train_and_evaluate_mlp(train_loader, test_loader, input_dim=BEST_D, device=device)

    # print comparison against the raw SVD baseline
    print("\n" + "="*50)
    print(f" Raw SVD (d=300) Baseline -> (from Task 4 results)")
    print(f" TF-IDF SVD (d=300) Score -> Acc: {acc:.4f} | F1: {f1:.4f}")
    print("="*50)