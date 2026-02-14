"""
utils.py - Shared helper functions used across all tasks.
Keeps things DRY so we don't copy-paste the same logic everywhere.
"""

import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import gc


def load_data_and_build_cooccurrence(json_path, window_size=5, cache_file="cooc_cache.pkl"):
    """
    Reads the CC-News JSON and builds a word-word co-occurrence dictionary.
    If a cached version exists on disk, we just load that instead.
    """

    # skip the heavy work if we already have a saved version
    if os.path.exists(cache_file):
        print(f"Found cached matrix at {cache_file}. Loading from disk... (This is fast!)")
        with open(cache_file, 'rb') as f:
            vocab, vocab2id, cooc_dict = pickle.load(f)
        return vocab, vocab2id, cooc_dict

    print("No cache found. Loading data from JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # build the vocabulary list and a quick lookup dict
    vocab = list(data.keys())
    vocab2id = {w: i for i, w in enumerate(vocab)}

    # pull out unique documents from the JSON structure
    docs = {}
    for word, occurrences in data.items():
        for occ in occurrences:
            doc_idx = occ[0]
            passage = occ[1]
            if doc_idx not in docs:
                docs[doc_idx] = passage

    print(f"Loaded {len(vocab)} vocabulary words and {len(docs)} unique documents.")

    # this dict will hold (word_i, word_j) -> weighted count
    cooc_dict = defaultdict(float)

    # slide a window across every document and count co-occurrences
    for doc_idx, text in tqdm(docs.items(), desc="Building Co-occurrence Matrix"):
        tokens = text.split()
        token_ids = [vocab2id[t] for t in tokens if t in vocab2id]

        for i, center_id in enumerate(token_ids):
            left = max(0, i - window_size)
            right = min(len(token_ids), i + window_size + 1)

            for j in range(left, right):
                if i != j:
                    context_id = token_ids[j]
                    # weight by inverse distance so closer words count more
                    distance = abs(i - j)
                    cooc_dict[(center_id, context_id)] += 1.0 / distance

    # free up memory since we're done with the raw data
    del docs
    del data
    gc.collect()

    # save so we never have to do this again
    print(f"Saving co-occurrence matrix to {cache_file} for future runs...")
    with open(cache_file, 'wb') as f:
        pickle.dump((vocab, vocab2id, cooc_dict), f)

    return vocab, vocab2id, cooc_dict


def load_and_build_term_doc_matrix(json_path):
    """
    Builds a sparse term-document matrix from the CC-News JSON.
    Each row is a word, each column is a document, and cells hold raw term frequencies.
    """
    from scipy.sparse import csr_matrix

    print("Loading JSON data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = list(data.keys())
    vocab2id = {w: i for i, w in enumerate(vocab)}

    rows = []
    cols = []
    vals = []

    # the raw doc IDs from JSON are sparse, so we remap them to 0, 1, 2, ...
    doc_id_map = {}
    current_col_idx = 0

    print("Building Sparse Term-Document Matrix...")
    for word, occurrences in tqdm(data.items()):
        word_id = vocab2id[word]

        for occ in occurrences:
            raw_doc_id = occ[0]
            passage = occ[1]

            # assign this doc a contiguous column index if we haven't seen it
            if raw_doc_id not in doc_id_map:
                doc_id_map[raw_doc_id] = current_col_idx
                current_col_idx += 1

            mapped_col_id = doc_id_map[raw_doc_id]

            # count how many times this word appears in the passage
            count = passage.split().count(word)
            if count > 0:
                rows.append(word_id)
                cols.append(mapped_col_id)
                vals.append(count)

    num_docs = len(doc_id_map)
    num_vocab = len(vocab)

    print(f"\nMatrix Shape: {num_vocab} terms x {num_docs} documents")
    Z = csr_matrix((vals, (rows, cols)), shape=(num_vocab, num_docs), dtype=np.float32)

    # clean up to free RAM
    print("Clearing raw python lists to free System RAM...")
    del data, rows, cols, vals, doc_id_map
    gc.collect()

    return Z, vocab, vocab2id


def get_nearest_neighbors(embeddings, vocab, vocab2id, word, k=5):
    """
    Finds the k most similar words to the given word using cosine similarity.
    """
    if word not in vocab2id:
        return []

    word_idx = vocab2id[word]
    word_vec = embeddings[word_idx]

    # normalize all embeddings to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # avoid dividing by zero
    norm_emb = embeddings / norms

    word_norm = np.linalg.norm(word_vec)
    if word_norm == 0:
        word_norm = 1e-10
    norm_word = word_vec / word_norm

    # dot product of unit vectors = cosine similarity
    sims = np.dot(norm_emb, norm_word)

    # grab top k, skipping the word itself (which would be index 0 after sort)
    top_indices = np.argsort(sims)[::-1][1:k+1]
    return [(vocab[i], sims[i]) for i in top_indices]


def load_vocab(json_path):
    """
    Loads just the vocabulary from the JSON and returns a word -> index mapping.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = list(data.keys())
    return {w: i for i, w in enumerate(vocab)}
