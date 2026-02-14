"""
task1.py - Train GloVe word embeddings from scratch on CC-News data.
Trains at multiple dimensions and saves embeddings, loss plots, and latency plots.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc
import time

from utils import load_data_and_build_cooccurrence, get_nearest_neighbors


class GloVeDataset(Dataset):
    """Wraps the co-occurrence dictionary into a PyTorch dataset with GloVe weighting."""

    def __init__(self, cooc_dict, x_max=100, alpha=0.75):
        self.i_idx, self.j_idx, self.counts, self.weights = [], [], [], []

        for (i, j), count in tqdm(cooc_dict.items(), desc="Preparing PyTorch Dataset"):
            self.i_idx.append(i)
            self.j_idx.append(j)
            self.counts.append(count)

            # GloVe weighting function: cap at 1.0 for frequent pairs
            weight = (count / x_max) ** alpha if count < x_max else 1.0
            self.weights.append(weight)

        # convert everything to tensors for fast GPU access
        self.i_idx = torch.tensor(self.i_idx, dtype=torch.long)
        self.j_idx = torch.tensor(self.j_idx, dtype=torch.long)
        self.counts = torch.tensor(self.counts, dtype=torch.float32)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def __len__(self):
        return len(self.i_idx)

    def __getitem__(self, idx):
        return self.i_idx[idx], self.j_idx[idx], self.counts[idx], self.weights[idx]


class GloVeModel(nn.Module):
    """The GloVe model: two embedding matrices (center + context) plus bias terms."""

    def __init__(self, vocab_size, embed_dim):
        super(GloVeModel, self).__init__()

        # center word embedding and context word embedding
        self.w_i = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.w_j = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # bias terms for center and context
        self.b_i = nn.Embedding(vocab_size, 1, sparse=True)
        self.b_j = nn.Embedding(vocab_size, 1, sparse=True)

        # initialize weights with a scaled uniform distribution
        initrange = (2.0 / (vocab_size + embed_dim))**0.5
        self.w_i.weight.data.uniform_(-initrange, initrange)
        self.w_j.weight.data.uniform_(-initrange, initrange)
        self.b_i.weight.data.zero_()
        self.b_j.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        # look up embeddings for the center and context words
        w_i = self.w_i(i_indices)
        w_j = self.w_j(j_indices)
        b_i = self.b_i(i_indices).squeeze()
        b_j = self.b_j(j_indices).squeeze()

        # GloVe prediction: dot(w_i, w_j) + b_i + b_j
        dot_product = torch.sum(w_i * w_j, dim=1)
        return dot_product + b_i + b_j


def train_glove_fast(dataset, vocab_size, embed_dim, epochs=15, batch_size=32768, lr=0.05, device='cuda'):
    """
    Trains GloVe by moving the entire dataset to GPU and doing manual batching.
    Returns the final embeddings, loss history, and total training time.
    """
    torch.cuda.empty_cache()

    model = GloVeModel(vocab_size, embed_dim).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    loss_history = []

    # move everything to GPU once to avoid repeated transfers
    print("Moving entire dataset to GPU for extreme speed...")
    i_idx = dataset.i_idx.to(device)
    j_idx = dataset.j_idx.to(device)
    counts = dataset.counts.to(device)
    weights = dataset.weights.to(device)
    log_counts = torch.log(counts)  # precompute log since GloVe loss uses log(X_ij)

    num_samples = len(i_idx)

    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        # shuffle the data each epoch for better convergence
        perm = torch.randperm(num_samples, device=device)
        i_idx_sh = i_idx[perm]
        j_idx_sh = j_idx[perm]
        log_counts_sh = log_counts[perm]
        weights_sh = weights[perm]

        # iterate through the data in chunks
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_i = i_idx_sh[start_idx:end_idx]
            batch_j = j_idx_sh[start_idx:end_idx]
            batch_log_counts = log_counts_sh[start_idx:end_idx]
            batch_weights = weights_sh[start_idx:end_idx]

            optimizer.zero_grad()
            predictions = model(batch_i, batch_j)

            # GloVe loss: weighted mean squared error between prediction and log(count)
            loss = torch.mean(batch_weights * (predictions - batch_log_counts)**2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end_idx - start_idx)

        avg_loss = total_loss / num_samples
        loss_history.append(avg_loss)

        # print progress every 10 epochs and at the start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Dim: {embed_dim} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    end_time = time.time()
    total_latency = end_time - start_time
    print(f"-> Training for d={embed_dim} completed in {total_latency:.2f} seconds.")

    # final embedding = sum of center and context embeddings (as per GloVe paper)
    final_embeddings = (model.w_i.weight.data + model.w_j.weight.data).cpu().numpy()

    # clean up GPU memory
    del model, optimizer, i_idx, j_idx, counts, weights, log_counts
    del i_idx_sh, j_idx_sh, log_counts_sh, weights_sh, perm
    torch.cuda.empty_cache()
    gc.collect()

    return final_embeddings, loss_history, total_latency


if __name__ == "__main__":
    # paths and hyperparameters
    JSON_PATH = "updated_vocab_document_dict.json"
    CACHE_PATH = "cooc_cache_w10.pkl"
    WINDOW_SIZE = 10
    EPOCHS = 50
    LR = 0.05
    DIMENSIONS = [50, 100, 200, 300]
    TEST_WORDS = ["government", "said", "India", "year"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the co-occurrence data (or read from cache)
    vocab, vocab2id, cooc_dict = load_data_and_build_cooccurrence(JSON_PATH, window_size=WINDOW_SIZE, cache_file=CACHE_PATH)
    dataset = GloVeDataset(cooc_dict)

    # free up the raw dict since the dataset has everything now
    del cooc_dict
    gc.collect()

    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # we'll collect metrics across all dimensions for the plots
    all_loss_histories = {}
    all_latencies = {}
    final_losses = {}

    # train a separate GloVe model for each embedding size
    for d in DIMENSIONS:
        print(f"\n--- Training GloVe for Dimension: {d} ---")
        embeddings, loss_hist, latency = train_glove_fast(
            dataset, len(vocab), embed_dim=d, epochs=EPOCHS, batch_size=32768, lr=LR, device=device
        )

        all_loss_histories[d] = loss_hist
        all_latencies[d] = latency
        final_losses[d] = loss_hist[-1]

        # save embeddings for use in Task 4
        np.save(f"embeddings/glove_{d}d.npy", embeddings)

        # show nearest neighbors as a quick sanity check
        print(f"\nTop-5 Nearest Neighbors (d={d}):")
        for word in TEST_WORDS:
            neighbors = get_nearest_neighbors(embeddings, vocab, vocab2id, word)
            print(f"  {word}: {[n[0] for n in neighbors]}")

    # --- generate report plots ---
    print("\nGenerating Plots for Report...")

    # plot 1: loss curves for all dimensions on one chart
    plt.figure(figsize=(10, 6))
    for d in DIMENSIONS:
        plt.plot(range(1, EPOCHS+1), all_loss_histories[d], label=f'd={d}', marker='o', markersize=3)
    plt.title("GloVe Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/glove_loss_curves.png")

    # plot 2: bar chart comparing training time across dimensions
    plt.figure(figsize=(8, 5))
    plt.bar([str(d) for d in DIMENSIONS], [all_latencies[d] for d in DIMENSIONS], color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title("Training Latency vs. Embedding Dimension")
    plt.xlabel("Embedding Dimension (d)")
    plt.ylabel("Total Training Time (Seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, d in enumerate(DIMENSIONS):
        plt.text(i, all_latencies[d] + 0.5, f"{all_latencies[d]:.1f}s", ha='center', va='bottom', fontweight='bold')
    plt.savefig("plots/glove_latency.png")

    # plot 3: how final converged loss changes with dimension
    plt.figure(figsize=(8, 5))
    plt.plot(DIMENSIONS, [final_losses[d] for d in DIMENSIONS], marker='s', color='purple', linestyle='--', linewidth=2, markersize=8)
    plt.title("Final Converged Loss vs. Embedding Dimension")
    plt.xlabel("Embedding Dimension (d)")
    plt.ylabel("Final Weighted MSE Loss")
    plt.xticks(DIMENSIONS)
    plt.grid(True)
    plt.savefig("plots/glove_final_loss.png")

    print("\nAll Training and Plotting complete! Check your /plots/ folder.")