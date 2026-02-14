"""
task4.py - NER using an MLP with pre-trained embeddings (GloVe and SVD).
Trains a simple neural network to predict NER tags from a single token's embedding.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from utils import load_vocab


class NERDataset(Dataset):
    """
    Converts CoNLL tokens into (embedding_vector, tag) pairs.
    Handles out-of-vocabulary words using mean-pooling as the UNK strategy.
    """

    def __init__(self, hf_dataset, vocab2id, embeddings):
        self.X = []
        self.y = []

        # the UNK vector is the average of all known embeddings
        self.unk_vector = np.mean(embeddings, axis=0)

        print("Processing dataset and applying OOV strategy...")
        for tokens, tags in zip(hf_dataset['tokens'], hf_dataset['ner_tags']):
            for word, tag in zip(tokens, tags):

                # try exact match first
                if word in vocab2id:
                    vec = embeddings[vocab2id[word]]

                # if that fails, try lowercase (catches "Government" -> "government")
                elif word.lower() in vocab2id:
                    vec = embeddings[vocab2id[word.lower()]]

                # if still not found, fall back to the UNK vector
                else:
                    vec = self.unk_vector

                self.X.append(vec)
                self.y.append(tag)

        # convert to tensors for PyTorch
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NER_MLP(nn.Module):
    """
    A simple MLP: one hidden layer with ReLU and dropout, then output to 9 NER classes.
    Input is a single token's embedding vector (no context).
    """

    def __init__(self, input_dim, hidden_dim=128, num_classes=9):
        super(NER_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate_mlp(train_loader, test_loader, input_dim, device, epochs=10):
    """
    Trains the MLP on the training set and evaluates on the test set.
    Returns token-level accuracy and macro-F1 (excluding 'O' tag).
    """
    model = NER_MLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # evaluation loop
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # compute metrics, excluding the 'O' tag (index 0) from F1 averaging
    labels_to_evaluate = [1, 2, 3, 4, 5, 6, 7, 8]
    macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=labels_to_evaluate)
    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy, macro_f1


if __name__ == "__main__":
    # paths
    JSON_PATH = "updated_vocab_document_dict.json"
    EMBEDDING_DIR = "embeddings"

    ALGORITHMS = ["glove", "svd"]
    DIMENSIONS = [50, 100, 200, 300]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the NER dataset and our vocabulary
    print("Loading CoNLL-2003 and Vocabulary...")
    conll = load_dataset("conll2003")
    vocab2id = load_vocab(JSON_PATH)

    # store results for the final comparison table
    results = {"glove": {}, "svd": {}}

    # train and evaluate an MLP for every (algorithm, dimension) combination
    for algo in ALGORITHMS:
        for d in DIMENSIONS:
            print(f"\n--- Evaluating {algo.upper()} at d={d} ---")

            # load the pre-trained embeddings we saved in Task 1 / Task 2
            emb_path = os.path.join(EMBEDDING_DIR, f"{algo}_{d}d.npy")
            embeddings = np.load(emb_path)

            # build datasets (this applies the OOV strategy automatically)
            train_dataset = NERDataset(conll['train'], vocab2id, embeddings)
            test_dataset = NERDataset(conll['test'], vocab2id, embeddings)

            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

            # train the MLP and grab metrics
            acc, f1 = train_and_evaluate_mlp(train_loader, test_loader, input_dim=d, device=device)
            results[algo][d] = {'acc': acc, 'f1': f1}

            print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

    # print a nice table summarizing everything
    print("\n" + "="*50)
    print(" FINAL RESULTS TABLE (TASK 4)")
    print("="*50)
    print(f"{'Model':<15} | {'Dim (d)':<8} | {'Accuracy':<10} | {'Macro-F1':<10}")
    print("-" * 50)

    for algo in ALGORITHMS:
        for d in DIMENSIONS:
            acc = results[algo][d]['acc']
            f1 = results[algo][d]['f1']
            print(f"{algo.upper()}-MLP{'':<7} | {d:<8} | {acc:.4f}     | {f1:.4f}")

    print("-" * 50)
    print(f"CRF (Baseline)  | N/A      | (from Task 3) | (from Task 3)")
    print("=" * 50)