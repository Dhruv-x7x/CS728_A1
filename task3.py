"""
task3.py - Named Entity Recognition using a CRF with hand-crafted features.
Trains on CoNLL-2003 and reports accuracy, F1, and feature importance.
"""

from datasets import load_dataset
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
import time

# load the CoNLL-2003 NER dataset from HuggingFace
print("Downloading CoNLL-2003 dataset...")
dataset = load_dataset("conll2003")

# get the human-readable tag names (B-PER, I-ORG, etc.) from the dataset metadata
ner_tags_mapping = dataset['train'].features['ner_tags'].feature.names


def word2features(sent, i):
    """
    Builds a feature dictionary for the word at position i in a sentence.
    These are the hand-crafted features the CRF will learn weights for.
    """
    word = sent[i]

    # features about the current word itself
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),        # the word in lowercase
        'word[-3:]': word[-3:],              # last 3 characters (suffix)
        'word[-2:]': word[-2:],              # last 2 characters (suffix)
        'word[:3]': word[:3],                # first 3 characters (prefix)
        'word.isupper()': word.isupper(),    # is the whole word uppercase? (e.g. "NATO")
        'word.istitle()': word.istitle(),    # is it title-cased? (e.g. "London")
        'word.isdigit()': word.isdigit(),    # is it a number? (e.g. "2024")
    }

    # features from the previous word (gives left context)
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        # no previous word means we're at the start of the sentence
        features['BOS'] = True

    # features from the next word (gives right context)
    if i < len(sent) - 1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        # no next word means we're at the end of the sentence
        features['EOS'] = True

    return features


def sent2features(sent):
    """Converts a full sentence into a list of feature dicts, one per token."""
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(tags):
    """Converts integer tag IDs to their string names (e.g. 1 -> 'B-PER')."""
    return [ner_tags_mapping[tag] for tag in tags]


# extract features from all splits
print("Extracting features from Train, Validation, and Test sets...")

X_train = [sent2features(s) for s in dataset['train']['tokens']]
y_train = [sent2labels(t) for t in dataset['train']['ner_tags']]

X_test = [sent2features(s) for s in dataset['test']['tokens']]
y_test = [sent2labels(t) for t in dataset['test']['ner_tags']]

# train the CRF model
print("\nTraining Conditional Random Field (CRF)...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,                        # L1 regularization strength
    c2=0.1,                        # L2 regularization strength
    max_iterations=100,
    all_possible_transitions=True   # let the model learn all tag-to-tag transitions
)

start_time = time.time()
crf.fit(X_train, y_train)
end_time = time.time()
print(f"-> CRF Training completed in {end_time - start_time:.2f} seconds.")

# run predictions on the test set
print("\nEvaluating on CoNLL-2003 Test Set...")
y_pred = crf.predict(X_test)

# we exclude the 'O' tag from F1 because it overwhelms the average
labels = list(crf.classes_)
labels.remove('O')

# compute overall metrics
f1 = metrics.flat_f1_score(y_test, y_pred, average='macro', labels=labels)
acc = metrics.flat_accuracy_score(y_test, y_pred)

print(f"\n======================================")
print(f"CRF Token-Level Accuracy: {acc:.4f}")
print(f"CRF Macro-F1 Score:       {f1:.4f}")
print(f"======================================\n")

# per-entity-type breakdown
print("Detailed Classification Report:")
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))


def print_top_features(state_features, top_k=15):
    """Shows the features with the largest learned weights (most important for classification)."""
    print(f"\nTop {top_k} Most Important Features learned by CRF:")
    sorted_features = sorted(state_features.items(), key=lambda x: abs(x[1]), reverse=True)

    for (attr, label), weight in sorted_features[:top_k]:
        print(f"{weight:8.4f} | {label:8} | {attr}")

print_top_features(crf.state_features_)