"""
Cyber threat detection with an artificial neural network using event profiles.

Each network connection record is turned into an "event profile": a short text
document of feature=value tokens. TF-IDF turns those documents into vectors, and
a neural network (multi-layer perceptron) learns to classify them. The same
vectors are used to train classic models so the comparison is fair.

Extension: binary Particle Swarm Optimization (PSO) picks a smaller set of
TF-IDF features, then an SVM is trained on only those features.

Every model is trained on kdd_train.csv and scored on kdd_test.csv, which it
never sees during training. kdd_test.csv also contains attack types that do not
appear in the training file, which makes the test harder and more realistic.

Run:  python cyber_threat_detection.py
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# Standard NSL-KDD mapping from attack name to attack category.
CATEGORY = {
    "normal": "normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probing / scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    # User to root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]


def load(path):
    df = pd.read_csv(path)
    names = df["labels"].str.strip()
    unknown = sorted(set(names) - set(CATEGORY))
    if unknown:
        raise ValueError(f"Unknown attack names in {path}: {unknown}")
    y = names.map(CATEGORY)
    return df.drop(columns=["labels"]), y


def event_profiles(df):
    """One text document per record, e.g. 'protocol_type=tcp service=http flag=SF ...'.
    Numbers are rounded so near-identical values share a token."""
    cols = df.columns
    rows = []
    for rec in df.itertuples(index=False):
        tokens = []
        for col, val in zip(cols, rec):
            if isinstance(val, float):
                val = round(val, 2)
            tokens.append(f"{col}={val}")
        rows.append(" ".join(tokens))
    return rows


def scores(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }


def binary_pso(X, y, n_particles=12, iters=15, alpha=0.9, seed=SEED):
    """Binary PSO feature selection. Fitness rewards validation accuracy and
    (a little) using fewer features. Uses only the training data."""
    rng = np.random.default_rng(seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    n_feat = X.shape[1]

    def fitness(mask):
        if mask.sum() == 0:
            return 1.0
        clf = LogisticRegression(max_iter=300)
        clf.fit(X_tr[:, mask], y_tr)
        acc = clf.score(X_val[:, mask], y_val)
        return alpha * (1 - acc) + (1 - alpha) * (mask.sum() / n_feat)

    pos = rng.random((n_particles, n_feat)) < 0.5
    vel = rng.normal(0, 1, (n_particles, n_feat))
    cost = np.array([fitness(p) for p in pos])
    pbest, pbest_cost = pos.copy(), cost.copy()
    g = pbest_cost.argmin()
    gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]
    w, c1, c2 = 0.9, 1.5, 1.5

    for it in range(iters):
        r1, r2 = rng.random(vel.shape), rng.random(vel.shape)
        vel = (w * vel + c1 * r1 * (pbest.astype(float) - pos)
               + c2 * r2 * (gbest.astype(float) - pos))
        vel = np.clip(vel, -6, 6)
        pos = rng.random(vel.shape) < 1 / (1 + np.exp(-vel))
        cost = np.array([fitness(p) for p in pos])
        better = cost < pbest_cost
        pbest[better], pbest_cost[better] = pos[better], cost[better]
        g = pbest_cost.argmin()
        if pbest_cost[g] < gbest_cost:
            gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]
        print(f"  PSO iteration {it + 1}/{iters}: best cost {gbest_cost:.4f}, "
              f"features {gbest.sum()}/{n_feat}")
    return gbest


def plot_metric(results, metric, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = results.index
    vals = results[metric]
    bars = ax.bar(names, vals, color=["#c0392b" if n == "ANN (MLP)" else "#7f8c8d" for n in names])
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"{metric} on the NSL-KDD test set")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{metric.lower()}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="folder with kdd_train.csv and kdd_test.csv")
    ap.add_argument("--out", default="results", help="folder for results and graphs")
    ap.add_argument("--no-pso", action="store_true", help="skip the PSO extension (faster)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X_train_df, y_train = load(os.path.join(args.data, "kdd_train.csv"))
    X_test_df, y_test = load(os.path.join(args.data, "kdd_test.csv"))
    print(f"Train records: {len(y_train)}   Test records: {len(y_test)}")
    print("Train classes:", y_train.value_counts().to_dict())
    print("Test classes: ", y_test.value_counts().to_dict())

    # TF-IDF is fitted on the training data only, so nothing leaks from the test set.
    tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2)
    X_train = tfidf.fit_transform(event_profiles(X_train_df))
    X_test = tfidf.transform(event_profiles(X_test_df))
    print(f"Event vector size (TF-IDF features): {X_train.shape[1]}")

    models = {
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                                   max_iter=200, random_state=SEED),
        "SVM": LinearSVC(random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }

    results, preds = {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        results[name] = scores(y_test, p)
        print(f"{name:14s} accuracy {results[name]['Accuracy']:.2f}%  "
              f"macro F1 {results[name]['F1']:.2f}%  ({time.time() - t0:.1f}s)")

    if not args.no_pso:
        print("Running PSO feature selection on the training data...")
        mask = binary_pso(X_train.toarray(), y_train.values)
        svm = LinearSVC(random_state=SEED).fit(X_train[:, mask], y_train)
        p = svm.predict(X_test[:, mask])
        preds["SVM + PSO"] = p
        results["SVM + PSO"] = scores(y_test, p)
        print(f"SVM + PSO      accuracy {results['SVM + PSO']['Accuracy']:.2f}%  "
              f"using {mask.sum()} of {X_train.shape[1]} features")

    table = pd.DataFrame(results).T.round(2)
    table.to_csv(os.path.join(args.out, "results.csv"))
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(table.to_markdown())
    print("\n", table)

    cm = pd.DataFrame(confusion_matrix(y_test, preds["ANN (MLP)"], labels=CLASSES),
                      index=[f"true {c}" for c in CLASSES],
                      columns=[f"pred {c}" for c in CLASSES])
    cm.to_csv(os.path.join(args.out, "ann_confusion_matrix.csv"))
    print("\nANN confusion matrix:\n", cm)

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        plot_metric(table, metric, args.out)


if __name__ == "__main__":
    main()
