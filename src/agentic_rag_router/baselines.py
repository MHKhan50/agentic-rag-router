"""Reference routing policies the neural router must beat.

Each baseline returns either hard choices or class probabilities over
agents, so it can be scored with exactly the same functions as the router.
"""

from __future__ import annotations

import numpy as np


def fixed(n: int, agent: int) -> np.ndarray:
    return np.full(n, agent, dtype=int)


def random_uniform(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, k, size=n)


def majority_class(y_train: np.ndarray, n: int, k: int) -> np.ndarray:
    return fixed(n, int(np.bincount(y_train, minlength=k).argmax()))


def knn_proba(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray, k_agents: int, k: int = 15) -> np.ndarray:
    """Cosine kNN vote over oracle labels of the most similar training queries."""
    A = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    B = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = B @ A.T
    k = min(k, len(X_train))
    nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    probs = np.zeros((len(X), k_agents))
    for i, row in enumerate(nn):
        probs[i] = np.bincount(y_train[row], minlength=k_agents) / k
    return probs


def fit_logistic_regression(X_train, y_train, k_agents: int, C: float = 1.0, seed: int = 0):
    """Multinomial logistic regression on the embeddings; returns ``proba(X)``."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X_train, y_train)

    def proba(X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), k_agents))
        out[:, clf.classes_] = clf.predict_proba(X)
        return out

    return proba


def knn_correctness(X_train: np.ndarray, C_train: np.ndarray, X: np.ndarray, k: int = 15) -> np.ndarray:
    """Per-agent P(correct): the share of the k most similar training queries
    (cosine) that each agent answered correctly. Same output as the
    correctness router, so it is scored with the same policy."""
    A = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    B = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = B @ A.T
    k = min(k, len(X_train))
    nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    return C_train[nn].mean(axis=1).astype(float)


def fit_logistic_correctness(X_train: np.ndarray, C_train: np.ndarray, C: float = 1.0, seed: int = 0):
    """One binary logistic regression per agent for P(agent answers correctly);
    returns ``proba(X)`` with shape (n, agents)."""
    from sklearn.linear_model import LogisticRegression

    models = []
    for j in range(C_train.shape[1]):
        yj = C_train[:, j]
        if yj.min() == yj.max():
            models.append(float(yj[0]))
        else:
            models.append(LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X_train, yj))

    def proba(X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), len(models)))
        for j, m in enumerate(models):
            out[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        return out

    return proba
