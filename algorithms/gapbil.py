import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Fitness function
# -----------------------------
def fitness(ind, X, y, alpha=0.01):
    n_selected = ind.sum()
    if n_selected == 0:
        return 0
    X_sel = X[:, ind==1]
    clf = LogisticRegression(solver="liblinear", max_iter=2000)
    score = cross_val_score(clf, X_sel, y, cv=3, scoring="accuracy").mean()
    penalty = alpha * (n_selected / X.shape[1])
    return score - penalty

# -----------------------------
# PBIL-style GA for high-dimensional features
# Population-Based Incremental Learning
# -----------------------------
def pbil_ga(X, y, generations=50, pop_size=50, learning_rate=0.1,
            alpha=0.01, init_prob=0.05, max_features=100):
    n_features = X.shape[1]

    # Initialize feature probability vector
    p = np.full(n_features, init_prob)

    best_mask = None
    best_score = -np.inf

    for gen in range(generations):
        # Sample population from probabilities
        pop = (rng.random((pop_size, n_features)) < p).astype(int)

        # Enforce maximum number of features per individual
        for i in range(pop_size):
            if pop[i].sum() > max_features:
                ones = np.where(pop[i]==1)[0]
                off_idx = rng.choice(ones, size=pop[i].sum()-max_features, replace=False)
                pop[i, off_idx] = 0

        # Evaluate fitness
        fits = np.array([fitness(ind, X, y, alpha) for ind in pop])
        gen_best_idx = np.argmax(fits)

        # Update best solution
        if fits[gen_best_idx] > best_score:
            best_score = fits[gen_best_idx]
            best_mask = pop[gen_best_idx].copy()

        # Update probability vector toward best individual
        p = (1 - learning_rate) * p + learning_rate * best_mask

        print(f"Gen {gen}: best CV acc={best_score:.3f}, features={best_mask.sum()}")

    return best_mask, best_score

# -----------------------------
# Run PBIL GA
# -----------------------------
best_mask, best_score = pbil_ga(
    X, y,
    generations=30,
    pop_size=50,
    learning_rate=0.1,
    alpha=0.05,
    init_prob=0.02,
    max_features=250
)

print("\nFinal Best CV Accuracy:", best_score)