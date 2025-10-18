import numpy as np
import pandas as pd
import random

from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression


K_dim = XK.shape[1]
alpha = 0.05   # sparsity penalty


def bwo_minimize(
    objective_cv_func,        # function(mask) -> loss (lower is better). Should accept binary list/array
    n_features,
    npop=30,
    maxiter=20,
    elitism=2,
    crossover_prob=0.9,
    mutation_prob=0.02,       # per-gene bit-flip probability
    stagnation_inject_every=10,
    inject_fraction=0.25,
    seed=None,
    disp=False
):
    """
    A robust BWO-like evolutionary search for binary feature masks.
    objective_cv_func(mask) must return a scalar loss (lower=better).
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    assert elitism >= 0 and elitism < npop

    # initialize population: binary masks
    pop = [np.random.choice([0,1], size=n_features).astype(int) for _ in range(npop)]
    # Ensure no-all-zero individuals (otherwise objective should handle it)
    for i in range(len(pop)):
        if pop[i].sum() == 0:
            # flip a random bit on
            pop[i][np.random.randint(0, n_features)] = 1

    # evaluation cache (use tuple(mask) as key)
    eval_cache = {}

    def eval_mask(mask):
        key = tuple(int(b) for b in mask)
        if key in eval_cache:
            return eval_cache[key]
        val = objective_cv_func(mask)
        eval_cache[key] = val
        return val

    best_loss = float('inf')
    best_mask = None
    no_improve = 0

    start_time = time.time()

    for epoch in range(1, maxiter+1):
        # Evaluate and sort by loss (ascending)
        losses = [eval_mask(m) for m in pop]
        idx_sorted = np.argsort(losses)
        pop = [pop[i] for i in idx_sorted]
        losses = [losses[i] for i in idx_sorted]

        # update best
        if losses[0] < best_loss - 1e-12:
            best_loss = float(losses[0])
            best_mask = pop[0].copy()
            no_improve = 0
        else:
            no_improve += 1

        if disp:
            print(f"> ITER {epoch}/{maxiter} | best_loss={best_loss:.6f} | median={np.median(losses):.6f} | pop={len(pop)}")

        # elitism: preserve top `elitism`
        new_pop = [pop[i].copy() for i in range(elitism)]

        # generate offspring until we reach npop
        while len(new_pop) < npop:
            # tournament selection (size=3)
            def tournament():
                contenders = np.random.choice(npop, size=3, replace=False)
                best = min(contenders, key=lambda i: losses[i])
                return pop[best]

            p1 = tournament()
            p2 = tournament()

            # crossover (uniform crossover with prob)
            if np.random.rand() < crossover_prob:
                mask_child = p1.copy()
                # uniform crossover per bit
                swap = np.random.rand(n_features) < 0.5
                mask_child[swap] = p2[swap]
            else:
                # no crossover, clone parent
                mask_child = p1.copy()

            # mutation: bit-flip with mutation_prob
            flip = np.random.rand(n_features) < mutation_prob
            mask_child[flip] = 1 - mask_child[flip]

            # ensure not empty
            if mask_child.sum() == 0:
                mask_child[np.random.randint(0, n_features)] = 1

            new_pop.append(mask_child)

        pop = new_pop

        # stagnation injection
        if no_improve >= stagnation_inject_every:
            n_replace = max(1, int(inject_fraction * npop))
            # replace worst n_replace individuals with random masks
            for i in range(1, n_replace+1):
                mask = np.random.choice([0,1], size=n_features).astype(int)
                if mask.sum() == 0:
                    mask[np.random.randint(0, n_features)] = 1
                pop[-i] = mask
            if disp:
                print(f"[stagnation] injected {n_replace} random individuals at epoch {epoch}")
            no_improve = 0

    total_time = time.time() - start_time
    if disp:
        print("Total time: {:.3f}s".format(total_time))

    return best_loss, best_mask, eval_cache


# build classifier once (you may tune hyperparameters outside of this loop)
clf = LogisticRegression(
    penalty="l2",
    C=0.1,
    solver="liblinear",
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

def cv_feature_mask_loss(mask, X, y, alpha=0.05, cv_folds=5, scoring='balanced_accuracy'):
    """
    mask: binary array-like length n_features
    returns: loss = 1 - mean(cv_score) + sparsity_penalty
    """
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return 1.0  # huge loss for empty set

    X_sub = X[:, mask]

    # stratified CV to preserve class proportions
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # use cross_val_score (balanced accuracy better for imbalanced classes)
    scores = cross_val_score(clf, X_sub, y, cv=skf, scoring=scoring, n_jobs=1)
    mean_score = float(np.mean(scores))
    sparsity_pen = alpha * (mask.sum() / X.shape[1])
    loss = 1.0 - mean_score + sparsity_pen
    return loss


best_loss, best_mask, cache = bwo_minimize(
    objective_cv_func=lambda m: cv_feature_mask_loss(m, XK, y, alpha=0.05, cv_folds=5),
    n_features=K_dim,
    npop=30,
    maxiter=20,
    elitism=2,
    crossover_prob=0.9,
    mutation_prob=0.01,   # lower per-bit mutation because p is large
    stagnation_inject_every=8,
    inject_fraction=0.25,
    seed=42,
    disp=True
)

print("Best loss (cv):", best_loss)
print("Selected features:", int(best_mask.sum()), "/", K_dim)