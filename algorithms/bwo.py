import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


"""
BWO
"""
def _enforce_max_features(mask, max_features, rng=None):
    if max_features is None:
        return mask
    
    m = mask.copy()
    n_on = int(m.sum())
    if n_on > max_features:
        ones_idx = np.flatnonzero(m)
        if rng is None:
            turn_off = np.random.choice(ones_idx, size=(n_on - max_features), replace=False)
        else:
            turn_off = rng.choice(ones_idx, size=(n_on - max_features), replace=False)
        m[turn_off] = 0
        
    return m


def bwo_minimize(
    objective_cv_func,        
    n_features,
    npop=30,
    maxiter=50,
    elitism=2,
    crossover_prob=0.9,
    mutation_prob=0.02,      
    stagnation_inject_every=10,
    inject_fraction=0.25,
    seed=None,
    verbose=False,
    max_features=None         
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)
    else:
        rng = None

    assert elitism >= 0 and elitism < npop

    # initialize population: binary masks (respecting max_features)
    pop = []
    for _ in range(npop):
        mask = np.random.choice([0,1], size=n_features).astype(int)
        mask = _enforce_max_features(mask, max_features, rng=rng)
        if mask.sum() == 0:
            idx = rng.randint(0, n_features) if rng is not None else np.random.randint(0, n_features)
            mask[idx] = 1
        pop.append(mask)

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

    for epoch in range(1, maxiter+1):
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

        if verbose:
            median_loss = np.median(losses)
            avg_selected = np.mean([m.sum() for m in pop])
            print(f"> ITER {epoch}/{maxiter} | best_loss={best_loss:.6f} | median={median_loss:.6f} | avg_selected={avg_selected:.1f}")

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
                # uniform crossover per bit
                swap = np.random.rand(n_features) < 0.5
                child = p1.copy()
                child[swap] = p2[swap]
            else:
                # no crossover, clone parent
                child = p1.copy()

            # mutation: bit-flip with mutation_prob
            flip = np.random.rand(n_features) < mutation_prob
            if flip.any():
                child[flip] = 1 - child[flip]

            # Enforce max_features cap (hard truncation if necessary)
            child = _enforce_max_features(child, max_features, rng=rng)

            # ensure not empty
            if child.sum() == 0:
                idx = rng.randint(0, n_features) if rng is not None else np.random.randint(0, n_features)
                child[idx] = 1

            new_pop.append(child)

        pop = new_pop

        # stagnation injection
        if no_improve >= stagnation_inject_every:
            n_replace = max(1, int(inject_fraction * npop))
            # replace worst n_replace individuals with random masks 
            for i in range(1, n_replace+1):
                mask = np.random.choice([0,1], size=n_features).astype(int)
                mask = _enforce_max_features(mask, max_features, rng=rng)
                if mask.sum() == 0:
                    idx = rng.randint(0, n_features) if rng is not None else np.random.randint(0, n_features)
                    mask[idx] = 1
                pop[-i] = mask
            if verbose:
                print(f"[stagnation] injected {n_replace} random individuals at epoch {epoch}")
            no_improve = 0


    return best_loss, best_mask, eval_cache


def bwo_select(X_train_scaled, 
               y_train,
               npop=30, 
               maxiter=20, 
               elitism=2,
               crossover_prob=0.9, 
               mutation_prob=0.02,
               stagnation_inject_every=10, 
               inject_fraction=0.25,
               alpha=0.05, 
               cv_folds=5, 
               scoring='balanced_accuracy',
               seed=42, 
               verbose=False,
               ):
   
    print("Selecting best features from BWO...")
    n_features = X_train_scaled.shape[1]
    
    clf_bwo = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42
        )
    
    def objective(mask):
        mask = np.asarray(mask).astype(bool)
        if not np.any(mask):
            return 1.0  
        X_sub = X_train_scaled[:, mask]
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(clf_bwo, X_sub, y_train, cv=skf, scoring=scoring, n_jobs=1)
        sparsity_pen = alpha * (mask.sum() / n_features)
        return 1.0 - float(np.mean(scores)) + sparsity_pen

    best_loss, best_mask, _ = bwo_minimize(
        objective_cv_func=objective,
        n_features=n_features,
        npop=npop,
        maxiter=maxiter,
        elitism=elitism,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        stagnation_inject_every=stagnation_inject_every,
        inject_fraction=inject_fraction,
        seed=seed,
        verbose=verbose,
        max_features=250
    )

    selected_indices = np.where(best_mask == 1)[0]
    print("Done selecting best features from BWO.")
    
    return selected_indices
