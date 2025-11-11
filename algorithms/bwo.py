import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


"""
BWO (Black Widow Optimization)
"""
def _enforce_max_features(mask, max_features):
    if max_features is None:
        return mask
    
    m = mask.copy()
    n_on = int(m.sum())
    if n_on > max_features:
        ones_idx = np.flatnonzero(m)
        turn_off = rng.choice(ones_idx, size=(n_on - max_features), replace=False)
        m[turn_off] = 0
        
    return m


def bwo_minimize(
    objective_cv_func,        
    n_features,
    pop_size=None,
    generations=None,
    elitism=None,
    crossover_prob=None,
    mutation_prob=None,      
    stagnation_inject_every=None,
    inject_fraction=None,
    max_features=None         
):
    
    # initialize population
    pop = []
    for _ in range(pop_size):
        mask = np.random.choice([0,1], size=n_features).astype(int)
        mask = _enforce_max_features(mask, max_features)
        if mask.sum() == 0:
            idx = rng.randint(0, n_features)
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

    best_score = -float('inf')
    best_mask = None
    no_improve = 0

    for epoch in range(1, generations+1):
        scores = [eval_mask(m) for m in pop]
        idx_sorted = np.argsort(scores)[::-1]
        pop = [pop[i] for i in idx_sorted]
        scores = [scores[i] for i in idx_sorted]

        # update best
        if scores[0] > best_score + 1e-12:
            best_score = float(scores[0])
            best_mask = pop[0].copy()
            no_improve = 0
        else:
            no_improve += 1

        median_score = np.median(scores)
        n_features_best = best_mask.sum()
        logging.info(f"BWO | Gen {epoch} | best_score={best_score:.6f} | median_score={median_score:.6f} | n_features_best={n_features_best}")

        # elitism
        new_pop = [pop[i].copy() for i in range(elitism)]

        # generate offspring
        while len(new_pop) < pop_size:
            def tournament():
                contenders = np.random.choice(pop_size, size=3, replace=False)
                best = max(contenders, key=lambda i: scores[i])
                return pop[best]

            p1 = tournament()
            p2 = tournament()

            # crossover 
            if np.random.rand() < crossover_prob:
                swap = np.random.rand(n_features) < 0.5
                child = p1.copy()
                child[swap] = p2[swap]
            else:
                child = p1.copy()

            # mutation:
            flip = np.random.rand(n_features) < mutation_prob
            if flip.any():
                child[flip] = 1 - child[flip]

            # enforce limits
            child = _enforce_max_features(child, max_features)
            if child.sum() == 0:
                idx = rng.randint(0, n_features)
                child[idx] = 1

            new_pop.append(child)

        pop = new_pop

        # stagnation injection
        if no_improve >= stagnation_inject_every:
            n_replace = max(1, int(inject_fraction * pop_size))
            for i in range(1, n_replace+1):
                mask = np.random.choice([0,1], size=n_features).astype(int)
                mask = _enforce_max_features(mask, max_features)
                if mask.sum() == 0:
                    idx = rng.randint(0, n_features) 
                    mask[idx] = 1
                pop[-i] = mask
            
            logging.info(f"[stagnation] injected {n_replace} random individuals at epoch {epoch}")
            no_improve = 0


    return best_score, best_mask, eval_cache


def bwo_select(X_train_scaled, 
               y_train,
               pop_size=5, 
               generations=2, 
               elitism=2,
               crossover_prob=0.9, 
               mutation_prob=0.02,
               stagnation_inject_every=10, 
               inject_fraction=0.25,
               alpha=0.05, 
               max_features=150
               ):
   
    global rng
    rng = np.random.default_rng(42)
    
    logging.info("Selecting best features from BWO...")
    
    n_features = X_train_scaled.shape[1]
    
    clf_bwo = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="liblinear",
        max_iter=2000,
        random_state=42
        )
    
    def objective(mask):
        mask = np.asarray(mask).astype(bool)
        if not np.any(mask):
            return -1.0 
        
        X_sub = X_train_scaled[:, mask]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf_bwo, X_sub, y_train, cv=skf, scoring='accuracy', n_jobs=1)
        mean_score = float(np.mean(scores))
        sparsity_pen = alpha * (mask.sum() / n_features)
        score = mean_score - sparsity_pen
        return score

    best_score, best_mask, _ = bwo_minimize(
        objective_cv_func=objective,
        n_features=n_features,
        pop_size=pop_size,
        generations=generations,
        elitism=elitism,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        stagnation_inject_every=stagnation_inject_every,
        inject_fraction=inject_fraction,
        max_features=max_features
    )

    selected_indices = np.where(best_mask == 1)[0]
    logging.info("Done selecting best features from BWO.")
    
    return selected_indices
