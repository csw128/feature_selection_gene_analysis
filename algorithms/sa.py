import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from math import exp
import logging

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


"""
SA (Simulated Annealing)
"""
def fitness(ind, X, y, alpha=None, cache=None):
    key = tuple(np.nonzero(ind)[0])
    if cache is not None and key in cache:
        return cache[key]

    n_selected = ind.sum()
    if n_selected == 0:
        val = 0.0
        if cache is not None:
            cache[key] = val
        return val

    X_sel = X[:, ind]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    clf = LogisticRegression(solver="liblinear", max_iter=2000)
    score = cross_val_score(clf, X_sel, y, cv=skf, scoring="accuracy").mean()
    penalty = alpha * (n_selected / X.shape[1])
    val = score - penalty

    if cache is not None:
        cache[key] = val
    return val


def simulated_annealing_feature_selection(X, y,
                                          fitness_func,
                                          max_generations=None,
                                          max_features=None,
                                          alpha=None,
                                          initial_temp=None,
                                          cooling_rate=None,
                                          min_temp=None,
                                          ):
    
    rng = np.random.default_rng()
   
    n_features = X.shape[1]
    cache = {}

    # Initialize 
    n_init = max(1, min(max_features, max(1, max_features // 2)))
    init_indices = rng.choice(n_features, size=n_init, replace=False)
    current = np.zeros(n_features, dtype=bool)
    current[init_indices] = True
    current_score = fitness_func(current, X, y, alpha=alpha, cache=cache)

    best = current.copy()
    best_score = current_score

    T = initial_temp

    history = []

    for gen in range(1, max_generations + 1):
        if gen < 0.2 * max_generations:
        # Early stage - explore more broadly
          n_flips = rng.integers(5, 30)
        else:
        # Later stage - fine-tune locally
          n_flips = rng.integers(1, 4)

        candidate = current.copy()
        flip_indices = rng.choice(n_features, size=n_flips, replace=False)
        candidate[flip_indices] = ~candidate[flip_indices]

        # enforce max_features
        n_cand = candidate.sum()
        if n_cand > max_features:
            ones = np.nonzero(candidate)[0]
            n_to_turn_off = int(n_cand - max_features)
            off_idx = rng.choice(ones, size=n_to_turn_off, replace=False)
            candidate[off_idx] = False

        cand_score = fitness_func(candidate, X, y, alpha=alpha, cache=cache)

        # acceptance
        accept = False
        if cand_score >= current_score:
            accept = True
        else:
            if T > 0:
                prob = exp((cand_score - current_score) / T)
                if rng.random() < prob:
                    accept = True

        if accept:
            current = candidate
            current_score = cand_score

        if current_score > best_score:
            best_score = current_score
            best = current.copy()

        history.append((gen, best_score, best.sum()))
        
        logging.info(f"SA | Gen {gen:3d} | temp={T:.4f} | current_score={current_score:.4f} | best_score={best_score:.4f} | n_features={best.sum():4d}")

        T = max(min_temp, T * cooling_rate)

    return {
        "best_mask": best,
        "best_score": best_score,
    }


def sa_select(X_train_scaled, 
              y_train,
              max_generations=2,
              max_features=150,
              alpha=0.05,
              initial_temp=1.5,
              cooling_rate=0.99,
              min_temp=1e-4,
              ):
    
    logging.info("Selecting best features from SA...")
    
    res = simulated_annealing_feature_selection(
        X_train_scaled,               
        y_train,                      
        fitness_func=fitness,         
        max_generations=max_generations,
        max_features=max_features,
        alpha=alpha,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        min_temp=min_temp
    )

    best_mask = res.get("best_mask")
    selected_indices = np.where(np.asarray(best_mask, dtype=bool))[0]
    
    logging.info("Done selecting best features from SA.")
    
    return selected_indices