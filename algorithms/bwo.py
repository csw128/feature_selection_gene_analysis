import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(
    filename='run.log',
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
    max_features=None,
    alpha_children=None,
    crossover_prob=None,
    procreate_rate=None,        
    cannibalism_rate=None,        
    mutation_rate=None          
):
    
    # initialize population
    pop = []
    for _ in range(pop_size):
        mask = rng.integers(0, 2, size=n_features).astype(int)
        mask = _enforce_max_features(mask, max_features)
        if mask.sum() == 0:
            mask[rng.integers(0, n_features)] = 1
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
        logging.info(f"BWO | Gen {epoch} | best_score={best_score:.6f} | median_score={median_score:.6f} | n_features_best={best_mask.sum()}")

        # Reproduction
        nr = max(1, int(pop_size * procreate_rate))
        pop1 = pop[:nr]  
        pop2 = []

        n_pairs = max(1, n_features // 2)
        
        for _rep in range(nr):
            # choose two parents at random from pop1
            p_idx = rng.choice(len(pop1), size=2, replace=False)
            par1 = pop1[p_idx[0]].copy()
            par2 = pop1[p_idx[1]].copy()

            # generate children 
            children = []
            for _p in range(n_pairs):
                if rng.random() < crossover_prob:
                    child1 = (alpha_children * par1 + (1 - alpha_children) * par2).round().astype(int)
                    child2 = (alpha_children * par2 + (1 - alpha_children) * par1).round().astype(int)
                else:
                    child1 = par1.copy()
                    child2 = par2.copy()

                for c in (child1, child2):
                    c[:] = _enforce_max_features(c, max_features)
                    if c.sum() == 0:
                        c[rng.integers(0, n_features)] = 1
                    children.append(c)

            # cannibalism: destroy weaker parent
            s1 = eval_mask(par1)
            s2 = eval_mask(par2)
            if s1 >= s2:
                mother = par1.copy()
            else:
                mother = par2.copy()
                
            pop2.append(mother)
            
            # cannibalism: destroy some children
            n_survive = max(1, int(ceil((1.0 - cannibalism_rate) * len(children))))
            if n_survive >= len(children):
                survivors = children
            else:
                # random selection of survivors
                pick_idx = rng.choice(len(children), size=n_survive, replace=False)
                survivors = [children[i] for i in pick_idx]

            pop2.extend(survivors)

        # mutation 
        nm = int(max(0, round(nr * mutation_rate)))   
        pop3 = []
        if nm > 0:
            nm = min(nm, len(pop1))
            sel_idx = rng.choice(len(pop1), size=nm, replace=False)
            for idx in sel_idx:
                sol = pop1[idx].copy()
                # mutate one decision variable chosen at random
                flip_idx = rng.integers(0, n_features)
                sol[flip_idx] = 1 - sol[flip_idx]
                sol[:] = _enforce_max_features(sol, max_features)
                if sol.sum() == 0:
                    sol[rng.integers(0, n_features)] = 1
                pop3.append(sol)

        # New population 
        new_pop = pop2 + pop3

        # If too large, truncate by keeping the best ones (evaluate if needed)
        if len(new_pop) > pop_size:
            new_scores = [eval_mask(m) for m in new_pop]
            idx_sorted_new = np.argsort(new_scores)[::-1]
            new_pop = [new_pop[i] for i in idx_sorted_new[:pop_size]]

        pop = new_pop

    return best_score, best_mask, eval_cache


def bwo_select(X_train_scaled, 
               y_train,
               pop_size=50,
               generations=100, 
               alpha=0.05, 
               alpha_children=0.5,
               max_features=150,
               crossover_prob=0.7,
               procreate_rate=0.5,       
               cannibalism_rate=0.5,        
               mutation_rate=0.1         
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
        alpha_children=alpha_children,
        max_features=max_features,
        crossover_prob=crossover_prob,       
        procreate_rate=procreate_rate,        
        cannibalism_rate=cannibalism_rate,        
        mutation_rate=mutation_rate          
    )

    selected_indices = np.where(best_mask == 1)[0]
    logging.info("Done selecting best features from BWO.")
    
    return selected_indices
