import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


"""
GA (Genetic Algorithm)
"""
# helper functions
def init_population(pop_size, n_features, prob=None, max_features=None):
    pop = (rng.random((pop_size, n_features)) < prob).astype(int)
    if max_features is not None:
        for ind in pop:
            if ind.sum() > max_features:
                ones_idx = np.where(ind == 1)[0]
                rng.shuffle(ones_idx)
                ind[ones_idx[max_features:]] = 0
    return pop

def fitness(ind, X, y):
    if ind.sum() == 0:  
        return 0
    X_sel = X[:, ind == 1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(solver="liblinear", max_iter=2000)
    scores = cross_val_score(clf, X_sel, y, cv=skf, scoring="accuracy")
    return scores.mean()

def select(pop, fits):
    total = fits.sum()
    if total == 0:
        idx = rng.choice(len(pop), size=2, replace=True)
    else:
        probs = fits / total
        idx = rng.choice(len(pop), size=2, replace=True, p=probs)
    return pop[idx[0]], pop[idx[1]]

def crossover(p1, p2, rate=None):
    if rng.random() < rate:
        point = rng.integers(1, len(p1))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2
    else:
        return p1.copy(), p2.copy()

def mutate(ind, rate=None):
    flips = rng.random(ind.shape) < rate
    ind[flips] = 1 - ind[flips]
    return ind

def enforce_max_features(ind, max_features):
    if max_features is not None and ind.sum() > max_features:
        ones_idx = np.where(ind == 1)[0]
        rng.shuffle(ones_idx)
        ind[ones_idx[max_features:]] = 0

# GA main loop 
def simple_ga(X,
              y,
              pop_size=None,
              generations=None,
              crossover_rate=None,
              mutation_rate=None,
              max_features=None,
              elite_size=None):
    
    n_features = X.shape[1]
    pop = init_population(pop_size, n_features, prob=0.01, max_features=max_features)

    best_mask = None
    best_score = -np.inf

    for gen in range(generations):
        fits = np.array([fitness(ind, X, y) for ind in pop])
        gen_best_idx = np.argmax(fits)
        gen_best_score = fits[gen_best_idx]
        gen_best_mask = pop[gen_best_idx].copy()

        # update global best
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_mask = gen_best_mask.copy()

        logging.info(f"GA | Gen {gen} | best_score={gen_best_score:.3f} | n_features={gen_best_mask.sum()}")

        # Elitism 
        new_pop = []
        if elite_size is not None and elite_size > 0:
            elite_size = min(elite_size, pop_size)
            top_idxs = np.argsort(fits)[-elite_size:][::-1]
            for idx in top_idxs:
                elite_copy = pop[idx].copy()
                enforce_max_features(elite_copy, max_features)
                new_pop.append(elite_copy)

        # generate populations
        while len(new_pop) < pop_size:
            p1, p2 = select(pop, fits)
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            enforce_max_features(c1, max_features)
            enforce_max_features(c2, max_features)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop[:pop_size])

    return best_mask, best_score


def ga_select(X_train_scaled, y_train,
              pop_size=5,
              generations=2,
              crossover_rate=0.7,
              mutation_rate=0.01,
              max_features=150,
              elite_size=1
              ):

    global rng
    rng = np.random.default_rng(42)
    logging.info("Selecting best features from GA...")
    
    best_mask, best_score = simple_ga(
        X_train_scaled, y_train,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        max_features=max_features,
        elite_size=elite_size
    )
   
    best_mask = np.asarray(best_mask)
    selected_indices = np.where(best_mask == 1)[0]
    logging.info("Done selecting best features from GA.")
    
    return selected_indices