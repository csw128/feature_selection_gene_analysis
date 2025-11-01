import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


"""
PBIL (Population-Based Incremental Learning)
"""
# Fitness function
def fitness(ind, X, y, alpha=None):
    n_selected = ind.sum()
    if n_selected == 0:
        return 0
    
    X_sel = X[:, ind==1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(solver="liblinear", max_iter=2000)
    score = cross_val_score(clf, X_sel, y, cv=skf, scoring="accuracy").mean()
    penalty = alpha * (n_selected / X.shape[1])
    return score - penalty


# PBIL-style GA 
def pbil_select(X_train_scaled, 
                y_train, 
                generations=20, 
                pop_size=30, 
                learning_rate=0.3,
                alpha=0.1, 
                init_prob=0.05, 
                max_features=150,
                ):
    
    global rng
    rng = np.random.default_rng(42)
    
    print("Selecting best features from GA PBIL...")
    n_features = X_train_scaled.shape[1]

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
        fits = np.array([fitness(ind, X_train_scaled, y_train, alpha) for ind in pop])
        gen_best_idx = np.argmax(fits)

        # Update best solution
        if fits[gen_best_idx] > best_score:
            best_score = fits[gen_best_idx]
            best_mask = pop[gen_best_idx].copy()

        # Update probability vector toward best individual
        p = (1 - learning_rate) * p + learning_rate * best_mask

        print(f"PBIL | Gen {gen} | best_score={best_score:.3f} | n_features={best_mask.sum()}")
        
    # Return indices of selected features
    selected_indices = np.where(best_mask==1)[0]
    print("Done selecting best features from GA PBIL.")
    
    return selected_indices
