import numpy as np

'''
Filter method: Random selection
'''
def random_select(X_train, max_features=100, random_state=42):
    print("Selecting random features from random selection...")
    
    n_features = X_train.shape[1]
    rng = np.random.default_rng(random_state)

    # select random number of features between 1 to max_features
    n_select = rng.integers(1, max_features + 1)
    indices = rng.choice(n_features, size=n_select, replace=False)
    
    print("Done selecting random features from random selection.")

    return sorted(indices.tolist())
