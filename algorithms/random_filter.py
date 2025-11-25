import numpy as np
import logging

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


'''
Filter method: Random selection
'''
def random_select(X_train, max_features=150):
    logging.info("Selecting random features from random selection...")
    
    n_features = X_train.shape[1]
    
    rng = np.random.default_rng()    

    # select random number of features between 10 to max_features
    n_select = rng.integers(10, max_features + 1)
    indices = rng.choice(n_features, size=n_select, replace=False)
    
    logging.info("Done selecting random features from random selection.")

    return sorted(indices.tolist())
