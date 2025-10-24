from sklearn.datasets import make_classification
import numpy as np
from outer_cv import nested_cv_experiment
from utils import create_outer_foldlist


# Synthetic dataset
X, y = make_classification(
    n_samples=40,
    n_features=2000,
    n_informative=100,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

rng = np.random.default_rng(42)


foldlist = create_outer_foldlist(n_samples=40, n_folds=10, n_repeats=10)

summary, results, features = nested_cv_experiment(X, y, foldlist)



