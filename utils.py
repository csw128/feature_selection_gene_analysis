from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


def create_outer_foldlist(n_samples, n_folds=10, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)
    foldlist = []
    
    for _ in range(n_repeats):
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        foldlist.append([fold.tolist() for fold in folds])
    
    return foldlist


def normalize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(float))
    X_test_scaled = scaler.transform(X_test.astype(float))
    
    return X_train_scaled, X_test_scaled


def train_and_predict_labels(X_train, y_train, X_test, random_state=42):
    clf = SVC(kernel="linear", probability=True, random_state=random_state)
    clf.fit(X_train, y_train)
    
    return clf.predict(X_test)
