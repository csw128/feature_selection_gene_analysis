import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def create_outer_foldlist(n_samples: int, n_folds: int = 10, n_repeats: int = 10, seed: int = None) -> List[List[List[int]]]:
    rng = np.random.default_rng(seed)
    foldlist = []
    
    for _ in range(n_repeats):
        indices = np.arange(1, n_samples + 1)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        foldlist.append([fold.tolist() for fold in folds])
    
    return foldlist

# foldlist = create_outer_foldlist(n_samples=40, n_folds=10, n_repeats=10, seed=42)
# print(f"Example folds (first repeat): {foldlist[0]}")


def normalize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(float))
    X_test_scaled = scaler.transform(X_test.astype(float))
    
    return X_train_scaled, X_test_scaled


def evaluate_svm(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)




