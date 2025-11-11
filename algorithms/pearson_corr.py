import numpy as np
import logging

logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


'''
Filter method: Pearson correlation
'''
def pearson_select(X_train, y_train, max_features=100, random_state=42):
    logging.info("Selecting best features from Pearson Correlation...")
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)

    n_samples, n_features = X.shape
    if n_features == 0:
        return []

    # mean
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()

    # standard deviation
    X_std = X_centered.std(axis=0)
    y_std = y_centered.std()

    # compute absolute Pearson correlation
    corr = np.zeros(n_features)
    valid = X_std > 0
    if y_std > 0:
        corr[valid] = (X_centered[:, valid].T @ y_centered) / (n_samples * X_std[valid] * y_std)

    abs_corr = np.abs(corr)

    # randomly pick k between 1 and max_features
    rng = np.random.default_rng(random_state)
    k = rng.integers(1, max_features + 1)

    # select top-k features
    topk_idx = np.argpartition(-abs_corr, k-1)[:k]
    topk_idx = topk_idx[np.argsort(-abs_corr[topk_idx])]

    logging.info("Done selecting best features from Pearson Correlation.")

    return sorted(topk_idx.tolist())
