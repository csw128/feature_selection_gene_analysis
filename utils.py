import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, f1_score



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


def evaluate_svm(X_train, y_train, X_test, y_test, random_state=42):
    metrics = {
        'accuracy': np.nan,
        'balanced_accuracy': np.nan,
        'roc_auc': np.nan,
        'log_loss': np.nan,
        'f1': np.nan
    }

    # Fit classifier with probability estimates
    clf = SVC(kernel='linear', probability=True, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_test, y_pred))
    metrics['f1'] = float(f1_score(y_test, y_pred))
    
    # predicted probabilities required for log_loss and roc_auc
    y_proba = clf.predict_proba(X_test)
    
    # For roc_auc and log_loss we need both classes to be present in y_test
    unique_labels = np.unique(y_test)
    if y_proba is not None:
        metrics['log_loss'] = float(log_loss(y_test, y_proba, labels=clf.classes_))
       
        if len(unique_labels) == 2:
            pos_index = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, pos_index]))

    return metrics




