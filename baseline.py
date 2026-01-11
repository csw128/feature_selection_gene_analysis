from utils import normalize_train_test, train_and_predict_labels
import numpy as np


def baseline_cv_experiment(X, y, foldlist):

    n_samples = X.shape[0]
    
    # Maximum a priori 
    values, counts = np.unique(y, return_counts=True)
    global_majority_class = values[np.argmax(counts)]
    map_accuracy = np.mean(y == global_majority_class)

    # Loop over repeats
    svm_accuracies = []
    
    for repeat_idx, outer_folds in enumerate(foldlist):

        preds = np.full(n_samples, -1, dtype=int)

        # Loop over folds
        for test_fold_idx in range(len(outer_folds)):
            print(f"Repeat {repeat_idx+1}/{len(foldlist)}, k fold {test_fold_idx+1}/{len(outer_folds)}")
        
            # split train/test
            test_idx = outer_folds[test_fold_idx]
            train_idx = [i for idx, fold in enumerate(outer_folds) if idx != test_fold_idx for i in fold]

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            # normalize train/test
            X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)

            # SVM on all features
            y_pred = train_and_predict_labels(
                X_train_scaled, y_train, X_test_scaled
            )
            
            preds[test_idx] = y_pred

        acc = np.mean(preds == y)
        svm_accuracies.append(float(acc))

    # Aggregate summary
    summary = {
        "svm_all_features": {
            "accuracy_mean": float(np.mean(svm_accuracies)),
            "accuracy_std": float(np.std(svm_accuracies)),
        },
        "max_apriori": {
            "accuracy_mean": float(map_accuracy),
            "accuracy_std": 0.0,
        },
    }

    return {
        "summary": summary,
    }