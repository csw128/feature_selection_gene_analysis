from utils import normalize_train_test, train_and_predict_labels
import numpy as np


def baseline_cv_experiment(X, y, foldlist):

    n_samples = X.shape[0]

    baselines = ["svm_all_features", "max_apriori"]

    outer_results = {b: [] for b in baselines}

    # Loop over repeats
    for repeat_idx, outer_folds in enumerate(foldlist):

        preds_per_baseline = {b: np.full(n_samples, -1, dtype=int) for b in baselines}

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

            # (1) SVM on all features
            y_pred_svm = train_and_predict_labels(
                X_train_scaled, y_train, X_test_scaled
            )
            preds_per_baseline["svm_all_features"][test_idx] = y_pred_svm

            # (2) Maximum a priori (majority class)
            values, counts = np.unique(y_train, return_counts=True)
            majority_class = values[np.argmax(counts)]
            print(f'Majority class label in y_train: {majority_class}')

            y_pred_apriori = np.full(len(test_idx), majority_class, dtype=int)
            preds_per_baseline["max_apriori"][test_idx] = y_pred_apriori

        # after finishing all folds in this repeat, compute accuracy
        repeat_indices = sorted(set(sum(outer_folds, [])))
        y_true_repeat = y[repeat_indices]

        for baseline in baselines:
            preds = preds_per_baseline[baseline][repeat_indices]
            acc = (preds == y_true_repeat).mean()
            
            outer_results[baseline].append(float(acc))

    
    # Aggregate summary (mean ± std for accuracy)
    summary = {}
    for baseline, acc_list in outer_results.items():
        arr = np.array(acc_list, dtype=float)
        summary[baseline] = {
            "accuracy_mean": float(np.nanmean(arr)),
            "accuracy_std": float(np.nanstd(arr)),
        }

    return {
        "outer_results": outer_results,
        "summary": summary,
    }