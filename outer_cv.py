# Outer CV 

from utils import normalize_train_test, evaluate_svm
from algorithms.svmrfe import svm_rfe
from algorithms.gapbil import ga_select
from algorithms.bwo import bwo_select
import pandas as pd


## 8-2 split
def nested_cv_experiment(X, y, foldlist):
    outer_results = {
        'svm_rfe': [],
        'ga': [],
        'bwo': []
    }

    outer_selected_features = {
    'svm_rfe': [],
    'ga': [],
    'bwo': []
    }

    # Loop over outer repetitions
    for repeat_idx, outer_folds in enumerate(foldlist):
        
        n_folds = len(outer_folds)
        
        step = 2
        test_pairs = [(i, i+1) for i in range(0, n_folds, step) if i+1 < n_folds]
        
        # Loop over outer iterations (test fold = 1 fold)
        for iter_idx, (f1, f2) in enumerate(test_pairs):
            print(f"Repeat {repeat_idx+1}/{len(foldlist)}, k fold {iter_idx+1}/{len(test_pairs)} (test folds {f1+1},{f2+1})")
            
            # Split train/test
            test_idx = outer_folds[f1] + outer_folds[f2]
            
            train_idx = []
            for idx_fold, fold in enumerate(outer_folds):
                if idx_fold not in (f1, f2):
                    train_idx.extend(fold)
            
            train_idx = sorted(set(train_idx))
            test_idx = sorted(set(test_idx))
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            # Normalize train/test
            X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)
            print("Normalised current fold train test set.")
            
            # Run feature selection algorithms to return the best feature indices
            best_features_svm = svm_rfe(X_train_scaled, y_train)
            best_features_ga = ga_select(X_train_scaled, y_train, verbose=True)
            best_features_bwo = bwo_select(X_train_scaled, y_train, verbose=True)
            
            # Store best features for record
            outer_selected_features['svm_rfe'].append(best_features_svm)
            outer_selected_features['ga'].append(best_features_ga)
            outer_selected_features['bwo'].append(best_features_bwo)
            
            # print("outer selected features: ", outer_selected_features)

            # Evaluate SVM on the selected features 
            metrics_svm = evaluate_svm(X_train_scaled[:, best_features_svm], y_train,
                                   X_test_scaled[:, best_features_svm], y_test)
            
            metrics_ga = evaluate_svm(X_train_scaled[:, best_features_ga], y_train,
                                  X_test_scaled[:, best_features_ga], y_test)
            
            metrics_bwo = evaluate_svm(X_train_scaled[:, best_features_bwo], y_train,
                                   X_test_scaled[:, best_features_bwo], y_test)
            
            # Store results
            outer_results['svm_rfe'].append(metrics_svm)
            outer_results['ga'].append(metrics_ga)
            outer_results['bwo'].append(metrics_bwo)

            print("outer results: ", outer_results)
            
        #     break
        
        # break
    
    # Aggregate results
    summary = {}

    for algo, results in outer_results.items():
        # Convert list of dicts → DataFrame
        df = pd.DataFrame(results)
        
        # Compute mean and std for each metric
        mean_metrics = df.mean(numeric_only=True)
        std_metrics = df.std(numeric_only=True)
        
        # Combine into a summary dict
        summary[algo] = {
            metric: {"mean": mean_metrics[metric], "std": std_metrics[metric]}
            for metric in mean_metrics.index
        }

    return summary





## 9-1 split
# def nested_cv_experiment(X, y, foldlist):
#     outer_results = {
#         'svm_rfe': [],
#         'ga': [],
#         'bwo': []
#     }

#     outer_selected_features = {
#     'svm_rfe': [],
#     'ga': [],
#     'bwo': []
#     }


#     # Loop over outer repetitions
#     for repeat_idx, outer_folds in enumerate(foldlist):
        
#         # Loop over outer iterations (test fold = 1 fold)
#         for test_fold_idx in range(len(outer_folds)):
#             print(f"Repeat {repeat_idx+1}/{len(foldlist)}, k fold {test_fold_idx+1}/{len(outer_folds)}")
            
#             # Split train/test
#             test_idx = outer_folds[test_fold_idx]
#             train_idx = [i for fold in outer_folds if fold != test_fold_idx for i in fold]  # flatten
            
#             X_train, y_train = X[train_idx, :], y[train_idx]
#             X_test, y_test = X[test_idx, :], y[test_idx]
#             print("Split current fold train test set.")
            
#             # Normalize train/test
#             X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)
#             print("Normalised current fold train test set.")
            
#             # Run feature selection algorithms to return the best feature indices
#             best_features_svm = svm_rfe(X_train_scaled, y_train)
#             best_features_ga = ga_select(X_train_scaled, y_train, verbose=True)
#             best_features_bwo = bwo_select(X_train_scaled, y_train, verbose=True)
            
#             # Store best features for record
#             outer_selected_features['svm_rfe'].append(best_features_svm)
#             outer_selected_features['ga'].append(best_features_ga)
#             outer_selected_features['bwo'].append(best_features_bwo)
            
#             print("outer selected features: ", outer_selected_features)

#             # Evaluate SVM on the selected features 
#             metrics_svm = evaluate_svm(X_train_scaled[:, best_features_svm], y_train,
#                                    X_test_scaled[:, best_features_svm], y_test)
            
#             metrics_ga = evaluate_svm(X_train_scaled[:, best_features_ga], y_train,
#                                   X_test_scaled[:, best_features_ga], y_test)
            
#             metrics_bwo = evaluate_svm(X_train_scaled[:, best_features_bwo], y_train,
#                                    X_test_scaled[:, best_features_bwo], y_test)
            
#             # Store results
#             outer_results['svm_rfe'].append(metrics_svm)
#             outer_results['ga'].append(metrics_ga)
#             outer_results['bwo'].append(metrics_bwo)

#             print("outer results: ", outer_results)
            
#             break
        
#         break
    
#     # # Aggregate results
#     # summary = {algo: (np.mean(scores), np.std(scores)) for algo, scores in outer_results.items()}
#     # return summary

