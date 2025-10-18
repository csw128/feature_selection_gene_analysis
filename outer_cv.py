# Outer CV 

from utils import normalize_train_test, evaluate_svm
from algorithms.svmrfe import svm_rfe
from algorithms.gapbil import ga_select
from algorithms.bwo import bwo_select
import numpy as np


def nested_cv_experiment(X, y, foldlist):
    outer_results = {
        'svm_rfe': [],
        'ga': [],
        'bwo': []
    }

    outer_selected_features = {
    'svm_rfe': [],
    'ga': [],
    'de': [],
    'bwo': []
    }


    # Loop over outer repetitions
    for repeat_idx, outer_folds in enumerate(foldlist):
        print(f"Outer repeat {repeat_idx+1}/{len(foldlist)}")
        
        # Loop over outer iterations (test fold = 1 fold)
        for test_fold_idx in range(len(outer_folds)):
            
            # Split train/test
            test_idx = outer_folds[test_fold_idx]
            train_idx = [i for fold in outer_folds if fold != test_fold_idx for i in fold]  # flatten
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            # Normalize train/test
            X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)
            
            # --- Run feature selection algorithms ---
            # Each algorithm returns the best feature indices
            best_features_svm = svm_rfe(X_train_scaled, y_train)
            best_features_ga = ga_select(X_train_scaled, y_train)
            best_features_bwo = bwo_select(X_train_scaled, y_train)
            
            # Store best features for record
            outer_selected_features['svm_rfe'].append(best_features_svm)
            outer_selected_features['ga'].append(best_features_ga)
            outer_selected_features['bwo'].append(best_features_bwo)

            # Evaluate SVM on the selected features
            acc_svm = evaluate_svm(X_train_scaled[:, best_features_svm], y_train,
                                   X_test_scaled[:, best_features_svm], y_test)
            
            acc_ga = evaluate_svm(X_train_scaled[:, best_features_ga], y_train,
                                  X_test_scaled[:, best_features_ga], y_test)
            
            acc_bwo = evaluate_svm(X_train_scaled[:, best_features_bwo], y_train,
                                   X_test_scaled[:, best_features_bwo], y_test)
            
            # Store results
            outer_results['svm_rfe'].append(acc_svm)
            outer_results['ga'].append(acc_ga)
            outer_results['bwo'].append(acc_bwo)
    
    # Aggregate results
    summary = {algo: (np.mean(scores), np.std(scores)) for algo, scores in outer_results.items()}
    return summary
