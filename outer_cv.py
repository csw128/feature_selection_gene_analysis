from utils import normalize_train_test, train_and_predict_labels
from algorithms.random_filter import random_select
from algorithms.pearson_corr import pearson_select
from algorithms.svmrfe import svm_rfe
from algorithms.ga import ga_select
from algorithms.pbil import pbil_select
from algorithms.sa import sa_select
from algorithms.bwo import bwo_select
from collections import Counter
import numpy as np
import time
import logging

logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

logging.info("Logging - Test start.")


selectors = {
        'rand_sel': lambda Xtr, ytr: random_select(Xtr),
        'pear_corr': lambda Xtr, ytr: pearson_select(Xtr, ytr),
        'svm_rfe': lambda Xtr, ytr: svm_rfe(Xtr, ytr),
        'ga': lambda Xtr, ytr: ga_select(Xtr, ytr),
        'pbil': lambda Xtr, ytr: pbil_select(Xtr, ytr),
        'sa': lambda Xtr, ytr: sa_select(Xtr, ytr),
        'bwo': lambda Xtr, ytr: bwo_select(Xtr, ytr)
    }

# 9-1 split
def nested_cv_experiment(X, y, foldlist):
    
    start = time.time()
    
    n_samples = X.shape[0]
    
    outer_results = {algo: [] for algo in selectors.keys()}  
    outer_selected_features_agg = {algo: [] for algo in selectors.keys()}
    
    # Loop over outer repetitions
    for repeat_idx, outer_folds in enumerate(foldlist):
        
        preds_per_algo = {algo: np.full(n_samples, -1, dtype=int) for algo in selectors.keys()}
        per_iter_selected = {algo: [] for algo in selectors.keys()}

        # Loop over outer iterations (test fold = 1 fold)
        for test_fold_idx in range(len(outer_folds)):
            print(f"Repeat {repeat_idx+1}/{len(foldlist)}, k fold {test_fold_idx+1}/{len(outer_folds)}")
            logging.info(f"Repeat {repeat_idx+1}/{len(foldlist)}, k fold {test_fold_idx+1}/{len(outer_folds)}")
              
            # Split train/test
            test_idx = outer_folds[test_fold_idx]
            train_idx = [i for idx, fold in enumerate(outer_folds) if idx != test_fold_idx for i in fold]
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            # normalize train/test
            X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)
            
            # run all selectors and evaluations 
            for algo, selector_fn in selectors.items():
                best_features = selector_fn(X_train_scaled, y_train)
                best_features = np.asarray(best_features, dtype=int)
            
                # store per-iteration selection
                per_iter_selected[algo].append(best_features.copy())
                
                # train SVM on selected features
                y_pred = train_and_predict_labels(X_train_scaled[:, best_features], y_train,
                                                          X_test_scaled[:, best_features])
            
                preds_per_algo[algo][test_idx] = y_pred
        
        # after finishing all folds in this repeat, compute accuracy per algorithm
        repeat_indices = sorted(set(sum(outer_folds, [])))
        y_true_repeat = y[repeat_indices]
        
        logging.info(f"Finished repeat {repeat_idx+1}. Accuracies this repeat:")
        
        for algo in selectors.keys():
            preds = preds_per_algo[algo][repeat_indices]
            acc = (preds == y_true_repeat).mean()

            outer_results[algo].append(float(acc))

            # aggregate selected feature summary for this repeat    
            all_feats = np.concatenate(per_iter_selected[algo]) if per_iter_selected[algo] else np.array([], dtype=int)
            freq_counter = Counter(all_feats.tolist())
            union_feats = np.array(sorted(freq_counter.keys()), dtype=int)

            # store aggregated info for this repeat
            agg_info = {
                "repeat_idx": repeat_idx,
                "union_features": union_feats.tolist(),
                "feature_freq": dict(freq_counter),       
                "per_iteration_features": [arr.tolist() for arr in per_iter_selected[algo]]  
            }
            outer_selected_features_agg[algo].append(agg_info)

        for algo in selectors.keys():
            logging.info(f"{algo}: {outer_results[algo][-1]:.4f}")
         
        # break
    
    # Aggregate summary (mean ± std for accuracy)
    summary = {}
    for algo, acc_list in outer_results.items():
        arr = np.array(acc_list, dtype=float)
        summary[algo] = {
            "accuracy_mean": float(np.nanmean(arr)),
            "accuracy_std": float(np.nanstd(arr)),
        }
    
    end = time.time()
    print(f"Total runtime: {(end - start)/60} minutes")
    logging.info(f"Total runtime: {(end - start)/60} minutes")
    
    return {
        "outer_results": outer_results,
        "outer_selected_features_agg": outer_selected_features_agg,
        "summary": summary
    }