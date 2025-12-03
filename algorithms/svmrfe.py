from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


"""
SVM RFECV
"""
def svm_rfe(X_train_scaled, y_train, max_features=150):
    logging.info("Selecting best features from SVM RFECV...")
    
    # Define SVM with balanced class weight
    svm_model = SVC(kernel="linear", random_state=42)
    
    # 5-fold CV for RFECV
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # RFECV selector
    selector_rfecv = RFECV(
        estimator=svm_model,
        step=3,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1
    )
    
    # Fit selector on training data
    selector_rfecv.fit(X_train_scaled, y_train)
    
    # Get selected feature indices
    ranking = selector_rfecv.ranking_  
    n_features_selected = selector_rfecv.n_features_
    k = min(max_features, n_features_selected)
    top_indices = np.argsort(ranking)[:k]
    
    logging.info("Done selecting best features from SVM RFECV.")
    
    return np.sort(top_indices)