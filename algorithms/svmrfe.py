import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


"""
SVM RFECV
"""
# Define SVM with class weight balancing
svm_model = SVC(kernel="linear", class_weight="balanced", random_state=42)

# set to 5 fold for small dataset
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selector_rfecv = RFECV(
    estimator=svm_model,
    step=3,                   # drop 3 features per iteration
    cv=cv_strategy,
    scoring="accuracy",       
    n_jobs=-1
)

selector_rfecv.fit(X_train_scl, y_train_array)
print(f"Optimal number of features: {selector_rfecv.n_features_}")

selected_mask_rfecv = selector_rfecv.support_
selected_gene_names_rfecv = X_train.columns[selected_mask_rfecv]
print("Selected gene names:", list(selected_gene_names_rfecv))