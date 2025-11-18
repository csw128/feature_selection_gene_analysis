from sklearn.datasets import make_classification
import numpy as np
from outer_cv import nested_cv_experiment
from utils import create_outer_foldlist
import pandas as pd
import json

# # Synthetic dataset
# X, y = make_classification(
#     n_samples=40,
#     n_features=2000,
#     n_informative=100,
#     n_redundant=5,
#     n_repeated=0,
#     n_classes=2,
#     random_state=42
# )

 

# # foldlist = create_outer_foldlist(n_samples=40, n_folds=10, n_repeats=10)

foldlist = [
    [[32, 7, 39, 18], [4, 37, 16, 5], [25, 28, 21, 20], [35, 24, 10, 27], [31, 30, 26, 6], [29, 11, 23, 9], [36, 22, 19, 3], [0, 15, 17, 34], [12, 14, 38, 2], [33, 1, 13, 8]], 
    [[13, 29, 33, 26], [21, 15, 0, 6], [35, 30, 38, 31], [24, 18, 4, 25], [22, 17, 7, 27], [12, 8, 10, 39], [5, 1, 28, 9], [2, 3, 11, 14], [20, 19, 34, 32], [37, 23, 36, 16]], 
    [[3, 34, 30, 0], [28, 16, 5, 11], [37, 24, 17, 25], [12, 20, 22, 2], [10, 13, 32, 39], [18, 19, 26, 38], [6, 8, 14, 15], [27, 4, 9, 33], [21, 36, 29, 31], [23, 1, 35, 7]], 
    [[6, 15, 8, 5], [3, 32, 37, 16], [33, 22, 34, 36], [4, 25, 20, 7], [31, 2, 24, 11], [17, 38, 12, 14], [21, 0, 9, 10], [27, 29, 30, 26], [18, 39, 1, 19], [13, 23, 28, 35]], 
    [[6, 14, 7, 34], [28, 24, 26, 22], [33, 31, 18, 38], [25, 35, 20, 15], [11, 12, 37, 21], [10, 9, 23, 17], [4, 16, 1, 36], [3, 29, 0, 30], [19, 8, 39, 13], [32, 2, 27, 5]], 
    [[32, 3, 9, 21], [4, 39, 6, 10], [23, 5, 28, 2], [19, 11, 18, 33], [8, 30, 35, 25], [29, 31, 7, 1], [0, 15, 27, 20], [24, 26, 38, 16], [12, 36, 17, 22], [37, 34, 14, 13]], 
    [[19, 31, 23, 4], [10, 2, 14, 30], [28, 32, 15, 16], [12, 0, 34, 38], [35, 18, 37, 13], [24, 22, 3, 29], [8, 11, 33, 26], [20, 39, 7, 5], [1, 27, 6, 36], [17, 9, 25, 21]], 
    [[13, 11, 8, 31], [1, 24, 27, 32], [5, 16, 34, 26], [9, 19, 7, 28], [36, 17, 33, 37], [30, 15, 10, 0], [35, 22, 18, 14], [25, 39, 2, 23], [38, 4, 29, 6], [12, 3, 20, 21]], 
    [[13, 0, 39, 21], [35, 18, 7, 22], [38, 8, 34, 25], [17, 28, 14, 4], [6, 10, 12, 26], [19, 36, 37, 32], [30, 15, 5, 9], [11, 3, 20, 16], [1, 24, 29, 27], [2, 31, 23, 33]], 
    [[9, 24, 27, 13], [38, 31, 16, 29], [35, 23, 1, 8], [11, 4, 14, 2], [20, 39, 25, 5], [32, 0, 12, 15], [3, 28, 22, 17], [36, 26, 6, 33], [30, 34, 7, 19], [18, 37, 10, 21]]
    ]

# results = nested_cv_experiment(X, y, foldlist)


# # GB
# df_gb = pd.read_csv("dataset/GB/reduced.csv")
# X_gb = df_gb.drop(columns=["label"]).values 
# y_gb = df_gb["label"].values  

# golub_test_results = nested_cv_experiment(X_gb, y_gb, foldlist)

# with open("results/golub_test_results.json", "w") as f:
#     json.dump(golub_test_results, f, indent=2)
#     print("saved results to json")
    

# GSE48558
df_48558 = pd.read_csv("dataset/GSE48558/reduced.csv")
X_48558 = df_48558.drop(columns=["label"]).values 
y_48558 = df_48558["label"].values  

gse48558_test_results = nested_cv_experiment(X_48558, y_48558, foldlist)

with open("results/GSE48558_test_results.json", "w") as f:
    json.dump(gse48558_test_results, f, indent=2)
    print("saved results to json")
    

# # GSE25136
# df_25136 = pd.read_csv("dataset/GSE25136/reduced.csv")
# X_25136 = df_25136.drop(columns=["label"]).values 
# y_25136 = df_25136["label"].values  

# gse25136_test_results = nested_cv_experiment(X_25136, y_25136, foldlist)

# with open("results/GSE25136_test_results.json", "w") as f:
#     json.dump(gse25136_test_results, f, indent=2)
#     print("saved results to json")
    

# # GSE10072
# df_10072 = pd.read_csv("dataset/GSE10072/reduced.csv")
# X_10072 = df_10072.drop(columns=["label"]).values 
# y_10072 = df_10072["label"].values  

# gse10072_test_results = nested_cv_experiment(X_10072, y_10072, foldlist)

# with open("results/GSE10072_test_results.json", "w") as f:
#     json.dump(gse10072_test_results, f, indent=2)
#     print("saved results to json")
