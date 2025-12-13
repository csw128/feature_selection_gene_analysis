from utils import create_outer_foldlist
from baseline import baseline_cv_experiment
import pandas as pd
import json


def run_experiment(dataset_path, result_path, foldlist):
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"]).values 
    y = df["label"].values  

    results = baseline_cv_experiment(X, y, foldlist)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
        print(f'Saved results to {result_path}.')


## Run experiments for all datasets

# GB (leukemia)
# ALL=47, AML=25
print("Starting baseline measure for GB")
fl1 = create_outer_foldlist(72)
run_experiment("dataset/GB/reduced.csv", "results_baseline/GB_results_baseline.json", fl1) 
print("Completed for GB")
print("------------------------------------------------------------------------------")

# GSE48558 (leukemia)
# B ALL=27, AML=18
print("Starting baseline measure for GSE48558")
fl2 = create_outer_foldlist(45)
run_experiment("dataset/GSE48558/reduced.csv", "results_baseline/GSE48558_results_baseline.json", fl2) 
print("Completed for GSE48558")
print("------------------------------------------------------------------------------")

# GSE25136 (prostate)
# recurrent=39, non-recurrent=40
print("Starting baseline measure for GSE25136")
fl3 = create_outer_foldlist(79)
run_experiment("dataset/GSE25136/reduced.csv", "results_baseline/GSE25136_results_baseline.json", fl3) 
print("Completed for GSE25136")
print("------------------------------------------------------------------------------")

# GSE10072 (lung)
# tumor=58, non-tumor=49
print("Starting baseline measure for GSE10072")
fl4 = create_outer_foldlist(107)
run_experiment("dataset/GSE10072/reduced.csv", "results_baseline/GSE10072_results_baseline.json", fl4) 
print("Completed for GSE10072")
print("------------------------------------------------------------------------------")

# GSE10245 (lung)
# AC=40, SCC=18
print("Starting baseline measure for GSE10245")
fl5 = create_outer_foldlist(58)
run_experiment("dataset/GSE10245/reduced.csv", "results_baseline/GSE10245_results_baseline.json", fl5) 
print("Completed for GSE10245")
print("------------------------------------------------------------------------------")