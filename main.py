from utils import create_outer_foldlist
from outer_cv import nested_cv_experiment
import pandas as pd
import json
import logging

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def run_experiment(dataset_path, result_path, foldlist):
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"]).values 
    y = df["label"].values  

    results = nested_cv_experiment(X, y, foldlist)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
        print(f'Saved results to {result_path}.')


## Run experiments for all datasets

# GB (leukemia)
# ALL=47, AML=25
logging.info("Starting experiment for GB")
fl1 = create_outer_foldlist(72)
run_experiment("dataset/GB/reduced.csv", "results/GB_results.json", fl1) 
logging.info("Completed experiment for GB")
logging.info("------------------------------------------------------------------------------")

# GSE48558 (leukemia)
# B ALL=27, AML=18
logging.info("Starting experiment for GSE48558")
fl2 = create_outer_foldlist(45)
run_experiment("dataset/GSE48558/reduced.csv", "results/GSE48558_results.json", fl2) 
logging.info("Completed experiment for GSE48558")
logging.info("------------------------------------------------------------------------------")

# GSE25136 (prostate)
# recurrent=39, non-recurrent=40
logging.info("Starting experiment for GSE25136")
fl3 = create_outer_foldlist(79)
run_experiment("dataset/GSE25136/reduced.csv", "results/GSE25136_results.json", fl3) 
logging.info("Completed experiment for GSE25136")
logging.info("------------------------------------------------------------------------------")

# GSE10072 (lung)
# tumor=58, non-tumor=49
logging.info("Starting experiment for GSE10072")
fl4 = create_outer_foldlist(107)
run_experiment("dataset/GSE10072/reduced.csv", "results/GSE10072_results.json", fl4) 
logging.info("Completed experiment for GSE10072")
logging.info("------------------------------------------------------------------------------")

# GSE10245 (lung)
# AC=40, SCC=18
logging.info("Starting experiment for GSE10245")
fl5 = create_outer_foldlist(58)
run_experiment("dataset/GSE10245/reduced.csv", "results/GSE10245_results.json", fl5) 
logging.info("Completed experiment for GSE10245")
logging.info("------------------------------------------------------------------------------")
