#!/usr/bin/env python3
import pandas as pd
from itertools import product


def compare_BayesSpace_against_ground_truth(year, sample):
    rows = []
    for k in snakemake.config["bayesspace_parameters"]["k"]:
        for nreps in snakemake.config["bayesspace_parameters"]["nreps"]:
            for seed in snakemake.config["bayesspace_parameters"]["seed"]:
                file_path1 = f"results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.csv"
                file_path2 = f"results/ground_truths/{year}/{sample}.csv"
                result_dir = f"results/comparisons/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}/vs_ground_truth/"
                rows.append((file_path1, file_path2, result_dir))
    return rows

def compare_mclust_against_ground_truth(year, sample):
    rows = []
    for k in snakemake.config["mclust_parameters"]["k"]:
        for PCs in snakemake.config["mclust_parameters"]["PCs"]:
            for model in snakemake.config["mclust_parameters"]["model"]:
                file_path1 = f"results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.csv"
                file_path2 = f"results/ground_truths/{year}/{sample}.csv"
                result_dir = f"results/comparisons/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}/vs_ground_truth/"
                rows.append((file_path1, file_path2, result_dir))
    return rows


def compare_against_ground_truth():
    rows = []
    for model in ["BayesSpace", "mclust"]:
        for year in snakemake.config["ground_truth_columns"]:
            for sample in snakemake.config["ground_truth_samples"][year]:
                if model == "BayesSpace":
                    rows.extend(compare_BayesSpace_against_ground_truth(year, sample))
                elif model == "mclust":
                    rows.extend(compare_mclust_against_ground_truth(year, sample))
                else:
                   # Model is not supported
                   print(f"Model {model} is not supported.")
                   exit(1)
    return rows
    

# Create a manifest of all the comparisons
def main():
    all_rows = []

    all_rows.extend(compare_against_ground_truth())

    df = pd.DataFrame(all_rows, columns=["file_path1", "file_path2", "result_dir"])
    
    df.to_csv(snakemake.output[0], index=False)

if __name__ == "__main__":
    main()
