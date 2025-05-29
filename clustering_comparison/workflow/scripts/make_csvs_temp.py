import os
import csv
import pandas as pd
from pathlib import Path

# Path to your manifest
manifest_path = '/zata/zippy/kresgeb/clustering_comparison/results/comparisons/comparison_manifest.csv'
manifest_dir = Path(manifest_path).parent.resolve()

# Will store rows grouped by method
results_by_method = {}

def extract_metadata(file_path1, result_dir, base_dir):
    parts = Path(file_path1).parts

    year = parts[2]
    method = parts[3]

    metadata = {
        "year": year,
        "method": method,
        "model_variant": None,
        "k": None,
        "sample": None,
        "nreps": None,
        "seed": None,
        "PCs": None,
    }

    if method == 'mclust':
        metadata["model_variant"] = parts[4]

    for part in parts:
        if part.startswith("k="):
            metadata["k"] = part.split("=")[1]
        elif part.startswith("nreps="):
            if "_" in part:
                # e.g. "nreps=10000_seed=314.csv"
                nreps_part, seed_part = part.split("_")
                metadata["nreps"] = nreps_part.split("=")[1]
                metadata["seed"] = seed_part.split("=")[1].replace(".csv", "")
            else:
                metadata["nreps"] = part.split("=")[1].replace(".csv", "")
        elif part.startswith("seed="):
            metadata["seed"] = part.split("=")[1].replace(".csv", "")
        elif part.startswith("PCs="):
            metadata["PCs"] = part.split("=")[1].replace(".csv", "")
        elif part.startswith("15") or part.startswith("Br"):
            metadata["sample"] = part
    # Read ARI
    ari_path = f"/zata/zippy/kresgeb/clustering_comparison/{result_dir}ari.txt"
    print("Looking for ARI at:", ari_path)
    try:
        with open(ari_path, 'r') as f:
            metadata["ari"] = float(f.read().strip())
    except Exception as e:
        metadata["ari"] = None

    return metadata


# Read manifest
with open(manifest_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_path1 = row['file_path1']
        result_dir = row['result_dir']
        meta = extract_metadata(file_path1, result_dir, manifest_dir)
        # Add the contingency_table URL
        meta["contingency_table"] = f"https://users.wenglab.org/kresgeb/cluster-comparisons/{result_dir}contingency_table.csv"

        method_key = meta['method']
        if method_key not in results_by_method:
            results_by_method[method_key] = []
        results_by_method[method_key].append(meta)

# Determine all non-empty columns per method
for method, rows in results_by_method.items():
    # Only keep columns that have at least one non-null value
    df = pd.DataFrame(rows)
    non_empty_cols = df.columns[df.notna().any()].tolist()
    df = df[non_empty_cols]

    output_file = f"/zata/zippy/kresgeb/clustering_comparison/results/sheet/{method}.csv"
    df.to_csv(output_file, index=False)
    print(f"Wrote {output_file} with {len(df)} rows and columns: {non_empty_cols}")
