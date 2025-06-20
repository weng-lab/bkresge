import json
import sys
import os


# Redirect stdout and stderr to the log file
log_file = open(snakemake.log[0], "w", buffering=1)  # line-buffered
sys.stdout = log_file
sys.stderr = log_file


output_path =  snakemake.output["manifest"]#"/zata/zippy/kresgeb/clustering_comparison/results/vitessce_visualizations/visualization_manifest.json" 


def make_ground_truth_view(year, sample):
    if sample not in snakemake.config["ground_truth_samples"].get(year, []):
        return None
    gt_column = snakemake.config["ground_truth_columns"][year][0]
    return {
        "title": "Ground Truth (Manual Annotation)",
        "clusterAssignmentPath": f"/zata/zippy/kresgeb/clustering_comparison/results/ground_truths/{year}/{sample}.csv",
        "columnName": "ground_truth",
        "sourceColumnName": gt_column,
    }


def make_paper_bayesspace_view(year, sample):
    if year != "2024":
        return None
    path = f"/zata/zippy/kresgeb/clustering_comparison/results/cluster_assignments/2024/paper_bayesspace/{sample}.csv"
    return {
        "title": f"{ari_string(year, sample, path)}Paper BayesSpace k=9",
        "clusterAssignmentPath": path,
        "columnName": "bayes_space_paper",
        "sourceColumnName": "BayesSpace_harmony_09",
    }


def make_mclust_views(year, sample):
    views = []
    params = snakemake.config["mclust_parameters"]
    for model in params["model"]:
        for k in params["k"]:
            for pcs in params["PCs"]:
                path = f"/zata/zippy/kresgeb/clustering_comparison/results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={pcs}.csv"
                views.append({
                    "title": f"{ari_string(year, sample, path)}Mclust k={k} pcs={pcs} model={model}",
                    "clusterAssignmentPath": path,
                    "columnName": f"mclust_k{k}_pcs{pcs}",
                })
    return views


def make_bayesspace_views(year, sample):
    views = []
    params = snakemake.config["bayesspace_parameters"]
    for k in params["k"]:
        for nreps in params["nreps"]:
            for seed in params["seed"]:
                path = f"/zata/zippy/kresgeb/clustering_comparison/results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.csv"
                views.append({
                    "title": f"{ari_string(year, sample, path)}BayesSpace k={k} nreps={nreps} seed={seed}",
                    "clusterAssignmentPath": path,
                    "columnName": f"bayesspace_k{k}_nreps{nreps}_seed{seed}",
                })
    return views

def ari_string(year, sample, path_to_cluster_assignment):

    # If the sample does not have ground truth data, return an empty string
    if sample not in snakemake.config["ground_truth_samples"].get(year, []):
        return ""

    if "/cluster_assignments/" not in path_to_cluster_assignment:
        raise ValueError(f"Unexpected cluster assignment path: {path_to_cluster_assignment}")

    # Replace cluster_assignments -> comparisons
    comparison_path = path_to_cluster_assignment.replace("/cluster_assignments/", "/comparisons/")

    # Replace .csv -> /vs_ground_truth/ari.txt
    if not comparison_path.endswith(".csv"):
        raise ValueError(f"Cluster assignment path does not end in .csv: {path_to_cluster_assignment}")

    ari_path = comparison_path.replace(".csv", "/vs_ground_truth/ari.txt")

    if not os.path.exists(ari_path):
        raise FileNotFoundError(f"ARI file not found at {ari_path}")

    with open(ari_path, "r") as f:
        value = f.read().strip()

    return f"[ARI: {value}] "

def generate_screen(year, sample):
    views = []

    ground_truth_view = make_ground_truth_view(year, sample)
    if ground_truth_view:
        views.append(ground_truth_view)

    paper_bayespace_view = make_paper_bayesspace_view(year, sample)
    if paper_bayespace_view:
        views.append(paper_bayespace_view)

    views.extend(make_mclust_views(year, sample))
    views.extend(make_bayesspace_views(year, sample))

    screen_name = f"sample_{sample}"

    screen = {
        "name": screen_name,
        "sample": sample,
        "year": int(year),
        "outputDir": f"/zata/public_html/users/kresgeb/cluster-comparisons/results/vitessce_visualizations/{year}/{screen_name}",
        "views": views,
    }
    return screen


def generate_manifest():
    manifest = {"allScreens": []}
    for year, samples in snakemake.config["samples"].items():
        for sample in samples:
            screen = generate_screen(year, sample)
            manifest["allScreens"].append(screen)
    return manifest


def main():
    manifest = generate_manifest()
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest saved to {output_path}")

if __name__ == "__main__": 
    main()