all_samples = [(year, sample) for year in config["samples"].keys() for sample in config["samples"][year]]
ground_truth_samples = [(year, sample) for year in config["ground_truth_samples"].keys() for sample in config["ground_truth_samples"][year]]

def get_bayesspace_outputs(year, sample):
    """
    Generate the output file paths for BayesSpace clustering results.
    """
    k_values = config["bayesspace_parameters"]["k"]
    nreps = config["bayesspace_parameters"]["nreps"]
    seeds = config["bayesspace_parameters"]["seed"]
    
    return expand(
        "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.csv",
        year=year,
        sample=sample,
        k=k_values,
        nreps=nreps,
        seed=seeds
    )

def get_mclust_outputs(year, sample):
    """
    Generate the output file paths for Mclust clustering results.
    """
    k_values = config["mclust_parameters"]["k"]
    models = config["mclust_parameters"]["model"]
    PCs = config["mclust_parameters"]["PCs"]

    return expand(
        "results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.csv",
        year=year,
        sample=sample,
        model=models,
        k=k_values,
        PCs=PCs
    )

def get_sc3_outputs(year, sample):
    """
    Generate the output file paths for SC3 clustering results.
    """
    k_values = config["sc3_parameters"]["k"]
    seeds = config["sc3_parameters"]["seed"]
    
    return expand(
        "results/cluster_assignments/{year}/SC3/k={k}/{sample}/seed={seed}.csv",
        year=year,
        sample=sample,
        k=k_values,
        seed=seeds
    )