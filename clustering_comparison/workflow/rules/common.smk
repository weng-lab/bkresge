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
        "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}_nreps={nreps}_seed={seed}.csv",
        year=year,
        sample=sample,
        k=k_values,
        nreps=nreps,
        seed=seeds
    )