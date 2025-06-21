rule run_comparisons:
    input:
        # Extract all available ground truth data
        [f"results/ground_truths/{year}/{sample}.csv" for year, sample in ground_truth_samples],

        # Extract all available clustering data from 2024 BayesSpace
        [f"results/cluster_assignments/2024/paper_bayesspace/{sample}.csv" for year, sample in all_samples if year == "2024"],

        # Run BayesSpace clustering for all samples
        [get_bayesspace_outputs(year, sample) for year, sample in all_samples],

        # Run Mclust clustering for all samples
        [get_mclust_outputs(year, sample) for year, sample in all_samples],

        # Run SC3 clustering for all samples
        [get_sc3_outputs(year, sample) for year, sample in all_samples],
        
        manifest="results/comparisons/comparison_manifest.csv"
    output:
        done = "results/comparisons/comparisons.done"
    log:
        "results/logs/comparisons/run_comparisons.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/run_comparisons.R"