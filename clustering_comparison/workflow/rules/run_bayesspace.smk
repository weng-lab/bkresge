rule run_bayesspace:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}_nreps={nreps}_seed={seed}.csv"
    log:
        "results/logs/cluster_assignments/{year}/BayesSpace/k={k}/{sample}_nreps={nreps}_seed={seed}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/run_bayesspace.R"