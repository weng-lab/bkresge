rule run_bayesspace:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.csv",
        output_png = "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.png"
    log:
        "results/logs/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/run_bayesspace.R"