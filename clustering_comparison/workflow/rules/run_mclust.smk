rule run_mclust:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.csv",
        # TODO implement the cluster visualization
        # output_png = "results/cluster_assignments/{year}/BayesSpace/k={k}/{sample}/nreps={nreps}_seed={seed}.png"
    log:
        "results/logs/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/run_mclust.R"