rule run_sc3:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/{year}/SC3/k={k}/{sample}/seed={seed}.csv"
    log:
        "results/logs/cluster_assignments/{year}/SC3/k={k}/{sample}/seed={seed}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    threads:
        16
    script:
        "../scripts/run_sc3.R"