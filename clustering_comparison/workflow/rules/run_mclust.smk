rule run_mclust:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.csv",
        output_png = "results/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.png"
    log:
        "results/logs/cluster_assignments/{year}/mclust/{model}/k={k}/{sample}/PCs={PCs}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    # Can sometimes run into image header errors during plot creation that are transient and can be retried    
    retries:
        3
    script:
        "../scripts/run_mclust.R"