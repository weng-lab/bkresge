rule extract_2024_bayesspace:
    input:
        rdata = "resources/paper_data/2024/{sample}.RData"
    output:
        output_csv = "results/cluster_assignments/2024/paper_bayesspace/{sample}.csv"
    log:
        "results/logs/cluster_assignments/2024/paper_bayesspace/{sample}.log"
    params:
        columns = config["2024_clustering_columns"]
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/extract_column_data.R"