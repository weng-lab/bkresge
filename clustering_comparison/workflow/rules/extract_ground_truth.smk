rule extract_ground_truth:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        output_csv = "results/ground_truths/{year}/{sample}.csv"
    log:
        "results/logs/ground_truths/{year}/{sample}.log"
    params:
        columns = lambda wildcards: config["ground_truth_columns"][wildcards.year]
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/extract_column_data.R"