rule extract_ground_truth:
    input:
        rdata = "resources/paper_data/{year}/{sample}.RData"
    output:
        ground_truth = "results/ground_truths/{year}/{sample}.csv"
    log:
        "results/logs/ground_truths/{year}/{sample}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/extract_ground_truths.R"