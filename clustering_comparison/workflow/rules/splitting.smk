rule split_rdata_samples:
    input:
        rdata="resources/paper_data/{year}.RData"
    output:
        directory("results/samples/{year}")
    log:
        "results/logs/split_samples/{year}.log"
    container:
        "docker://autumnusomega/bioinformatics:cluster-comparison"
    script:
        "../scripts/split_samples.R"