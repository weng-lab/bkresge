rule download_paper_data:
    output:
        "resources/paper_data/{year}.RData"
    params:
        url = lambda wildcards: config["paper_data_urls"][wildcards.year]
    log:
        "results/logs/download_paper_data/{year}.log"
    shell:
        """
        mkdir -p $(dirname {output})
        curl -L -o {output} {params.url} &> {log}
        """
