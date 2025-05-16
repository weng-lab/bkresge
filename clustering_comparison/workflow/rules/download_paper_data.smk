rule download_paper_data:
    output:
        "resources/paper_data/{year}/{sample}.RData"
    params:
        url = lambda wildcards: f"https://users.wenglab.org/kresgeb/cluster-comparisons/paper_data/{wildcards.year}/{wildcards.sample}.RData"
    log:
        "results/logs/download_paper_data/{year}/{sample}.log"
    shell:
        """
        mkdir -p $(dirname {output})
        curl -fL {params.url} -o {output} > {log} 2>&1
        """