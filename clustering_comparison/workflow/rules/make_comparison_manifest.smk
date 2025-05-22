rule make_comparison_manifest:
    output:
        "resources/comparison/comparison_manifest.csv"
    conda:
        "../envs/simple_pandas.yaml"
    script:
        "../scripts/make_comparison_manifest.py"