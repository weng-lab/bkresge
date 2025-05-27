rule make_comparison_manifest:
    output:
        "results/comparisons/comparison_manifest.csv"
    conda:
        "../envs/simple_pandas.yaml"
    script:
        "../scripts/make_comparison_manifest.py"