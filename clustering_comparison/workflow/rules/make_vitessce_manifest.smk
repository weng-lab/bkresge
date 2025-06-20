rule make_vitessce_manifest:
    input:
        comparisons_done = "results/comparisons/comparisons.done"
    output:
        manifest = "results/vitessce_visualizations/visualization_manifest.json"
    log:
        "results/logs/vitessce_visualizations/make_vitessce_manifest.log"
    conda:
        "../envs/simple_pandas.yaml"
    script:
        "../scripts/make_vitessce_manifest.py"