suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SC3)
    library(ggplot2)
    library(ggspavis)
#    library(BiocParallel)
})

input_rdata <- "/zata/zippy/kresgeb/clustering_comparison/resources/paper_data/2024/Br6522_mid.RData"


# Load RData
load(input_rdata)
if (!exists("spe")) stop("No object named 'spe' found in RData file.")

# This renaming is necessary for SC3 to work properly
# avoids: Error in sc3_prepare(object, gene_filter, pct_dropout_min, pct_dropout_max,  :
#  There is no `feature_symbol` column in the `rowData` slot of your dataset! Please write your gene/transcript names to `rowData(object)$feature_symbol`!
rowData(spe)$feature_symbol <- rowData(spe)$gene_name

# SC3 requires the logcounts assay to be a matrix, not a DelayedArray
assay(spe, "logcounts") <- as.matrix(assay(spe, "logcounts"))

# Register MulticoreParam with 5 workers
# register(MulticoreParam(workers = 5)) # fork-based parallelism (Unix/macOS only)

sc3_results <- sc3(spe, ks = 9, n_cores = 100, biology = FALSE, gene_filter = FALSE)
