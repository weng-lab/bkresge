suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SC3)
    library(ggplot2)
    library(ggspavis)
    library(BiocParallel)
})

# Access Snakemake input/output/log
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["output_csv"]]
log_file <- snakemake@log[[1]]

# Params
k <- as.integer(snakemake@wildcards[["k"]])
seed <- as.integer(snakemake@wildcards[["seed"]])
n_cores <- if (exists("snakemake")) snakemake@threads else 1

# Redirect stdout and stderr to log
log_con <- file(log_file, open = "wb")
sink(log_con, type = "output")
sink(log_con, type = "message")
on.exit(
    {
        sink(type = "message")
        sink(type = "output")
        close(log_con)
    },
    add = TRUE
)

message("== SC3 clustering started ==")
message(Sys.time())

# Load RData
load(input_rdata)
if (!exists("spe")) stop("No object named 'spe' found in RData file.")

# This renaming is necessary for SC3 to work properly
# avoids: Error in sc3_prepare(object, gene_filter, pct_dropout_min, pct_dropout_max,  :
#  There is no `feature_symbol` column in the `rowData` slot of your dataset! Please write your gene/transcript names to `rowData(object)$feature_symbol`!
rowData(spe)$feature_symbol <- rowData(spe)$gene_name

# SC3 requires the logcounts assay to be a matrix, not a DelayedArray
assay(spe, "logcounts") <- as.matrix(assay(spe, "logcounts"))

sc3_results <- sc3(spe, ks = k, n_cores = n_cores, biology = FALSE, gene_filter = FALSE, rand_seed = seed)


message("SC3 clustering completed.")
message(Sys.time())

# Extract barcodes (column names) and SC3 cluster assignments
cluster_col_name <- paste0("sc3_", k, "_clusters")
cluster_df <- data.frame(
    barcode = colnames(sc3_results),
    cluster = colData(sc3_results)[[cluster_col_name]]
)

# Write to CSV
write.table(
    cluster_df,
    file = output_csv,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = c(1) # Only quote the first column (barcode)
)

message("Cluster assignments written to: ", output_csv)
