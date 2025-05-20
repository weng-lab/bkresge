suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(mclust)
})

# Access Snakemake variables
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["output_csv"]]
log_file <- snakemake@log[[1]]
model_name <- snakemake@wildcards[["model"]]
G <- as.integer(snakemake@wildcards[["k"]])
pc_count <- as.integer(snakemake@wildcards[["PCs"]])

# Logging
log_con <- file(log_file, open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")

message("== Mclust clustering started ==")
message(paste("Input:", input_rdata))
message(paste("Model:", model_name, "| G:", G))
message(paste("Output:", output_csv))

# Load RData
load(input_rdata)
if (!exists("spe")) stop("No object named 'spe' found in RData file.")

# Extract top PCs (number of PCs is specified in the Snakemake wildcards)
pcs <- reducedDims(spe)$PCA[, 1:pc_count]

# Run Mclust
message("Running Mclust...")
mclust_result <- Mclust(pcs, modelNames = model_name, G = G)

if (is.null(mclust_result)) stop("Mclust returned NULL")

# Save result
cluster_df <- data.frame(
    barcode = colnames(spe),
    cluster = mclust_result$classification
)
write.csv(cluster_df, file = output_csv, row.names = FALSE)

message("== Mclust clustering complete ==")
sink(type = "output")
sink(type = "message")
close(log_con)
