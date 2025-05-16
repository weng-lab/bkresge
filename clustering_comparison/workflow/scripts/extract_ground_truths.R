#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(SpatialExperiment)
})

# Access Snakemake input/output/config
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["ground_truth"]]
log_file <- snakemake@log[[1]]
year <- snakemake@wildcards[["year"]]
sample <- snakemake@wildcards[["sample"]]
config <- snakemake@config

# Redirect stdout and stderr to the log file
sink(log_file)
cat("Starting ground truth extraction\n")
cat("Year: ", year, " | Sample: ", sample, "\n")
cat("Input RData: ", input_rdata, "\n")

# Load the RData file (should contain 'spe_sub')
load(input_rdata)
if (!exists("spe_sub")) stop("No object named 'spe_sub' found in RData file.")

# Get ground truth column(s) from config
columns <- config$ground_truth_columns[[year]]
cat("Using annotation columns: ", paste(columns, collapse = ", "), "\n")

# Check if columns exist
available <- colnames(colData(spe_sub))
missing <- setdiff(columns, available)
if (length(missing) > 0) {
    stop("Missing ground truth column(s): ", paste(missing, collapse = ", "))
}

# Extract relevant data
df <- as.data.frame(colData(spe_sub)[, columns, drop = FALSE])
df$barcode <- colnames(spe_sub)

# Write to CSV
write.csv(df, file = output_csv, row.names = FALSE)
cat("Ground truth written to: ", output_csv, "\n")
sink()
