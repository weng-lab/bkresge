#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(SpatialExperiment)
})

# Access Snakemake input/output/config
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["output_csv"]]
log_file <- snakemake@log[[1]]
year <- snakemake@wildcards[["year"]]
sample <- snakemake@wildcards[["sample"]]
columns <- snakemake@params[["columns"]]


# Redirect stdout and stderr to the log file
sink(log_file)
cat("Starting column extraction\n")
cat("Year: ", year, " | Sample: ", sample, "\n")
cat("Columns: ", paste(columns, collapse = ", "), "\n")
cat("Input RData: ", input_rdata, "\n")

# Load the RData file (should contain 'spe')
load(input_rdata)
if (!exists("spe")) stop("No object named 'spe' found in RData file.")


# Check if columns exist
available <- colnames(colData(spe))
missing <- setdiff(columns, available)
if (length(missing) > 0) {
    stop("Missing column(s): ", paste(missing, collapse = ", "))
}

# Extract relevant data
df <- data.frame(
    barcode = colnames(spe),
    as.data.frame(colData(spe)[, columns, drop = FALSE])
)

# Write to CSV
write.csv(df, file = output_csv, row.names = FALSE)
cat("Column data written to: ", output_csv, "\n")
