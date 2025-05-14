#!/usr/bin/env Rscript


suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(sessioninfo)
})

# Logging
sink(snakemake@log[[1]])
# sink(snakemake@log[[1]], type = "message")

# Load the data
input_file <- snakemake@input[[1]]
output_dir <- snakemake@output[[1]]

Sys.time()
print(paste("Loading from:", input_file))
load(input_file) # loads an object named `spe`

if (!exists("spe")) {
    stop("The object 'spe' is not found in the loaded RData file.")
}

# Ensure output directory exists
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Split by sample_id
sample_ids <- unique(colData(spe)$sample_id)

Sys.time()
cat("Found samples:", paste(sample_ids, collapse = ", "), "\n")

for (sample in sample_ids) {
    print(Sys.time())
    cat("Processing sample:", sample, "\n")
    spe_sub <- spe[, colData(spe)$sample_id == sample]
    save(spe_sub, file = file.path(output_dir, paste0(sample, ".RData")))
}


## Reproducibility information
print("Reproducibility information:")
Sys.time()
proc.time()
options(width = 120)
session_info()
