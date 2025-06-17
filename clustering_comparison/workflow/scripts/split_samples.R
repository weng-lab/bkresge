#!/usr/bin/env Rscript


suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(sessioninfo)
})


# Change 2024->2021 to run on 2021 data
input_file <- "/zata/public_html/users/kresgeb/cluster-comparisons/paper_data/2021.RData"
output_dir <- "/zata/public_html/users/kresgeb/cluster-comparisons/paper_data/2021/"

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

spe_full <- spe

for (sample in sample_ids) {
    print(Sys.time())
    cat("Processing sample:", sample, "\n")
    spe <- spe_full[, colData(spe_full)$sample_id == sample]
    save(spe, file = file.path(output_dir, paste0(sample, ".RData")))
}


## Reproducibility information
print("Reproducibility information:")
Sys.time()
proc.time()
options(width = 120)
session_info()
