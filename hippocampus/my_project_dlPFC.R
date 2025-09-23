#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(RcppML)
    library(Matrix)
    library(sessioninfo)
})
# Logging
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/projection_k_80.log"
sink(log_file, append = FALSE, split = TRUE)
options(width = 120)
log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

# Threads
threads <- 64
options(RcppML.threads = threads) # for RcppML 0.5.5
# setRcppMLthreads(threads) #for RcppML v.0.3.7 (latest CRAN) as of 9/9/25
log_msg(sprintf("Running with %d threads", threads))

seed <- 1029
log_msg(sprintf("Seed: %d", seed))

# Verbose option for RcppML v0.5.5
options(RcppML.verbose = TRUE)

# Paths
nmf_path <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/nmf_x_k_80.rda"
srt_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
proj_out <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/proj_srt_k_80.rda"

log_msg("===== Starting Projection =====")

# Load NMF object
load(nmf_path, verbose = TRUE)
stopifnot(exists("x"))
log_msg("Loaded NMF object")

# Load SRT object
obj_names <- load(srt_path, verbose = TRUE)
srt <- get(obj_names)
stopifnot(inherits(srt, "SpatialExperiment"))
log_msg("Loaded SRT object")

# The object uses Ensebml IDs as rownames
# I should not ever WANT this,
# so lets just replace them with the gene_names right now
# Replace Ensembl IDs with gene_name as rownames
log_msg("Replacing rownames(srt) with gene_name from rowData...")

gene_names <- as.character(rowData(srt)$gene_name)

# Handle NAs
na_idx <- which(is.na(gene_names) | gene_names == "")
if (length(na_idx) > 0) {
    gene_names[na_idx] <- rownames(srt)[na_idx] # fallback to Ensembl ID
    log_msg(sprintf("Replaced %d NA/blank gene_name entries with Ensembl IDs", length(na_idx)))
}

# Count duplicates before fixing
dup_count <- sum(duplicated(gene_names))
log_msg(sprintf("Number of duplicated gene_name entries: %d", dup_count))

# Make names unique (adds .1, .2, etc. suffixes)
gene_names <- make.unique(gene_names)

# Apply to srt
rownames(srt) <- gene_names
log_msg("Rowname replacement complete")

# Match genes
log_msg("Determining the common genes...")
common_genes <- intersect(rownames(x@w), rownames(srt))
log_msg(sprintf("Number of common genes: %d", length(common_genes)))

log_msg("Filtering w and Y by common genes...")
w <- x@w[common_genes, ]
log_msg(sprintf("Dimensions of w: %s", dim(w)))
Y <- assay(srt, "logcounts")[common_genes, ]
log_msg(sprintf("Dimensions of Y: %s", dim(Y)))

# Project: estimate H for SRT given fixed W
set.seed(seed)
log_msg("Projecting SRT into snRNA NMF basis...")
proj <- project(w, Y, L1 = 0) # no L1 penalty (tweak if desired)
log_msg("Projection Complete")

# Rescale so loadings per spot sum to 1
proj <- t(proj)
proj <- apply(proj, 2, function(z) z / sum(z))
log_msg("Rescaling complete")

# # Prepend 'my_' to projection column names since HPC uses this convention
# colnames(proj) <- paste0("my_", colnames(proj))

# Add to SRT metadata
colData(srt) <- cbind(colData(srt), proj)

# Save result
log_msg("Saving results...")
save(srt, proj, file = proj_out)
log_msg(paste("Projection saved to:", proj_out))

# Session info
print(sessionInfo())
print(session_info())
log_msg("===== Finished Projection =====")

sink()
