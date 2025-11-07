#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(here)
    library(SpatialExperiment)
    library(RcppML)
    library(SingleCellExperiment)
    library(Matrix)
    library(sessioninfo)
})

# Set the project root for 'here' package
here::i_am("scripts/my_nmf_dlPFC.R")

# Load shared utility functions
source(here("scripts", "utils.R"))

# Logging
log_file <- setup_log(prefix = "nmf")

# Threads
threads <- 64
options(RcppML.threads = threads) # for RcppML 0.5.5
# setRcppMLthreads(threads) #for RcppML v.0.3.7 (latest CRAN) as of 9/9/25
log_msg(sprintf("Running with %d threads", threads))

# Verbose option for RcppML v0.5.5
options(RcppML.verbose = TRUE)

# Paths
snrna_seq_data_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
path_for_x <- here("data", "nmf_x.rds")
path_for_appended_snrna <- here("data", "snrna_with_nmf.rds")

# k <- 100
k <- 80
seed <- 1029

log_msg("===== Starting NMF pipeline =====")
log_msg(paste("Loading data from:", snrna_seq_data_path))
load_and_rename(snrna_seq_data_path, new_names = "snrna")

if (!inherits(snrna, "SingleCellExperiment")) {
    stop("Loaded object is not a SingleCellExperiment, cannot continue.")
}

log_msg("Data successfully loaded.")
log_msg(sprintf("snrna object: %s", class(snrna)))
log_msg(sprintf(
    "Dimensions of logcounts: %d genes x %d cells",
    nrow(assay(snrna, "logcounts")), ncol(assay(snrna, "logcounts"))
))

# Estimate memory footprint
size_bytes <- object.size(assay(snrna, "logcounts"))
log_msg(sprintf("Size of logcounts assay: %.2f MB", size_bytes / (1024^2)))

# Run NMF
log_msg(sprintf("Running NMF with k=%d, seed=%d...", k, seed))
start_time <- Sys.time()

x <- RcppML::nmf(
    assay(snrna, "logcounts"),
    k = k,
    tol = 1e-06,
    maxit = 1000,
    verbose = TRUE,
    L1 = 0.1,
    seed = seed,
    mask_zeros = FALSE,
    diag = TRUE,
    nonneg = TRUE
)

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "mins")
log_msg(sprintf("NMF completed in %.2f minutes", as.numeric(elapsed)))

# Save result
log_msg(paste("Saving NMF result to:", normalizePath(path_for_x, mustWork = FALSE)))
saveRDS(x, file = path_for_x)
log_msg("Save complete.")

# Append NMF results to SingleCellExperiment object and save
nmf_matrix <- as.matrix(t(x$H))
colnames(nmf_matrix) <- paste0("nmf", seq_len(ncol(nmf_matrix)))

log_msg("Appending NMF results to SingleCellExperiment object...")
colData(snrna) <- cbind(colData(snrna), nmf_matrix)

log_msg(paste("Saving updated SingleCellExperiment to:", normalizePath(path_for_appended_snrna, mustWork = FALSE)))
saveRDS(snrna, file = path_for_appended_snrna)
log_msg("Save complete.")


# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished NMF process =====")

# Close log
close_log()
