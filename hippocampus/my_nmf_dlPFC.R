#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(RcppML)
    library(SingleCellExperiment)
    library(Matrix)
    library(sessioninfo)
})



# Open log file (append = FALSE to overwrite each run)
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/nmf.log"
sink(log_file, append = FALSE, split = TRUE) # split=TRUE keeps console + file
options(width = 120)

log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}
# Threads
threads <- 100
options(RcppML.threads = threads) # for RcppML 0.5.5
# setRcppMLthreads(threads) #for RcppML v.0.3.7 (latest CRAN) as of 9/9/25
log_msg(sprintf("Running with %d threads", threads))

# Paths
snrna_seq_data_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
path_for_x <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/nmf_x.rda"

log_msg("===== Starting NMF pipeline =====")
log_msg(paste("Loading data from:", snrna_seq_data_path))

obj_names <- load(snrna_seq_data_path, verbose = TRUE)
stopifnot(length(obj_names) == 1)

snrna <- get(obj_names)

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
log_msg("Running NMF with RcppML...")
start_time <- Sys.time()

x <- RcppML::nmf(
    assay(snrna, "logcounts"),
    k = 100,
    tol = 1e-06,
    maxit = 1000,
    verbose = TRUE,
    L1 = 0.1,
    seed = 1135,
    mask_zeros = FALSE,
    diag = TRUE,
    nonneg = TRUE
)

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "mins")
log_msg(sprintf("NMF completed in %.2f minutes", as.numeric(elapsed)))

# Save result
log_msg(paste("Saving NMF result to:", path_for_x))
save(x, file = path_for_x)
log_msg("Save complete.")

# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished NMF process =====")

# Close sink
sink()
