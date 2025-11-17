#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(here)
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(RcppML)
    library(sessioninfo)
    library(duckplyr)
    library(readr)
})

# --------------------------------------------------------------
# Set project root and import utilities
# --------------------------------------------------------------
my_relative_path <- "scripts/batched_projection.R"
here::i_am(my_relative_path)
source(here("scripts", "utils.R"))

# --------------------------------------------------------------
# Global setup
# --------------------------------------------------------------
threads <- 64
options(RcppML.threads = threads)
options(RcppML.verbose = TRUE)

setup_log(prefix = "batched_projection")
snapshot_script(here(my_relative_path))
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg(sprintf("Running batched projection with %d threads", threads))

# --------------------------------------------------------------
# Paths and parameters
# --------------------------------------------------------------
nmf_summary_tsv <- here("data", "batched_nmf", "summary.tsv")
srt_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
proj_outdir <- here("data", "batched_projection")
summary_tsv <- file.path(proj_outdir, "summary.tsv")

if (!dir.exists(proj_outdir)) {
    dir.create(proj_outdir, recursive = TRUE)
}

# --------------------------------------------------------------
# Load SRT once
# --------------------------------------------------------------
log_msg("===== Loading and processing SRT object =====")
log_msg(paste("Loading SRT object from:", srt_path))
load_and_rename(srt_path, new_names = "srt", verbose = TRUE)
stopifnot(inherits(srt, "SpatialExperiment"))

# The object uses Ensebml IDs as rownames
# I should not ever WANT this,
# so lets just replace them with the gene_names right now
# Replace Ensembl IDs with gene_name as rownames
log_msg("Replacing rownames(srt) with gene_name from rowData...")
gene_names <- as.character(rowData(srt)$gene_name)
na_idx <- which(is.na(gene_names) | gene_names == "")
if (length(na_idx) > 0) {
    gene_names[na_idx] <- rownames(srt)[na_idx] # fallback to Ensembl IDs
    log_msg(sprintf("Replaced %d NA/blank gene_name entries with Ensembl IDs", length(na_idx)))
}
# Make names unique (adds .1, .2, etc. suffixes)
gene_names <- make.unique(gene_names)

# apply new rownames
rownames(srt) <- gene_names
log_msg("SRT preprocessing complete.")

# --------------------------------------------------------------
# Load NMF batch summary
# --------------------------------------------------------------
if (!file.exists(nmf_summary_tsv)) {
    stop("NMF summary TSV not found at: ", nmf_summary_tsv)
}
nmf_summary <- read_tsv(nmf_summary_tsv, show_col_types = FALSE)
log_msg(sprintf("Loaded %d NMF runs from summary.", nrow(nmf_summary)))

# --------------------------------------------------------------
# Setup for output summary TSV
# --------------------------------------------------------------

if (file.exists(summary_tsv)) {
    summary_df <- read_tsv(summary_tsv, show_col_types = FALSE)
    log_msg(sprintf("Loaded existing summary TSV with %d rows", nrow(summary_df)))
} else {
    summary_df <- tibble(
        timestamp = character(),
        k = integer(),
        seed = integer(),
        tol = numeric(),
        L1 = numeric(),
        elapsed_min = numeric(),
        nmf_input = character(),
        projection_output = character(),
        n_common_genes = integer()
    )
    write_tsv(summary_df, summary_tsv)
    log_msg(sprintf("Initialized empty summary TSV at: %s", summary_tsv))
}


# --------------------------------------------------------------
# Run projections for each NMF result
# --------------------------------------------------------------

# --- Config ---
skip_completed <- TRUE # set FALSE to force re-run even if already completed

for (i in seq_len(nrow(nmf_summary))) {
    nmf_path <- nmf_summary$output_path[i]
    k <- nmf_summary$k[i]
    seed <- nmf_summary$seed[i]
    tol <- nmf_summary$tol[i]
    L1 <- nmf_summary$L1[i]

    run_prefix <- sprintf("proj_k%d_seed%d_tol%.0e_L1%.1f", k, seed, tol, L1)
    proj_out_path <- file.path(proj_outdir, paste0(run_prefix, ".rds"))

    log_msg("--------------------------------------------")
    log_msg(sprintf(
        "Starting projection %d / %d: k=%d, seed=%d, tol=%.0e, L1=%.1f",
        i, nrow(nmf_summary), k, seed, tol, L1
    ))
    log_msg("--------------------------------------------")

    # --- Check for completed run ---
    already_done <- FALSE
    if (skip_completed && nrow(summary_df) > 0) {
        match_row <- summary_df |>
            filter(k == !!k, seed == !!seed, tol == !!tol, L1 == !!L1)
        if (nrow(match_row) > 0) {
            already_done <- TRUE
            ts <- lubridate::ymd_hms(match_row$timestamp, tz = "UTC")
            ago <- lubridate::as.period(Sys.time() - ts, unit = "minute")
            log_msg(sprintf(
                "Skipping run (already completed)\n\tcompleted %s ago at %s UTC\n\tlocation: %s",
                lubridate::time_length(ago, "hour") %>%
                    {
                        \(x) sprintf("%.1f hours", x)
                    }(),
                format(ts, "%Y-%m-%dT%H:%M:%SZ"),
                match_row$projection_output
            ))
        }
    }
    if (already_done) next

    start_time <- Sys.time()

    # Load NMF model
    nmf_x <- readRDS(nmf_path)
    w <- nmf_x@w

    # Match genes
    common_genes <- intersect(rownames(w), rownames(srt))
    log_msg(sprintf("Common genes: %d", length(common_genes)))

    w <- w[common_genes, , drop = FALSE]
    Y <- assay(srt, "logcounts")[common_genes, , drop = FALSE]

    # Projection
    log_msg("Projecting SRT into snRNA NMF basis...")
    set.seed(seed)
    proj <- project(w, Y, L1 = 0)
    proj <- t(proj)

    # Rescale so loadings per spot sum to 1
    proj <- apply(proj, 2, function(z) z / sum(z))
    log_msg("Projection complete and rescaled.")

    # Save result
    saveRDS(proj, proj_out_path)
    log_msg(paste("Saved projection result to:", proj_out_path))

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

    # --- Construct row ---
    new_row <- tibble(
        timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
        k = k, seed = seed, tol = tol, L1 = L1,
        elapsed_min = elapsed,
        nmf_input = nmf_path,
        projection_output = proj_out_path,
        n_common_genes = length(common_genes)
    )
    # --- Append to TSV file ---
    write_tsv(new_row, summary_tsv, append = TRUE)
    log_msg("Appended run metadata to summary.tsv")
}

log_msg("===== Finished batched projection =====")
close_log()
