#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(here)
    library(RcppML)
    library(SingleCellExperiment)
    library(sessioninfo)
    library(duckplyr)
    library(readr)
    library(lubridate)
})

# --------------------------------------------------------------
# Set project root and import utilities
# --------------------------------------------------------------
my_relative_path <- "scripts/batched_nmf.R"
here::i_am(my_relative_path)

source(here("scripts", "utils.R"))

# --------------------------------------------------------------
# Global setup
# --------------------------------------------------------------
threads <- 64
options(RcppML.threads = threads)
options(RcppML.verbose = TRUE)

setup_log(prefix = "batched_nmf")
snapshot_script(here(my_relative_path))
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg(sprintf("Running batched NMF with %d threads", threads))

snrna_seq_data_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
batched_outdir <- here("data", "batched_nmf")

if (!dir.exists(batched_outdir)) {
    dir.create(batched_outdir, recursive = TRUE)
}

# --------------------------------------------------------------
# Parameter grid
# --------------------------------------------------------------
k_values <- c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
# k_values <- c(80)
seeds <- c(42, 1029, 31415, 120301, 2025)
# seeds <- c(1029)
tolerances <- c(1e-05, 1e-06, 1e-07)
# tolerances <- c(1e-06)
L1s <- c(0, 0.1, 0.2, 0.5, 0.7)
# L1s <- c(0.1)

param_grid <- expand.grid(
    k = k_values,
    seed = seeds,
    tol = tolerances,
    L1 = L1s
)

# --------------------------------------------------------------
# Load data once
# --------------------------------------------------------------
log_msg("===== Loading snRNA-seq data once for all runs =====")
log_msg(paste("Loading data from:", snrna_seq_data_path))
load_and_rename(snrna_seq_data_path, new_names = "snrna")

if (!inherits(snrna, "SingleCellExperiment")) {
    stop("Loaded object is not a SingleCellExperiment, cannot continue.")
}

log_msg("Data successfully loaded.")
log_msg(sprintf(
    "Dimensions of logcounts: %d genes x %d cells",
    nrow(assay(snrna, "logcounts")),
    ncol(assay(snrna, "logcounts"))
))
log_msg(sprintf(
    "Memory footprint: %.2f MB",
    object.size(assay(snrna, "logcounts")) / (1024^2)
))


#--------------------------------------------------------------
# Setup for summary TSV
#--------------------------------------------------------------
summary_tsv <- file.path(batched_outdir, "summary.tsv")

# Load existing summary or create empty one
if (file.exists(summary_tsv)) {
    summary_df <- read_tsv(summary_tsv, show_col_types = FALSE)
    log_msg(paste("Loaded existing summary TSV with", nrow(summary_df), "rows"))
} else {
    summary_df <- tibble(
        timestamp = character(),
        k = integer(),
        seed = integer(),
        tol = numeric(),
        L1 = numeric(),
        elapsed_min = numeric(),
        output_path = character()
    )
    write_tsv(summary_df, summary_tsv)
    log_msg(paste("Initialized empty summary TSV at:", summary_tsv))
}

# --------------------------------------------------------------
# Run over grid
# --------------------------------------------------------------

# --- Config ---
skip_completed <- TRUE # set FALSE to force re-run even if already completed

for (i in seq_len(nrow(param_grid))) {
    k <- param_grid$k[i]
    seed <- param_grid$seed[i]
    tol <- param_grid$tol[i]
    L1 <- param_grid$L1[i]

    run_prefix <- sprintf("nmf_k%d_seed%d_tol%.0e_L1%.1f", k, seed, tol, L1)

    log_msg("--------------------------------------------")
    log_msg(sprintf(
        "Starting run %d / %d: k=%d, seed=%d, tol=%.0e, L1=%.1f",
        i,
        nrow(param_grid),
        k,
        seed,
        tol,
        L1
    ))
    log_msg("--------------------------------------------")

    # Check for completed run
    already_done <- FALSE
    if (skip_completed && nrow(summary_df) > 0) {
        match_row <- summary_df %>%
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
                match_row$output_path
            ))
        }
    }
    if (already_done) {
        next
    }

    # --- Run NMF ---
    start_time <- Sys.time()

    x <- RcppML::nmf(
        assay(snrna, "logcounts"),
        k = k,
        tol = tol,
        maxit = 1000,
        verbose = TRUE,
        L1 = L1,
        seed = seed,
        mask_zeros = FALSE,
        diag = TRUE,
        nonneg = TRUE
    )

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))
    log_msg(sprintf("Run completed in %.2f minutes", elapsed))

    out_path <- file.path(batched_outdir, paste0(run_prefix, ".rds"))
    saveRDS(x, out_path)
    log_msg(paste("Saved NMF result to:", out_path))

    # Construct new row for summary TSV
    new_row <- tibble(
        timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
        k = k,
        seed = seed,
        tol = tol,
        L1 = L1,
        elapsed_min = elapsed,
        output_path = out_path
    )

    # Save to summary TSV
    write_tsv(new_row, summary_tsv, append = TRUE)
    log_msg("Appended run metadata to summary.tsv")
}

log_msg("===== Batch NMF Complete =====")
close_log()
