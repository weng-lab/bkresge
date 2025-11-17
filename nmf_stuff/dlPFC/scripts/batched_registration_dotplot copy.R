#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(here)
    library(SingleCellExperiment)
    library(SpatialExperiment)
    library(RcppML)
    library(dplyr)
    library(ggplot2)
    library(scater)
    library(sessioninfo)
    library(mclust)
    library(viridis)
    library(tidyr)
})

# ------------------------------
# Setup
# ------------------------------
my_relative_path <- "scripts/batched_registration_dotplot.R"
here::i_am(my_relative_path)
source(here("scripts", "utils.R"))

setup_log(prefix = "batched_registration_dotplot")
snapshot_script(here(my_relative_path))
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Starting batched registration dotplot =====")

# ------------------------------
# Paths - adjust if needed
# ------------------------------
# baseline SCE and SPE that do NOT have nmf/projection columns appended permanently
dlpfc_spe_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
dlpfc_sce_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"

nmf_summary_csv <- here("data", "batched_nmf", "summary.csv")
proj_summary_csv <- here("data", "batched_projection", "summary.csv")

output_root <- here("output", "registration_dotplot")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

master_summary_csv <- file.path(output_root, "summary.csv")
if (!file.exists(master_summary_csv)) {
    write.table(
        data.frame(
            timestamp = character(), k = integer(), seed = integer(), tol = numeric(),
            nmf_input = character(), projection_input = character(),
            n_common_genes = integer(),
            n_specific = integer(), n_general = integer(),
            elapsed_min = numeric(), output_dir = character(),
            stringsAsFactors = FALSE
        ),
        file = master_summary_csv, sep = ",", row.names = FALSE, col.names = TRUE
    )
    log_msg(paste("Initialized master summary CSV at:", master_summary_csv))
}

# Column names in sce/spe that contain cell-type and domain annotations
cell_type_col_name <- "cellType_hc"
domain_col_name <- "BayesSpace_harmony_09"

get_col <- function(obj, colname) {
    if (!colname %in% colnames(colData(obj))) {
        stop(sprintf("Column '%s' not found in colData", colname))
    }
    return(as.factor(colData(obj)[[colname]]))
}

# ------------------------------
# Load baseline sce & spe (once)
# ------------------------------
log_msg("Loading base SCE and SPE (once)")
log_msg(paste("Loading SPE from:", dlpfc_spe_path))
load_and_rename(dlpfc_spe_path, "spe", verbose = TRUE) # expects object named 'spe' after load
log_msg("Loaded SPE")

log_msg(paste("Loading SCE from:", dlpfc_sce_path))
load_and_rename(dlpfc_sce_path, "sce", verbose = TRUE)
log_msg("Loaded SCE")

stopifnot(inherits(sce, "SingleCellExperiment"))
stopifnot(inherits(spe, "SpatialExperiment"))

# Ensure SPE rownames are gene symbols (as in projection code)
log_msg("Ensuring SPE rownames are gene names")
gene_names <- as.character(rowData(spe)$gene_name)
na_idx <- which(is.na(gene_names) | gene_names == "")
if (length(na_idx) > 0) {
    gene_names[na_idx] <- rownames(spe)[na_idx]
    log_msg(sprintf("Replaced %d NA gene_name entries with Ensembl IDs", length(na_idx)))
}
rownames(spe) <- make.unique(gene_names)

# ------------------------------
# Read NMF and projection summaries
# ------------------------------
if (!file.exists(nmf_summary_csv)) stop("NMF summary CSV not found at: ", nmf_summary_csv)
if (!file.exists(proj_summary_csv)) stop("Projection summary CSV not found at: ", proj_summary_csv)

nmf_summary <- read.csv(nmf_summary_csv, stringsAsFactors = FALSE)
proj_summary <- read.csv(proj_summary_csv, stringsAsFactors = FALSE)

log_msg(sprintf("Found %d NMF runs and %d projections", nrow(nmf_summary), nrow(proj_summary)))

# Match projections to nmf runs using nmf_input path column from projection summary
# projection summary has nmf_input column named 'nmf_input' per batched_projection script
if (!"nmf_input" %in% colnames(proj_summary)) {
    # try alternative column names; otherwise assume projection records are in same order
    stop("Projection summary does not contain 'nmf_input' column to link back to NMF runs.")
}

# Create a lookup from nmf input path -> projection output path
proj_lookup <- setNames(proj_summary$projection_output, proj_summary$nmf_input)

# ------------------------------
# Iterate over NMF runs
# ------------------------------
for (i in seq_len(nrow(nmf_summary))) {
    nmf_row <- nmf_summary[i, ]
    nmf_path <- nmf_row$output_path
    k <- nmf_row$k
    seed <- nmf_row$seed
    tol <- nmf_row$tol

    run_prefix <- sprintf("k%d_seed%d_tol%.0e", k, seed, tol)
    run_outdir <- file.path(output_root, paste0("run_", run_prefix))
    dir.create(run_outdir, recursive = TRUE, showWarnings = FALSE)
    plots_dir <- file.path(run_outdir, "plots")
    dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

    log_msg("--------------------------------------------")
    log_msg(sprintf("Starting run %d/%d: %s", i, nrow(nmf_summary), run_prefix))
    log_msg(paste("nmf_path:", nmf_path))

    # find projection for this nmf
    proj_path <- proj_lookup[[nmf_path]]
    if (is.null(proj_path) || !file.exists(proj_path)) {
        log_msg(sprintf("WARNING: No projection found for nmf_path: %s. Skipping this run.", nmf_path))
        next
    }
    log_msg(paste("projection path:", proj_path))

    # ------------------------------
    # Load NMF object (RcppML nmf)
    # ------------------------------
    nmf_x <- tryCatch(
        {
            readRDS(nmf_path)
        },
        error = function(e) {
            log_msg(sprintf("Failed to read NMF object at %s: %s", nmf_path, as.character(e)))
            return(NULL)
        }
    )

    if (is.null(nmf_x)) next

    # construct nmf matrix of shape (cells x patterns) consistent with earlier code
    # Earlier code used: nmf_matrix <- as.matrix(t(x$h)); colnames -> nmf1.. ; appended to colData(snrna)
    nmf_mat <- tryCatch(
        {
            mm <- as.matrix(t(nmf_x$h))
            colnames(mm) <- paste0("nmf", seq_len(ncol(mm)))
            mm
        },
        error = function(e) {
            log_msg(sprintf("Failed to build nmf matrix from NMF object: %s", as.character(e)))
            NULL
        }
    )

    if (is.null(nmf_mat)) next
    log_msg(sprintf("NMF matrix: %d rows, %d cols", nrow(nmf_mat), ncol(nmf_mat)))
    log_msg(sprintf("Duplicate NMF rownames: %d", anyDuplicated(rownames(nmf_mat))))

    # ------------------------------
    # Load projection (spots x patterns)
    # ------------------------------
    proj_mat <- tryCatch(
        {
            readRDS(proj_path)
        },
        error = function(e) {
            log_msg(sprintf("Failed to read projection at %s: %s", proj_path, as.character(e)))
            NULL
        }
    )

    if (is.null(proj_mat)) next
    log_msg(sprintf("Projection matrix: %d rows, %d cols", nrow(proj_mat), ncol(proj_mat)))
    log_msg(sprintf("Duplicate projection rownames: %d", anyDuplicated(rownames(proj_mat))))
    # # Ensure projection rows correspond to SPE columns (spots) where possible
    # # If rownames missing but dims match, assign assuming same order
    # if (is.null(rownames(proj_mat))) {
    #     if (nrow(proj_mat) == ncol(spe)) {
    #         rownames(proj_mat) <- colnames(spe)
    #     }
    # }

    start_time <- Sys.time()
    # ------------------------------
    # Prepare local copies of sce and spe for this run
    # ------------------------------
    sce_run <- sce
    spe_run <- spe
    log_msg("Attaching NMF matrix to SCE colData...")
    log_msg(sprintf(
        "SCE colData rows: %d, unique rownames: %d",
        nrow(colData(sce_run)), length(unique(rownames(colData(sce_run))))
    ))
    # Attach NMF columns to sce_run$colData (temporary)
    # If dimensions match by rows, cbind; otherwise try to match by rownames
    if (nrow(nmf_mat) == nrow(colData(sce_run))) {
        new_nmf_df <- as.data.frame(nmf_mat, stringsAsFactors = FALSE)
        colData(sce_run) <- cbind(colData(sce_run), new_nmf_df)
        log_msg("NMF matrix attached by row order")
    } else if (!is.null(rownames(nmf_mat)) && all(rownames(nmf_mat) %in% rownames(colData(sce_run)))) {
        new_nmf_df <- as.data.frame(nmf_mat, stringsAsFactors = FALSE)
        # reorder to match sce_run
        new_nmf_df <- new_nmf_df[rownames(colData(sce_run)), , drop = FALSE]
        colData(sce_run) <- cbind(colData(sce_run), new_nmf_df)
    } else {
        log_msg("WARNING: NMF matrix rows do not match SCE colData rows; skipping this run.")
        next
    }

    # Attach projection to spe_run$colData temporarily
    # projection matrix assumed rows = spots (colData rows)
    log_msg("Attaching projection matrix to SPE colData...")
    log_msg(sprintf(
        "SPE colData rows: %d, unique rownames: %d",
        nrow(colData(spe_run)), length(unique(rownames(colData(spe_run))))
    ))
    if (nrow(proj_mat) == nrow(colData(spe_run))) {
        proj_df <- as.data.frame(proj_mat, stringsAsFactors = FALSE)
        colData(spe_run) <- cbind(colData(spe_run), proj_df)
        log_msg("Projection matrix attached by row order")
    } else if (!is.null(rownames(proj_mat)) && all(rownames(proj_mat) %in% rownames(colData(spe_run)))) {
        proj_df <- as.data.frame(proj_mat, stringsAsFactors = FALSE)
        proj_df <- proj_df[rownames(colData(spe_run)), , drop = FALSE]
        colData(spe_run) <- cbind(colData(spe_run), proj_df)
    } else {
        log_msg("WARNING: Projection rows do not match SPE colData rows; skipping this run.")
        next
    }

    # Now identify nmf columns in the (now augmented) sce_run colData
    nmf_cols <- grep("^nmf[0-9]+$", colnames(colData(sce_run)), value = TRUE)
    if (length(nmf_cols) == 0) {
        log_msg("No nmf columns found after attaching matrix; skipping.")
        next
    }
    log_msg(sprintf("Found %d nmf patterns for plotting", length(nmf_cols)))

    # ------------------------------
    # ECDF of nonzero counts (nuclei + spots)
    # ------------------------------
    log_msg("Creating ECDF plot for nonzero NMF pattern weights...")
    # Count nonzeros
    nonzero_nuclei <- colSums(as.matrix(colData(sce_run)[, nmf_cols]) > 0)
    nonzero_spots <- colSums(as.matrix(colData(spe_run)[, nmf_cols]) > 0)

    # Build long dataframe with source column
    nonzero_df <- bind_rows(
        data.frame(nonzero_count = nonzero_nuclei, source = "Nuclei"),
        data.frame(nonzero_count = nonzero_spots, source = "Spots")
    )

    # Single ECDF plot with facet_wrap
    p_ecdf <- ggplot(nonzero_df, aes(x = log10(nonzero_count))) +
        stat_ecdf(geom = "step") +
        coord_cartesian(xlim = c(0, 5)) +
        facet_wrap(~source, ncol = 1, scales = "fixed") +
        labs(
            x = "log10(# with nonzero weight) per NMF pattern",
            y = "ECDF",
            title = "ECDF of nonzero NMF pattern weights"
        ) +
        theme_minimal()

    ggsave(
        filename = file.path(plots_dir, "ecdf_nonzero_nuclei_spots_nmf_patterns_faceted.pdf"),
        plot = p_ecdf, width = 6, height = 8, dpi = 300
    )

    # ------------------------------
    # Dotplot: nuclei proportion (presence) and scaled avg
    # ------------------------------
    log_msg("Preparing data for dotplot of NMF pattern presence by cell type...")

    nmf_weight_matrix <- as.matrix(colData(sce_run)[, nmf_cols])
    # binary presence
    nuclei_nmf_nonzero_binary <- nmf_weight_matrix > 0
    nuclei_nmf_presence <- data.frame(
        cell_type = get_col(sce_run, cell_type_col_name),
        nuclei_nmf_nonzero_binary,
        check.names = FALSE
    )

    nuclei_nmf_prop_summary <- nuclei_nmf_presence %>%
        group_by(cell_type) %>%
        add_tally(name = "total") %>%
        group_by(cell_type, total) %>%
        summarize(across(all_of(nmf_cols), sum), .groups = "drop") %>%
        pivot_longer(cols = all_of(nmf_cols), names_to = "nmf", values_to = "n") %>%
        mutate(prop = n / total)

    # scaled avg
    nmf_scaled_matrix <- apply(nmf_weight_matrix, 2, scale)
    nuclei_nmf_scaled <- data.frame(
        cell_type = get_col(sce_run, cell_type_col_name),
        nmf_scaled_matrix,
        check.names = FALSE
    )

    nuclei_nmf_scaled_summary <- nuclei_nmf_scaled %>%
        group_by(cell_type) %>%
        summarize(across(all_of(nmf_cols), mean), .groups = "drop") %>%
        pivot_longer(cols = all_of(nmf_cols), names_to = "nmf", values_to = "scaled_avg")

    # ------------------------------
    # Abundance, entropy and GMM split
    # ------------------------------
    log_msg("Computing specificity / entropy and GMM split...")

    pseudo_count <- 1e-9
    specificity_df <- nuclei_nmf_prop_summary %>%
        group_by(nmf) %>%
        summarize(
            mean_prop = mean(prop),
            sd_prop = sd(prop),
            prop_cv = sd(prop) / mean(prop),
            entropy = -sum(((prop + pseudo_count) / sum(prop + pseudo_count)) * log((prop + pseudo_count) / sum(prop + pseudo_count)))
        )

    # Mclust on entropy (G=2)
    gmm_fit <- Mclust(specificity_df$entropy, G = 2, verbose = FALSE)
    specificity_df$entropy_cluster <- gmm_fit$classification
    cluster_means <- tapply(specificity_df$entropy, specificity_df$entropy_cluster, mean)
    specificity_df$pattern_type <- ifelse(
        specificity_df$entropy_cluster == which.max(cluster_means),
        "general", "specific"
    )

    log_msg("Plotting entropy histogram with GMM cluster split...")

    split_plot <- ggplot(specificity_df, aes(x = entropy, fill = pattern_type)) +
        geom_histogram(
            bins = max(5, floor(k / 2)), # dynamic bins with lower bound for stability
            color = "white",
            position = "identity",
            alpha = 0.6
        ) +
        scale_fill_manual(values = c("specific" = "#E69F00", "general" = "#56B4E9")) +
        theme_minimal(base_size = 14) +
        labs(
            title = sprintf("Entropy split (GMM, k=%d): General vs Specific patterns", k),
            x = "Entropy",
            y = "Count",
            fill = "Pattern type"
        ) +
        theme(
            plot.title = element_text(hjust = 0.5),
            legend.position = "top"
        )

    ggsave(
        filename = file.path(plots_dir, "dotplot_entropy_bimodal_split.pdf"),
        plot = split_plot,
        height = 6, width = 8, dpi = 300
    )
    log_msg("Saved entropy split histogram plot.")

    # Patterns abundance thresholds
    allowed_nmf <- setdiff(nmf_cols, character(0)) # no manual removal by default
    nonzero_spots_clean <- nonzero_spots[!is.na(nonzero_spots) & names(nonzero_spots) %in% allowed_nmf]
    abundance_threshold <- 1050
    low_abundance_nmf <- names(nonzero_spots_clean[nonzero_spots_clean < abundance_threshold])
    high_abundance_nmf <- setdiff(allowed_nmf, low_abundance_nmf)

    nmf_info <- data.frame(
        nmf = allowed_nmf,
        abundance = ifelse(allowed_nmf %in% low_abundance_nmf, "low", "high"),
        stringsAsFactors = FALSE
    ) %>%
        left_join(specificity_df[, c("nmf", "pattern_type")], by = "nmf")

    # staircase center_of_mass ordering
    staircase_df <- nuclei_nmf_scaled_summary %>%
        group_by(nmf) %>%
        summarize(center_of_mass = weighted.mean(
            x = as.numeric(factor(cell_type)),
            w = pmax(scaled_avg, 0)
        )) %>%
        ungroup()

    nmf_info <- nmf_info %>%
        left_join(staircase_df, by = "nmf") %>%
        mutate(order_group = case_when(
            TRUE ~ 5
        )) %>%
        arrange(order_group, center_of_mass)

    nmf_order <- nmf_info$nmf
    nmf_order <- nmf_order[!is.na(nmf_order)]

    # Merge summaries for plotting
    dot_df <- left_join(
        nuclei_nmf_prop_summary[, c("cell_type", "nmf", "prop")],
        nuclei_nmf_scaled_summary[, c("cell_type", "nmf", "scaled_avg")],
        by = c("cell_type", "nmf")
    ) %>%
        mutate(
            nmf_f = factor(nmf, levels = nmf_order),
            cell_type = factor(cell_type, levels = unique(cell_type))
        ) %>%
        filter(!is.na(nmf) & nmf %in% nmf_order)

    # Plot 1: nuclei dotplot
    p1 <- ggplot(dot_df, aes(x = nmf_f, y = cell_type, size = prop, color = scaled_avg)) +
        geom_point(stroke = 0, alpha = 0.9) +
        scale_size(range = c(0, 3), name = "Proportion") +
        scale_color_viridis_c(option = "F", direction = -1, name = "Scaled avg") +
        theme_bw(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
            axis.text.y = element_text(size = 8),
            axis.title = element_blank(), panel.grid = element_blank(), legend.position = "right"
        ) +
        labs(
            title = "Dotplot of NMF patterns vs. snRNA-seq cell types",
            subtitle = sprintf("Dataset: dlPFC (run: %s)", run_prefix)
        )

    ggsave(
        filename = file.path(plots_dir, "dotplot_nmf_patterns_by_cell_type.pdf"),
        plot = p1, height = 8, width = 16
    )

    # ------------------------------
    # Spot dotplot by domain using spe_run
    # ------------------------------
    log_msg("Creating SRT projection dotplot of NMF patterns by anatomical domain...")

    nmf_weight_matrix_spe <- as.matrix(colData(spe_run)[, nmf_cols])
    spots_nmf_nonzero_binary <- nmf_weight_matrix_spe > 0
    spots_nmf_presence <- data.frame(
        domain = get_col(spe_run, domain_col_name),
        spots_nmf_nonzero_binary,
        check.names = FALSE
    )

    spots_nmf_prop_summary <- spots_nmf_presence %>%
        group_by(domain) %>%
        add_tally(name = "total") %>%
        group_by(domain, total) %>%
        summarize(across(all_of(nmf_cols), sum), .groups = "drop") %>%
        pivot_longer(cols = all_of(nmf_cols), names_to = "nmf", values_to = "n") %>%
        mutate(prop = n / total)

    nmf_scaled_matrix_spe <- apply(nmf_weight_matrix_spe, 2, scale)
    spots_nmf_scaled <- data.frame(
        domain = get_col(spe_run, domain_col_name),
        nmf_scaled_matrix_spe,
        check.names = FALSE
    )

    spots_nmf_scaled_summary <- spots_nmf_scaled %>%
        group_by(domain) %>%
        summarize(across(all_of(nmf_cols), mean), .groups = "drop") %>%
        pivot_longer(cols = all_of(nmf_cols), names_to = "nmf", values_to = "scaled_avg")

    spot_dot_df <- left_join(
        spots_nmf_prop_summary[, c("domain", "nmf", "prop")],
        spots_nmf_scaled_summary[, c("domain", "nmf", "scaled_avg")],
        by = c("domain", "nmf")
    ) %>%
        filter(!is.na(nmf) & nmf %in% high_abundance_nmf) %>%
        mutate(
            nmf_f = factor(nmf, levels = nmf_order[nmf_order %in% high_abundance_nmf]),
            domain = factor(domain, levels = unique(domain))
        )

    p2 <- ggplot(spot_dot_df, aes(x = nmf_f, y = domain, size = prop, color = scaled_avg)) +
        geom_point(stroke = 0, alpha = 0.9) +
        scale_size(range = c(0, 3), name = "Proportion") +
        scale_color_viridis_c(option = "F", direction = -1, name = "Scaled avg") +
        theme_bw(base_size = 12) +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
            axis.text.y = element_text(size = 8),
            axis.title = element_blank(), panel.grid = element_blank(), legend.position = "right"
        ) +
        labs(
            title = "Dotplot of NMF patterns vs. SRT anatomical domains",
            subtitle = sprintf("High-abundance NMF patterns (run: %s)", run_prefix)
        )

    ggsave(
        filename = file.path(plots_dir, "dotplot_nmf_patterns_by_srt_domain.pdf"),
        plot = p2, height = 6, width = 12
    )

    # ------------------------------
    # Summarize this run and append to master CSV
    # ------------------------------
    n_specific <- sum(specificity_df$pattern_type == "specific", na.rm = TRUE)
    n_general <- sum(specificity_df$pattern_type == "general", na.rm = TRUE)
    n_common <- length(intersect(rownames(nmf_mat), rownames(spe)))

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

    new_row <- data.frame(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        k = k, seed = seed, tol = tol,
        nmf_input = nmf_path, projection_input = proj_path,
        n_common_genes = n_common,
        n_specific = n_specific, n_general = n_general,
        elapsed_min = elapsed,
        output_dir = run_outdir,
        stringsAsFactors = FALSE
    )

    write.table(new_row, file = master_summary_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    log_msg(sprintf("Appended run summary to %s", master_summary_csv))

    # cleanup for next run
    rm(nmf_x, nmf_mat, proj_mat, proj_df, new_nmf_df, sce_run, spe_run, proj_df)
    gc()
} # end for runs

log_msg("===== All runs complete for batched_registration_dotplot =====")
close_log()
