suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(RcppML)
    library(SingleCellExperiment)
    library(Matrix)
    library(sessioninfo)
    library(clue)
})

# --- Setup logging and paths ---
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/explore_x.log"
sink(log_file, append = FALSE, split = TRUE) # overwrite log each run
options(width = 120)

log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

snrna_seq_output_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"
path_for_x <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/nmf_x.rda"
plot_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/plots_v0.5.5"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
log_msg("===== Starting exploration of NMF result x =====")
log_msg(paste("Loading data from:", snrna_seq_output_path))
load(snrna_seq_output_path, verbose = TRUE)
snrna <- spatial_hpc_snrna_seq
log_msg("Data successfully loaded.")

log_msg(paste("Loading data from:", path_for_x))
load(path_for_x, verbose = TRUE)
log_msg("Data successfully loaded.")

# At this point you should have: nmf_new, nmf_paper matrices (cells x k)

# --- Define helper for comparison ---
compare_nmf <- function(nmf_new, nmf_paper, plot_dir) {
    log_msg("Comparing nmf_new vs nmf_paper")

    # 1. Basic summaries
    log_msg("Summary stats for nmf_new:")
    print(summary(as.vector(nmf_new)))
    log_msg("Summary stats for nmf_paper:")
    print(summary(as.vector(nmf_paper)))

    # 2. Top-left 10 x 10 sanity check
    log_msg("Top-left 10x10 of nmf_new:")
    print(nmf_new[1:10, 1:10])
    log_msg("Top-left 10x10 of nmf_paper:")
    print(nmf_paper[1:10, 1:10])

    # 3. Boxplots
    png(file.path(plot_dir, "boxplot_components.png"), width = 1200, height = 600)
    par(mfrow = c(1, 2))
    boxplot(nmf_new, outline = FALSE, main = "nmf_new", ylab = "Loadings", xlab = "Components")
    boxplot(nmf_paper, outline = FALSE, main = "nmf_paper", ylab = "Loadings", xlab = "Components")
    dev.off()

    # 4. Density overlay
    png(file.path(plot_dir, "density_overlay.png"), width = 800, height = 600)
    plot(density(as.vector(nmf_new)),
        col = "blue", lwd = 2,
        main = "Distribution of loadings", xlab = "Loading value"
    )
    lines(density(as.vector(nmf_paper)), col = "red", lwd = 2)
    legend("topright",
        legend = c("nmf_new", "nmf_paper"),
        col = c("blue", "red"), lwd = 2
    )
    dev.off()

    # 5. Per-column mean comparison
    png(file.path(plot_dir, "per_column_mean_comparison.png"), width = 800, height = 600)
    nmf_new_means <- colMeans(nmf_new)
    nmf_paper_means <- colMeans(nmf_paper)
    plot(nmf_new_means, nmf_paper_means,
        xlab = "nmf_new means", ylab = "nmf_paper means",
        main = "Per-column mean comparison"
    )
    abline(0, 1, col = "red", lty = 2)
    dev.off()

    # 6. Correlation heatmap with legend/scale
    suppressPackageStartupMessages(library(pheatmap))

    cor_mat <- cor(nmf_new, nmf_paper)

    heatmap_file <- file.path(plot_dir, "correlation_heatmap.png")
    png(heatmap_file, width = 1200, height = 1000)
    pheatmap(
        cor_mat,
        cluster_rows = TRUE,
        cluster_cols = TRUE,
        show_rownames = TRUE,
        show_colnames = TRUE,
        main = "Correlation: nmf_new vs nmf_paper",
        color = colorRampPalette(c("blue", "white", "red"))(100),
        fontsize = 8
    )
    dev.off()

    log_msg("Plots saved to:")
    log_msg(plot_dir)
}

# Prepare matrices
nmf_new <- t(x@h) # cells x 100
nmf_paper <- as.matrix(colData(snrna)[, paste0("nmf", 1:100)])

# --- Run comparison ---
compare_nmf(nmf_new, nmf_paper, plot_dir)

log_msg("===== Finished NMF comparison =====")

# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
sink() # close log
