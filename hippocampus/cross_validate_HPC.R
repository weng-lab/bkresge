suppressPackageStartupMessages({
    library(RcppML)
    library(singlet)
    library(sessioninfo)
    library(SingleCellExperiment)
    library(SpatialExperiment)
})

# Open log file (append = FALSE to overwrite each run)
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/cross_validate.log"
sink(log_file, append = FALSE, split = TRUE) # split=TRUE keeps console + file
options(width = 120)

log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

path_to_data <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"
path_for_cvnmf <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/cvnmf.rda"

# # Seed
# seed <- 123
# set.seed(seed)
# log_msg(sprintf("Seed %d", seed))

# Threads
threads <- 64
log_msg(sprintf("Running with %d threads", threads))

log_msg(sprintf("Loading data from %s", path_to_data))
obj_names <- load(path_to_data, verbose = TRUE)
stopifnot(length(obj_names) == 1)

snrna <- get(obj_names)

if (!inherits(snrna, "SingleCellExperiment")) {
    stop("Loaded object is not a SingleCellExperiment, cannot continue.")
}
log_msg("Data successfully loaded.")

# Run cross-validation
log_msg("Running cross-validation with singlet::cross_validate_nmf...")
cvnmf <- cross_validate_nmf(
    assay(snrna, "logcounts"),
    ranks = c(5, 10, 50, 100, 125, 150, 200),
    n_replicates = 3,
    tol = 1e-03,
    maxit = 100,
    verbose = 3,
    L1 = 0.1,
    L2 = 0,
    threads = threads,
    test_density = 0.2
)
log_msg(sprintf("Saving cross-validation results to %s", path_for_cvnmf))
save(cvnmf, file = path_for_cvnmf)
log_msg("Cross-validation results saved.")


# Plot results
# Built-in singlet plot (plot1)
p1 <- plot(cvnmf)
ggsave(
    filename = "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/cross_validate_plot1.png",
    plot = p1,
    width = 8,
    height = 6,
    dpi = 300
)

# Custom ggplot version (plot2)
df <- as.data.frame(cvnmf)
filtered_df <- df %>%
    dplyr::group_by(k, rep) %>%
    dplyr::filter(iter == max(iter)) %>%
    dplyr::ungroup()

p2 <- ggplot(filtered_df, aes(x = k, y = test_error, color = as.factor(rep))) +
    geom_line() +
    labs(
        title = "Test Error vs. k for Different Reps",
        x = "k",
        y = "Test Error",
        color = "Rep"
    ) +
    theme_minimal()

ggsave(
    filename = "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/cross_validate_plot2.png",
    plot = p2,
    width = 8,
    height = 6,
    dpi = 300
)



log_msg("Cross-validation complete.")


# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())

# Close sink
sink()
