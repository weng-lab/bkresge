library(SingleCellExperiment)
library(SpatialExperiment)
library(dplyr)
library(ggplot2)
library(scater)
library(sessioninfo)

##### Paths
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/registration_dotplot.log"
hpc_spe_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"
hpc_sce_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"
dlpfc_spe_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
dlpfc_sce_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
plot_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/plots"

##### Logging
# Open log file (append = FALSE to overwrite each run)
sink(log_file, append = FALSE, split = TRUE) # split=TRUE keeps console + file
options(width = 120)
log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}



##### Loading data
load_and_rename <- function(path, new_name) {
    obj_names <- load(path, verbose = TRUE)
    if (length(obj_names) != 1) {
        stop(paste("Expected 1 object in", path, "but got", length(obj_names)))
    }
    assign(new_name, get(obj_names), envir = .GlobalEnv)
    rm(list = obj_names, envir = .GlobalEnv) # clean up original name
}

log_msg("Loading in data...")

load_and_rename(hpc_sce_path, "sce")
load_and_rename(hpc_spe_path, "spe")

log_msg("Data loading complete")

# Identify the NMF pattern columns (look like nmf1, nmf47, etc.)
nmf_cols <- grep("^nmf[0-9]+$", colnames(colData(sce)), value = TRUE)
log_msg(sprintf("Found %d nmf patterns", length(nmf_cols)))
# Make sure the spe has the same nmf columns
missing_nmf_cols <- setdiff(nmf_cols, colnames(colData(spe)))
if (length(missing_nmf_cols) > 0) {
    stop(paste("The following nmf columns are missing in spe:", paste(missing_nmf_cols, collapse = ", ")))
}

##### Non-zero plots (nucluei and spots [post-projection] combined) #####

# Count nonzeros
nonzero_nuclei <- colSums(as.matrix(colData(sce)[, nmf_cols]) > 0)
nonzero_spots <- colSums(as.matrix(colData(spe)[, nmf_cols]) > 0)

# Build long dataframe with source column
nonzero_df <- dplyr::bind_rows(
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

# Save as PDF (one page, two facets stacked)
ggsave(
    filename = file.path(plot_dir, "ecdf_nonzero_nuclei_spots_nmf_patterns_faceted.pdf"),
    plot = p_ecdf,
    width = 6, height = 8, dpi = 300
)

##### Dotplot ######




# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished Making Plots=====")

# Close sink
sink()
