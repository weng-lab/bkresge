library(SingleCellExperiment)
library(SpatialExperiment)
library(dplyr)
library(ggplot2)
library(scater)
library(sessioninfo)
library(mclust)
library(viridis)

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
dataset <- "HPC"
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

log_msg("Creating ECDF plot for nonzero NMF pattern weights...")
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

ecdf_plot_file <- file.path(plot_dir, "ecdf_nonzero_nuclei_spots_nmf_patterns_faceted.pdf")
# Save as PDF (one page, two facets stacked)
log_msg(sprintf("Saving ECDF plot to %s", ecdf_plot_file))
ggsave(
    filename = ecdf_plot_file,
    plot = p_ecdf,
    width = 6, height = 8, dpi = 300
)

##### Dotplots ######

# Convert to a numeric matrix for easier manipulation
nmf_weight_matrix <- as.matrix(colData(sce)[, nmf_cols])

#### NMF patterns generated via snRNA-seq gene expression (nuclei: cell type X NMF pattern) #####

### Data Preparation ###

log_msg("Preparing data for dotplot of NMF pattern presence by cell type...")

# Binary presence/absence matrix
nuclei_nmf_nonzero_binary <- nmf_weight_matrix > 0

# Add cell type information to the binary matrix
nuclei_nmf_presence <- data.frame(
    superfine_cell_class = colData(sce)$superfine.cell.class,
    nuclei_nmf_nonzero_binary,
    check.names = FALSE
)

# Summarize nuclei-level NMF pattern presence proportion by cell type

log_msg("Summarizing NMF pattern presence per cell type...")

# Calculate, for each cell type -> NMF pattern:
# - n: number of nuclei with nonzero (TRUE) weights
# - total: total number of nuclei in that cell type
# - prop: n / total (proportion of nuclei with nonzero weights)
nuclei_nmf_prop_summary <- nuclei_nmf_presence %>%
    # Group by cell type to calculate per-cluster statistics
    group_by(superfine_cell_class) %>%
    # Add total nuclei count for each cell type
    add_tally(name = "total") %>%
    # Group by cell type and total for summarisation
    group_by(superfine_cell_class, total) %>%
    # Summarize: for each nmf column, count TRUE values (sum since TRUE == 1)
    summarise(across(all_of(nmf_cols), sum), .groups = "drop") %>%
    # Pivot to long format: one row per cell type -> nmf pattern
    tidyr::pivot_longer(
        cols = all_of(nmf_cols),
        names_to = "nmf",
        values_to = "n"
    ) %>%
    # Compute proportion
    mutate(prop = n / total)

log_msg("NMF pattern proportion summary created successfully")

log_msg("Preparing data for dotplot of average scaled NMF weights by cell type...")

# Scale NMF weights across nuclei (columns)
nmf_scaled_matrix <- apply(nmf_weight_matrix, 2, scale)

# Add cell type information to the scaled matrix
nuclei_nmf_scaled <- data.frame(
    superfine_cell_class = colData(sce)$superfine.cell.class,
    nmf_scaled_matrix,
    check.names = FALSE,
    row.names = rownames(colData(sce)) # keep nuclei barcodes as row names
)

# Summarize average scaled NMF weights per cell type
log_msg("Summarizing average scaled NMF weights per cell type...")

nuclei_nmf_scaled_summary <- nuclei_nmf_scaled %>%
    # Group by cell type
    group_by(superfine_cell_class) %>%
    # For each nmf column, compute mean scaled weight
    summarise(across(all_of(nmf_cols), mean), .groups = "drop") %>%
    # Pivot to long format
    tidyr::pivot_longer(
        cols = all_of(nmf_cols),
        names_to = "nmf",
        values_to = "scaled_avg"
    )

log_msg("Average scaled NMF weight summary created successfully")


### Merging Summaries and Plotting ###


## Deciding on NMF pattern order ##
if (dataset == "HPC") {
    manual_removed_nmf <- c("nmf2", "nmf3", "nmf16") # NA patterns identified manually
    sex_specific_nmf <- c("nmf28", "nmf37") # add sex specific patterns
} else {
    stop(sprintf("Dataset: %s not recognized for cell type ordering.", dataset))
}

abundance_threshold <- 1050 # number of spots in SRT projection below which patterns are removed

allowed_nmf <- setdiff(nmf_cols, manual_removed_nmf)

# Keep only allowed NMF patterns and remove NAs
nonzero_spots_clean <- nonzero_spots[!is.na(nonzero_spots) & names(nonzero_spots) %in% allowed_nmf]

# Patterns with < threshold
low_abundance_nmf <- names(nonzero_spots_clean[nonzero_spots_clean < abundance_threshold])

# Patterns with >= threshold
high_abundance_nmf <- setdiff(allowed_nmf, low_abundance_nmf)

pseudo_count <- 1e-9

specificity_df <- nuclei_nmf_prop_summary %>%
    group_by(nmf) %>%
    summarise(
        mean_prop = mean(prop),
        sd_prop = sd(prop),
        prop_cv = sd(prop) / mean(prop),
        entropy = -sum(((prop + pseudo_count) / sum(prop + pseudo_count)) * log((prop + pseudo_count) / sum(prop + pseudo_count)))
    )

# Plot distribution to visually check for bimodality
# ggplot(specificity_df, aes(x = entropy)) +
#   geom_histogram(bins = 30, fill = "steelblue", color = "white") +
#   geom_density(color = "black", linewidth = 1) +
#   theme_minimal(base_size = 14) +
#   labs(title = "Distribution of NMF entropy", x = "Entropy", y = "Density")

# Fit Gaussian mixture model (2 components)
gmm_fit <- Mclust(specificity_df$entropy, G = 2)

# Add the component classification
specificity_df$entropy_cluster <- gmm_fit$classification

# Check the means to identify which cluster is which
cluster_means <- tapply(specificity_df$entropy, specificity_df$entropy_cluster, mean)

# The higher mean corresponds to "general", lower mean to "specific"
# Assign labels accordingly
specificity_df$pattern_type <- ifelse(
    specificity_df$entropy_cluster == which.max(cluster_means),
    "general", "specific"
)

# # Optional: visualize the split
# ggplot(specificity_df, aes(x = entropy, fill = pattern_type)) +
#   geom_histogram(bins = 30, color = "white", position = "identity", alpha = 0.6) +
#   scale_fill_manual(values = c("specific" = "#E69F00", "general" = "#56B4E9")) +
#   theme_minimal(base_size = 14) +
#   labs(title = "Bimodal entropy split: General vs Specific patterns",
#        x = "Entropy", y = "Count", fill = "Pattern type")

log_msg(sprintf(
    "Identified %d specific and %d general NMF patterns based on entropy.",
    sum(specificity_df$pattern_type == "specific"),
    sum(specificity_df$pattern_type == "general")
))


# Merge abundance and specificity info #

nmf_info <- data.frame(
    nmf = allowed_nmf,
    abundance = ifelse(allowed_nmf %in% low_abundance_nmf, "low", "high"),
    stringsAsFactors = FALSE
) %>%
    left_join(specificity_df[, c("nmf", "pattern_type")], by = "nmf")

## Compute a "staircase" index for ordering within specific patterns ##
# Use the center of mass of scaled_avg across superfine cell classes
# (higher y-position = later in order)

staircase_df <- nuclei_nmf_scaled_summary %>%
    group_by(nmf) %>%
    summarise(center_of_mass = weighted.mean(
        x = as.numeric(factor(superfine_cell_class)),
        w = pmax(scaled_avg, 0)
    )) %>%
    ungroup()

nmf_info <- nmf_info %>%
    left_join(staircase_df, by = "nmf")

## Build final ordering ##
# Order priority:
#   0: manually removed
#   1: low-abundance general
#   2: low-abundance specific
#   3: high-abundance general
#   4: high-abundance specific

nmf_info <- nmf_info %>%
    mutate(
        order_group = case_when(
            nmf %in% sex_specific_nmf ~ 0,
            abundance == "low" & pattern_type == "general" ~ 1,
            abundance == "low" & pattern_type == "specific" ~ 2,
            abundance == "high" & pattern_type == "general" ~ 3,
            abundance == "high" & pattern_type == "specific" ~ 4,
            TRUE ~ 5
        )
    ) %>%
    arrange(order_group, center_of_mass)

nmf_order <- nmf_info$nmf
nmf_order <- nmf_order[!is.na(nmf_order)]


log_msg(sprintf(
    "Ordering complete: %d total NMFs (manual: %d, low: %d, high: %d)",
    length(nmf_order),
    length(manual_removed_nmf),
    sum(nmf_info$abundance == "low", na.rm = TRUE),
    sum(nmf_info$abundance == "high", na.rm = TRUE)
))

### Prepare dotplot dataframe ###

dot_df <- left_join(
    nuclei_nmf_prop_summary[, c("superfine_cell_class", "nmf", "prop")],
    nuclei_nmf_scaled_summary[, c("superfine_cell_class", "nmf", "scaled_avg")],
    by = c("superfine_cell_class", "nmf")
)

# Make the factors ordered
dot_df <- dot_df %>%
    mutate(
        nmf_f = factor(nmf, levels = nmf_order),
        superfine_cell_class = factor(superfine_cell_class,
            levels = unique(superfine_cell_class)
        ) # bottom-up order (as in original)
    )

# Filter the dot dataframe to only include named NMFs in the computed order
dot_df <- dot_df %>%
    filter(!is.na(nmf) & nmf %in% nmf_order)

### Plot dotplot ###

p1 <- ggplot(dot_df, aes(
    x = nmf_f, y = superfine_cell_class,
    size = prop, color = scaled_avg
)) +
    geom_point(stroke = 0, alpha = 0.9) +
    scale_size(range = c(0, 3), name = "Proportion") +
    scale_color_viridis_c(option = "F", direction = -1, name = "Scaled avg") +
    theme_bw(base_size = 12) +
    theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size = 8),
        axis.title = element_blank(),
        panel.grid = element_blank(),
        legend.position = "right"
    ) +
    labs(
        title = "Dotplot of NMF patterns vs. snRNA-seq cell types",
        subtitle = sprintf(
            "Low vs. high abundance and general vs. specific split (dataset: %s)",
            dataset
        )
    )

### Save plot ###

ggsave(
    filename = file.path(plot_dir, "dotplot_nmf_patterns_by_cell_type.pdf"),
    plot = p1,
    height = 8,
    width = 16
)

log_msg("Dotplot successfully generated and saved.")





# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished Making Plots=====")

# Close sink
sink()
