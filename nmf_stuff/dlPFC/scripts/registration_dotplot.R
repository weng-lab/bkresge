suppressPackageStartupMessages({
    library(here)
    library(SingleCellExperiment)
    library(SpatialExperiment)
    library(dplyr)
    library(ggplot2)
    library(scater)
    library(sessioninfo)
    library(mclust)
    library(viridis)
})

# Set the project root for 'here' package
here::i_am("scripts/registration_dotplot.R")

# Load shared utility functions
source(here("scripts", "utils.R"))

# Logging
log_file <- setup_log(prefix = "registration_dotplot")

##### Paths
dlpfc_spe_path <- here("data", "srt_with_nmf.rda")
dlpfc_sce_path <- here("data", "snrna_with_nmf.rds")

# The column in colData(sce) that has the cell type annotations
cell_type_col_name <- "cellType_hc"
domain_col_name <- "BayesSpace_harmony_09"

get_col <- function(obj, colname) {
    if (!colname %in% colnames(colData(obj))) {
        stop(sprintf("Column '%s' not found in colData", colname))
    }
    return(as.factor(colData(obj)[[colname]]))
}


log_msg("Loading in data...")
dataset <- "dlPFC"
load_and_rename(dlpfc_spe_path, "spe", verbose = TRUE)
sce <- readRDS(dlpfc_sce_path)
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

ecdf_plot_file <- here("output", "plots", "ecdf_nonzero_nuclei_spots_nmf_patterns_faceted.pdf")
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
    cell_type = get_col(sce, cell_type_col_name),
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
    group_by(cell_type) %>%
    # Add total nuclei count for each cell type
    add_tally(name = "total") %>%
    # Group by cell type and total for summarization
    group_by(cell_type, total) %>%
    # Summarize: for each nmf column, count TRUE values (sum since TRUE == 1)
    summarize(across(all_of(nmf_cols), sum), .groups = "drop") %>%
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
    cell_type = get_col(sce, cell_type_col_name),
    nmf_scaled_matrix,
    check.names = FALSE,
    row.names = rownames(colData(sce)) # keep nuclei barcodes as row names
)

# Summarize average scaled NMF weights per cell type
log_msg("Summarizing average scaled NMF weights per cell type...")

nuclei_nmf_scaled_summary <- nuclei_nmf_scaled %>%
    # Group by cell type
    group_by(cell_type) %>%
    # For each nmf column, compute mean scaled weight
    summarize(across(all_of(nmf_cols), mean), .groups = "drop") %>%
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
    manual_isolated_nmf <- c("nmf28", "nmf37") # add sex specific patterns
} else if (dataset == "dlPFC") {
    manual_removed_nmf <- c()
    manual_isolated_nmf <- c()
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
    summarize(
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

# Optional: visualize the split
split_plot <- ggplot(specificity_df, aes(x = entropy, fill = pattern_type)) +
    geom_histogram(bins = 30, color = "white", position = "identity", alpha = 0.6) +
    scale_fill_manual(values = c("specific" = "#E69F00", "general" = "#56B4E9")) +
    theme_minimal(base_size = 14) +
    labs(
        title = "Bimodal entropy split: General vs Specific patterns",
        x = "Entropy", y = "Count", fill = "Pattern type"
    )

ggsave(
    filename = here("output", "plots", "dotplot_entropy_bimodal_split.pdf"),
    plot = split_plot,
    height = 6, width = 8
)

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
# Use the center of mass of scaled_avg across cell types
# (higher y-position = later in order)

staircase_df <- nuclei_nmf_scaled_summary %>%
    group_by(nmf) %>%
    summarize(center_of_mass = weighted.mean(
        x = as.numeric(factor(cell_type)),
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
            nmf %in% manual_isolated_nmf ~ 0,
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
    nuclei_nmf_prop_summary[, c("cell_type", "nmf", "prop")],
    nuclei_nmf_scaled_summary[, c("cell_type", "nmf", "scaled_avg")],
    by = c("cell_type", "nmf")
)

# Make the factors ordered
dot_df <- dot_df %>%
    mutate(
        nmf_f = factor(nmf, levels = nmf_order),
        cell_type = factor(cell_type,
            levels = unique(cell_type)
        ) # bottom-up order (as in original)
    )

# Filter the dot dataframe to only include named NMFs in the computed order
dot_df <- dot_df %>%
    filter(!is.na(nmf) & nmf %in% nmf_order)

### Plot dotplot ###

p1 <- ggplot(dot_df, aes(
    x = nmf_f, y = cell_type,
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
    filename = here("output", "plots", "dotplot_nmf_patterns_by_cell_type.pdf"),
    plot = p1,
    height = 8,
    width = 16
)

log_msg("Dotplot successfully generated and saved.")

### SRT projection spot plot ####
log_msg("Creating SRT projection dotplot of NMF patterns by anatomical domain...")

# Convert to a numeric matrix for easier manipulation
nmf_weight_matrix_spe <- as.matrix(colData(spe)[, nmf_cols])

# Binary presence/absence matrix for spots
spots_nmf_nonzero_binary <- nmf_weight_matrix_spe > 0

# Add domain information to the binary matrix
spots_nmf_presence <- data.frame(
    domain = get_col(spe, domain_col_name),
    spots_nmf_nonzero_binary,
    check.names = FALSE
)

# Summarize spot-level NMF pattern presence proportion by domain
log_msg("Summarizing SRT NMF pattern presence per anatomical domain...")

# Summarize presence per domain
spots_nmf_prop_summary <- spots_nmf_presence %>%
    group_by(domain) %>%
    add_tally(name = "total") %>%
    group_by(domain, total) %>%
    summarize(across(all_of(nmf_cols), sum), .groups = "drop") %>%
    tidyr::pivot_longer(
        cols = all_of(nmf_cols),
        names_to = "nmf",
        values_to = "n"
    ) %>%
    mutate(prop = n / total)

log_msg("Summarizing average scaled NMF weights per domain...")

# Scale weights across all spots (columns)
nmf_scaled_matrix_spe <- apply(nmf_weight_matrix_spe, 2, scale)

# Add domain information
spots_nmf_scaled <- data.frame(
    domain = get_col(spe, domain_col_name),
    nmf_scaled_matrix_spe,
    check.names = FALSE
)

# Summarize average scaled NMF weights by domain
spots_nmf_scaled_summary <- spots_nmf_scaled %>%
    group_by(domain) %>%
    summarize(across(all_of(nmf_cols), mean), .groups = "drop") %>%
    tidyr::pivot_longer(
        cols = all_of(nmf_cols),
        names_to = "nmf",
        values_to = "scaled_avg"
    )

log_msg("Merging proportion and scaled summaries for spot dotplot...")

# Merge summaries
spot_dot_df <- left_join(
    spots_nmf_prop_summary[, c("domain", "nmf", "prop")],
    spots_nmf_scaled_summary[, c("domain", "nmf", "scaled_avg")],
    by = c("domain", "nmf")
)

# Restrict to high-abundance nmfs, keeping the same ordering from previous plot
spot_dot_df <- spot_dot_df %>%
    filter(!is.na(nmf) & nmf %in% high_abundance_nmf) %>%
    mutate(
        nmf_f = factor(nmf, levels = nmf_order[nmf_order %in% high_abundance_nmf]),
        domain = factor(domain, levels = unique(domain))
    )

# Confirm inclusion counts
log_msg(sprintf(
    "Spot dotplot includes %d high-abundance NMF patterns across %d domains.",
    length(unique(spot_dot_df$nmf_f)),
    length(unique(spot_dot_df$domain))
))
# Plot spot dotplot
p2 <- ggplot(spot_dot_df, aes(
    x = nmf_f, y = domain,
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
        title = "Dotplot of NMF patterns vs. SRT anatomical domains",
        subtitle = sprintf(
            "High-abundance NMF patterns only (dataset: %s)",
            dataset
        )
    )

# Save spot dotplot
ggsave(
    filename = here("output", "plots", "dotplot_nmf_patterns_by_srt_domain.pdf"),
    plot = p2,
    height = 6,
    width = 12
)

# Session info
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())
log_msg("===== Finished Making Plots=====")

# Close sink
sink()
