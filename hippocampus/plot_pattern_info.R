#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(tidyr)
    library(dplyr)
    library(RcppML)
})

#------ Parameters ------
sample_type <- "HPC" # or "HPC"
k <- 100 # NMF rank

# Base directories
base_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf"
sample_dir <- file.path(base_dir, sample_type)

# Construct file names
nmf_file <- if (k == 100) "nmf_x.rda" else sprintf("nmf_x_k_%d.rda", k)
proj_file <- if (k == 100) "proj_srt.rda" else sprintf("proj_srt_k_%d.rda", k)

# Full paths
nmf_data_path <- file.path(sample_dir, nmf_file)
proj_srt_path <- file.path(sample_dir, proj_file)

# Plot directory (create subdirectory for this k)
plot_dir <- file.path(sample_dir, "plots", sprintf("k_%d", k))
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# Print paths to check
cat("NMF path:", nmf_data_path, "\n")
cat("Projection path:", proj_srt_path, "\n")
cat("Plot dir:", plot_dir, "\n")


#--- Load NMF results ---
load(nmf_data_path, verbose = TRUE)

#--- Extract W and H ---
W <- x$w # genes x patterns
H <- x$h # patterns x spots

#--- Convert to tidy format ---
df_W <- as.data.frame(W) %>%
    tibble::rownames_to_column("gene") %>%
    pivot_longer(
        cols = -gene,
        names_to = "pattern",
        values_to = "contribution"
    )

df_H <- as.data.frame(H) %>%
    tibble::rownames_to_column("pattern") %>%
    pivot_longer(
        cols = -pattern,
        names_to = "spot",
        values_to = "expression"
    )

#--- Boxplot for W (gene contributions) ---
pW <- ggplot(df_W, aes(x = pattern, y = contribution)) +
    geom_boxplot(fill = "skyblue", color = "black", outlier.size = 0.5) +
    theme_bw() +
    labs(
        title = "Distribution of gene contributions (W)",
        x = "Pattern",
        y = "Gene contribution"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

#--- Boxplot for H (spot contributions) ---
pH <- ggplot(df_H, aes(x = pattern, y = expression)) +
    geom_boxplot(fill = "salmon", color = "black", outlier.size = 0.5) +
    theme_bw() +
    labs(
        title = "Distribution of spot expression values (H)",
        x = "Pattern",
        y = "Spot expression"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))


#--- Save plots ---
path_w_plot <- file.path(plot_dir, "nmf_gene_contributions_boxplot.png")
path_h_plot <- file.path(plot_dir, "nmf_spot_expression_boxplot.png")

ggsave(path_w_plot, pW, width = 20, height = 8, dpi = 300)
ggsave(path_h_plot, pH, width = 20, height = 8, dpi = 300)

#--- Summaries ---
sum_W <- df_W %>%
    group_by(pattern) %>%
    summarise(total_contribution = sum(contribution), .groups = "drop")

sum_H <- df_H %>%
    group_by(pattern) %>%
    summarise(total_expression = sum(expression), .groups = "drop")

#--- Histogram / barplots ---
pW_sum <- ggplot(sum_W, aes(x = pattern, y = total_contribution)) +
    geom_col(fill = "skyblue", color = "black") +
    theme_bw() +
    labs(
        title = "Total gene contributions per pattern (W)",
        x = "Pattern",
        y = "Sum of contributions"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

pH_sum <- ggplot(sum_H, aes(x = pattern, y = total_expression)) +
    geom_col(fill = "salmon", color = "black") +
    theme_bw() +
    labs(
        title = "Total spot expression per pattern (H)",
        x = "Pattern",
        y = "Sum of expression"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

#--- Save ---
path_w_sum_plot <- file.path(plot_dir, "nmf_gene_contributions_sum.png")
path_h_sum_plot <- file.path(plot_dir, "nmf_spot_expression_sum.png")

ggsave(path_w_sum_plot, pW_sum, width = 20, height = 8, dpi = 300)
ggsave(path_h_sum_plot, pH_sum, width = 20, height = 8, dpi = 300)

# --- Percent zeros per pattern ---
# Calculate percent zeros for W
pct_zero_W <- df_W %>%
    group_by(pattern) %>%
    summarise(
        percent_zero = mean(contribution == 0) * 100,
        .groups = "drop"
    ) %>%
    mutate(type = "Gene Contribution")

# Calculate percent zeros for H
pct_zero_H <- df_H %>%
    group_by(pattern) %>%
    summarise(
        percent_zero = mean(expression == 0) * 100,
        .groups = "drop"
    ) %>%
    mutate(type = "Spot Expression")

# Combine
pct_zeros <- bind_rows(pct_zero_W, pct_zero_H)

# Plot percent zeros
p_zero <- ggplot(pct_zeros, aes(x = factor(pattern), y = percent_zero, fill = type)) +
    geom_bar(stat = "identity", width = 0.9) +
    facet_wrap(~type, nrow = 2, scales = "free_y") +
    labs(
        title = "Percentage of Zero Values per NMF Pattern",
        x = "NMF Pattern",
        y = "Percent Zero (%)"
    ) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

# Save percent zeros plot
path_zero_plot <- file.path(plot_dir, "nmf_percent_zeros_per_pattern.png")
ggsave(path_zero_plot, p_zero, width = 20, height = 8, dpi = 300)

# --- Load projection results ---
load(proj_srt_path, verbose = TRUE) # loads srt and proj

# Convert proj (spots x patterns) into tidy format
df_proj <- as.data.frame(proj) %>%
    tibble::rownames_to_column("spot") %>%
    pivot_longer(
        cols = -spot,
        names_to = "pattern",
        values_to = "proj_expression"
    )

# --- Boxplot for projection spot expression ---
p_proj <- ggplot(df_proj, aes(x = pattern, y = proj_expression)) +
    geom_boxplot(fill = "lightgreen", color = "black", outlier.size = 0.5) +
    theme_bw() +
    labs(
        title = "Distribution of projected spot expression per pattern",
        x = "Pattern",
        y = "Projected spot expression"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# Save projection boxplot
path_proj_plot <- file.path(plot_dir, "nmf_projection_spot_expression_boxplot.png")
ggsave(path_proj_plot, p_proj, width = 20, height = 8, dpi = 300)

# --- Percent zeros in projection ---
pct_zero_proj <- df_proj %>%
    group_by(pattern) %>%
    summarise(
        percent_zero = mean(proj_expression == 0) * 100,
        .groups = "drop"
    ) %>%
    mutate(type = "Projected Spot Expression")

p_proj_zero <- ggplot(pct_zero_proj, aes(x = factor(pattern), y = percent_zero)) +
    geom_col(fill = "lightgreen", color = "black", width = 0.8) +
    labs(
        title = "Percentage of Zero Values per NMF Pattern (Projection)",
        x = "NMF Pattern",
        y = "Percent Zero (%)"
    ) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

# Save percent zeros projection plot
path_proj_zero_plot <- file.path(plot_dir, "nmf_projection_percent_zeros_per_pattern.png")
ggsave(path_proj_zero_plot, p_proj_zero, width = 20, height = 8, dpi = 300)

# --- Total projected expression per pattern ---
sum_proj <- df_proj %>%
    group_by(pattern) %>%
    summarise(total_proj_expression = sum(proj_expression), .groups = "drop")

# Barplot of total projected expression
p_proj_sum <- ggplot(sum_proj, aes(x = pattern, y = total_proj_expression)) +
    geom_col(fill = "lightgreen", color = "black") +
    theme_bw() +
    labs(
        title = "Total projected spot expression per pattern",
        x = "Pattern",
        y = "Sum of projected expression"
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# Save total projected expression plot
path_proj_sum_plot <- file.path(plot_dir, "nmf_projection_sum_expression.png")
ggsave(path_proj_sum_plot, p_proj_sum, width = 20, height = 8, dpi = 300)
