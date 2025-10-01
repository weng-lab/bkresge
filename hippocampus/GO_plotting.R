library(ggplot2)
library(ggwordcloud)
library(patchwork)

# --- Config ---
# output_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_plots"
# output_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_plots_one_over"
# output_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_plots_specificity_02"
output_dir <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_plots_percentile_01"
nmf_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/nmf_x_k_80.rda"
# go_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80.rda"
# go_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_one_over.rda"
# go_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_specificity_02.rda"
go_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/go_analysis_k_80_percentile_01.rda"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load objects ---
load(go_file, verbose = TRUE) # gives 'go'
load(nmf_file, verbose = TRUE) # gives 'x'

# --- Logging ---
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_plotting.log"
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_plotting_one_over.log"
# log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_plotting_specificity_02.log"
log_file <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/GO_plotting_percentile_01.log"
sink(log_file, append = FALSE, split = TRUE)
options(width = 120)
log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

# --- Plotting function ---
plot_go_pattern <- function(pattern_num, top_n = 30) {
    enrich_res <- go[[pattern_num]]
    df <- enrich_res@result

    if (nrow(df) == 0) {
        log_msg(sprintf("No enriched terms for pattern %s.", pattern_num))
        return(NULL)
    }

    # --- Top GO terms ---
    df <- df[order(df$qvalue), ]
    df <- head(df, top_n)
    df$neglogq <- -log10(df$qvalue)
    df$Description <- factor(df$Description, levels = rev(df$Description))

    p_go <- ggplot(df, aes(x = Description, y = neglogq)) +
        geom_col(fill = "steelblue") +
        geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
        coord_flip() +
        labs(
            title = paste("Pattern", pattern_num, "Top Enriched GO Terms"),
            x = "GO Term",
            y = "-log10(q-value)"
        ) +
        theme_minimal(base_size = 12)

    # --- Wordcloud ---
    # Genes highlighted in enriched terms
    gene_list <- unique(unlist(strsplit(df$geneID, split = "/")))

    # All genes from the NMF for this pattern
    w <- x@w[, pattern_num]
    names(w) <- rownames(x@w)

    # Restrict to genes that exist in the enrichment marker genes / input genes
    w <- w[names(w) %in% enrich_res@gene]
    w <- w[!is.na(w)]
    if (length(w) == 0) {
        log_msg(sprintf("No valid genes for wordcloud for pattern %s.", pattern_num))
        return(NULL)
    }

    # Build dataframe for wordcloud
    df_wc <- data.frame(
        word = names(w),
        freq = w,
        in_term = ifelse(names(w) %in% gene_list, "inGO", "notGO")
    )

    # Optional: limit number of words for clarity
    top_n_words <- min(100, nrow(df_wc))
    df_wc <- df_wc[order(df_wc$freq, decreasing = TRUE), ][1:top_n_words, ]

    p_wc <- ggplot(df_wc, aes(label = word, size = freq, color = in_term)) +
        geom_text_wordcloud_area(
            grid_size = 0.25, # smaller = denser
            eccentricity = 0.5 # lower = more circular, less spread
        ) +
        scale_size_area(max_size = 12) + # smaller size can fit more words
        scale_color_manual(values = c("inGO" = "green3", "notGO" = "grey70")) +
        theme_minimal() +
        labs(title = paste("Pattern", pattern_num, "Genes Highlighted by GO"))

    # --- Combine side by side ---
    combined <- p_wc | p_go

    # --- Output file ---
    out_file <- file.path(output_dir, sprintf("pattern_%s_plots.pdf", pattern_num))

    ggsave(out_file, combined, width = 12, height = 6)

    log_msg(sprintf("Saved plots for pattern %s to %s", pattern_num, out_file))
}

# --- Loop through all patterns ---
for (i in seq_along(go)) {
    log_msg(sprintf("Processing pattern %s...", i))
    try(plot_go_pattern(i))
}
