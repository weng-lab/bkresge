suppressPackageStartupMessages({
    library(mclust) # for adjustedRandIndex
    library(dplyr) # for inner_join
})

# Read the manifest file and outputs/logs from Snakemake
manifest <- read.csv(snakemake@input[["manifest"]], stringsAsFactors = FALSE)
done_file <- snakemake@output[["done"]]
log_file <- snakemake@log[[1]]

# Set up logging
log_con <- file(log_file, open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")

cat("=== Starting clustering comparison ===\n")
cat(sprintf("Time started: %s\n", Sys.time()))
cat(sprintf("Loaded manifest with %d rows\n\n", nrow(manifest)))

for (i in seq_len(nrow(manifest))) {
    file1 <- manifest$file_path1[i]
    file2 <- manifest$file_path2[i]
    result_dir <- manifest$result_dir[i]

    cat("------------------------------------------------------------\n")
    cat(sprintf("Processing row %d\n", i))
    cat(sprintf("  file1: %s\n", file1))
    cat(sprintf("  file2: %s\n", file2))
    cat(sprintf("  result_dir: %s\n", result_dir))

    if (!dir.exists(result_dir)) {
        dir.create(result_dir, recursive = TRUE)
        cat("  Created result directory.\n")
    }

    # Load data
    df1 <- read.csv(file1, stringsAsFactors = FALSE)
    df2 <- read.csv(file2, stringsAsFactors = FALSE)

    #  Handle alternate cluster column names (ex. 2024 paper BayesSpace assignments)
    cluster_col1 <- setdiff(colnames(df1), "barcode")[1]
    if (cluster_col1 != "cluster") {
        cat(sprintf("  Note: In file1, cluster column is named '%s'. Renaming to 'cluster'.\n", cluster_col1))
        names(df1)[names(df1) == cluster_col1] <- "cluster"
    }

    cluster_col2 <- setdiff(colnames(df2), "barcode")[1]
    if (cluster_col2 != "cluster") {
        cat(sprintf("  Note: In file2, cluster column is named '%s'. Renaming to 'cluster'.\n", cluster_col2))
        names(df2)[names(df2) == cluster_col2] <- "cluster"
    }

    # Basic checks
    if (!all(c("barcode", "cluster") %in% colnames(df1))) {
        stop(paste("file1 is missing required columns in row", i))
    }
    if (!all(c("barcode", "cluster") %in% colnames(df2))) {
        stop(paste("file2 is missing required columns in row", i))
    }

    # Check barcode set equality
    barcodes1 <- df1$barcode
    barcodes2 <- df2$barcode

    missing_in_1 <- setdiff(barcodes2, barcodes1)
    missing_in_2 <- setdiff(barcodes1, barcodes2)

    if (length(missing_in_1) > 0 || length(missing_in_2) > 0) {
        cat("  Mismatch in barcode sets:\n")
        if (length(missing_in_1) > 0) {
            cat("    Barcodes in file2 but not in file1:", paste(missing_in_1, collapse = ", "), "\n")
        }
        if (length(missing_in_2) > 0) {
            cat("    Barcodes in file1 but not in file2:", paste(missing_in_2, collapse = ", "), "\n")
        }
        stop("  Cluster CSVs have mismatched barcode sets.")
    }

    # Merge and align (ARI requires both sets to be in the same order)
    merged_df <- inner_join(df1, df2, by = "barcode", suffix = c(".1", ".2"))
    cat("  Successfully merged data frames by barcode.\n")

    # Calculate ARI
    ari <- adjustedRandIndex(merged_df$cluster.1, merged_df$cluster.2)
    cat(sprintf("  Adjusted Rand Index: %.4f\n", ari))

    # Save ARI to file
    ari_file <- file.path(result_dir, "ari.txt")
    write(sprintf("%.4f", ari), file = ari_file)
    cat("  ARI saved to", ari_file, "\n")

    #  Save contingency table
    ct <- table(merged_df$cluster.1, merged_df$cluster.2)
    ct_file <- file.path(result_dir, "contingency_table.csv")
    write.csv(as.data.frame.matrix(ct), file = ct_file, row.names = TRUE)
    cat("  Contingency table saved to", ct_file, "\n")
}

cat("\n=== Done! ===\n")
cat(sprintf("Time ended: %s\n", Sys.time()))

# Mark completion for Snakemake (this is the specified output of the rule)
file.create(done_file)

# Close the log
sink(type = "output")
sink(type = "message")
close(log_con)
