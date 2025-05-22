suppressPackageStartupMessages({
    library(mclust) # for adjustedRandIndex
    library(dplyr) # for inner_join
})



file1 <- "/zata/zippy/kresgeb/clustering_comparison/results/cluster_assignments/2024/BayesSpace/k=9/Br6522_mid/nreps=10000_seed=314.csv"
file2 <- "/zata/zippy/kresgeb/clustering_comparison/results/cluster_assignments/2024/BayesSpace/k=9/Br6522_mid/nreps=10000_seed=30122.csv"

# ---- Load data ----
df1 <- read.csv(file1, stringsAsFactors = FALSE)
df2 <- read.csv(file2, stringsAsFactors = FALSE)

# ---- Basic checks ----
if (!all(c("barcode", "cluster") %in% colnames(df1))) {
    stop("file1 is missing required columns: 'barcode', 'cluster'")
}
if (!all(c("barcode", "cluster") %in% colnames(df2))) {
    stop("file2 is missing required columns: 'barcode', 'cluster'")
}

# ---- Check barcode set equality ----
barcodes1 <- df1$barcode
barcodes2 <- df2$barcode

missing_in_1 <- setdiff(barcodes2, barcodes1)
missing_in_2 <- setdiff(barcodes1, barcodes2)

if (length(missing_in_1) > 0 || length(missing_in_2) > 0) {
    cat("Mismatch in barcode sets:\n")
    if (length(missing_in_1) > 0) {
        cat("  Barcodes in file2 but not in file1:", paste(missing_in_1, collapse = ", "), "\n")
    }
    if (length(missing_in_2) > 0) {
        cat("  Barcodes in file1 but not in file2:", paste(missing_in_2, collapse = ", "), "\n")
    }
    stop("Cluster CSVs have mismatched barcode sets.")
}

# ---- Merge and align ----
merged_df <- inner_join(df1, df2, by = "barcode", suffix = c(".1", ".2"))

# ---- Calculate ARI ----
ari <- adjustedRandIndex(merged_df$cluster.1, merged_df$cluster.2)
cat(sprintf("Adjusted Rand Index: %.4f\n", ari))

# ---- Contingency table ----
cat("Contingency table:\n")
ct <- table(merged_df$cluster.1, merged_df$cluster.2)
print(ct)
