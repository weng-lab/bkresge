#!/usr/bin/env Rscript

library(duckplyr)
library(tidyr)
library(arrow)

# --------------------------
# Config
# --------------------------
summary_tsv <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/summary.tsv"
output_parquet <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/all_patterns.parquet"

summary_df <- read.delim(summary_tsv, stringsAsFactors = FALSE)


gene_meta <- read_parquet(
    "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/snrna/gene_meta.parquet"
)

stopifnot(!anyDuplicated(gene_meta$gene_id))

stopifnot(!is.unsorted(gene_meta$gene_id))


# --------------------------
# Convert one NMF object to tall
# --------------------------
nmf_to_tall <- function(x, params, run_id) {
    W <- x@w
    gene_names <- rownames(W)

    stopifnot(nrow(W) == nrow(gene_meta))

    meta_names <- gene_meta$gene_name

    # ---- Sanity check: positional match of gene names ----
    mismatches <- which(gene_names != meta_names)

    if (length(mismatches) > 0) {
        bad <- mismatches[seq_len(min(10, length(mismatches)))]

        stop(
            sprintf(
                paste(
                    "Gene name mismatch between NMF W and gene_meta",
                    "run_id = %s",
                    "First %d mismatches:",
                    "%s",
                    sep = "\n"
                ),
                run_id,
                length(mismatches),
                paste(
                    sprintf(
                        "row %d: W='%s'  meta='%s'",
                        bad,
                        gene_names[bad],
                        meta_names[bad]
                    ),
                    collapse = "\n"
                )
            )
        )
    }
    # ------------------------------------------------------

    pattern_nums <- colnames(W)

    W_df <- as.data.frame(W)
    W_df$gene_name <- gene_names
    W_df$gene_idx <- seq_len(nrow(W_df))

    W_df %>%
        pivot_longer(
            cols = all_of(pattern_nums),
            names_to = "pattern_number",
            values_to = "weight"
        ) %>%
        mutate(
            pattern_number = as.integer(sub("nmf", "", pattern_number)),
            run_id = run_id,
            k = params$k,
            seed = params$seed,
            tolerance = params$tol,
            L1 = params$L1
        ) %>%
        left_join(
            gene_meta %>%
                mutate(gene_idx = row_number()) %>%
                select(gene_idx, gene_id),
            by = c("gene_idx")
        ) %>%
        select(
            run_id,
            pattern_number,
            gene_name,
            gene_id,
            weight,
            k,
            seed,
            tolerance,
            L1
        )
}

# --------------------------
# Read all runs into memory and combine
# --------------------------
all_tall <- bind_rows(lapply(seq_len(nrow(summary_df)), function(i) {
    file <- summary_df$output_path[i]
    cat("Reading run_id =", i, "file:", file, "\n")

    x <- readRDS(file)
    params <- summary_df[i, c("k", "seed", "tol", "L1")]

    nmf_to_tall(x, params, run_id = i)
}))

cat(
    "Combined all runs into one tall data frame with",
    nrow(all_tall),
    "rows.\n"
)
cat("Now writing...\n")

# --------------------------
# Write to partitioned parquet
# --------------------------
write_dataset(
    all_tall,
    path = "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/all_patterns_partitioned_corrected",
    format = "parquet",
    partitioning = "k",
    existing_data_behavior = "overwrite"
)

cat("Done!")

# #!/usr/bin/env Rscript

# library(dplyr)
# library(tidyr)
# library(arrow)

# summary_tsv <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/subset.tsv"
# output_dir <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/subset_nmf_tall_dataset"

# summary_df <- read.delim(summary_tsv, stringsAsFactors = FALSE)

# nmf_to_tall <- function(x, params, run_id) {
#     W <- x@w
#     genes <- rownames(W)
#     patterns <- colnames(W)

#     W_df <- as.data.frame(W, row.names = genes)
#     W_df$gene <- genes

#     W_df %>%
#         pivot_longer(
#             cols = all_of(patterns),
#             names_to = "pattern",
#             values_to = "loading"
#         ) %>%
#         mutate(
#             pattern = as.integer(sub("nmf", "", pattern)),
#             run_id = run_id,
#             k = params$k,
#             seed = params$seed,
#             tol = params$tol,
#             L1 = params$L1
#         ) %>%
#         select(run_id, pattern, gene, loading, k, seed, tol, L1)
# }

# # --------------------------
# # Stream each run to its own Parquet file
# # --------------------------
# dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# for (i in seq_len(nrow(summary_df))) {
#     file <- summary_df$output_path[i]
#     cat("Processing run_id =", i, "file:", file, "\n")

#     x <- readRDS(file)
#     params <- summary_df[i, c("k", "seed", "tol", "L1")]
#     tall <- nmf_to_tall(x, params, run_id = i)

#     # Write each run to a unique Parquet file
#     out_file <- file.path(output_dir, paste0("run_", i, ".parquet"))
#     write_parquet(tall, out_file, compression = "snappy")
# }

# cat("All runs written to Parquet files in:", output_dir, "\n")
