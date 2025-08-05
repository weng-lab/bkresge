library(BayesSpace)
library(mclust)
# library(dplyr)
library(spatialLIBD)

# load("/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData", verbose = TRUE)

spe <- spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")

output_csv <- "/zata/zippy/kresgeb/clustering_comparison/results/my_bs_output.csv"

seed <- 030122

set.seed(seed)

message(paste("BayesSpace Version:", packageVersion("BayesSpace")))

message((paste("Seed:", seed)))

message(paste("BayesSpace clustering started at:", Sys.time()))

# Run BayesSpace
message("Running BayesSpace spatialCluster...")
spe <- spatialCluster(
    spe,
    use.dimred = "HARMONY", # Use "PCA" if HARMONY is not available
    q = 9,
    nrep = 10000,
)

# Write cluster assignments to CSV
message("Saving cluster assignments...")
df <- data.frame(
    barcode = colnames(spe),
    sample_id = colData(spe)$sample_id,
    cluster = colData(spe)$spatial.cluster
)
write.csv(df, file = output_csv, row.names = FALSE)

message(paste("BayesSpace clustering finished at:", Sys.time()))

# Get sample IDs
samples <- unique(colData(spe)$sample_id)

# Compute ARI for each sample
ari_per_sample <- sapply(samples, function(sid) {
    idx <- colData(spe)$sample_id == sid
    cl1 <- colData(spe)$spatial.cluster[idx]
    cl2 <- colData(spe)$BayesSpace_harmony_09[idx]

    # Check for at least 2 unique labels in each clustering
    if (length(unique(cl1)) > 1 && length(unique(cl2)) > 1) {
        adjustedRandIndex(cl1, cl2)
    } else {
        NA # Not defined if only one cluster
    }
})

# Convert to data.frame for easier use
ari_df <- data.frame(
    sample_id = names(ari_per_sample),
    ARI = unname(ari_per_sample)
)

print(ari_df)
