library("SpatialExperiment")
library("BayesSpace")

# Define file paths
filtered_spe_path <- "/zata/zippy/kresgeb/scratch/spe_filtered_final.Rdata"
filtered_spe_path_data <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rdata/spe/01_build_spe/spe_filtered_final.Rdata"
clustered_spe_path <- "/zata/zippy/kresgeb/psych_screen/comparisons/spe_clustered.Rdata"

# Ensure the data is in scratch
if (!file.exists(filtered_spe_path)) {
    message(paste("Failed to find", filtered_spe_path, "... copying from", filtered_spe_path_data, "..."))
    file.copy(from = filtered_spe_path_data, to = filtered_spe_path)
}

# Same seed as 2024 paper
set.seed(030122)

# Run BayesSpace clustering if needed, otherwise simply load the results of clustering
if (!file.exists(clustered_spe_path)) {
    message(paste("Loading spe data from", filtered_spe_path, "..."))
    load(file = filtered_spe_path, verbose = TRUE)

    Sys.time()

    ### BayesSpace on Batch Corrected
    message("Running BayesSpace...")
    spe <- spatialCluster(spe, use.dimred = "HARMONY", q = 9, nrep = 10000)

    save(spe, file = clustered_spe_path)
    message(paste("Saved at", clustered_spe_path))
} else {
    message(paste("BayesSpace run found at", clustered_spe_path, " clustering will NOT be rerun"))
    message(paste("Loading clustered spe data from", clustered_spe_path, "..."))
    load(file = clustered_spe_path, verbose = TRUE)
}
