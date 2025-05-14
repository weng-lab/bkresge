library("SpatialExperiment")
library("BayesSpace")


filtered_spe_path <- "/zata/zippy/kresgeb/scratch/spe_filtered_final.Rdata"
filtered_spe_path_data <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rdata/spe/01_build_spe/spe_filtered_final.Rdata"

sample_id <- "Br6522_mid"

# Function to run BayesSpace clustering on a specific sample
#' @param spe A SpatialExperiment object containing the spatial transcriptomics data.
#' @param sample_id The ID of the sample to cluster.
#' @param q The number of clusters to use for clustering.
#' @param provided_seed The seed for random number generation. 0 means no seed is set.
#' @return A SpatialExperiment object with the clustering results.
#' @examples
#' spe_clustered <- cluster_sample(spe, sample_id = "Br6522_mid", q = 9, provided_seed = 030122)
#' @export
cluster_sample <- function(spe, sample_id, q = 9, provided_seed = 0) {
    # Subset to the specified sample
    spe_sub <- spe[, spe$sample_id == sample_id]

    # Run BayesSpace clustering
    # Set the seed for reproducibility
    if (provided_seed != 0) {
        message(paste("Setting seed to", provided_seed))
        set.seed(provided_seed)
    }
    message(paste("Running BayesSpace clustering for sample", sample_id, "with q =", q))
    spe_clustered <- spatialCluster(spe_sub, q = q, nrep = 10000, use.dimred = "HARMONY")

    return(spe_clustered)
}

# Ensure the data is in scratch
if (!file.exists(filtered_spe_path)) {
    message(paste("Failed to find", filtered_spe_path, "... copying from", filtered_spe_path_data, "..."))
    file.copy(from = filtered_spe_path_data, to = filtered_spe_path)
}

# Load the filtered spe data
message(paste("Loading spe data from", filtered_spe_path, "..."))
load(file = filtered_spe_path, verbose = TRUE)

# Check if the sample_id exists in the data
if (!sample_id %in% unique(spe$sample_id)) {
    stop(paste("Sample ID", sample_id, "not found in the data."))
}

spe_clustered <- cluster_sample(spe, sample_id, provided_seed = 030122)

# Save the clustered data
clustered_spe_path <- paste0("/zata/zippy/kresgeb/psych_screen/comparisons/spe_clustered_", sample_id, ".Rdata")
save(spe_clustered, file = clustered_spe_path)
message(paste("Saved clustered data at", clustered_spe_path))

# Function to run BayesSpace clustering on a specific sample
#' @param spe A SpatialExperiment object containing the spatial transcriptomics data.
#' @param sample_id The ID of the sample to cluster.
#' @param q The number of clusters to use for clustering.
#' @param provided_seed The seed for random number generation. 0 means no seed is set.
#' @return A SpatialExperiment object with the clustering results.
#' @examples
#' spe_clustered <- cluster_sample(spe, sample_id = "Br6522_mid", q = 9, provided_seed = 030122)
#' @export
cluster_sample <- function(spe, sample_id, q = 9, provided_seed = 0) {
    # Subset to the specified sample
    spe_sub <- spe[, spe$sample_id == sample_id]

    # Run BayesSpace clustering
    # Set the seed for reproducibility
    if (provided_seed != 0) {
        message(paste("Setting seed to", provided_seed))
        set.seed(provided_seed)
    }
    spe_clustered <- spatialCluster(spe_sub, q = q, nrep = 10000, use.dimred = "HARMONY")

    # Add spatial coordinates to the colData
    # This is necessary to ensure that the spatial coordinates are preserved in a form that can be used for visualization
    # and further analysis
    colData(spe_clustered)$pxl_col_in_fullres <- spatialCoords(spe_clustered)[, "pxl_col_in_fullres"]
    colData(spe_clustered)$pxl_row_in_fullres <- spatialCoords(spe_clustered)[, "pxl_row_in_fullres"]

    return(spe_clustered)
}
