library("SpatialExperiment")
library("BayesSpace")
library("ggplot2")

# Define file paths
my_cluster_path <- "/zata/zippy/kresgeb/psych_screen/comparisons/spe_clustered.Rdata"
paper_cluster_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rdata/spe/clustering_results/bayesSpace_harmony_9/clusters.csv"

# Load in my cluster data
load(my_cluster_path, verbose = TRUE) # spe is the object

# Load in the paper's cluster data
paper_clusters <- read.csv(paper_cluster_path, row.names = 1)


spe_keys <- spe$key


# Match spots present in both datasets (Should be all of them)
common_keys <- intersect(spe_keys, rownames(paper_clusters))

# Extract corresponding clusters
my_clusters <- spe$spatial.cluster[match(common_keys, spe_keys)]
paper_clusters_subset <- paper_clusters[common_keys, "cluster"]

# Create contingency table
cluster_comparison_table <- table(Paper = paper_clusters_subset, Mine = my_clusters)

# Print the table
print(cluster_comparison_table)

# Calculate the adjusted Rand index
