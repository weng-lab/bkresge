library("BayesSpace")
library("SpatialExperiment")
library("ggplot2")


# Load spe with BayesSpace clustering
load("/zata/zippy/kresgeb/psych_screen/comparisons/spe_clustered_Br6522_mid.Rdata", verbose = TRUE)

# visualize the clustering results using clusterPlot function
clusterPlot(spe_clustered) +
    ggtitle("BayesSpace Clustering") +
    theme(plot.title = element_text(hjust = 0.5))
