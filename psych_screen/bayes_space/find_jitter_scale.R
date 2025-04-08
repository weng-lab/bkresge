library("SpatialExperiment")
library("BayesSpace")
library("ggplot2")
library("parallel")
library("sessioninfo")

# Define file paths
full_spe_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rdata/spe/01_build_spe/spe_filtered_final.Rdata"
sample_spe_path <- "/zata/zippy/kresgeb/scratch/sample_spe.Rdata"
tuning_plot_path <- "/zata/zippy/kresgeb/psych_screen/output/qTune_plot.png"
cluster_plot_path <- "/zata/zippy/kresgeb/psych_screen/output/cPlot.jpg"
enhanced_cluster_plot_path <- "/zata/zippy/kresgeb/psych_screen/output/ecPlot2.jpg"
cluster_spe_path <- "/zata/zippy/kresgeb/scratch/cluster_sample_spe.Rdata"
enhanced_spe_path <- "/zata/zippy/kresgeb/scratch/enhanced_cluster_sample_spe2.Rdata"

Sys.time()

message(paste("Loading cluster data from:", cluster_spe_path, "..."))
load(file = cluster_spe_path, verbose = TRUE)

enhanced_spe <- spatialEnhance(sample_spe, init = colData(sample_spe)$spatial.cluster, q = 9, use.dimred = "HARMONY", cores = 64L, verbose = TRUE, jitter.scale = 0.75, save.chain = TRUE, nrep = 2500, burn.in = 100)

message(mcmcChain(enhanced_spe, "Ychange"))

Sys.time()
