library("SpatialExperiment")
library("BayesSpace")
library("ggplot2")
library("parallel")
library("sessioninfo")

# Define file paths
cluster_spe_path <- "/zata/zippy/kresgeb/psych_screen/output/bayes_space/spe_clusterd_unique_rownames.Rdata"
base_chain_path <- "/zata/zippy/kresgeb/psych_screen/output/bayes_space/jitter/"

# ~10mins on sample_spe with 64L cores and 2500 nrep on z013
# ~xmins on sample_spe with 64L cores and 2500 nrep on slurm

args <- commandArgs(trailingOnly = TRUE)

# Expecting one argument: jitter_scale
if (length(args) < 1) {
    stop("Usage: Rscript your_script.R <jitter_scale>")
} else {
    message(paste("jitter-scale:", args))
}

jitter_scale <- as.numeric(args[1])

Sys.time()

message(paste("Loading cluster data from:", cluster_spe_path, "..."))
load(file = cluster_spe_path, verbose = TRUE)

# spe <- sample_spe

Sys.time()

colData(spe)$pxl_col_in_fullres <- spatialCoords(spe)[, "pxl_col_in_fullres"]
colData(spe)$pxl_row_in_fullres <- spatialCoords(spe)[, "pxl_row_in_fullres"]

spe.enhanced <- spatialEnhance(spe, init = colData(spe)$spatial.cluster, q = 9, use.dimred = "HARMONY", cores = 64L, verbose = TRUE, jitter.scale = jitter_scale, save.chain = TRUE, nrep = 200, burn.in = 100)

chain <- mcmcChain(spe.enhanced, "Ychange")

chain_path <- paste0(base_chain_path, "chain_", jitter_scale, ".Rdata")
message(paste("Saving chain to", chain_path, "..."))
save(chain, file = chain_path)

Sys.time()

## Reproducibility information
print("Reproducibility information:")
Sys.time()
proc.time()
options(width = 120)
session_info()
