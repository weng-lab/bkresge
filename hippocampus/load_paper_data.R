library(SummarizedExperiment)
library(SpatialExperiment)
library(humanHippocampus2024)
library(ExperimentHub)

# Paths
hpc_spe_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"
hpc_sce_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"
dlpfc_spe_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
dlpfc_sce_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"

message("Loading from files...")

# Function to load and rename
load_and_rename <- function(path, new_name) {
  obj_names <- load(path, verbose = TRUE)
  if (length(obj_names) != 1) {
    stop(paste("Expected 1 object in", path, "but got", length(obj_names)))
  }
  assign(new_name, get(obj_names), envir = .GlobalEnv)
  rm(list = obj_names, envir = .GlobalEnv)  # clean up original name
}

# Load each dataset with standardized names
load_and_rename(hpc_spe_path, "hpc_spe")
load_and_rename(hpc_sce_path, "hpc_sce")
load_and_rename(dlpfc_spe_path, "dlpfc_spe")
load_and_rename(dlpfc_sce_path, "dlpfc_sce")
