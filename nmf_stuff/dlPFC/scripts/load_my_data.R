library(SummarizedExperiment)
library(SpatialExperiment)
library(humanHippocampus2024)
library(ExperimentHub)

# Paths
dlpfc_spe_path <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/srt_with_nmf.rda"
dlpfc_sce_path <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/snrna_with_nmf.rda"

message("Loading from files...")

# Function to load and rename
load_and_rename <- function(path, new_name) {
  obj_names <- load(path, verbose = TRUE)
  if (length(obj_names) != 1) {
    stop(paste("Expected 1 object in", path, "but got", length(obj_names)))
  }
  assign(new_name, get(obj_names), envir = .GlobalEnv)
  rm(list = obj_names, envir = .GlobalEnv) # clean up original name
}

# Load each dataset with standardized names
load_and_rename(dlpfc_spe_path, "spe")
load_and_rename(dlpfc_sce_path, "sce")
