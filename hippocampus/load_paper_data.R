library(SummarizedExperiment)
library(SpatialExperiment)
library(humanHippocampus2024)
library(ExperimentHub)

spe_output_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"
snrna_seq_output_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"
message("Loading from files...")
load(spe_output_path, verbose = TRUE)
load(snrna_seq_output_path, verbose = TRUE)
