library(SummarizedExperiment)
library(SpatialExperiment)
library(humanHippocampus2024)
library(ExperimentHub)

spe_output_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"
snrna_seq_output_path <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_snrna_seq.Rdata"

#>     cache
ehub <- ExperimentHub()

## Load the datasets of the package
myfiles <- query(ehub, "humanHippocampus2024")

spatial_hpc_spe <- myfiles[["EH9605"]]
message("Loaded spatial_hpc_spe from ExperimentHub")
spatial_hpc_snrna_seq <- myfiles[["EH9606"]]
message("Loaded spatial_hpc_snrna_seq from ExperimentHub")

# Save the loaded data to an RData file
message("Saving spatial_hpc_spe and spatial_hpc_snrna_seq to RData files...")
save(spatial_hpc_spe, spatial_hpc_snrna_seq, file = spe_output_path)
message(paste("spe data saved to", spe_output_path))
save(spatial_hpc_snrna_seq, file = snrna_seq_output_path)
message(paste("snrna_seq data saved to", snrna_seq_output_path))
