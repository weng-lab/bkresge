suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(zellkonverter)
    library(here)
    library(sessioninfo)
})

# --------------------------------------------------------------
# Set project root and import utilities
# --------------------------------------------------------------
my_relative_path <- "scripts/one-timers/convert_paper_data.R"
here::i_am(my_relative_path)

source(here("scripts", "utils.R"))

# --------------------------------------------------------------
# Global setup
# --------------------------------------------------------------

setup_log(prefix = "one_timers_convert_paper_data")
snapshot_script(here(my_relative_path))
log_msg("===== Session Info =====")
print(sessionInfo())
print(session_info())

snrna_seq_data_path <- "/data/zusers/kresgeb/psych_encode/spatialDLPFC_snRNAseq_fetch/2024_snRNA.RData"
snrna_converted_output_path <- here(
    "data",
    "snrna",
    "converted_snRNA_data.h5ad"
)
srt_data_path <- "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData"
srt_converted_output_path <- here("data", "converted_srt_data.h5ad")


# --------------------------------------------------------------

# Load data
load_and_rename(snrna_seq_data_path, new_names = c("snrna"), verbose = TRUE)
# load_and_rename(srt_data_path, new_names = c("srt"), verbose = TRUE)

# Convert snRNA-seq data
log_msg("Converting snRNA-seq data to AnnData format...")
# Print basic info
log_msg(sprintf(
    "snrna dimensions: %d genes x %d spots",
    nrow(snrna),
    ncol(snrna)
))
log_msg(sprintf("Assays: %s", paste0(names(assays(snrna)), collapse = ", ")))
log_msg(sprintf(
    "colData columns: %s",
    paste0(colnames(colData(snrna)), collapse = ", ")
))
log_msg(sprintf(
    "rowData columns: %s",
    paste0(colnames(rowData(snrna)), collapse = ", ")
))

log_msg(sprintf("Writing to %s...", snrna_converted_output_path))
writeH5AD(snrna, snrna_converted_output_path)
log_msg("snRNA-seq data conversion complete.")
