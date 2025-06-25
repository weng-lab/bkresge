suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(zellkonverter)
})

OUT_DIR <- "/zata/zippy/kresgeb/psych_screen/paper_data_processing/paper_data"
IN_DIR <- "/zata/public_html/users/kresgeb/cluster-comparisons/paper_data"
YEARS <- c("2021", "2024")

# NOTE: can also grab from spe <- spatialLIBD::fetch_data(type = "spatialDLPFC_Visium") for 2024 data and spe <- spatialLIBD::fetch_data(type = "spe") for 2021 data

# Ensure output directory exists
if (!dir.exists(OUT_DIR)) {
    dir.create(OUT_DIR, recursive = TRUE)
}

for (year in YEARS) {
    rdata_path <- file.path(IN_DIR, paste0(year, ".RData"))
    message(sprintf("Loading %s...", rdata_path))
    load(rdata_path, verbose = TRUE) # assumes 'spe' object is loaded

    if (!exists("spe")) {
        warning(sprintf("No 'spe' object found in %s, skipping.", rdata_path))
        next
    }

    # Print basic info
    message(sprintf("spe dimensions: %d genes x %d spots", nrow(spe), ncol(spe)))
    message("Assays: ", paste0(names(assays(spe)), collapse = ", "))
    message("colData columns: ", paste0(colnames(colData(spe)), collapse = ", "))
    message("rowData columns: ", paste0(colnames(rowData(spe)), collapse = ", "))

    # Write to .h5ad
    h5ad_path <- file.path(OUT_DIR, paste0(year, ".h5ad"))
    message(sprintf("Writing to %s...", h5ad_path))
    writeH5AD(spe, h5ad_path)

    # Clean up
    rm(spe)
    gc()
}
