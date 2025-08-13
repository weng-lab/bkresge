suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(zellkonverter)
})

OUT_PATH <- "/data/zusers/kresgeb/hippocampus/python_converted/spe_2.h5ad"
IN_PATH <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"

# TODO: Uncomment when ready
# load(IN_PATH, verbose = TRUE)

# if (!exists("spatial_hpc_spe")) {
#     stop("No 'spatial_hpc_spe' found in ", IN_PATH)
# }

# Convert dateImg to character for h5ad compatibility
colData(spatial_hpc_spe)$dateImg <- as.character(colData(spatial_hpc_spe)$dateImg)

sce <- as(spatial_hpc_spe, "SingleCellExperiment")

message("Adding spatial coordinates to reducedDims (obsm)...")
spatial_coords <- as.matrix(spatialCoords(spatial_hpc_spe))

# Remove dimnames to force NumPy array instead of pandas DataFrame
dimnames(spatial_coords) <- NULL

# Force integer
storage.mode(spatial_coords) <- "integer"

reducedDims(sce)[["spatial"]] <- spatial_coords
message("Spatial coordinates added!")

# Optimized hex to RGB array converter
hex_to_rgb_array <- function(ras) {
    h <- nrow(ras)
    w <- ncol(ras)
    n <- h * w

    ras_vec <- as.vector(ras)
    r_hex <- substr(ras_vec, 2, 3)
    g_hex <- substr(ras_vec, 4, 5)
    b_hex <- substr(ras_vec, 6, 7)

    hex2int <- function(hex_vec) {
        as.integer(strtoi(hex_vec, base = 16L))
    }

    r_int <- hex2int(r_hex)
    g_int <- hex2int(g_hex)
    b_int <- hex2int(b_hex)

    rgb_array <- array(0L, dim = c(h, w, 3))
    # Need to use byrow = TRUE since R is column-major by default(?!?!?!?!) Crazyyyy
    rgb_array[, , 1] <- matrix(r_int, nrow = h, ncol = w, byrow = TRUE)
    rgb_array[, , 2] <- matrix(g_int, nrow = h, ncol = w, byrow = TRUE)
    rgb_array[, , 3] <- matrix(b_int, nrow = h, ncol = w, byrow = TRUE)

    # rgb_array <- aperm(rgb_array, c(2, 1, 3)) # does a transpose, maybe wanted? need to check in downstream
    rgb_array
}

img_df <- imgData(spatial_hpc_spe)
spatial_list <- list()

for (sid in unique(img_df$sample_id)) {
    message(sprintf("Processing sample: %s", sid))
    sample_rows <- img_df[img_df$sample_id == sid, ]
    # message("Sample rows:")
    # show(sample_rows)

    images_list <- list()
    scalefactors_list <- list()
    metadata_list <- list()

    for (i in seq_len(nrow(sample_rows))) {
        message(sprintf("  Image %d/%d (image_id: %s)", i, nrow(sample_rows), sample_rows$image_id[i]))

        img_obj <- sample_rows$data[[i]]
        ras <- imgRaster(img_obj)
        message("    Raster extracted")

        rgb_array <- hex_to_rgb_array(ras)
        message("    Converted raster to RGB array")
        message(sprintf("    RGB array shape: (%d, %d, %d)", dim(rgb_array)[1], dim(rgb_array)[2], dim(rgb_array)[3]))

        images_list[[sample_rows$image_id[i]]] <- rgb_array

        sf <- sample_rows$scaleFactor[i]
        if (sample_rows$image_id[i] %in% c("hires", "lowres")) {
            scalefactors_list[[paste0("tissue_", sample_rows$image_id[i], "_scalef")]] <- sf
            message(sprintf("    Recorded scaleFactor tissue_%s_scalef = %g", sample_rows$image_id[i], sf))
        }
    }

    spatial_list[[as.character(sid)]] <- list(
        metadata = metadata_list,
        images = images_list,
        scalefactors = scalefactors_list
    )
    message(sprintf("Finished sample %s\n", sid))
}

metadata(sce)$spatial <- spatial_list

message("Writing SingleCellExperiment to H5AD...")
writeH5AD(sce, OUT_PATH)
message("Write complete!")







# suppressPackageStartupMessages({
#     library(SpatialExperiment)
#     library(zellkonverter)
# })

# OUT_PATH <- "/data/zusers/kresgeb/hippocampus/python_converted/spe.h5ad"
# IN_PATH <- "/data/zusers/kresgeb/hippocampus/R_download/spatial_hpc_spe.Rdata"


# message(sprintf("Loading %s...", IN_PATH))
# load(IN_PATH, verbose = TRUE)

# if (!exists("spatial_hpc_spe")) {
#     warning(sprintf("No 'spatial_hpc_spe' object found in %s, skipping.", IN_PATH))
#     next
# }

# # Print basic info
# message(sprintf("spatial_hpc_spe dimensions: %d genes x %d spots", nrow(spatial_hpc_spe), ncol(spatial_hpc_spe)))
# message("Assays: ", paste0(names(assays(spatial_hpc_spe)), collapse = ", "))
# message("colData columns: ", paste0(colnames(colData(spatial_hpc_spe)), collapse = ", "))
# message("rowData columns: ", paste0(colnames(rowData(spatial_hpc_spe)), collapse = ", "))

# # Have to convert dateImg to character to avoid issues with H5AD conversion
# colData(spatial_hpc_spe)$dateImg <- as.character(colData(spatial_hpc_spe)$dateImg)

# # Write to .h5ad (WARNING: strips out image and spatial coordinate [pxl_col_in_fullres, etc])
# message(sprintf("Writing to %s...", OUT_PATH))
# writeH5AD(spatial_hpc_spe, OUT_PATH)

# # Clean up
# rm(spatial_hpc_spe)
# gc()
