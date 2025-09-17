suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(zellkonverter)
})

# OUT_PATH <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/srt.h5ad"
OUT_PATH <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/srt.h5ad"
# IN_PATH <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/proj_srt.rda"
IN_PATH <- "/zata/zippy/kresgeb/hippocampus/my_output/nmf/2024_dlpfc/proj_srt.rda"

load(IN_PATH, verbose = TRUE)

if (!exists("srt")) {
    stop("No 'srt' found in ", IN_PATH)
}


# Convert dateImg to character for h5ad compatibility, only needed for HPC???
if ("dateImg" %in% colnames(colData(srt))) {
    colData(srt)$dateImg <- as.character(colData(srt)$dateImg)
} else {
    message("No 'dateImg' column found in colData(srt); skipping conversion.")
}

sce <- as(srt, "SingleCellExperiment")

message("Adding spatial coordinates to reducedDims (obsm)...")
spatial_coords <- as.matrix(spatialCoords(srt))

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

img_df <- imgData(srt)
spatial_list <- list()

total_samples <- length(unique(img_df$sample_id))
sample_idx <- 0

for (sid in unique(img_df$sample_id)) {
    sample_idx <- sample_idx + 1
    message(sprintf("Processing sample: %s", sid))
    sample_rows <- img_df[img_df$sample_id == sid, ]

    images_list <- list()
    scalefactors_list <- list()
    metadata_list <- list()

    for (i in seq_len(nrow(sample_rows))) {
        message(sprintf(
            "  Image %d/%d (image_id: %s)",
            i, nrow(sample_rows), sample_rows$image_id[i]
        ))

        img_obj <- sample_rows$data[[i]]
        ras <- imgRaster(img_obj)
        message("    Raster extracted")

        rgb_array <- hex_to_rgb_array(ras)
        message("    Converted raster to RGB array")
        message(sprintf(
            "    RGB array shape: (%d, %d, %d)",
            dim(rgb_array)[1], dim(rgb_array)[2], dim(rgb_array)[3]
        ))

        images_list[[sample_rows$image_id[i]]] <- rgb_array

        sf <- sample_rows$scaleFactor[i]
        if (sample_rows$image_id[i] %in% c("hires", "lowres")) {
            scalefactors_list[[paste0("tissue_", sample_rows$image_id[i], "_scalef")]] <- sf
            message(sprintf(
                "    Recorded scaleFactor tissue_%s_scalef = %g",
                sample_rows$image_id[i], sf
            ))
        }
    }

    spatial_list[[as.character(sid)]] <- list(
        metadata = metadata_list,
        images = images_list,
        scalefactors = scalefactors_list
    )

    pct_done <- (sample_idx / total_samples) * 100
    message(sprintf(
        "Finished sample %s (%d/%d processed, %.2f%% done)\n",
        sid, sample_idx, total_samples, pct_done
    ))
}
# This will add to the .uns in the AnnData specifically .uns["spatial"]
metadata(sce)$spatial <- spatial_list

message("Writing SingleCellExperiment to H5AD...")
writeH5AD(sce, OUT_PATH)
message("Write complete!")
