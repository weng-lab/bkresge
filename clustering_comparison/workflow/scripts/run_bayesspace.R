suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(SpatialExperiment)
    library(BayesSpace)
    library(ggplot2)
})

# Access Snakemake input/output/log
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["output_csv"]]
output_png <- snakemake@output[["output_png"]]
log_file <- snakemake@log[[1]]

# Params
k <- as.integer(snakemake@wildcards[["k"]])
nreps <- as.integer(snakemake@wildcards[["nreps"]])
seed <- as.integer(snakemake@wildcards[["seed"]])

# Redirect stdout and stderr to log
log_con <- file(log_file, open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")

message("== BayesSpace clustering started ==")
message(paste("Start time:", Sys.time()))
message(paste("Input RData:", input_rdata))
message(paste("k:", k, "| nreps:", nreps, "| seed:", seed))
message(paste("Output CSV:", output_csv))
message(paste("Output PNG:", output_png))

# Load the RData file (should contain 'spe')
load(input_rdata)
if (!exists("spe")) stop("No object named 'spe' found in RData file.")

# Set seed
set.seed(seed)

# Run BayesSpace
message("Running BayesSpace spatialCluster...")
spe <- spatialCluster(
    spe,
    use.dimred = "PCA", # TODO: use "HARMONY" if available
    q = k,
    nrep = nreps,
)

# Write cluster assignments to CSV
message("Saving cluster assignments...")
df <- data.frame(
    barcode = colnames(spe),
    cluster = colData(spe)$spatial.cluster
)
write.csv(df, file = output_csv, row.names = FALSE)

message(paste("BayesSpace clustering finished at:", Sys.time()))

# Add spatial coordinates to the colData
# This is necessary to ensure that the spatial coordinates are preserved in a form that can be used for visualization
# and further analysis
# Row and column coordinates are somehow different from the Vitessce visualization. (Swapping pxl_row and pxl_col did not fix this)
# colData(spe)$pxl_col_in_fullres <- spatialCoords(spe)[, "pxl_row_in_fullres"]
# colData(spe)$pxl_row_in_fullres <- spatialCoords(spe)[, "pxl_col_in_fullres"]
colData(spe)$pxl_col_in_fullres <- spatialCoords(spe)[, "pxl_col_in_fullres"]
colData(spe)$pxl_row_in_fullres <- spatialCoords(spe)[, "pxl_row_in_fullres"]

# Create a figure with the clustering results
figure <- clusterPlot(spe) +
    ggtitle("BayesSpace Clustering") +
    theme(plot.title = element_text(hjust = 0.5))

# Save the figure
message("Saving figure...")
ggsave(
    filename = output_png,
    plot = figure,
    device = "png",
    width = 8,
    height = 6,
    dpi = 300,
    bg = "white"
)
message("Figure saved.")
message("== BayesSpace clustering finished ==")

# # Close log connection cleanly
sink(type = "message")
sink(type = "output")
close(log_con)
