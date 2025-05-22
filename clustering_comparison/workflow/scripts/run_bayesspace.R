suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(SpatialExperiment)
    library(BayesSpace)
    library(ggplot2)
    library(ggspavis)
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


colData(spe)$spatial.cluster <- factor(colData(spe)$spatial.cluster)

# Create a figure with the clustering results
plot <- plotVisium(spe, annotate = "spatial.cluster") +
    ggtitle("BayesSpace clustering")

ggsave(output_png, plot, width = 6, height = 5, bg = "white")

message("Figure saved.")
message("== BayesSpace clustering finished ==")

# # Close log connection cleanly
sink(type = "message")
sink(type = "output")
close(log_con)
