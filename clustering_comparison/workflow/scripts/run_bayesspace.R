suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(SpatialExperiment)
    library(BayesSpace)
})

# Access Snakemake input/output/log
input_rdata <- snakemake@input[["rdata"]]
output_csv <- snakemake@output[["output_csv"]]
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

# # Close log connection cleanly
sink(type = "message")
sink(type = "output")
close(log_con)
