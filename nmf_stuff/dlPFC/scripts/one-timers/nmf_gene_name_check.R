suppressPackageStartupMessages({
    library(here)
    library(RcppML)
    library(SingleCellExperiment)
    library(sessioninfo)
    library(duckplyr)
    library(readr)
    library(lubridate)
})
print(sessionInfo())
print(session_info())

x <- readRDS(
    "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/nmf_k10_seed42_tol1e-05_L10.0.rds"
)

print(head(rownames(x@w), 50))
