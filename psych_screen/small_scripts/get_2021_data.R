library("SpatialExperiment")
spe <- spatialLIBD::fetch_data(type = "spe")

save(spe, file = "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2021.RData")
