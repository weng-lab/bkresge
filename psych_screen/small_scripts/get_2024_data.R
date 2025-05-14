library("SpatialExperiment")
spe <- spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")

save(spe, file = "/data/zusers/kresgeb/psych_encode/spatialLIBD_fetch_data/2024.RData")
