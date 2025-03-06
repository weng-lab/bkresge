spe <- spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")

zellkonverter::writeH5AD(spe, "./full_visium.h5ad")
