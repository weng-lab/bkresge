spe <- spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")

zellkonverter::writeH5AD(spe, "/zata/zippy/kresgeb/psych_screen/paper_data_processing/full_visium.h5ad")
