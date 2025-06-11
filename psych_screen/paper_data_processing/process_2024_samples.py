# Load all the 2024 paper samples, create the config files and process and structure the data


import os
import pandas as pd
import squidpy as sq
import scanpy as sc
import multiprocessing
import warnings
import scipy
import json
import numpy as np
from vitessce.data_utils import (
    to_diamond,
    rgb_img_to_ome_zarr,
    optimize_adata,
)

SPACERANGER_SOURCE_DIR = (
    "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger"
)
OUTPUT_DIR = "/zata/public_html/users/kresgeb/psych_encode/spatialDLPFC"
TEMPLATE_CONFIG_PATH = "/zata/zippy/kresgeb/psych_screen/paper_data_processing/template_configs/template_config_2024.json"
BAYESSPACE_CLUSTERS_DIR = "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rdata/spe/clustering_results"
WHITELIST_PATH = "/zata/zippy/kresgeb/psych_screen/paper_data_processing/whitelist.txt"
FULL_VISIUM_PATH = (
    "/zata/zippy/kresgeb/psych_screen/paper_data_processing/full_visium.h5ad"
) # ZellKonverter of spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")
COLOR_DATA_PATH = (
    "/zata/zippy/kresgeb/psych_screen/paper_data_processing/colors/k16_like_manual.json"
)

# Suppress the specific UserWarnings about unique names
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
)
# Suppress the specific UserWarnings about unique names
warnings.filterwarnings(
    "ignore",
    message="Observation names are not unique. To make them unique, call `.obs_names_make_unique`.",
)


def main():
    # all subdirectories in the source directory (exclude the names.txt)
    sample_names = [
        entry.name for entry in os.scandir(SPACERANGER_SOURCE_DIR) if entry.is_dir()
    ]

    # Make all directories if they do not exist
    for sample_name in sample_names:
        os.makedirs(name=os.path.join(
            OUTPUT_DIR, "data", sample_name), exist_ok=True)
        os.makedirs(
            name=os.path.join(OUTPUT_DIR, "configs", sample_name), exist_ok=True
        )

    pool = multiprocessing.Pool(processes=min(
        os.cpu_count() / 2, len(sample_names)))
    pool.map(process_sample, sample_names)

    # Close the pool to free resources
    pool.close()
    pool.join()


# Loosely based on https://github.com/vitessce/vitessce-python/blob/main/demos/human-lymph-node-10x-visium/src/create_zarr.py
def process_sample(sample_name):
    data_output_path = os.path.join(
        OUTPUT_DIR, "data", sample_name, "data.h5ad.zarr")
    image_output_path = os.path.join(
        OUTPUT_DIR, "data", sample_name, "image.ome.zarr")
    source_outs_path = os.path.join(
        SPACERANGER_SOURCE_DIR, sample_name, "outs")

    adata = sq.read.visium(source_outs_path)
    adata.var_names_make_unique()

    # Retrieve the data from the R pipeline used in the paper
    full_visium = sc.read_h5ad(FULL_VISIUM_PATH)
    # The shortened name, ex. Br8667_mid
    sample_id = "_".join(sample_name.split("_")[1:3])
    sample_visium = full_visium[full_visium.obs["sample_id"] == sample_id]

    # # Calculate QC metrics
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # # Perform basic filtering (much more generous than source)
    # sc.pp.filter_cells(adata, min_genes=100)
    # sc.pp.filter_genes(adata, min_cells=10)
    # adata = adata[adata.obs["pct_counts_mt"] < 30]

    # Remove any remaining spots that the R pipeline in the paper (scran discard) says should be removed
    # NOTE: apparently this still leaves spots that do not have a cluster assigned to them, somehow...
    discard_spots = sample_visium.obs[
        sample_visium.obs["scran_discard"] == "TRUE"
    ].index
    adata = adata[~adata.obs.index.isin(discard_spots)]

    # Clustering
    add_cluster_data(adata, sample_name, k_tuple=(9, 16, 28))

    # Remove all spots that do not have an assigned cluster in BayesSpace k=9
    spots_missing_cluster_data = adata.obs[pd.isna(
        adata.obs["bayes_space_k=9"])].index
    adata = adata[~adata.obs.index.isin(spots_missing_cluster_data)]

    # Perform normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    # Determine the top 100 highly variable genes.
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=100)

    # Genes of Interest are all the highly variable genes and any additional ones in the whitelist provided
    set_genes_of_interest(adata, WHITELIST_PATH)

    # Dimensionality reduction
    sc.pp.pca(adata, mask_var="genes_of_interest")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # Add manual layers if they exist, store whether it exists or not
    has_manual_layers = add_manual_layers(adata, sample_visium)

    # If there is manual layer data, make sure no Nones/Null make it through (they break the visualization)
    if has_manual_layers:
        spots_missing_manual_layer_data = adata.obs[pd.isna(
            adata.obs["manual_layers"])].index
        adata = adata[~adata.obs.index.isin(spots_missing_manual_layer_data)]

    # Hierarchical clustering of genes for optimal gene ordering
    X_goi_arr = adata[:, adata.var["genes_of_interest"]].X.toarray()
    X_goi_index = adata[:, adata.var["genes_of_interest"]].var.copy().index
    Z = scipy.cluster.hierarchy.linkage(
        X_goi_arr.T, method="average", optimal_ordering=True
    )

    # Get the hierarchy-based ordering of genes.
    num_cells = adata.obs.shape[0]
    goi_index_ordering = scipy.cluster.hierarchy.leaves_list(Z)
    genes_of_interest = X_goi_index.values[goi_index_ordering].tolist()
    all_genes = adata.var.index.values.tolist()
    not_goi = adata.var.loc[~adata.var["genes_of_interest"]
                            ].index.values.tolist()

    def get_orig_index(gene_id):
        return all_genes.index(gene_id)

    var_index_ordering = list(map(get_orig_index, genes_of_interest)) + list(
        map(get_orig_index, not_goi)
    )

    # Create a new *ordered* gene expression dataframe.
    adata = adata[:, var_index_ordering].copy()
    adata.obsm["X_goi"] = adata[:, adata.var["genes_of_interest"]].X.copy()

    # Scale the spatial data to align with the image
    scale_factor = get_scale_factor(sample_name)
    adata.obsm["spatial"] = adata.obsm["spatial"] * scale_factor

    # Create the diamond visualizations for the spots
    adata.obsm["segmentations"] = np.zeros((num_cells, 4, 2))
    radius = 7
    for i in range(num_cells):
        adata.obsm["segmentations"][i, :, :] = to_diamond(
            adata.obsm["spatial"][i, 0], adata.obsm["spatial"][i, 1], radius
        )

    # Write img_arr to OME-Zarr.
    # Need to convert images from interleaved to non-interleaved (color axis should be first).
    img_hires = adata.uns["spatial"][sample_name]["images"]["hires"]
    img_arr = np.transpose(img_hires, (2, 0, 1))
    rgb_img_to_ome_zarr(
        img_arr,
        image_output_path,
        axes="cyx",
        chunks=(1, 256, 256),
        img_name="H & E Image",
    )

    # Optimize and write anndata
    adata = optimize_adata(
        adata,
        obs_cols=(["manual_layers"] if has_manual_layers else [])
        + ["bayes_space_k=9", "bayes_space_k=16", "bayes_space_k=28"],
        var_cols=["highly_variable", "genes_of_interest"],
        obsm_keys=["X_goi", "spatial", "segmentations", "X_umap", "X_pca"],
        optimize_X=True,
        # Vitessce plays nicely with dense matrices saved with chunking
        to_dense_X=True,
    )
    adata.write_zarr(data_output_path, chunks=[adata.shape[0], 10])

    # Create the config files from the template
    create_configuration_file(sample_name, has_manual_layers)



def create_configuration_file(sample_name, has_manual_layers=False):
    for suffix in ["", "_single_column"]:
        # Adjust template and output paths
        template_path = TEMPLATE_CONFIG_PATH
        output_file_name = "config.json"

        if suffix:
            template_path = TEMPLATE_CONFIG_PATH.replace(".json", f"{suffix}.json")
            output_file_name = f"config{suffix}.json"

        output_file_path = os.path.join(
            OUTPUT_DIR, "configs", sample_name, output_file_name
        )

        # Load the template
        with open(template_path, "r") as f:
            data = json.load(f)

        # Replace <<Sample_Name>> with the actual sample name
        data_str = json.dumps(data)
        data_str = data_str.replace("<<Sample_Name>>", sample_name)
        data = json.loads(data_str)

        if not has_manual_layers:
            # Remove "Manually Annotated Layers" from obsSets
            datasets = data.get("datasets", [])
            for dataset in datasets:
                for file in dataset.get("files", []):
                    options = file.get("options", {})
                    obs_sets = options.get("obsSets", [])
                    options["obsSets"] = [
                        entry for entry in obs_sets if entry["name"] != "Manually Annotated Layers"
                    ]

        data = add_color_data(data)

        # Write the updated config
        with open(output_file_path, "w") as file:
            json.dump(data, file, indent=2)


def hex_to_rgb(hex_color):
    """Converts hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))


def add_color_data(config_data):
    """
    Fills the 'obsSetColor' section in the config file from the color palette file.

    :param config_path: Path to the config file to be updated.
    :return: Updated config data with filled 'obsSetColor' section.
    """

    # Load the sets color file
    with open(COLOR_DATA_PATH, "r") as sets_file:
        sets_data = json.load(sets_file)

    # Initialize the 'obsSetColor' structure
    obs_set_color = {"A": []}

    # Iterate through each set in the sets color file
    for set_entry in sets_data["sets"]:
        set_name = set_entry["setName"]
        for color_entry in set_entry["colors"]:
            label = color_entry["label"]
            hex_color = color_entry["hex"]
            rgb_color = hex_to_rgb(hex_color)

            # Build the path and color entry for the 'obsSetColor'
            path = [set_name]
            if label:
                path.append(label)

            color_entry = {"path": path, "color": rgb_color}

            # Append to the appropriate place in obsSetColor
            obs_set_color["A"].append(color_entry)

    # Fill the 'obsSetColor' section of the config data
    config_data["coordinationSpace"]["obsSetColor"] = obs_set_color

    return config_data


# Returns True if manual layer data exists
def add_manual_layers(adata, sample_visium):

    manual_layers = sample_visium.obs["manual_layer_label"]

    if manual_layers.notna().any():
        # Replace 'Layer X' with 'LX', but keep 'WM' unchanged
        rename_dict = {f"Layer {i}": f"L{i}" for i in range(1, 7)}
        manual_layers = manual_layers.cat.rename_categories(rename_dict)
        adata.obs["manual_layers"] = adata.obs_names.map(manual_layers)
        return True
    else:
        return False


# TODO improve efficiency, currently opens and searches the clusters.csv once for EACH sample--despite having the data for ALL samples
# This compounds for more entries in k_tuple
# TODO change to save under one obs entry (bayes_space) with multiple columns where each column is a resolution (simpler and MAY be better for performance/compression?)
def add_cluster_data(adata, sample_name, k_tuple=(9,)):
    # The shortened name that exists in the clustering results csv
    # Ex. Br8667_mid
    sample_id = "_".join(sample_name.split("_")[1:3])

    for k in k_tuple:
        # Read the data from clusters.csv
        clusters_path = os.path.join(
            BAYESSPACE_CLUSTERS_DIR, f"bayesSpace_harmony_{k}", "clusters.csv"
        )
        full_cluster_data = pd.read_csv(clusters_path)

        # Extract only the relevant data
        filtered_cluster_data = full_cluster_data[
            full_cluster_data["key"].str.contains(sample_id)
        ].copy()
        filtered_cluster_data.loc[:, "cell_id"] = filtered_cluster_data["key"].apply(
            lambda x: x.split("_")[0]
        )
        filtered_cluster_data = filtered_cluster_data[["cell_id", "cluster"]]
        filtered_cluster_data.set_index("cell_id", inplace=True)

        # Convert to format as found in leiden
        filtered_cluster_data = (
            filtered_cluster_data.astype(int).astype(str).astype("category")
        )

        # Rename the clusters to something more easy for user to understand based on rename_dict
        rename_dict = {
            str(i): f"Spatial Domain {int(k):02d}D{int(i):02d}" for i in range(1, k + 1)
        }
        # uncomment me for renaming
        # filtered_cluster_data["cluster"] = filtered_cluster_data[
        #     "cluster"
        # ].cat.rename_categories(rename_dict)

        # Add the data to the AnnData object
        adata.obs[f"bayes_space_k={k}"] = adata.obs_names.map(
            filtered_cluster_data["cluster"]
        )


def filter_by_scran(adata, sample_visium):
    pass


def set_genes_of_interest(adata, whitelist_path):

    if "highly_variable" in adata.var:
        adata.var["genes_of_interest"] = adata.var["highly_variable"].copy()
    else:
        adata.var["genes_of_interest"] = pd.Series(
            False, index=adata.var.index)
    with open(whitelist_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#") and line in adata.var.index:
                adata.var.loc[line, "genes_of_interest"] = True


def get_scale_factor(sample_name):
    json_path = os.path.join(
        SPACERANGER_SOURCE_DIR, sample_name, "outs", "spatial", "scalefactors_json.json"
    )
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("tissue_hires_scalef")


if __name__ == "__main__":
    main()
