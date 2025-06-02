import squidpy as sq
import os
import warnings
import json
from vitessce.data_utils import (
    to_diamond,
    rgb_img_to_ome_zarr,
    optimize_adata,
)
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress the specific UserWarnings about unique names
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
)

TEMPLATE_CONFIG_PATH = "/zata/zippy/kresgeb/clustering_comparison/resources/template_config.json"
MANIFEST_JSON_PATH = "/zata/zippy/kresgeb/clustering_comparison/resources/test_vitessce_manifest.json"

def save_adata(adata, output_path):
        # Optimize and write anndata
    adata = optimize_adata(
        adata,
        obsm_keys = ['spatial', 'segmentations'],
        optimize_X=True,
        # Vitessce plays nicely with dense matrices saved with chunking
        to_dense_X=True,
    )
    adata.write_zarr(output_path, chunks=[adata.shape[0], 10])


def save_image(adata, image_output_path):

    sample_key = next(iter(adata.uns["spatial"]))
    # Need to convert images from interleaved to non-interleaved (color axis should be first).
    img_hires = adata.uns["spatial"][sample_key]["images"]["hires"]
    img_arr = np.transpose(img_hires, (2, 0, 1))
    rgb_img_to_ome_zarr(
        img_arr,
        image_output_path,
        axes="cyx",
        chunks=(1, 256, 256),
        img_name="H & E Image",
    )

def create_segmentations(sample_path, adata):

    # Scale the spatial data to align with the image
    scale_factor = get_scale_factor(sample_path)
    adata.obsm["spatial"] = adata.obsm["spatial"] * scale_factor

    # Create the diamond visualizations for the spots
    num_cells = adata.obs.shape[0] #TODO: should filter out any non-assigned cells in an earlier step! (compare against cluster assignment csv entries)
    adata.obsm["segmentations"] = np.zeros((num_cells, 4, 2))
    radius = 7
    for i in range(num_cells):
        adata.obsm["segmentations"][i, :, :] = to_diamond(
            adata.obsm["spatial"][i, 0], adata.obsm["spatial"][i, 1], radius
        )
    return adata

def load_data(sample_name, year):
    source_dir = "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger" if year == 2024 else "/data/zusers/kresgeb/psych_encode/HumanPilot10X/reorganized"
    
    # Load the AnnData object from the folder in the source directory that contains the sample_name
    # (!!! ASSUMES ONLY ONE FOLDER MATCHES !!!)
    sample_folder = [f for f in os.listdir(source_dir) if sample_name in f]
    assert len(sample_folder) == 1, f"Expected one folder match for {sample_name}, got {len(sample_folder)}"

    sample_path = os.path.join(source_dir, sample_folder[0])
    outs_folder = os.path.join(sample_path, "outs")

    adata = sq.read.visium(outs_folder)
    adata.var_names_make_unique()

    return sample_path, adata

def get_scale_factor(sample_path):
    json_path = os.path.join(
        sample_path, "outs", "spatial", "scalefactors_json.json"
    )
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("tissue_hires_scalef")

def add_cluster_assignments(adata, cluster_assignment_path, column_name):

    # Load cluster assignments
    cluster_df = pd.read_csv(cluster_assignment_path)

    # Check if 'barcode' column exists
    if 'barcode' not in cluster_df.columns:
        raise ValueError("CSV must contain a 'barcode' column.")

    # Determine cluster assignment column
    cluster_columns = [col for col in cluster_df.columns if col != 'barcode']
    if len(cluster_columns) != 1:
        raise ValueError("CSV must contain exactly one cluster assignment column besides 'barcode'.")
    
    cluster_col = cluster_columns[0]
    if cluster_col != 'cluster':
        print(f"Note: Using '{cluster_col}' as the cluster assignment column instead of 'cluster'.")

    # Set index to barcode
    cluster_df = cluster_df.set_index('barcode')

    # Check barcodes in adata
    original_n = adata.n_obs
    common_barcodes = adata.obs_names.intersection(cluster_df.index)
    removed_n = original_n - len(common_barcodes)

    if removed_n > 0:
        print(f"Removed {removed_n} spots not found in the cluster assignment file.")

    # Subset adata
    adata = adata[common_barcodes].copy()

    # Assign the cluster labels
    adata.obs[column_name] = cluster_df.loc[adata.obs_names, cluster_col].astype(str)

    return adata


def add_view_to_config(config_json, view_title, column_name, adata):

     # Add to obsSets
    obs_set_entry = {
        "name": view_title,
        "path": f"obs/{column_name}"
    }
    config_json["datasets"][0]["files"][0]["options"]["obsSets"].append(obs_set_entry)

    # Add to obsSetSelection
    unique_values = sorted(map(str, adata.obs[column_name].dropna().unique()))
    selection_list = [[view_title, val] for val in unique_values]
    config_json["coordinationSpace"]["obsSetSelection"][view_title] = selection_list

    # Add to layout 
    view_index = len(config_json["layout"])
    layout_entry = {
        "component": "spatial",
        "props": {
            "title": view_title
        },
        "coordinationScopes": {
            "obsType": "A",
            "spatialImageLayer": "A",
            "spatialSegmentationLayer": "A",
            "spatialZoom": "A",
            "spatialTargetX": "A",
            "spatialTargetY": "A",
            "obsColorEncoding": "A",
            "obsSetSelection": view_title
        },
        "x": 0,
        "y": view_index * 2,  # stack vertically
        "w": 6,
        "h": 2
    }
    config_json["layout"].append(layout_entry)

    return config_json


def create_screen(screen_json):

    # Open the template config file
    with open(TEMPLATE_CONFIG_PATH, "r") as f:
        config_json = json.load(f)

    # Load sample adata
    sample_name = screen_json["sample"]
    year = screen_json["year"]
    output_dir = screen_json["outputDir"]
    sample_path, adata = load_data(sample_name, year)

    # For each view in screen_json...
    for view in screen_json["views"]:
        view_title = view["title"]
        cluster_assignment_path = view["clusterAssignmentPath"]
        column_name = view["columnName"]

        # Add cluster assignments to adata.obs
        adata = add_cluster_assignments(adata, cluster_assignment_path, column_name)

        # Add view to config
        config_json = add_view_to_config(config_json, view_title, column_name, adata)

    # Create segmentations
    adata = create_segmentations(sample_path, adata)

    # Save adata
    adata_path = Path(output_dir) / "adata.h5ad.zarr"
    save_adata(adata, adata_path)

    # Save image.ome.zarr
    image_path = Path(output_dir) / "image.ome.zarr"
    save_image(adata, image_path)

     # Fill in URLs in config using users.wenglab.org base
    try:
        url_suffix = str(output_dir).split("/users/", 1)[1]
    except IndexError:
        raise ValueError(f"Expected '/users/' in outputDir path: {output_dir}")

    base_url = f"https://users.wenglab.org/{url_suffix}"
    config_json["datasets"][0]["files"][0]["url"] = f"{base_url}/adata.h5ad.zarr"
    config_json["datasets"][0]["files"][1]["url"] = f"{base_url}/image.ome.zarr"

    # Save config
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"Created screen in {output_dir} with config at {config_path}. (URL to config: {base_url}/config.json)")


def main():
    # Load the manifest json
    with open(MANIFEST_JSON_PATH, 'r') as f:
        manifest_json = json.load(f)
    
    # For each screen in the manifest, call create_screen
    for screen in manifest_json.get("allScreens", []):
        print(f"Creating screen: {screen['name']}")
        create_screen(screen)

if __name__ == "__main__":
    main()
