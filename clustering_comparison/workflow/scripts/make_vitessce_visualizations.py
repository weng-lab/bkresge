from datetime import datetime
import math
import sys
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

# Redirect stdout and stderr to the log file
log_file = open(snakemake.log[0], "w", buffering=1)  # line-buffered
sys.stdout = log_file
sys.stderr = log_file

TEMPLATE_CONFIG_PATH = snakemake.input["template"]  # Path to the template config file
MANIFEST_JSON_PATH = snakemake.input["manifest"]  # Path to the manifest JSON file

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
    num_cells = adata.obs.shape[0] # should filter out any non-assigned cells in an earlier step! (compare against cluster assignment csv entries)
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

def add_cluster_assignments(adata, cluster_assignment_path, column_name, source_column_name='cluster'):
    # Load cluster assignments
    cluster_df = pd.read_csv(cluster_assignment_path)

    # Check if 'barcode' column exists
    if 'barcode' not in cluster_df.columns:
        raise ValueError("CSV must contain a 'barcode' column.")

    if source_column_name not in cluster_df.columns:
        if len(cluster_df.columns) == 2:
            new_source_column_name = cluster_df.columns[1]
            print(f"CSV located at {cluster_assignment_path} does not contain the requested column:'{source_column_name}',\nbut it contains only one non-barcode column named '{new_source_column_name}'. Using that instead.")
            source_column_name = new_source_column_name
        else:
            raise ValueError(f"CSV located at {cluster_assignment_path} does not contain the requested column:'{source_column_name}'.\n    Since there is more than one non-barcode column, please specify the correct column name to use for cluster assignments.\n    Options are: {', '.join(cluster_df.columns[1:])}")
        

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
    adata.obs[column_name] = cluster_df.loc[adata.obs_names, source_column_name].astype(str)

    return adata


def add_view_to_config(config_json, view_title, column_name, adata, grid_layout):

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
        **grid_layout[view_index]  # Use the grid layout for x, y, w, h
    }
    config_json["layout"].append(layout_entry)

    return config_json

def create_grid_layout(num_views):
    # create a structure that stores x, y, w, h relative to a grid for each view_index to be used elsewhere
    layout = []

    if num_views == 0:
        return layout
    
    if num_views > 144:
        raise ValueError("Number of views exceeds the maximum grid size of 12x12 (144 views), attempted to create a grid layout for {num_views} views.")

    best_config = None
    best_score = float('inf')

    for rows in range(1, num_views + 1):
        cols = math.ceil(num_views / rows)
        if rows > 12 or cols > 12:
            continue

        view_w = 12 // cols
        view_h = 12 // rows

        total_cells_used = view_w * view_h * num_views
        unused_space = 144 - total_cells_used
        aspect_ratio_diff = abs(view_w - view_h)

        # Score = space wasted + penalty for squashed/stretch views
        score = unused_space + (aspect_ratio_diff * 50)

        if score < best_score:
            best_score = score
            best_config = (rows, cols, view_w, view_h)

    rows, cols, view_w, view_h = best_config

    for idx in range(num_views):
        row = idx // cols
        col = idx % cols
        x = col * view_w
        y = row * view_h
        layout.append({
            "x": x,
            "y": y,
            "w": view_w,
            "h": view_h,
        })
    # print(f"Unused space: {144 - view_w * view_h * num_views}, Aspect ratio difference: {abs(view_w - view_h)}, Score: {best_score}")
    return layout


def create_screen(screen_json):

    # Open the template config file
    with open(TEMPLATE_CONFIG_PATH, "r") as f:
        config_json = json.load(f)

    # Load sample adata
    sample_name = screen_json["sample"]
    year = screen_json["year"]
    output_dir = screen_json["outputDir"]
    sample_path, adata = load_data(sample_name, year)

    grid_layout = create_grid_layout(len(screen_json["views"]))

    # For each view in screen_json...
    for view in screen_json["views"]:
        view_title = view["title"]
        print(f"\tProcessing view: {view_title}")
        cluster_assignment_path = view["clusterAssignmentPath"]
        column_name = view["columnName"]
        source_column_name = view.get("sourceColumnName")
        # if sourceColumnName is specified, pass it to add_cluster_assignments
        if source_column_name is None:
            source_column_name = 'cluster'

        # Add cluster assignments to adata.obs
        adata = add_cluster_assignments(adata, cluster_assignment_path, column_name, source_column_name)

        # Add view to config
        config_json = add_view_to_config(config_json, view_title, column_name, adata, grid_layout)

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

    print(f"[{datetime.now().isoformat()}] Starting Vitessce visualization creation...")
    # Load the manifest json
    with open(MANIFEST_JSON_PATH, 'r') as f:
        manifest_json = json.load(f)
    
    # For each screen in the manifest, call create_screen
    for screen in manifest_json.get("allScreens", []):
        print(f"[{datetime.now().isoformat()}] Creating screen: {screen['name']}")
        create_screen(screen)
    
    # Make the done file
    done_file = Path(snakemake.output[0])
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.touch()

    print(f"[{datetime.now().isoformat()}] Finished Vitessce visualization creation. Done file created at {done_file}.")


if __name__ == "__main__":
    main()
