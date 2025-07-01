import os
import pandas as pd
import squidpy as sq
import scanpy as sc
import warnings
import json
import numpy as np
from vitessce.data_utils import (
    to_diamond,
    rgb_img_to_ome_zarr,
    optimize_adata,
    to_uint8
)

# Paths dictionary with configs for 2021 and 2024
PATHS = {
    "2021": {
        "visium_source_dir": "/data/zusers/kresgeb/psych_encode/HumanPilot10X/reorganized",
        "output_dir": "/zata/public_html/users/kresgeb/psych_screen/HumanPilot10X",
        "template_config": "/zata/zippy/kresgeb/psych_screen/paper_data_processing/template_configs/template_config_2021.json",
        "full_adata_path": "/zata/zippy/kresgeb/psych_screen/paper_data_processing/paper_data/2021.h5ad",
    },
    "2024": {
        "visium_source_dir": "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger",
        "output_dir": "/zata/public_html/users/kresgeb/psych_screen/spatialDLPFC",
        "template_config": "/zata/zippy/kresgeb/psych_screen/paper_data_processing/template_configs/template_config_2024.json",
        "full_adata_path": "/zata/zippy/kresgeb/psych_screen/paper_data_processing/paper_data/2024.h5ad", 
        "color_data_path": "/zata/zippy/kresgeb/psych_screen/paper_data_processing/colors/k16_like_manual.json", # Not currently used since all color data is in the template config
    },
}

warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
)
warnings.filterwarnings(
    "ignore",
    message="Observation names are not unique. To make them unique, call `.obs_names_make_unique`.",
)

def load_visium_data(visium_source_dir, sample_name):
    """
    Load Visium data using Squidpy and scale spatial coordinates using the hires scale factor.

    :param visium_source_dir: Path to the base Visium data directory for the year.
    :param sample_name: Name of the sample (e.g., "151507" or "Br8667_mid").
    :return: AnnData object with scaled spatial coordinates.
    """
    # Find the folder that contains this sample
    matching_dirs = [d for d in os.listdir(visium_source_dir) if sample_name in d]
    assert len(matching_dirs) == 1, f"Expected 1 match for {sample_name}, found {len(matching_dirs)}"
    sample_dir = os.path.join(visium_source_dir, matching_dirs[0])
    outs_path = os.path.join(sample_dir, "outs")

    # Load the Visium spatial data
    adata = sq.read.visium(outs_path)
    adata.var_names_make_unique()

    # Load and apply the scale factor
    scale_json_path = os.path.join(outs_path, "spatial", "scalefactors_json.json")
    with open(scale_json_path, "r") as f:
        scale_data = json.load(f)
    scale_factor = scale_data["tissue_hires_scalef"]
    adata.obsm["spatial"] = adata.obsm["spatial"] * scale_factor

    return adata

def transfer_spatial_and_image_metadata(sample_adata, visium_reference_adata, sample_name):
    """
    Transfer spatial coordinates and image metadata (uns["spatial"]) from Visium reference to sample_adata.

    :param sample_adata: AnnData object with expression and obs data from the paper pipeline.
    :param visium_reference_adata: Visium AnnData object containing spatial/image metadata.
    :param sample_name: Sample identifier string.
    :return: Modified sample_adata with spatial and image metadata.
    """
    # Align and transfer spatial coordinates
    spatial_df = pd.DataFrame(
        visium_reference_adata.obsm["spatial"],
        index=visium_reference_adata.obs_names
    )
    sample_adata.obsm["spatial"] = spatial_df.loc[sample_adata.obs_names].values

    # Use the actual key in uns["spatial"], which is usually the folder name (not sample_id)
    spatial_key = list(visium_reference_adata.uns["spatial"].keys())[0]
    sample_adata.uns["spatial"] = {
        sample_name: visium_reference_adata.uns["spatial"][spatial_key]
    }

    return sample_adata

def add_segmentations(adata, radius=7):
    """
    Add diamond-shaped segmentations to adata using spatial coordinates.

    :param adata: AnnData object with obsm["spatial"] coordinates.
    :param radius: Radius for diamond shape.
    :return: AnnData object with segmentations added to obsm["segmentations"].
    """
    num_cells = adata.shape[0]
    segmentations = np.zeros((num_cells, 4, 2))
    for i in range(num_cells):
        x, y = adata.obsm["spatial"][i]
        segmentations[i] = to_diamond(x, y, radius)
    adata.obsm["segmentations"] = segmentations
    return adata

def determine_obs_cols(adata, year):
    """
    Determine which obs columns to keep based on the year. And performs renaming of specific columns.

    :param adata: AnnData object.
    :param year: Year of the dataset (2021 or 2024).
    :return: List of obs columns to keep.
    """
    obs_cols = []

    if year == "2021":
        if "spatialLIBD" in adata.obs:
            adata.obs["manual_layers"] = adata.obs["spatialLIBD"]
            obs_cols.append("manual_layers")

    elif year == "2024":
        # Handle manual layers if they exist
        if "manual_layer_label" in adata.obs and adata.obs["manual_layer_label"].notna().any():
            adata.obs["manual_layers"] = adata.obs["manual_layer_label"]
            obs_cols.append("manual_layers")

        # Remap BayesSpace harmony cluster columns to new names
        for k in [9, 16]:
            old_col = f"BayesSpace_harmony_{k:02d}"
            new_col = f"bayes_space_k={k}"
            if old_col in adata.obs:
                adata.obs[new_col] = adata.obs[old_col]
                obs_cols.append(new_col)

    return obs_cols

def write_ome_zarr_image(adata, output_path, sample_name):
    """
    Write the high-resolution RGB image from AnnData to OME-Zarr format.

    :param adata: AnnData object with uns["spatial"][sample_name]["images"]["hires"].
    :param output_path: Destination path for image.ome.zarr.
    :param sample_name: Sample name key used in uns["spatial"].
    """

    # Extract the RGB image from uns
    try:
        img_hires = adata.uns["spatial"][sample_name]["images"]["hires"]
    except KeyError as e:
        raise ValueError(f"Missing hires image for sample {sample_name}") from e

    # Need to convert images from interleaved to non-interleaved (color axis should be first).
    img_arr = np.transpose(img_hires, (2, 0, 1))  # shape: (3, height, width)

    # Save image to OME-Zarr using Vitessce helper
    rgb_img_to_ome_zarr(
        img_arr,
        output_path,
        axes="cyx",              # color-y-x
        chunks=(1, 256, 256),    # default chunking
        img_name="H & E Image"   # name shown in Vitessce
    )

def remove_unassigned_spots(sample_adata, obs_cols): #TODO: This does NOT seemingly actually remove unassigned spots for 2024 (unless they are already removed in the paper pipeline???) Should look into this
    """
    Remove spots from sample_adata that have missing or empty values
    in any of the specified obs columns.

    :param sample_adata: AnnData object
    :param obs_cols: list of column names in sample_adata.obs to check
    :return: filtered AnnData with unassigned spots removed
    """
    initial_spot_count = sample_adata.n_obs
    filtered_adata = sample_adata.copy()

    for col in obs_cols:
        if col not in filtered_adata.obs:
            print(f"Warning: Column '{col}' not found in obs. Skipping.")
            continue

        series = filtered_adata.obs[col]

        # Check missing values
        missing_mask = series.isna()

        # For string columns, also check empty or whitespace-only strings
        if pd.api.types.is_string_dtype(series):
            missing_mask = missing_mask | (series.str.strip() == "")
        
        missing_spots = filtered_adata.obs.index[missing_mask]

        if len(missing_spots) > 0:
            print(f"Removing {len(missing_spots)} spots due to missing/unassigned values in '{col}'")
            filtered_adata = filtered_adata[~filtered_adata.obs.index.isin(missing_spots)]

    removed_spots = initial_spot_count - filtered_adata.n_obs
    if removed_spots != 0:
        print(f"Total spots removed: {removed_spots} / {initial_spot_count}")
    return filtered_adata

def create_configuration_file(year, sample_name, obs_cols):
    """
    Create and save a configuration file for Vitessce visualization.

    :param year: Year of the dataset (2021 or 2024).
    :param sample: The name of the sample (e.g., "151507" or "Br8667_mid").
    :param obs_cols: List of observation columns to include (usually cluster assignments).
    """
    # Creating both single and multi-column configurations
    for suffix in ["", "_single_column"]:

        # Adjust template and output paths
        template_path = PATHS[year]["template_config"]
        output_file_name = "config.json"

        if suffix:
            template_path = template_path.replace(".json", f"{suffix}.json")
            output_file_name = f"config{suffix}.json"

        output_file_path = os.path.join(
            PATHS[year]["output_dir"], "configs", sample_name, output_file_name
        )

        # Load the template
        with open(template_path, "r") as f:
            data = json.load(f)

        # Replace <<Sample_Name>> with the actual sample name
        data_str = json.dumps(data)
        data_str = data_str.replace("<<Sample_Name>>", sample_name)
        data = json.loads(data_str)

        # Remove the manual layers obsSet if it does not exist in obs_cols
        # Note: This is only relevant for 2024, where the manual layers are not always present
        if "manual_layers" not in obs_cols:
            # Remove "Manually Annotated Layers" from obsSets
            datasets = data.get("datasets", [])
            for dataset in datasets:
                for file in dataset.get("files", []):
                    options = file.get("options", {})
                    obs_sets = options.get("obsSets", [])
                    options["obsSets"] = [
                        entry for entry in obs_sets if entry["name"] != "Manually Annotated Layers"
                    ]


        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Write the updated config
        with open(output_file_path, "w") as file:
            json.dump(data, file, indent=2)

def main():
    # For 2021 and 2024, process the data
    for year in ["2021", "2024"]:
        visium_source_dir = PATHS[year]["visium_source_dir"]
        output_dir = PATHS[year]["output_dir"]
        template_config_path = PATHS[year]["template_config"]
        full_adata_path = PATHS[year]["full_adata_path"]

        # Create directories if they do not exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)

        # Load the full AnnData object (for the year, has all samples)
        print(f"Loading full AnnData object for year {year} from {full_adata_path}")
        full_adata = sc.read_h5ad(full_adata_path)

        # Get all sample names
        sample_names = full_adata.obs["sample_id"].unique()

        # Process each sample
        for sample_name in sample_names:
            print(f"Processing sample: {sample_name} for year {year}")

            # Filter the full AnnData object for the current sample
            sample_adata = full_adata[full_adata.obs["sample_id"] == sample_name].copy()

            # Get the visium data for the sample
            visium_reference_adata = load_visium_data(visium_source_dir, sample_name)

            # Index by HGNC gene names instead of Ensembl IDs
            sample_adata.var["gene_name_orig"] = sample_adata.var["gene_name"]
            sample_adata.var.index = sample_adata.var["gene_name"].astype(str)
            sample_adata.var.index.name = None  # Clear index name to avoid conflict
            sample_adata.var_names_make_unique()
        
            # Add the spatial and the image data to the sample AnnData object
            sample_adata = transfer_spatial_and_image_metadata(sample_adata, visium_reference_adata, sample_name)

            # Add segmentations to the sample AnnData object
            sample_adata = add_segmentations(sample_adata)

            # Save the image (OME-Zarr format)
            image_output_path = os.path.join(output_dir, "data", sample_name, "image.ome.zarr")
            write_ome_zarr_image(sample_adata, image_output_path, sample_name)

            # Determine which obs columns to keep based on the year (also renames columns)
            obs_cols = determine_obs_cols(sample_adata, year)

            # Remove spots with unassigned clusters/assignments
            sample_adata = remove_unassigned_spots(sample_adata, obs_cols)

            # Precalculate logcounts uint8 layer (not currently used, but useful for Vitessce)
            # sample_adata.layers["logcounts_uint8"] = to_uint8(sample_adata.layers["logcounts"], norm_along="global")

            # Optimize the AnnData object
            optimized_adata = optimize_adata(
                sample_adata,
                obs_cols= obs_cols,
                obsm_keys=["spatial", "segmentations"],
                layer_keys=["logcounts"], # ["logcounts", "logcounts_uint8"] # Uncomment if you want to include the uint8 layer
                optimize_X=True,
                # Vitessce plays nicely with dense matrices saved with chunking
                to_dense_X=True,
            )
        
            # Save the optimized AnnData object
            optimized_adata_path = os.path.join(output_dir, "data", sample_name, "data.h5ad.zarr")
            optimized_adata.write_zarr(optimized_adata_path, chunks=[optimized_adata.shape[0], 10])

            # Create the configuration file for Vitessce visualization
            create_configuration_file(year, sample_name, obs_cols)





if __name__ == "__main__":
    main()