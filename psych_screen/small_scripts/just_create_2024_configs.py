import multiprocessing
import os
import json


SPACERANGER_SOURCE_DIR = (
    "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger"
)
OUTPUT_DIR = "/zata/public_html/users/kresgeb/psych_encode/spatialDLPFC"
TEMPLATE_CONFIG_PATH = "/zata/zippy/kresgeb/psych_screen/paper_data_processing/template_configs/2024_test_config.json"
COLOR_DATA_PATH = (
    "/zata/zippy/kresgeb/psych_screen/paper_data_processing/colors/output.json"
)


def main():
    # all subdirectories in the source directory (exclude the names.txt)
    sample_names = [
        entry.name for entry in os.scandir(SPACERANGER_SOURCE_DIR) if entry.is_dir()
    ]
    samples_with_MA = [
        "DLPFC_Br6522_ant_manual_alignment_all",
        "DLPFC_Br6522_mid_manual_alignment_all",
        "DLPFC_Br8667_post_manual_alignment_all",
    ]

    # Make all directories if they do not exist
    for sample_name in sample_names:
        os.makedirs(name=os.path.join(OUTPUT_DIR, "data", sample_name), exist_ok=True)
        os.makedirs(
            name=os.path.join(OUTPUT_DIR, "configs", sample_name), exist_ok=True
        )
        create_configuration_file(
            sample_name=sample_name, has_manual_layers=sample_name in samples_with_MA
        )


def create_configuration_file(sample_name, has_manual_layers=False):
    output_file_path = os.path.join(OUTPUT_DIR, "configs", sample_name, "config.json")

    with open(TEMPLATE_CONFIG_PATH, "r") as f:
        data = json.load(f)

    # Adjust for sample name
    # Convert the data to a string
    data_str = json.dumps(data)
    # Replace <<Sample_Name>> with the actual sample name
    data_str = data_str.replace("<<Sample_Name>>", sample_name)
    # Convert the string back to a dictionary
    data = json.loads(data_str)

    data = add_color_data(data)

    if not has_manual_layers:
        # Find the "Manually Annotated Layers" entry in obsSets and remove it
        datasets = data.get("datasets", [])
        for dataset in datasets:
            files = dataset.get("files", [])
            for file in files:
                options = file.get("options", {})
                obs_sets = options.get("obsSets", [])
                options["obsSets"] = [
                    entry
                    for entry in obs_sets
                    if entry["name"] != "Manually Annotated Layers"
                ]

    print(data)
    # Write the updated data to a new JSON file
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=2)


def hex_to_rgb(hex_color):
    """Converts hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


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


if __name__ == "__main__":
    main()
