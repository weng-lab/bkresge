# Reorganizes and lightly reformats the file structure provided by the 2021
# paper to be more in line with the usual Visium file output structure.

import os
import glob
import shutil


SOURCE_DIR = "/data/zusers/kresgeb/psych_encode/HumanPilot10X/raw"
OUTPUT_DIR = "/data/zusers/kresgeb/psych_encode/HumanPilot10X/reorganized"
SAMPLE_NAMES = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]


def main():
    for sample in SAMPLE_NAMES:
        reorganize_sample(sample)


def reorganize_sample(sample_name):
    sample_source_dir = os.path.join(SOURCE_DIR, sample_name)
    sample_output_dir = os.path.join(OUTPUT_DIR, sample_name, "outs")

    # Make necessary directories in the destination directory
    os.makedirs(
        os.path.join(OUTPUT_DIR, sample_name, "outs", "analysis", "manual_layers"),
        exist_ok=True,
    )
    os.makedirs(os.path.join(OUTPUT_DIR, sample_name, "outs", "spatial"), exist_ok=True)

    # Copy over both the filtered and raw counts files
    for source_file_path in glob.glob(
        os.path.join(sample_source_dir, "*_feature_bc_matrix.h5")
    ):
        new_file_name = "_".join(os.path.basename(source_file_path).split("_")[1:])
        shutil.copy2(source_file_path, os.path.join(sample_output_dir, new_file_name))

    # Copy over the tissue_positions_list
    shutil.copy2(
        os.path.join(sample_source_dir, "tissue_positions_list.txt"),
        os.path.join(sample_output_dir, "spatial", "tissue_positions_list.csv"),
    )

    # Copy over high and low res tissue images (H&E staining)
    for res in ["hires", "lowres"]:
        shutil.copy2(
            os.path.join(sample_source_dir, f"tissue_{res}_image.png"),
            os.path.join(sample_output_dir, "spatial", f"tissue_{res}_image.png"),
        )

    # Copy over the scale factors
    shutil.copy2(
        os.path.join(sample_source_dir, "scalefactors_json.json"),
        os.path.join(sample_output_dir, "spatial", "scalefactors_json.json"),
    )

    # Copy over the Layer data (just the barcodes for each)
    for source_file_path in glob.glob(
        os.path.join(sample_source_dir, "Layers", "*_barcodes.txt")
    ):
        new_file_name = "_".join(os.path.basename(source_file_path).split("_")[1:])
        shutil.copy2(
            source_file_path,
            os.path.join(sample_output_dir, "analysis", "manual_layers", new_file_name),
        )


if __name__ == "__main__":
    main()
