import os
from PIL import Image
import matplotlib.pyplot as plt


# Function to generate a collage from a base path and names.txt file
def generate_collage(base_path, output_filename, slices_per_sample):
    # Define the path to the names.txt file
    names_file = os.path.join(base_path, "names.txt")

    # Read the sample names from names.txt (excluding the last entry which is the file name itself)
    with open(names_file, "r") as f:
        sample_names = [line.strip() for line in f.readlines()]

    # Remove the last entry (file name) from the list
    sample_names = sample_names[:-1]

    # Function to get the full image path for each sample
    def get_image_path(sample_name):
        return os.path.join(
            base_path, sample_name, "outs/spatial/tissue_lowres_image.png"
        )

    # Load images for the collage
    images = []
    for sample_name in sample_names:
        image_path = get_image_path(sample_name)
        if os.path.exists(image_path):
            img = Image.open(image_path)
            images.append(img)
        else:
            print(f"Image for {sample_name} not found at {image_path}")

    # Determine collage layout (e.g., 4 images per row)
    n_images = len(images)
    n_cols = slices_per_sample
    n_rows = (n_images // n_cols) + (1 if n_images % n_cols != 0 else 0)

    # Create the collage
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes to make indexing easier
    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")  # Hide axes
        axes[i].set_title(sample_names[i])

    # Hide any unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    # Save the collage as a file
    output_path = os.path.join("/zata/zippy/kresgeb/psych_screen", output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # Close the plot to avoid memory issues
    plt.close()

    print(f"Collage saved to {output_path}")


# Generate collages for both datasets
generate_collage(
    "/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger",
    "2024_tissue_collage.png",
    3,
)
generate_collage(
    "/data/zusers/kresgeb/psych_encode/HumanPilot10X/reorganized",
    "2021_tissue_collage.png",
    4,
)
