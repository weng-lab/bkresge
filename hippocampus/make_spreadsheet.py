#!/usr/bin/env python3
import pandas as pd

def make_vitessce_links(
    input_csv="/zata/zippy/kresgeb/hippocampus/srt_unique_sample_id_brnum_position_sorted_custom.csv",
    output_csv="/zata/zippy/kresgeb/hippocampus/vitessce_spreadsheet.csv"
    # output_csv="/zata/zippy/kresgeb/hippocampus/my_output/nmf/HPC/vitessce_spreadsheet.csv"
):
    """
    Reads a CSV of sample_ids and appends a vitessce_link column.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV.
    """
    # Load the CSV
    df = pd.read_csv(input_csv)

    # Base URL (already URL-encoded up to the configs path)
    base_url = (
        "https://vitessce.io/#?edit=false&url="
        "https%3A%2F%2Fusers.wenglab.org%2Fkresgeb%2Fhippocampus%2Fconfigs%2F"
        # "https%3A%2F%2Fusers.wenglab.org%2Fkresgeb%2Fhippocampus%2Fnmf_compare%2Fconfigs%2F"
    )

    # Construct vitessce_link column
    df["vitessce_link"] = df["sample_id"].apply(
        lambda sid: f"{base_url}{sid}_config.json"
    )

    # Save to new CSV
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved updated CSV with Vitessce links to {output_csv}")


if __name__ == "__main__":
    make_vitessce_links()
