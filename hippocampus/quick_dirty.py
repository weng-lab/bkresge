import os
import csv

csv_path = "/zata/zippy/kresgeb/hippocampus/srt_unique_sample_id_brnum_position_sorted_custom.csv"
dir_path = "/data/zusers/kresgeb/hippocampus/geo_reformatted/srt"

# 1. Read sample_ids from CSV
csv_sample_ids = []
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        csv_sample_ids.append(row["sample_id"])

csv_sample_ids_set = set(csv_sample_ids)

# 2. Get directory names and extract sample_ids
dir_entries = os.listdir(dir_path)
dir_sample_ids = []
for d in dir_entries:
    # Expect format like: GSM8226199_V12F14-051_A1
    parts = d.split("_", 1)
    if len(parts) == 2:
        sample_id = parts[1]
        dir_sample_ids.append(sample_id)

dir_sample_ids_set = set(dir_sample_ids)

# 3. Compare
only_in_csv = csv_sample_ids_set - dir_sample_ids_set
only_in_dir = dir_sample_ids_set - csv_sample_ids_set
in_both = csv_sample_ids_set & dir_sample_ids_set

print(f"Samples only in CSV ({len(only_in_csv)}): {sorted(only_in_csv)}\n")
print(f"Samples only in directories ({len(only_in_dir)}): {sorted(only_in_dir)}\n")
print(f"Samples in both ({len(in_both)}): {sorted(in_both)}\n")
