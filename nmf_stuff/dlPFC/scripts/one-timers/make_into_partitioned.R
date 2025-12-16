library(duckplyr)
library(arrow)

parquet_path <- "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/all_patterns.parquet"

ds <- open_dataset(parquet_path, format = "parquet")

cat("Schema of the Parquet dataset:\n")
print(ds$schema)

write_dataset(
    ds,
    path = "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/all_patterns_partitioned",
    format = "parquet",
    partitioning = "k",
    existing_data_behavior = "overwrite"
)