# Load the full TSV, keeping all columns
df <- read.delim(
    "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/batched_nmf/summary.tsv",
    check.names = FALSE
)

# Select 10 random rows
set.seed(123) # for reproducibility
subset_df <- df[sample(nrow(df), 100), ]

# Save the subset
write.table(
    subset_df,
    "/zata/zippy/kresgeb/nmf_stuff/dlPFC/data/subset.tsv",
    sep = "\t",
    quote = FALSE,
    row.names = FALSE # do not write row names
)
