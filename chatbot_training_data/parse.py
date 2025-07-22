import pandas as pd

# Path to GTF
gtf_path = "/zata/zippy/kresgeb/chatbot_training_data/gencode.v48.annotation.gtf.gz"

print("Loading GTF file...")

# Load GTF with pandas (skips comment lines)
gtf = pd.read_csv(
    gtf_path, 
    sep="\t", 
    comment="#", 
    header=None, 
    names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
)

# Keep only gene features
genes = gtf[gtf["feature"] == "gene"].copy()

print(f"Found {len(genes)} gene features in GTF.")

print("Parsing gene attributes...")
# Parse attributes
def parse_attributes(attr_str):
    attrs = {}
    for entry in attr_str.strip().split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if " " not in entry:
            continue  # skip malformed
        key, value = entry.split(" ", 1)
        attrs[key] = value.strip('"')
    return attrs

print("Extracting gene_id and gene_name...")
# Extract gene_id and gene_name
genes_attrs = genes["attribute"].apply(parse_attributes)
genes["gene_id"] = genes_attrs.apply(lambda x: x.get("gene_id", ""))
genes["gene_name"] = genes_attrs.apply(lambda x: x.get("gene_name", ""))
genes["gene_type"] = genes_attrs.apply(lambda x: x.get("gene_type", ""))
genes = genes[genes["gene_type"] == "protein_coding"]
print(f"Filtered to {len(genes)} protein-coding genes.")

print("Extracting chromosome, start, end, and strand...")
# Select and save output
output = genes[["gene_id", "gene_name", "chrom", "start", "end"]]
print("Saving output to gene_coordinates.tsv...")
output.to_csv("/zata/zippy/kresgeb/chatbot_training_data/gene_coordinates.tsv", sep="\t", index=False)