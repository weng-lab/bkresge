import pandas as pd
import json
import random

# === Config ===
INPUT_TSV = "chatbot_training_data/gene_coordinates.tsv"
OUTPUT_CSV = "chatbot_training_data/finetune_dataset.csv"
PROMPT_CONFIG = "chatbot_training_data/prompt_config.json"
IGSCREEN_MODE = "browser"  # could be "icres" too
N_GENES_PER_TEMPLATE = 100
RANDOM_STATE = 42
MISSPELLED_GENE_PROB = 0.05  # Probability of misspelling a gene name

COLUMN_ORDER = [
    "prompt",
    "prompt_template",
    "gene_misspelled",
    "gene",
    "chr_name",
    "start",
    "end",
    "igscreen_link_gene",
    "igscreen_link_region"
]

def misspell_gene_name(gene, existing_genes, max_attempts = 10):

    # Don't mangle ensembl gene names
    if gene.startswith("ENSG") or gene.startswith("ENSMUSG"):
        return gene, False
    
    if len(gene) < 3:
        return gene  # avoid mangling very short names

    for _ in range(max_attempts):
        typo_type = random.choice(["swap", "delete", "duplicate"])
        typo = gene

        if typo_type == "swap" and len(gene) >= 3:
            i = random.randint(0, len(gene) - 2)
            typo = list(gene)
            typo[i], typo[i + 1] = typo[i + 1], typo[i]
            typo = "".join(typo)

        elif typo_type == "delete":
            i = random.randint(0, len(gene) - 1)
            typo = gene[:i] + gene[i+1:]

        elif typo_type == "duplicate":
            i = random.randint(0, len(gene) - 1)
            typo = gene[:i] + gene[i] + gene[i:]

        if typo not in existing_genes:
            return typo, True

    return gene, False  # fallback, same as original

def main():
    # === Load Inputs ===
    genes = pd.read_csv(INPUT_TSV, sep="\t")
    with open(PROMPT_CONFIG) as f:
        templates = json.load(f)

    existing_genes = set(genes["gene_name"])

    # Collect all link types across templates for column creation
    all_link_types = sorted({lt for entry in templates for lt in entry["link_types"]})

    # === Build Output Rows ===
    rows = []

    for i, template in enumerate(templates):
        prompt_template = template["prompt_template"]
        link_types = template["link_types"]

        sampled_genes = genes.sample(n=min(N_GENES_PER_TEMPLATE, len(genes)), random_state=RANDOM_STATE + i)

        for _, row in sampled_genes.iterrows():
            real_gene = row["gene_name"]
            chr_name = row["chrom"].replace("chr", "")
            start = row["start"]
            end = row["end"]

            # Determine which fields are present in the template
            contains_gene = "{gene}" in prompt_template
            contains_chr = "{chr}" in prompt_template
            contains_start = "{start}" in prompt_template
            contains_end = "{end}" in prompt_template
            

            # Determine whether to use typo
            use_typo = ("{gene}" in prompt_template) and (random.random() < MISSPELLED_GENE_PROB)
            prompted_gene, use_typo = misspell_gene_name(real_gene, existing_genes) if use_typo else (real_gene, False)

           
            format_kwargs = {}
            if contains_gene:
                format_kwargs["gene"] = prompted_gene
            if contains_chr:
                format_kwargs["chr"] = chr_name
            if contains_start:
                format_kwargs["start"] = start
            if contains_end:
                format_kwargs["end"] = end

            prompt = prompt_template.format(**format_kwargs)

            # Construct links
            links = {}
            for lt in all_link_types:
                if lt in link_types:
                    if lt == "gene" and contains_gene:
                        links[f"igscreen_link_{lt}"] = (
                            f"https://igscreen.wenglab.org/gene/{real_gene}"
                        )
                    elif lt == "region" and contains_chr and contains_start and contains_end:
                        links[f"igscreen_link_{lt}"] = (
                            f"https://igscreen.wenglab.org/region/chr{chr_name}:{start}-{end}/{IGSCREEN_MODE}"
                        )
                    else:
                        links[f"igscreen_link_{lt}"] = ""
                else:
                    links[f"igscreen_link_{lt}"] = ""

            # Build row
            row_data = {
                "prompt": prompt,
                "prompt_template": prompt_template,
                "gene_misspelled": use_typo,
                **links
            }
            if contains_gene:
                row_data["gene"] = real_gene
            if contains_chr:
                row_data["chr_name"] = chr_name
            if contains_start:
                row_data["start"] = start
            if contains_end:
                row_data["end"] = end

            rows.append(row_data)

    # === Save ===
    df_out = pd.DataFrame(rows)
    # Ensure start/end are integers where present
    for col in ["start", "end"]:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").astype("Int64")
    df_out.to_csv(OUTPUT_CSV, index=False, columns=COLUMN_ORDER)
    print(f"Saved {len(df_out)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
