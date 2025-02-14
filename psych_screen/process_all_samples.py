import os
import squidpy as sq
import scanpy as sc
import multiprocessing
import warnings
import scipy
import json
import numpy as np
from vitessce.data_utils import (
    to_diamond,
    rgb_img_to_ome_zarr,
    optimize_adata,
)

SPACERANGER_SOURCE_DIR = '/data/zusers/kresgeb/psych_encode/spatialDLPFC/processed-data/rerun_spaceranger'
OUTPUT_DIR = '/zata/public_html/users/kresgeb/psych_encode_spatialDLPFC'
TEMPLATE_CONFIG_PATH = '/zata/zippy/kresgeb/psych_screen/template_config.json'
WHITELIST_PATH = '/zata/zippy/kresgeb/psych_screen/whitelist.txt'

# Suppress the specific UserWarning about unique names
warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.")


def main():
    # all subdirectories in the source directory (exclude the names.txt)
    sample_names = [entry.name for entry in os.scandir(SPACERANGER_SOURCE_DIR) if entry.is_dir()]

    # Make all directories if they do not exist
    for sample_name in sample_names:
        os.makedirs(name=os.path.join(OUTPUT_DIR, 'data', sample_name), exist_ok=True)
        os.makedirs(name=os.path.join(OUTPUT_DIR, 'configs', sample_name), exist_ok=True)

    # Create the config files from the template
    for sample_name in sample_names:
        create_configuration_file(sample_name)
    
    pool = multiprocessing.Pool(processes=30)
    pool.map(process_sample, sample_names)

    # Close the pool to free resources
    pool.close()
    pool.join()

# Based on https://github.com/vitessce/vitessce-python/blob/main/demos/human-lymph-node-10x-visium/src/create_zarr.py
def process_sample(sample_name):
    data_output_path = os.path.join(OUTPUT_DIR, 'data', sample_name, 'data.h5ad.zarr')
    image_output_path = os.path.join(OUTPUT_DIR, 'data', sample_name, 'image.ome.zarr')
    source_path = os.path.join(SPACERANGER_SOURCE_DIR, sample_name, 'outs')

    adata = sq.read.visium(source_path)
    adata.var_names_make_unique()

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Perform basic filtering (much more generous than source)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=10)
    adata = adata[adata.obs["pct_counts_mt"] < 30]

    # Perform normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    # Determine the top 300 highly variable genes.
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=300)

    # Genes of Interest are all the highly variable genes and any additional ones in the whitelist provided
    set_genes_of_interest(adata, WHITELIST_PATH)

    # Dimensionality reduction and clustering
    sc.pp.pca(adata, mask_var='genes_of_interest')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden", flavor="igraph", n_iterations=2)

    # Hierarchical clustering of genes for optimal gene ordering
    X_goi_arr = adata[:, adata.var['genes_of_interest']].X.toarray()
    X_goi_index = adata[:, adata.var['genes_of_interest']].var.copy().index
    Z = scipy.cluster.hierarchy.linkage(X_goi_arr.T, method='average', optimal_ordering=True)

    # Get the hierarchy-based ordering of genes.
    num_cells = adata.obs.shape[0]
    goi_index_ordering = scipy.cluster.hierarchy.leaves_list(Z)
    genes_of_interest = X_goi_index.values[goi_index_ordering].tolist()
    all_genes = adata.var.index.values.tolist()
    not_goi = adata.var.loc[~adata.var['genes_of_interest']].index.values.tolist()

    def get_orig_index(gene_id):
        return all_genes.index(gene_id)

    var_index_ordering = list(map(get_orig_index, genes_of_interest)) + list(map(get_orig_index, not_goi))

    # Create a new *ordered* gene expression dataframe.
    adata = adata[:, var_index_ordering].copy()
    adata.obsm["X_goi"] = adata[:, adata.var['genes_of_interest']].X.copy()

    # Scale the spatial data to align with the image
    scale_factor = get_scale_factor(sample_name)
    adata.obsm['spatial'] = (adata.obsm['spatial'] * scale_factor)

    # Create the diamond visualizations for the spots
    adata.obsm['segmentations'] = np.zeros((num_cells, 4, 2))
    radius = 7
    for i in range(num_cells):
        adata.obsm['segmentations'][i, :, :] = to_diamond(adata.obsm['spatial'][i, 0], adata.obsm['spatial'][i, 1], radius)
    
    # Write img_arr to OME-Zarr.
    # Need to convert images from interleaved to non-interleaved (color axis should be first).
    img_hires = adata.uns['spatial'][sample_name]['images']['hires']
    img_arr = np.transpose(img_hires, (2, 0, 1))
    rgb_img_to_ome_zarr(img_arr, image_output_path, axes="cyx", chunks=(1, 256, 256), img_name="H & E Image")

    # Optimize and write anndata
    adata = optimize_adata(
        adata,
        obs_cols=["leiden"],
        var_cols=["highly_variable", "genes_of_interest"],
        obsm_keys=["X_goi", "spatial", "segmentations", "X_umap", "X_pca"],
        optimize_X=True,
        # Vitessce plays nicely with dense matrices saved with chunking
        to_dense_X=True,
    )
    adata.write_zarr(data_output_path, chunks=[adata.shape[0], 10])


def create_configuration_file(sample_name):
    output_file_path = os.path.join(OUTPUT_DIR, 'configs', sample_name, 'config.json')

    with open(TEMPLATE_CONFIG_PATH, 'r') as f:
        data = json.load(f)
    
    # Convert the data to a string
    data_str = json.dumps(data)
    
    # Replace <<Sample_Name>> with the actual sample name
    data_str = data_str.replace("<<Sample_Name>>", sample_name)
    
    # Convert the string back to a dictionary
    data = json.loads(data_str)
    
    # Write the updated data to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=2)

def set_genes_of_interest(adata, whitelist_path):
    adata.var['genes_of_interest'] = adata.var['highly_variable'].copy()
    with open(whitelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#') and line in adata.var.index:
                adata.var.loc[line, 'genes_of_interest'] = True

def get_scale_factor(sample_name):
    json_path = os.path.join(SPACERANGER_SOURCE_DIR, sample_name, 'outs', 'spatial', 'scalefactors_json.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('tissue_hires_scalef')

if __name__ == "__main__":
    main()