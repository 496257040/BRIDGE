import scanpy as sc
import numpy as np
import pandas as pd
import os
import re
import scipy.io
import gzip
data_dir = "E:\数据集\GSE200996\GSE200996_RAW"
def parse_sample_info_from_dirname(dirname):
    info = {
        "sample_id": "unknown",
        "patient": "unknown",
        "tissue": "unknown",
        "disease": "HNSCC",
        "cancer_type": "HNSCC",
        "treatment_drug": "Nivolumab or Nivolumab & Ipilimumab",
        "treatment_timepoint": "none",
    }
    parts = dirname.split("_")
    gsm_id = parts[0] if parts[0].startswith("GSM") else "unknown"
    info["sample_id"] = gsm_id
    tissue_keywords = ["PBMC", "tumor"]
    for keyword in tissue_keywords:
        if keyword in dirname:
            info["tissue"] = keyword
            break
    simplified = re.sub(r'^GSM\d+_raw_feature_bc_matrix_', '', dirname)
    simplified = re.sub(r'^GSM\d+_', '', simplified) if simplified == dirname else simplified
    patient_match = re.match(r'((?:P\d+)(?:-P\d+)*)', simplified)
    if patient_match:
        info["patient"] = patient_match.group(1)
    tx_match = re.search(r'(pre-Tx|post-Tx|pre_Tx|post_Tx|preTx|postTx)', simplified, re.IGNORECASE)
    if tx_match:
        raw_tx = tx_match.group(1).lower()
        if "pre" in raw_tx:
            info["treatment_timepoint"] = "pre-treatment"
        elif "post" in raw_tx:
            info["treatment_timepoint"] = "post-treatment"
    return info
def find_h5_files(data_dir):
    all_files = os.listdir(data_dir)
    h5_pattern = re.compile(r'^(GSM\d+_.+?)\.h5$')
    sample_files = {}
    for f in all_files:
        match = h5_pattern.match(f)
        if match:
            sample_id = match.group(1)
            sample_files[sample_id] = os.path.join(data_dir, f)
    return sample_files
def read_h5_file(file_path, file_type=None):
    adata = sc.read_10x_h5(file_path)
    return adata
sample_files = find_h5_files(data_dir)
adatas = []
metadata_records = []
for prefix, files in sorted(sample_files.items()):
    adata = read_h5_file(files)
    adata.var_names_make_unique()
    if "gene_ids" in adata.var.columns:
        adata.var["gene_ids"] = adata.var["gene_ids"].astype(str)
    for col in adata.var.columns:
        if pd.api.types.is_categorical_dtype(adata.var[col]):
            adata.var[col] = adata.var[col].astype(str)
    info = parse_sample_info_from_dirname(prefix)
    adata.obs_names = [f"{info['sample_id']}_{cell_id}" for cell_id in adata.obs_names]
    adata.obs_names_make_unique()
    adata.obs["sample_id"] = info["sample_id"]
    adata.obs["patient"] = info["patient"]
    adata.obs["tissue"] = info["tissue"]
    adata.obs["disease"] = info["disease"]
    adata.obs["cancer_type"] = info["cancer_type"]
    adata.obs["treatment_drug"] = info["treatment_drug"]
    adata.obs["treatment_timepoint"] = info["treatment_timepoint"]
    adata.obs["sample_prefix"] = prefix
    adatas.append(adata)
    metadata_records.append(info)
common_genes = adatas[0].var_names
for ad in adatas[1:]:
    common_genes = common_genes.intersection(ad.var_names)
print(f"\n公共基因数量: {len(common_genes)}")
# 对每个adata只保留公共基因，并确保顺序一致
for i in range(len(adatas)):
    adatas[i] = adatas[i][:, common_genes].copy()
var_cols_to_keep = []
if all("gene_ids" in ad.var.columns for ad in adatas):
    var_cols_to_keep.append("gene_ids")
for i in range(len(adatas)):
    keep_cols = [c for c in var_cols_to_keep if c in adatas[i].var.columns]
    adatas[i].var = adatas[i].var[keep_cols].copy()
metadata_df = pd.DataFrame(metadata_records)
metadata_path = r"C:\Users\49625\Desktop\大创相关\code\data\GSE200996_HNSCC\GSE200996_sample_metadata.csv"
os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
metadata_df.to_csv(metadata_path, index=False)
adata_combined = sc.concat(adatas, join="inner", merge="same", index_unique="-")
adata_combined.var_names_make_unique()
del adatas
#质控
adata_combined.var["mt"] = adata_combined.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata_combined, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts > 200, :]
adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts < 5000, :]
adata_combined = adata_combined[adata_combined.obs.pct_counts_mt < 20, :]
sc.pp.filter_genes(adata_combined, min_cells=3)
output_dir = "C:\\Users\\49625\\Desktop\\大创相关\\code\\data\\GSE200996_HNSCC\\filtered_h5ad"
os.makedirs(output_dir, exist_ok=True)
unique_sample_ids = metadata_df["sample_id"].unique()
sample_summary = []
for sample_id in unique_sample_ids:
    if sample_id == "unknown" or pd.isna(sample_id):
        continue
    sample_metadata = metadata_df[metadata_df["sample_id"] == sample_id].iloc[0]
    sample_cells = adata_combined.obs['sample_id'] == sample_id
    if sample_cells.sum() == 0:
        print(f"样本 {sample_id} 在质控后没有细胞，跳过")
        continue
    adata_sample = adata_combined[sample_cells, :].copy()
    adata_sample.uns["sample_metadata"] = sample_metadata.to_dict()
    output_filename = f"{sample_id}_filtered.h5ad"
    output_path = os.path.join(output_dir, output_filename)
    adata_sample.write_h5ad(output_path, compression="gzip")
    sample_summary.append({
        "sample_id": sample_id, 
        "patient": sample_metadata.get("patient", "unknown"),
        "tissue": sample_metadata.get("tissue", "unknown"),
        "disease": sample_metadata.get("disease", "unknown"),
        "cancer_type": sample_metadata.get("cancer_type", "none"),
        "treatment_drug": sample_metadata.get("treatment_drug", "none"),
        "treatment_timepoint":  sample_metadata.get("treatment_timepoint", "none"),
        "n_cells": adata_sample.n_obs,
        "n_genes": adata_sample.n_vars,
        "file_path": output_path
    })
print(f"\n质控后样本处理完成,结果已保存至: {output_dir}")