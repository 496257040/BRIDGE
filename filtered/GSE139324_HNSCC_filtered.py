import scanpy as sc
import numpy as np
import pandas as pd
import os
import re
import scipy.io
import gzip
data_dir = "E:\\数据集\\GSE139324\\GSE139324_RAW"
def parse_sample_info_from_dirname(dirname):
    info = {
        "patient": "unknown",
        "tissue": "unknown",
        "disease": "unknown",
        "cancer_type": "none",
        "treatment_drug": "none",
        "treatment_timepoint": "none",
        "sample_id": dirname
    }
    parts = dirname.split("_")
    gsm_id = parts[0] if parts[0].startswith("GSM") else "unknown"
    info["sample_id"] = gsm_id
    disease_keywords = {
        "HNSCC": {"disease": "cancer", "cancer_type": "HNSCC"},
        "HD": {"disease": "healthy", "cancer_type": "none"},
        "Healthy": {"disease": "healthy", "cancer_type": "none"},
    }
    tissue_keywords = ["PBMC", "TIL", "Tonsil"]
    patient_pattern = re.compile(r'(?:HD(?:_tonsil)?_|HNSCC_)(\d+)')
    for part in parts[1:]:
        # 匹配疾病状态
        for keyword, mapping in disease_keywords.items():
            if keyword.upper() == part.upper():
                info["disease"] = mapping["disease"]
                info["cancer_type"] = mapping["cancer_type"]
                break
        # 匹配患者编号
        patient_match = patient_pattern.search(dirname)
        if patient_match:
            patient_full = patient_match.group(0)
            info["patient"] = patient_full
        # 匹配组织来源
        for tissue_kw in tissue_keywords:
            if tissue_kw.upper() == part.upper():
                info["tissue"] = part
                break
    return info
def find_10x_sample_groups(data_dir):
    all_files = os.listdir(data_dir)
    barcode_pattern = re.compile(r'^(.+?)_(barcodes\.tsv(?:\.gz)?)$')
    feature_pattern = re.compile(r'^(.+?)_(features\.tsv(?:\.gz)?|genes\.tsv(?:\.gz)?)$')
    matrix_pattern = re.compile(r'^(.+?)_(matrix\.mtx(?:\.gz)?)$')
    sample_files = {}
    for f in all_files:
        for pattern, ftype in [
            (barcode_pattern, "barcodes"),
            (feature_pattern, "features"),
            (matrix_pattern, "matrix")
        ]:
            match = pattern.match(f)
            if match:
                prefix = match.group(1)
                if prefix not in sample_files:
                    sample_files[prefix] = {}
                sample_files[prefix][ftype] = os.path.join(data_dir, f)
                break
    complete_samples = {}
    for prefix, files in sample_files.items():
        if len(files) == 3 and all(k in files for k in ["barcodes", "features", "matrix"]):
            complete_samples[prefix] = files
        else:
            print(f"[警告] 样本 {prefix} 文件不完整，已跳过。现有文件: {list(files.keys())}")
    return complete_samples
def read_10x_from_files(barcodes_path, features_path, matrix_path):
    # 读取barcodes
    if barcodes_path.endswith(".gz"):
        barcodes = pd.read_csv(barcodes_path, header=None, compression="gzip")[0].values
    else:
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values
    # 读取features/genes
    if features_path.endswith(".gz"):
        features = pd.read_csv(features_path, header=None, sep="\t", compression="gzip")
    else:
        features = pd.read_csv(features_path, header=None, sep="\t")
    gene_ids = features[0].values
    gene_names = features[1].values if features.shape[1] > 1 else gene_ids
    # 读取matrix
    if matrix_path.endswith(".gz"):
        import subprocess
        # scipy.io.mmread 可直读取matrix
    if matrix_path.endswith(".gz"):
        import subprocess
        # scipy.io.mmread 可直接读取gzip文件
        matrix = scipy.io.mmread(matrix_path).T.tocsr()
    else:
        matrix = scipy.io.mmread(matrix_path).T.tocsr()
    # 构建AnnData
    adata = sc.AnnData(X=matrix)
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = pd.Index(gene_names)
    adata.var["gene_ids"] = gene_ids
    return adata
sample_groups = find_10x_sample_groups(data_dir)
adatas = []
metadata_records = []
for prefix, files in sorted(sample_groups.items()):
    # 读取数据
    adata = read_10x_from_files(
        files["barcodes"],
        files["features"],
        files["matrix"]
    )
    adata.var_names_make_unique()
    # 从文件名解析元数据
    info = parse_sample_info_from_dirname(prefix)
    # 统一贴标签 —— 对所有样本使用相同的字段
    # 缺失的字段统一标记为 "none" 或 "unknown"
    adata.obs["sample_id"] = info["sample_id"]
    adata.obs["patient"] = info["patient"]
    adata.obs["tissue"] = info["tissue"]
    adata.obs["disease"] = info["disease"]
    adata.obs["cancer_type"] = info["cancer_type"]
    adata.obs["treatment_drug"] = info["treatment_drug"]
    adata.obs["treatment_timepoint"] = info["treatment_timepoint"]
    adata.obs["sample_prefix"] = prefix
    # 给barcode加前缀防止重复
    adata.obs_names = [f"{info['sample_id']}_{bc}" for bc in adata.obs_names]
    adatas.append(adata)
    metadata_records.append(info)
# 输出元数据汇总表
metadata_df = pd.DataFrame(metadata_records)
metadata_df.to_csv("C:\\Users\\49625\\Desktop\\大创相关\\code\\data\\GSE139324_sample_metadata.csv", index=False)
print(f"\n样本元数据汇总表已保存至: GSE139324_sample_metadata.csv")
#合并所有样本
adata_combined = sc.concat(adatas, join="inner", merge="same")
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
output_dir = "C:\\Users\\49625\\Desktop\\大创相关\\code\\data\\GSE139324_filtered_h5ad"
os.makedirs(output_dir, exist_ok=True)
unique_sample_ids = metadata_df["sample_id"].unique()
print(f"找到 {len(unique_sample_ids)} 个唯一样本ID")
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
        "n_cells": adata_sample.n_obs,
        "n_genes": adata_sample.n_vars,
        "file_path": output_path
    })
    
