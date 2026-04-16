import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import gzip
import scipy
from scipy import sparse
from scipy.io import mmread
import anndata as ad
import re
from typing import Dict, List, Optional, Tuple
print("正在运行...")

#======================生成标签==========================
def parse_sample_info_from_dirname(dirname):
    info = {      #标签类型,要包括以下几个，记得标准化命名
        "sample_id": "unknown",
        "patient": "unknown",
        "tissue": "unknown",
        "disease": "CRC",
        "cancer_type": "CRC",
        "treatment_drug": "none or capecitabine & oxaliplatin or 5FU & oxaliplatin & leucovorin & bevacizumab",
        "treatment_timepoint": "none",
        "cell_sorting": "none",
    }
    parts = dirname.split("_")
    # 提取样本ID，通常是以GSM开头的部分
    gsm_id = parts[0] if parts[0].startswith("GSM") else "unknown"
    info["sample_id"] = gsm_id
    # 提取组织来源和疾病状态
    tissue_keywords = ["PBMC", "tumor","LM"]
    for keyword in tissue_keywords:
        if keyword in dirname:
            info["tissue"] = keyword
            break
    simplified = re.sub(r'^GSM\d+_raw_feature_bc_matrix_', '', dirname)
    simplified = re.sub(r'^GSM\d+_', '', simplified) if simplified == dirname else simplified
    # 提取患者编号，假设格式为P1, P2等，或者P1-P2等
    # 提取一个或多个以短横线连接的完整患者编号（如 COL07 或 COL07-COL12）
    patient_match = re.search(r'((?:COL\d+)(?:-COL\d+)*)', simplified)
    if patient_match:
        info['patient'] = patient_match.group(1)  # 直接获取完整编号
    tx_match = re.search(r'(post-treatment)', simplified, re.IGNORECASE)
    # 提取治疗时间点，假设包含pre-Tx或post-Tx等关键词
    if tx_match:
        raw_tx = tx_match.group(1).lower()
        if "post-treatment" in raw_tx:
            info["treatment_timepoint"] = "post-treatment"

    # 编译正则表达式模式（保持不变）
    tg_match = re.compile(
        r'(capecitabine\s*[&+]\s*oxaliplatin|5FU\s*[&+]\s*oxaliplatin\s*[&+]\s*leucovorin\s*[&+]\s*bevacizumab)',
        re.IGNORECASE)

    # 在目标字符串（例如 filename）中进行匹配
    match_result = tg_match.search(filename)  # 或使用 match 方法，根据需求
    # 提取治疗药物
    if match_result:
        raw_tg = match_result.group(1)
        if "capecitabine & oxaliplatin" in raw_tg:
            info["treatment_drug"] = "capecitabine & oxaliplatin"
        elif "5FU & oxaliplatin & leucovorin & bevacizumab" in raw_tg:
            info["treatment_drug"] = "5FU & oxaliplatin & leucovorin & bevacizumab"
    # 提取细胞分选信息
    sort_match = re.search(r'sorted_(.+?)_(PBMC|tumor|LM|Tumor|TUMOR|Normal)', simplified, re.IGNORECASE)
    if sort_match:
        info["cell_sorting"] = sort_match.group(1)
    print(info)
    return info

 #===========================生成gsm样本的详细信息，供标准化标签使用==========================
 #从GEO系列矩阵文件中提取特定GSM样本的详细信息
def extract_sample_info(series_matrix_file, target_gsm):

    desired_attributes = ["Sample_geo_accession","Sample_characteristics_treatment", "Sample_characteristics_tissue","Sample_characteristics_age","Sample_characteristics_sex","Sample_source_name_ch1"]

    # 读取文件
    with open(series_matrix_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析文件内容
    sample_data = {}
    current_attribute = None
    gsm_columns = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')

        if line.startswith('!'):
            attr_name = line[1:].split('\t')[0] if '\t' in line else line[1:]

            if attr_name.lower() in ['sample_geo_accession']:
                gsm_columns = parts[1:]

            current_attribute = attr_name

        if current_attribute and len(parts) > 1:
            if current_attribute not in sample_data:
                sample_data[current_attribute] = {}

            for i, gsm in enumerate(gsm_columns):
                if i + 1 < len(parts):
                    sample_data[current_attribute][gsm] = parts[i + 1]


    # 提取目标GSM样本的指定属性信息
    if target_gsm in gsm_columns and sample_data:
        print(f"\n{'=' * 60}")
        print(f"样本 {target_gsm} 的指定属性信息:")
        print('=' * 60)

        result = {}
        for attr, samples_dict in sample_data.items():
            # 只提取我们感兴趣的属性
            if attr in desired_attributes and target_gsm in samples_dict:
                if attr == ''"Sample_characteristics_age"'' :
                    if sample_data[attr][target_gsm] == '"age: 58"':
                        value = "COL07"
                    elif sample_data[attr][target_gsm] == '"age: 65"':
                        value = "COL12"
                    elif sample_data[attr][target_gsm] == '"age: 35"':
                        value = "COL15"
                    elif sample_data[attr][target_gsm] == '"age: 71"':
                        value = "COL16"
                    elif sample_data[attr][target_gsm] == '"age: 52"':
                        value = "COL17"
                    elif sample_data[attr][target_gsm] == '"age: 46"':
                        value = "COL18"
                    result[attr] = value
                elif attr == ''"Sample_characteristics_sex"'':
                    attr1 = ''"Sample_characteristics_age"''
                    if sample_data[attr1][target_gsm] == '"age: 58"':
                        value = "none"
                    elif sample_data[attr1][target_gsm] == '"age: 65"':
                        value = "none"
                    elif sample_data[attr1][target_gsm] == '"age: 35"':
                        value = "capecitabine & oxaliplatin"
                    elif sample_data[attr1][target_gsm] == '"age: 71"':
                        value = "none"
                    elif sample_data[attr1][target_gsm] == '"age: 52"':
                        value = "capecitabine & oxaliplatin"
                    elif sample_data[attr1][target_gsm] == '"age: 46"':
                        value = "5FU & oxaliplatin & leucovorin & bevacizumab"
                    result[attr] = value
                elif attr == ''"Sample_source_name_ch1"'':
                    value = "CRC"
                    result[attr] = value
                else:
                    value = samples_dict[target_gsm]
                    result[attr] = value
        # 保存结果到字符串
        if result:
            # 将结果字典的所有值用下划线连接成字符串
            result_str = "_".join(str(value) for value in result.values() if value)
            result_clean = (result_str.replace('"', '').
                            replace("tissue: matched liver metastases", "LM").
                            replace('treatment: treated', 'post-treatment').
                            replace('tissue: PBMC', 'PBMC').
                            replace('tissue: primary colorectal cancer', 'tumor').
                            replace('treatment: treatment-naïve', 'none'))
            print(f"信息已转换为字符串: {result_clean}")
            return result_clean

if __name__ == "__main__":
    # 文件路径
    series_matrix_file = "GSE178318_series_matrix.txt"
    input_dir=r'C:\Users\14584\Desktop\数据贴标签 - 0407\GSE178318'
    out_dir=r'C:\Users\14584\Desktop\数据贴标签 - 0407\GSE178318\各个样本数据'
    # 1. 使用提供的样本信息字典
    sample_info_list = []
    for a in range(5387660,5387675):
        target_gsm='"'+"GSM"+str(a) +'"'   # 替换为您要查找的GSM编号
        filename=extract_sample_info(series_matrix_file,target_gsm)
        output_info=parse_sample_info_from_dirname(filename)
        sample_info_list.append(output_info)
    print("正在质控和分组...")

    # 2. 根据barcode格式创建映射表
    sample_mapping = {}

    for sample in sample_info_list:
        patient = sample['patient']
        tissue = sample['tissue']

        # 将tissue转换为barcode中的缩写
        if tissue == 'tumor':
            tissue_abbr = "CRC"  # tumor对应CRC
        elif tissue == 'PBMC':
            tissue_abbr = "PBMC"  # PBMC对应PBMC
        elif tissue == 'LM':
            tissue_abbr = "LM"  # LM对应LM
        else:
            tissue_abbr = tissue.upper()  # 其他情况转为大写

        sample_mapping[(patient, tissue_abbr)] = sample
        print(f"创建映射: ({patient}, {tissue_abbr}) -> {sample['sample_id']}")

    # 3. 手动读取10x Genomics数据
    matrix_file = "GSE178318_matrix.mtx.gz"
    barcodes_file = "GSE178318_barcodes.tsv.gz"
    features_file = "GSE178318_features.tsv.gz"

    if not all([os.path.exists(matrix_file), os.path.exists(barcodes_file), os.path.exists(features_file)]):
        exit(1)

    # 读取矩阵
    with gzip.open(matrix_file, 'rb') as f:
        X = mmread(f).T.tocsr()

    # 读取条形码
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = pd.read_csv(f, header=None, sep='\t')[0].values

    # 读取特征
    with gzip.open(features_file, 'rt') as f:
        features_df = pd.read_csv(f, header=None, sep='\t')

    # 处理特征文件
    if features_df.shape[1] == 2:
        features_df[2] = "Gene Expression"
    elif features_df.shape[1] == 1:
        features_df[1] = features_df[0]
        features_df[2] = "Gene Expression"

    if features_df.shape[1] >= 3:
        gene_ids = features_df[0].values
        gene_symbols = features_df[1].values
        gene_types = features_df[2].values

        var = pd.DataFrame({
            'gene_ids': gene_ids,
            'gene_symbols': gene_symbols,
            'feature_types': gene_types
        })
        var.index = gene_symbols
    else:
        gene_ids = features_df[0].values
        gene_symbols = gene_ids
        gene_types = ["Gene Expression"] * len(gene_ids)

        var = pd.DataFrame({
            'gene_ids': gene_ids,
            'gene_symbols': gene_symbols,
            'feature_types': gene_types
        })
        var.index = gene_symbols

    # 创建obs DataFrame
    obs = pd.DataFrame(index=barcodes)

    # 创建AnnData对象
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()

    # 4. 根据barcode格式分配细胞到样本
    cell_to_sample = {}
    unassigned_cells = []

    # 打印前几个barcode示例
    print("\n前10个barcode示例:")
    for i, barcode in enumerate(adata.obs_names[:10]):
        print(f"  {i}: {barcode}")

    for i, barcode in enumerate(adata.obs_names):
        barcode_str = str(barcode)
        assigned = False

        # 方法1: 尝试标准格式 _COLxx_CRC
        parts = barcode_str.split('_')
        if len(parts) >= 3:
            for j in range(len(parts) - 1):
                if parts[j].startswith("COL") and parts[j][3:].isdigit():
                    patient_part = parts[j]
                    if j + 1 < len(parts):
                        tissue_abbr = parts[j + 1]
                    else:
                        tissue_abbr = ""

                    # 检查映射
                    if (patient_part, tissue_abbr) in sample_mapping:
                        cell_to_sample[i] = sample_mapping[(patient_part, tissue_abbr)]
                        assigned = True
                        break

        # 方法2: 尝试从barcode中查找COLxx和CRC/LM/PBMC
        if not assigned:
            for (patient_key, tissue_key), sample_info in sample_mapping.items():
                # 如果barcode包含患者信息和组织信息
                if patient_key in barcode_str and tissue_key in barcode_str:
                    cell_to_sample[i] = sample_info
                    assigned = True
                    break

        if not assigned:
            unassigned_cells.append(i)

        # 打印进度
        if i % 10000 == 0 and i > 0:
            print(f"  已处理 {i} 个细胞，已分配 {len(cell_to_sample)} 个，未分配 {len(unassigned_cells)} 个")

    # 5. 质量控制
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    # 设置过滤阈值
    min_genes = 200
    max_genes = 5000
    max_mt_percent = 20

    # 应用过滤
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    adata = adata[adata.obs['pct_counts_mt'] < max_mt_percent, :]
    sc.pp.filter_genes(adata, min_cells=3)

    # 6. 按样本分割数据并保存
    output_dir = "gsm_samples"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    sample_cell_counts = {}

    # 按样本ID分组细胞索引
    sample_to_cells = {}
    for cell_idx, sample_info in cell_to_sample.items():
        sample_id = sample_info['sample_id']
        if cell_idx < adata.n_obs:  # 确保索引在范围内
            if sample_id not in sample_to_cells:
                sample_to_cells[sample_id] = []
            sample_to_cells[sample_id].append(cell_idx)

    # 创建每个样本的AnnData并保存
    for sample_id, cell_indices in sample_to_cells.items():
        if not cell_indices:
            continue

        # 获取这个样本的信息
        sample_info = next((s for s in sample_info_list if s['sample_id'] == sample_id), None)
        if not sample_info:
            print(f"警告: 未找到样本 {sample_id} 的信息")
            continue

        # 提取这个样本的细胞
        sample_adata = adata[cell_indices, :].copy()

        # 添加所有元数据信息
        for key, value in sample_info.items():
            sample_adata.obs[key] = str(value)

        # 额外添加一些通用元数据
        sample_adata.obs['dataset'] = 'GSE178318'
        sample_adata.obs['organism'] = 'Human'

        # 保存为h5ad文件
        output_file = os.path.join(output_dir, f"{sample_id}_filtered.h5ad")
        sample_adata.write_h5ad(output_file)

        sample_cell_counts[sample_id] = sample_adata.n_obs
        saved_files.append(output_file)

        print(f"已保存样本 {sample_id}: {sample_adata.n_obs} 个细胞")

    # 保存未分配的细胞
    if unassigned_cells and len(unassigned_cells) > 0:
        unassigned_indices = [idx for idx in unassigned_cells if idx < adata.n_obs]

        if unassigned_indices:
            unassigned_adata = adata[unassigned_indices, :].copy()

            # 添加基本的元数据
            unassigned_adata.obs['sample_id'] = 'unassigned'
            unassigned_adata.obs['patient'] = 'unknown'
            unassigned_adata.obs['tissue'] = 'unknown'
            unassigned_adata.obs['disease'] = 'colorectal_cancer_liver_metastasis'
            unassigned_adata.obs['dataset'] = 'GSE178318'

            output_file = os.path.join(output_dir, "unassigned_cells_filtered.h5ad")
            unassigned_adata.write_h5ad(output_file)

            saved_files.append(output_file)
            print(f"已保存未分配细胞: {unassigned_adata.n_obs} 个细胞")

    # 创建样本统计信息
    if sample_cell_counts:
        stats_df = pd.DataFrame({
            'sample_id': list(sample_cell_counts.keys()),
            'cell_count': list(sample_cell_counts.values())
        })

        # 添加样本信息到统计表
        for sample_info in sample_info_list:
            sample_id = sample_info['sample_id']
            if sample_id in stats_df['sample_id'].values:
                for key, value in sample_info.items():
                    if key != 'sample_id':  # sample_id已经是列
                        stats_df.loc[stats_df['sample_id'] == sample_id, key] = str(value)

        # 保存统计信息
        stats_file = os.path.join(output_dir, "sample_statistics.csv")
        stats_df.to_csv(stats_file, index=False)

        print(f"\n样本统计信息保存为: {stats_file}")

    print("\n运行完成！")
    print(f"已处理样本数量: {len(sample_cell_counts)}")
    print(f"已分配细胞总数: {sum(sample_cell_counts.values())}")
    print(f"未分配细胞数量: {len(unassigned_cells) if unassigned_cells else 0}")
    print(f"输出文件保存在: {output_dir}")