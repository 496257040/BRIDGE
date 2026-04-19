import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import harmonypy
np.random.seed(114514)
qc_dir = r"E:\数据集\filtered_h5ad\CRC1合并"
adatas = []
# 定义列名映射
column_mapping = {
    'gsm_id': 'sample_id',
    'patient_id': 'patient',
    'tissue_type': 'tissue',
    'disease_type': 'disease',
    'treatment': 'treatment_drug',
    'timepoint': 'treatment_timepoint',
    'batch': 'dataset'
}
for subfolder in os.listdir(qc_dir):
    subfolder_path = os.path.join(qc_dir, subfolder)
    # 确保是文件夹
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的.h5ad文件
        for f in sorted(os.listdir(subfolder_path)):
            if f.endswith(".h5ad"):
                filepath = os.path.join(subfolder_path, f)
                ad = sc.read_h5ad(filepath)
                # 在添加元数据前，先统一列名
                # 应用重命名
                for old_col, new_col in column_mapping.items():
                    if old_col in ad.obs.columns:
                        ad.obs[new_col] = ad.obs[old_col]
                        ad.obs.drop(old_col, axis=1, inplace=True)
                # 添加来源信息到obs中
                ad.obs['dataset'] = subfolder
                ad.obs['sample_file'] = f
                adatas.append(ad)
# 合并前检查列名是否统一
if adatas:
    # 找出所有数据集的共有列
    common_cols = set(adatas[0].obs.columns)
    for ad in adatas[1:]:
        common_cols = common_cols.intersection(set(ad.obs.columns))
    # 只保留共有列
    for i in range(len(adatas)):
        adatas[i] = adatas[i][:, :].copy()  # 保持所有基因
        # 只保留共有列
        cols_to_keep = [col for col in adatas[i].obs.columns if col in common_cols]
        adatas[i].obs = adatas[i].obs[cols_to_keep]
    # 合并
    adata = sc.concat(adatas, join="inner", index_unique="-")
    adata.var_names_make_unique()
    del adatas
    print(f"\n合并完成: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")
    print(f"合并后共有列名: {list(adata.obs.columns)}")
else:
    print("没有找到任何 .h5ad 文件!")
# 标准化
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
# 高变基因筛选
sc.pp.highly_variable_genes(
    adata, 
    n_top_genes=2000, 
    batch_key="sample_id",
    flavor='seurat_v3'  
)
adata = adata[:, adata.var.highly_variable].copy()
print(f"\n高变基因筛选后: {adata.n_vars} 个基因")
# 回归与缩放
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(adata, max_value=10)
# PCA降维
sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
# Harmony批次效应校正
try:
    from harmonypy import run_harmony
    pca_matrix = adata.obsm["X_pca"]
    meta_data = pd.DataFrame({"sample_id": adata.obs["sample_id"].values}, index=adata.obs_names)
    harmony_out = run_harmony(
        pca_matrix,
        meta_data,
        "sample_id",
        max_iter_harmony=20
    )
    Z_corr = harmony_out.Z_corr
    print(f"Z_corr shape: {Z_corr.shape}")
    print(f"期望 shape: ({adata.n_obs}, {pca_matrix.shape[1]})")
    if Z_corr.shape == (adata.n_obs, pca_matrix.shape[1]):
        # 新版：已经是 (n_cells, n_components)，直接赋值
        adata.obsm["X_pca_harmony"] = np.array(Z_corr)
        print("检测到新版harmonypy输出格式，直接赋值（无需转置）")
    elif Z_corr.shape == (pca_matrix.shape[1], adata.n_obs):
        # 旧版：是 (n_components, n_cells)，需要转置
        adata.obsm["X_pca_harmony"] = np.array(Z_corr.T)
        print("检测到旧版harmonypy输出格式，已转置后赋值")
    else:
        raise ValueError(f"Z_corr shape异常: {Z_corr.shape}")
    print(f"成功写入 X_pca_harmony，shape: {adata.obsm['X_pca_harmony'].shape}")
    use_rep = "X_pca_harmony"
except ImportError:
    print("[提示] 未安装harmonypy，跳过批次校正")
    use_rep = "X_pca"
except Exception as e:
    print(f"[警告] Harmony运行出错: {e}")
    use_rep = "X_pca"
# 邻域图构建与UMAP可视化
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30, use_rep=use_rep)
sc.tl.umap(adata)
# Leiden聚类
sc.tl.leiden(adata, resolution=0.8, key_added="leiden")

marker_genes = {
    # ====== 核心免疫细胞大类 ======
    "T cells": ["CD3D", "CD3E"],
    "CD4+ T cells": ["CD4", "IL7R"],
    "CD8+ T cells": ["CD8A", "CD8B"],
    "NK cells": ["NKG7", "GNLY"],
    "B cells": ["CD79A", "MS4A1"],
    "Plasma cells": ["MZB1", "SDC1"],
    "Monocytes": ["CD14", "LYZ"],
    "Macrophages": ["CD68", "CD163"],
    "cDC (常规树突状细胞)": ["CLEC9A", "CD1C"],  # cDC1:CLEC9A, cDC2:CD1C
    "pDC (浆细胞样树突状细胞)": ["LILRA4", "GZMB"],
    "Neutrophils": ["CSF3R", "S100A9"],
    # ====== 关键功能/状态亚群 (对分析至关重要) ======
    "Regulatory T cells (Treg)": ["FOXP3", "IL2RA"],
    "Exhausted CD8+ T cells": ["PDCD1", "HAVCR2"],
    "Cycling T cells (增殖)": ["MKI67", "TOP2A"],
    # ====== 非免疫/基质细胞 (CD45-) ======
    "Epithelial cells": ["EPCAM", "KRT19"],
    "CRC Tumor cells (结直肠肿瘤细胞)": ["CEACAM5", "CEACAM6"],  # CRC特异性
    "Fibroblasts": ["COL1A1", "DCN"],
    "Cancer-associated fibroblasts (CAF)": ["FAP", "ACTA2"],  # 癌症相关成纤维细胞
    "Endothelial cells": ["PECAM1", "VWF"],
}
cell_type_annotation = {
    "0": "CD8+ T", "1": "CD8+ T", "2": "T cells", "3": "CD4+T",
    "4": "Monocytes", "5": "NK cells", "6": "???", "7": "Treg",
    "8": "CRC tumor cells", "9": "B cells", "10": "Cycling T cells", "11": "Plasma cells",
    "12": "CTL", "13": "Fibroblasts", "14": "?????", "15": "T cells (ambiguous)", "16": "pDC", 
}
marker_list = [gene for genes in marker_genes.values() for gene in genes]
marker_list_filtered = [g for g in marker_list if g in adata.var_names]  
dotplot = sc.pl.dotplot(
    adata,
    var_names=marker_list_filtered,
    groupby="leiden",
    standard_scale="var",
    show=False,
    figsize=(12, 8)
)
plt.tight_layout()
save_path = r"C:\Users\49625\Desktop\大创相关\code\image\Dotplot_marker_genes_CRC.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
adata.obs["cell_type"] = adata.obs["leiden"].map(cell_type_annotation)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, legend_loc="on data",
           title="Leiden Clustering (r=0.8)", frameon=False)
sc.pl.umap(adata, color="tissue", ax=axes[1], show=False,
           title="Tissue Source", frameon=False)
sc.pl.umap(adata,color="cell_type",title="Annotated Cell Types",ax=axes[2],show=False,
    frameon=True,size=8,legend_loc="on data",legend_fontsize=8)
plt.tight_layout()
plt.savefig(r"C:\Users\49625\Desktop\大创相关\code\image\UMAP_Leiden_Harmony.png", dpi=300, bbox_inches="tight")
plt.show()
output_final = r"E:\数据集\filtered_h5ad\CRC1合并\adata_final.h5ad"
adata.write(output_final)
print(f"\n聚类结果已保存至: {output_final}")
