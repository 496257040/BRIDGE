import scanpy as sc
import celltypist
from celltypist import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered.h5ad")
print(f"数据维度: {adata.shape}")
print(f"obs列: {list(adata.obs.columns)}")
leiden_key = "leiden_0.5"
# 检查是否已有该分辨率的聚类结果
existing_leiden = [col for col in adata.obs.columns if "leiden" in col.lower()]
print(f"已有的Leiden列: {existing_leiden}")
if leiden_key not in adata.obs.columns:
    # 也检查preview版本
    if "leiden_0.5_preview" in adata.obs.columns:
        adata.obs[leiden_key] = adata.obs["leiden_0.5_preview"]
        print(f"✅ 已将 'leiden_0.5_preview' 重命名为 '{leiden_key}'")
    else:
        print(f"未找到 {leiden_key}，正在计算...")
        sc.tl.leiden(adata, resolution=0.5, random_state=114514, key_added=leiden_key)
        print(f"✅ Leiden聚类完成")
n_clusters = adata.obs[leiden_key].nunique()
print(f"聚类数: {n_clusters}")
print(f"各聚类细胞数:\n{adata.obs[leiden_key].value_counts().sort_index()}")
# CellTypist自动注释
# 下载模型
#models.download_models(force_update=False) 
model_name = "Immune_All_Low.pkl"
try:
    model = models.Model.load(model=model_name)
    print(f"✅ 模型加载成功: {model_name}")
except:
    print(f"模型 {model_name} 未找到，正在下载...")
    models.download_models(force_update=False, model=model_name)
    model = models.Model.load(model=model_name)
    print(f"✅ 模型下载并加载成功: {model_name}")
print(f"模型包含的细胞类型数: {len(model.cell_types)}")
# 检查当前数据状态
print("\n检查数据归一化状态...")
if hasattr(adata, "raw") and adata.raw is not None:
    print("检测到 adata.raw 存在")
    x_sample = adata.raw.X[:5, :5]
    if hasattr(x_sample, "toarray"):
        x_sample = x_sample.toarray()
    print(f"  raw.X 前5x5值:\n  {x_sample}")
x_check = adata.X[:5, :5]
if hasattr(x_check, "toarray"):
    x_check = x_check.toarray()
print(f"adata.X 前5x5值:\n  {x_check}")
max_val = adata.X.max()
if hasattr(max_val, "toarray"):
    max_val = max_val.toarray().item()
print(f"adata.X 最大值: {max_val:.4f}")
# 如果X看起来已经是log归一化的（最大值通常<15），直接使用
# 如果X是原始计数（最大值很大），需要先归一化
if max_val > 20:
    print("⚠️ X看起来可能是原始计数，创建归一化副本用于CellTypist...")
    adata_ct = adata.copy()
    sc.pp.normalize_total(adata_ct, target_sum=1e4)
    sc.pp.log1p(adata_ct)
else:
    print("✅ X看起来已经是log归一化数据，可直接用于CellTypist")
    adata_ct = adata.copy()
#  运行CellTypist预测
print("\n正在运行CellTypist预测...")
predictions = celltypist.annotate(
    adata_ct,
    model=model_name,
    majority_voting=True,
    over_clustering=leiden_key 
)
adata_result = predictions.to_adata()
adata.obs["celltypist_predicted"] = adata_result.obs["predicted_labels"]
adata.obs["celltypist_majority"] = adata_result.obs["majority_voting"]
if "conf_score" in adata_result.obs.columns:
    adata.obs["celltypist_conf"] = adata_result.obs["conf_score"]
print("\n✅ CellTypist注释完成！")
print(f"\n各聚类的多数投票注释结果:")
cluster_annotation = adata.obs.groupby(leiden_key)["celltypist_majority"].agg(
    lambda x: x.value_counts().index[0]
)
for cluster_id, cell_type in cluster_annotation.items():
    n_cells = (adata.obs[leiden_key] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {cell_type} ({n_cells} cells)")
# 释放副本
del adata_ct, adata_result
# 保存结果
output_path = r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist.h5ad"
adata.write_h5ad(output_path)