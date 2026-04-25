import scanpy as sc
import celltypist
from celltypist import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist.h5ad")
adata.obs["cell_annotation"] = adata.obs["celltypist_majority"].astype("category")
leiden_key = "leiden_0.5"
n_clusters = adata.obs[leiden_key].nunique()
# 运行rank_genes_groups（Wilcoxon检验）
print("正在进行差异表达分析（Wilcoxon检验）...")
sc.tl.rank_genes_groups(
    adata,
    groupby="cell_annotation",
    method="wilcoxon",
    pts=True,  # 计算表达比例
    n_genes=200,
    use_raw=True if adata.raw is not None else False
)
print("✅ 差异表达分析完成")
# 提取每种细胞类型的Top 4基因
n_top_genes = 4
cell_types_ordered = []
genes_ordered = []
gene_to_celltype = {}
# 获取所有注释类型，按cluster编号排序
cluster_to_ct = {}
for cluster_id in sorted(adata.obs[leiden_key].cat.categories, key=lambda x: int(x)):
    ct = adata.obs.loc[adata.obs[leiden_key] == cluster_id, "celltypist_majority"].value_counts().index[0]
    cluster_to_ct[cluster_id] = ct
# 去重但保持顺序（同一注释可能对应多个cluster）
seen_ct = set()
unique_ct_ordered = []
cluster_ct_pairs = []
for cluster_id, ct in cluster_to_ct.items():
    cluster_ct_pairs.append((cluster_id, ct))
    if ct not in seen_ct:
        seen_ct.add(ct)
        unique_ct_ordered.append(ct)
print(f"\n注释类型顺序（按cluster编号排列）:")
for i, ct in enumerate(unique_ct_ordered):
    clusters = [cid for cid, c in cluster_ct_pairs if c == ct]
    print(f"  {i+1}. {ct} (Cluster: {', '.join(clusters)})")
# 从rank_genes_groups中提取每种细胞类型的top基因
result = adata.uns["rank_genes_groups"]
print(f"\n每种细胞类型的Top {n_top_genes} 特征基因:")
for ct in unique_ct_ordered:
    # 获取该细胞类型的差异基因排名
    ct_idx = list(result["names"].dtype.names).index(ct) if ct in result["names"].dtype.names else None
    if ct_idx is not None or ct in result["names"].dtype.names:
        top_genes = [result["names"][i][ct] for i in range(n_top_genes)]
        top_scores = [result["scores"][i][ct] for i in range(n_top_genes)]
        top_logfc = [result["logfoldchanges"][i][ct] for i in range(n_top_genes)]
        print(f"\n  {ct}:")
        for g, s, lfc in zip(top_genes, top_scores, top_logfc):
            print(f"    {g:15s}  score={s:.2f}  logFC={lfc:.2f}")
        # 避免基因重复：如果某基因已被前面的细胞类型选中，则跳过取下一个
        added = 0
        gene_idx = 0
        while added < n_top_genes and gene_idx < 200:
            gene = result["names"][gene_idx][ct]
            if gene not in gene_to_celltype:
                genes_ordered.append(gene)
                gene_to_celltype[gene] = ct
                added += 1
            gene_idx += 1
        cell_types_ordered.append(ct)
print(f"\n总计选择基因数: {len(genes_ordered)}")
print(f"基因顺序（前20个）: {genes_ordered[:20]}")
# 生成阶梯状DotPlot
cluster_labels = {}
for cluster_id, ct in cluster_to_ct.items():
    cluster_labels[cluster_id] = f"C{cluster_id}: {ct}"
adata.obs["cluster_label"] = adata.obs[leiden_key].map(cluster_labels).astype("category")
# 5.2 按照注释类型顺序排列cluster
# 同一注释类型的cluster排在一起
ordered_labels = []
for ct in unique_ct_ordered:
    matching_clusters = sorted(
        [cid for cid, c in cluster_ct_pairs if c == ct],
        key=lambda x: int(x)
    )
    for cid in matching_clusters:
        ordered_labels.append(cluster_labels[cid])
# 设置category顺序
adata.obs["cluster_label"] = adata.obs["cluster_label"].cat.reorder_categories(ordered_labels)
# 5.3 构建基因分组字典（用于DotPlot中的基因分隔线）
var_group_positions = []
var_group_labels = []
current_pos = 0
for ct in unique_ct_ordered:
    ct_genes = [g for g in genes_ordered if gene_to_celltype[g] == ct]
    if len(ct_genes) > 0:
        var_group_positions.append((current_pos, current_pos + len(ct_genes) - 1))
        # 截断过长的名称
        label = ct if len(ct) <= 20 else ct[:17] + "..."
        var_group_labels.append(label)
        current_pos += len(ct_genes)
# 5.4 绘制DotPlot
# print("正在绘制DotPlot...")
# dp = sc.pl.dotplot(
#     adata,
#     var_names=genes_ordered,
#     groupby="cluster_label",
#     categories_order=ordered_labels,
#     var_group_positions=var_group_positions,
#     var_group_labels=var_group_labels,
#     var_group_rotation=45,
#     standard_scale="var",  # 按基因标准化，使阶梯更明显
#     cmap="Reds",
#     colorbar_title="Scaled\nExpression",
#     size_title="Fraction of\ncells (%)",
#     figsize=(max(len(genes_ordered) * 0.5, 14), max(len(ordered_labels) * 0.5, 8)),
#     show=False,
#     return_fig=True,
#     use_raw=True if adata.raw is not None else False
# )
# # 添加分隔线使阶梯更明显
# dp.add_totals(color=["#e6e6e6", "#cccccc"]).style(
#     cmap="Reds",
#     dot_edge_color="black",
#     dot_edge_lw=0.3,
#     grid=True
# )
# fig_path_dot = r"C:\Users\49625\Desktop\大创相关\code\image\CRC_dotplot_staircase.png"
# dp.savefig(fig_path_dot, dpi=200, bbox_inches="tight")
# print(f"✅ DotPlot已保存至: {fig_path_dot}")
# plt.show()
# 同时生成UMAP注释图
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
# 图1：按Leiden聚类着色
sc.pl.umap(adata, color=leiden_key, ax=axes[0], show=False,
           title=f"Leiden Clustering (res=0.5, {n_clusters} clusters)",
           legend_loc="on data", legend_fontsize=8, frameon=True)
# 图2：按CellTypist注释着色
sc.pl.umap(adata, color="celltypist_majority", ax=axes[1], show=False,
           title="CellTypist Annotation (majority voting)",
           legend_loc="on data", legend_fontsize=7, frameon=True)
plt.tight_layout()
fig_path_umap = r"C:\Users\49625\Desktop\大创相关\code\image\CRC_umap_annotations.png"
plt.savefig(fig_path_umap, dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ UMAP注释图已保存至: {fig_path_umap}")
