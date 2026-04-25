import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# 字体设置
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist_annotated.h5ad")
print(f"数据维度: {adata.shape}")
print(f"obs列: {list(adata.obs.columns)}")
possible_source_cols = [col for col in adata.obs.columns 
                        if col.lower() in ["source", "tissue", "origin", 
                                            "tissue_type", "sample_type",
                                            "condition", "group"]]
print(f"可能的组织来源列: {possible_source_cols}")
if len(possible_source_cols) > 0:
    source_col = possible_source_cols[0]
    print(f"使用 '{source_col}' 列作为组织来源")
print(f"组织来源类别: {sorted(adata.obs[source_col].unique())}")
print(f"各来源细胞数:\n{adata.obs[source_col].value_counts()}")
annot_col = "celltypist_majority"
print(f"\n使用细胞注释列: '{annot_col}'")
print(f"细胞类型数: {adata.obs[annot_col].nunique()}")
# 计算各组织来源中每种细胞类型的比例
source_order = ["tumor", "normal", "LM", "PBMC"]
available_sources = [s for s in source_order if s in adata.obs[source_col].values]
print(f"可用的组织来源（按指定顺序）: {available_sources}")
count_table = pd.crosstab(
    adata.obs[annot_col], 
    adata.obs[source_col]
)
# 只保留目标来源列
count_table = count_table[[s for s in available_sources if s in count_table.columns]]
# 计算每个来源内的比例（列归一化）
prop_table = count_table.div(count_table.sum(axis=0), axis=1) * 100
# 按tumor中的比例降序排列细胞类型
if "tumor" in prop_table.columns:
    sort_col = "tumor"
elif len(prop_table.columns) > 0:
    sort_col = prop_table.columns[0]
prop_table = prop_table.sort_values(by=sort_col, ascending=False)
count_table = count_table.loc[prop_table.index]
print(f"\n细胞比例表 (%):")
print(prop_table.round(2).to_string())
# 图1 —— 堆叠柱状图（每个来源一根柱子）
n_celltypes = len(prop_table.index)
if n_celltypes <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_celltypes]
else:
    colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_celltypes))
color_dict = {ct: colors[i] for i, ct in enumerate(prop_table.index)}
fig1, ax1 = plt.subplots(figsize=(10, 8))
# 堆叠柱状图
bottom = np.zeros(len(available_sources))
x_pos = np.arange(len(available_sources))
bar_width = 0.6
for ct in prop_table.index:
    values = [prop_table.loc[ct, s] if s in prop_table.columns else 0 
              for s in available_sources]
    ax1.bar(x_pos, values, bar_width, bottom=bottom, 
            label=ct, color=color_dict[ct], edgecolor="white", linewidth=0.3)
    bottom += values
ax1.set_xticks(x_pos)
ax1.set_xticklabels(available_sources, fontsize=12, fontweight="bold")
ax1.set_ylabel("Cell Proportion (%)", fontsize=12)
ax1.set_title("Cell Type Composition by Tissue Origin", fontsize=14, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, 
           ncol=1 if n_celltypes <= 15 else 2, frameon=True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
fig1_path = r"C:\Users\49625\Desktop\大创相关\code\image\barplot_stacked_by_source.png"
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
# plt.show()
print(f"✅ 堆叠柱状图已保存至: {fig1_path}")
# 图2 —— 分组柱状图（每种细胞类型一组）
prop_long = prop_table.reset_index().melt(
    id_vars=annot_col, 
    var_name="Tissue Origin", 
    value_name="Proportion (%)"
)
# 来源配色
source_colors = {
    "tumor": "#E64B35",
    "normal": "#4DBBD5", 
    "LM": "#00A087",
    "PBMC": "#3C5488"
}
fig2, ax2 = plt.subplots(figsize=(max(n_celltypes * 1.2, 14), 7))
n_sources = len(available_sources)
bar_width = 0.8 / n_sources
x_pos = np.arange(n_celltypes)
for i, source in enumerate(available_sources):
    values = [prop_table.loc[ct, source] if source in prop_table.columns else 0 
              for ct in prop_table.index]
    offset = (i - n_sources / 2 + 0.5) * bar_width
    bars = ax2.bar(x_pos + offset, values, bar_width, 
                   label=source, color=source_colors.get(source, f"C{i}"),
                   edgecolor="white", linewidth=0.3)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(prop_table.index, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Cell Proportion (%)", fontsize=12)
ax2.set_title("Cell Type Proportion Comparison Across Tissue Origins", 
              fontsize=14, fontweight="bold")
ax2.legend(title="Tissue Origin", fontsize=10, title_fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
fig2_path = r"C:\Users\49625\Desktop\大创相关\code\image\barplot_grouped_by_source.png"
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
# plt.show()
print(f"✅ 分组柱状图已保存至: {fig2_path}")
# 图3 —— 热图展示比例
fig3, ax3 = plt.subplots(figsize=(8, max(n_celltypes * 0.4, 6)))
sns.heatmap(
    prop_table,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Proportion (%)", "shrink": 0.8},
    ax=ax3,
    square=False
)
ax3.set_title("Cell Type Proportion Heatmap\nby Tissue Origin", 
              fontsize=13, fontweight="bold", pad=15)
ax3.set_xlabel("Tissue Origin", fontsize=11)
ax3.set_ylabel("Cell Type", fontsize=11)
ax3.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig3_path = r"C:\Users\49625\Desktop\大创相关\code\image\heatmap_proportion_by_source.png"
fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
# plt.show()
print(f"✅ 比例热图已保存至: {fig3_path}")
# 图4 —— 按组织来源分面的UMAP
fig4, axes = plt.subplots(1, len(available_sources), 
                           figsize=(6 * len(available_sources), 6))
if len(available_sources) == 1:
    axes = [axes]
for i, source in enumerate(available_sources):
    mask = adata.obs[source_col] == source
    n_cells = mask.sum()
    
    # 绘制灰色背景（所有细胞）
    axes[i].scatter(
        adata.obsm["X_umap"][:, 0],
        adata.obsm["X_umap"][:, 1],
        s=0.5, c="lightgrey", alpha=0.3, rasterized=True
    )
    
    # 高亮当前来源的细胞，按注释着色
    adata_sub = adata[mask]
    cell_types_sub = adata_sub.obs[annot_col].values
    unique_ct = sorted(set(cell_types_sub))
    
    for ct in unique_ct:
        ct_mask = cell_types_sub == ct
        axes[i].scatter(
            adata_sub.obsm["X_umap"][ct_mask, 0],
            adata_sub.obsm["X_umap"][ct_mask, 1],
            s=1.5, c=[color_dict.get(ct, "grey")], 
            alpha=0.6, label=ct, rasterized=True
        )
    
    axes[i].set_title(f"{source}\n(n={n_cells:,})", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("UMAP1", fontsize=10)
    if i == 0:
        axes[i].set_ylabel("UMAP2", fontsize=10)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
# 统一图例放在最后
handles = [Patch(facecolor=color_dict[ct], label=ct) 
           for ct in prop_table.index if ct in color_dict]
fig4.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc="center left",
            fontsize=7, ncol=1, frameon=True, title="Cell Type", title_fontsize=9)
plt.suptitle("UMAP Split by Tissue Origin", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig4_path = r"C:\Users\49625\Desktop\大创相关\code\image\umap_split_by_source.png"
fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
# plt.show()
print(f"✅ 分面UMAP已保存至: {fig4_path}")
