import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist_annotated.h5ad")
print(f"数据维度: {adata.shape}")
print(f"obs列: {list(adata.obs.columns)}")
tp_col = "treatment_timepoint"
annot_col = "celltypist_majority"
print(f"治疗时间点列: '{tp_col}'")
print(f"治疗时间点类别: {sorted(adata.obs[tp_col].astype(str).unique())}")
print(f"各时间点细胞数:\n{adata.obs[tp_col].value_counts()}")
print(f"\n细胞注释列: '{annot_col}'")
print(f"细胞类型数: {adata.obs[annot_col].nunique()}")
tp_order = ["none", "pre-treatment", "post-treatment"]
# 检查实际存在的时间点
available_tp = [t for t in tp_order if t in adata.obs[tp_col].values]
# 如果有些类别不在预设顺序中，也加进来
extra_tp = [t for t in adata.obs[tp_col].unique() if t not in tp_order]
if extra_tp:
    available_tp.extend(sorted(extra_tp))
    print(f"⚠️ 发现额外时间点类别: {extra_tp}，已追加")
print(f"可用的治疗时间点（按逻辑顺序）: {available_tp}")
# 交叉计数表
count_table = pd.crosstab(
    adata.obs[annot_col],
    adata.obs[tp_col]
)
# 只保留目标时间点列并按指定顺序排列
count_table = count_table[[t for t in available_tp if t in count_table.columns]]
# 计算每个时间点内的比例（列归一化）
prop_table = count_table.div(count_table.sum(axis=0), axis=1) * 100
# 按pre-treatment中的比例降序排列细胞类型
if "pre-treatment" in prop_table.columns:
    sort_col = "pre-treatment"
elif len(prop_table.columns) > 0:
    sort_col = prop_table.columns[0]
prop_table = prop_table.sort_values(by=sort_col, ascending=False)
count_table = count_table.loc[prop_table.index]
print(f"\n细胞比例表 (%):")
print(prop_table.round(2).to_string())
# 绘制堆叠柱状图
n_celltypes = len(prop_table.index)
# 配色
if n_celltypes <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_celltypes]
else:
    colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_celltypes))
color_dict = {ct: colors[i] for i, ct in enumerate(prop_table.index)}
fig1, ax1 = plt.subplots(figsize=(8, 8))
bottom = np.zeros(len(available_tp))
x_pos = np.arange(len(available_tp))
bar_width = 0.55
for ct in prop_table.index:
    values = [prop_table.loc[ct, t] if t in prop_table.columns else 0
              for t in available_tp]
    ax1.bar(x_pos, values, bar_width, bottom=bottom,
            label=ct, color=color_dict[ct], edgecolor="white", linewidth=0.3)
    bottom += values
ax1.set_xticks(x_pos)
ax1.set_xticklabels(available_tp, fontsize=11, fontweight="bold")
ax1.set_ylabel("Cell Proportion (%)", fontsize=12)
ax1.set_title("Cell Type Composition by Treatment Timepoint",
              fontsize=14, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
           ncol=1 if n_celltypes <= 15 else 2, frameon=True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
fig1_path = r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_celltype_composition_by_timepoint.png"
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
print(f"\n✅ 堆叠柱状图已保存到: {fig1_path}")
# 分组柱状图
tp_colors = {
    "none": "#91D1C2",
    "pre-treatment": "#F39B7F",
    "post-treatment": "#8491B4"
}
fig2, ax2 = plt.subplots(figsize=(max(n_celltypes * 1.1, 14), 7))
n_tp = len(available_tp)
bar_width = 0.75 / n_tp
x_pos = np.arange(n_celltypes)
for i, tp in enumerate(available_tp):
    values = [prop_table.loc[ct, tp] if tp in prop_table.columns else 0
              for ct in prop_table.index]
    offset = (i - n_tp / 2 + 0.5) * bar_width
    ax2.bar(x_pos + offset, values, bar_width,
            label=tp, color=tp_colors.get(tp, f"C{i}"),
            edgecolor="white", linewidth=0.3)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(prop_table.index, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Cell Proportion (%)", fontsize=12)
ax2.set_title("Cell Type Proportion Comparison Across Treatment Timepoints",
              fontsize=14, fontweight="bold")
ax2.legend(title="Treatment Timepoint", fontsize=10, title_fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
fig2_path = r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_celltype_proportion_comparison_by_timepoint.png"
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
print(f"\n✅ 分组柱状图已保存到: {fig2_path}")
# 热图展示比例
fig3, ax3 = plt.subplots(figsize=(7, max(n_celltypes * 0.4, 6)))
sns.heatmap(
    prop_table,
    annot=True,
    fmt=".1f",
    cmap="BuPu",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Proportion (%)", "shrink": 0.8},
    ax=ax3,
    square=False
)
ax3.set_title("Cell Type Proportion Heatmap\nby Treatment Timepoint",
              fontsize=13, fontweight="bold", pad=15)
ax3.set_xlabel("Treatment Timepoint", fontsize=11)
ax3.set_ylabel("Cell Type", fontsize=11)
ax3.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig3_path = r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_celltype_proportion_heatmap_by_timepoint.png"
fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
print(f"\n✅ 热图已保存到: {fig3_path}")
# 治疗前后变化差异图
if "pre-treatment" in prop_table.columns and "post-treatment" in prop_table.columns:
    diff = prop_table["post-treatment"] - prop_table["pre-treatment"]
    diff = diff.sort_values()
    fig4, ax4 = plt.subplots(figsize=(10, max(n_celltypes * 0.35, 6)))
    bar_colors = ["#E64B35" if v > 0 else "#3C5488" for v in diff.values]
    ax4.barh(range(len(diff)), diff.values, color=bar_colors,
             edgecolor="white", linewidth=0.3, height=0.7)
    ax4.set_yticks(range(len(diff)))
    ax4.set_yticklabels(diff.index, fontsize=9)
    ax4.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax4.set_xlabel("Δ Proportion (%) : post-treatment − pre-treatment", fontsize=11)
    ax4.set_title("Change in Cell Type Proportion\n(Post-treatment vs Pre-treatment)",
                  fontsize=13, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E64B35", label="Increased after treatment"),
        Patch(facecolor="#3C5488", label="Decreased after treatment")
    ]
    ax4.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    fig4_path = r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_celltype_proportion_change_post_vs_pre_treatment.png"
    fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ 治疗前后变化差异图已保存到: {fig4_path}")
else:
    print("\n⚠️ 无法绘制治疗前后变化差异图：缺少 'pre-treatment' 或 'post-treatment' 数据")
