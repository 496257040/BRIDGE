import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist_annotated.h5ad")
print(f"原始数据维度: {adata.shape}")
tp_col = "treatment_timepoint"
tissue_col = "tissue"
annot_col = "celltypist_majority"
# 仅保留 pre-treatment 和 post-treatment
adata_sub = adata[adata.obs[tp_col].isin(["pre-treatment", "post-treatment"])].copy()
print(f"筛选后数据维度: {adata_sub.shape}")
print(f"\n治疗时间点分布:\n{adata_sub.obs[tp_col].value_counts()}")
print(f"\n组织来源分布:\n{adata_sub.obs[tissue_col].value_counts()}")
print(f"\n细胞类型数: {adata_sub.obs[annot_col].nunique()}")
# 确认组织来源类别
tissue_order = ["LM", "normal", "PBMC", "tumor"]
available_tissue = [t for t in tissue_order if t in adata_sub.obs[tissue_col].values]
extra_tissue = [t for t in adata_sub.obs[tissue_col].unique() if t not in tissue_order]
if extra_tissue:
    available_tissue.extend(sorted(extra_tissue))
    print(f"⚠️ 发现额外组织来源: {extra_tissue}")
print(f"可用组织来源（排序后）: {available_tissue}")
tp_list = ["pre-treatment", "post-treatment"]
prop_dict = {}
count_dict = {}
for tp in tp_list:
    for tis in available_tissue:
        mask = (adata_sub.obs[tp_col] == tp) & (adata_sub.obs[tissue_col] == tis)
        subset = adata_sub.obs.loc[mask, annot_col]
        n_total = len(subset)
        if n_total == 0:
            print(f"  ⚠️ {tp} | {tis}: 无细胞，跳过")
            continue
        counts = subset.value_counts()
        props = (counts / n_total * 100)
        key = f"{tp}|{tis}"
        prop_dict[key] = props
        count_dict[key] = counts
        print(f"  ✔ {tp} | {tis}: {n_total} 个细胞")
# 整合为DataFrame
all_celltypes = sorted(adata_sub.obs[annot_col].unique())
prop_df = pd.DataFrame(index=all_celltypes)
for key, props in prop_dict.items():
    prop_df[key] = props
prop_df = prop_df.fillna(0)
print(f"\n比例矩阵维度: {prop_df.shape}")
print(prop_df.round(2).to_string())
# 图1 —— 分面堆叠柱状图（2行×N列面板）
n_ct = len(all_celltypes)
if n_ct <= 20:
    cmap_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_ct]
else:
    cmap_colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_ct))
color_dict = {ct: cmap_colors[i] for i, ct in enumerate(all_celltypes)}
n_tis = len(available_tissue)
fig1, axes1 = plt.subplots(
    2, n_tis, figsize=(4 * n_tis, 10),
    sharey=True
)
if n_tis == 1:
    axes1 = axes1.reshape(2, 1)
for row_idx, tp in enumerate(tp_list):
    for col_idx, tis in enumerate(available_tissue):
        ax = axes1[row_idx, col_idx]
        key = f"{tp}|{tis}"
        if key not in prop_dict:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_title(f"{tis}\n({tp})", fontsize=10, fontweight="bold")
            continue
        bottom = 0
        for ct in all_celltypes:
            val = prop_df.loc[ct, key] if key in prop_df.columns else 0
            ax.bar(0, val, bottom=bottom, color=color_dict[ct],
                   width=0.6, edgecolor="white", linewidth=0.3)
            bottom += val
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        if row_idx == 0:
            ax.set_title(f"{tis}", fontsize=12, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(f"{tp}\n\nProportion (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
# 统一图例
handles = [plt.Rectangle((0, 0), 1, 1, fc=color_dict[ct]) for ct in all_celltypes]
fig1.legend(handles, all_celltypes,
            bbox_to_anchor=(1.01, 0.5), loc="center left",
            fontsize=7, ncol=1 if n_ct <= 18 else 2, frameon=True,
            title="Cell Type", title_fontsize=9)
fig1.suptitle("Cell Type Composition: Treatment × Tissue Origin",
              fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig1_path = r"C:\Users\49625\Desktop\大创相关\code\image\facet_stacked_treatment_tissue.png"
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
print(f"✅ 分面堆叠柱状图已保存至: {fig1_path}")
# 图2 —— 双热图（pre vs post，按组织来源分列）
heatmap_pre = pd.DataFrame(index=all_celltypes, columns=available_tissue, dtype=float)
heatmap_post = pd.DataFrame(index=all_celltypes, columns=available_tissue, dtype=float)
for tis in available_tissue:
    key_pre = f"pre-treatment|{tis}"
    key_post = f"post-treatment|{tis}"
    if key_pre in prop_df.columns:
        heatmap_pre[tis] = prop_df[key_pre]
    if key_post in prop_df.columns:
        heatmap_post[tis] = prop_df[key_post]
heatmap_pre = heatmap_pre.fillna(0)
heatmap_post = heatmap_post.fillna(0)
# 按pre-treatment的行总和排序
row_order = heatmap_pre.sum(axis=1).sort_values(ascending=False).index
heatmap_pre = heatmap_pre.loc[row_order]
heatmap_post = heatmap_post.loc[row_order]
vmax = max(heatmap_pre.max().max(), heatmap_post.max().max())
fig2, (ax_pre, ax_post, ax_cbar) = plt.subplots(
    1, 3, figsize=(14, max(n_ct * 0.38, 7)),
    gridspec_kw={"width_ratios": [1, 1, 0.05]}
)
sns.heatmap(
    heatmap_pre, ax=ax_pre, annot=True, fmt=".1f",
    cmap="YlOrRd", vmin=0, vmax=vmax,
    linewidths=0.5, linecolor="white",
    cbar=False, square=False
)
ax_pre.set_title("Pre-treatment", fontsize=13, fontweight="bold", pad=10)
ax_pre.set_xlabel("Tissue Origin", fontsize=11)
ax_pre.set_ylabel("Cell Type", fontsize=11)
ax_pre.tick_params(axis="y", rotation=0)
sns.heatmap(
    heatmap_post, ax=ax_post, annot=True, fmt=".1f",
    cmap="YlOrRd", vmin=0, vmax=vmax,
    linewidths=0.5, linecolor="white",
    cbar_ax=ax_cbar, square=False
)
ax_post.set_title("Post-treatment", fontsize=13, fontweight="bold", pad=10)
ax_post.set_xlabel("Tissue Origin", fontsize=11)
ax_post.set_yticklabels([])
ax_post.set_ylabel("")
ax_cbar.set_ylabel("Proportion (%)", fontsize=10)
fig2.suptitle("Cell Type Proportion by Tissue Origin\n(Pre vs Post Treatment)",
              fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
fig2_path = r"C:\Users\49625\Desktop\大创相关\code\image\heatmap_dual_pre_post_tissue.png"
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
print(f"✅ 双热图已保存至: {fig2_path}")
# 图3 —— 气泡图展示治疗前后各组织中细胞比例变化
bubble_data = []
for tis in available_tissue:
    key_pre = f"pre-treatment|{tis}"
    key_post = f"post-treatment|{tis}"
    for ct in all_celltypes:
        val_pre = prop_df.loc[ct, key_pre] if key_pre in prop_df.columns else 0
        val_post = prop_df.loc[ct, key_post] if key_post in prop_df.columns else 0
        delta = val_post - val_pre
        bubble_data.append({
            "tissue": tis,
            "cell_type": ct,
            "delta": delta,
            "abs_delta": abs(delta),
            "direction": "Increased" if delta > 0 else "Decreased"
        })
bubble_df = pd.DataFrame(bubble_data)
# 过滤掉变化过小的点（|Δ| < 0.5%），避免图太拥挤
bubble_plot = bubble_df[bubble_df["abs_delta"] >= 0.5].copy()
if len(bubble_plot) > 0:
    fig3, ax3 = plt.subplots(figsize=(max(n_tis * 3, 10), max(n_ct * 0.35, 7)))
    # 编码x和y
    tissue_map = {t: i for i, t in enumerate(available_tissue)}
    ct_order_bubble = bubble_plot.groupby("cell_type")["abs_delta"].max().sort_values(ascending=True).index
    ct_map = {ct: i for i, ct in enumerate(ct_order_bubble)}
    bubble_plot["x"] = bubble_plot["tissue"].map(tissue_map)
    bubble_plot["y"] = bubble_plot["cell_type"].map(ct_map)
    # 气泡大小缩放
    size_scale = 800 / max(bubble_plot["abs_delta"].max(), 1)
    colors_bubble = bubble_plot["direction"].map(
        {"Increased": "#E64B35", "Decreased": "#3C5488"}
    )
    scatter = ax3.scatter(
        bubble_plot["x"], bubble_plot["y"],
        s=bubble_plot["abs_delta"] * size_scale,
        c=colors_bubble, alpha=0.7, edgecolors="grey", linewidth=0.5
    )
    ax3.set_xticks(range(len(available_tissue)))
    ax3.set_xticklabels(available_tissue, fontsize=11, fontweight="bold")
    ax3.set_yticks(range(len(ct_order_bubble)))
    ax3.set_yticklabels(ct_order_bubble, fontsize=9)
    ax3.set_xlabel("Tissue Origin", fontsize=12)
    ax3.set_ylabel("Cell Type", fontsize=12)
    ax3.set_title("Cell Proportion Change After Treatment\n(Bubble Size = |Δ%|, filtered |Δ| ≥ 0.5%)",
                  fontsize=13, fontweight="bold")
    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_color = [
        Patch(facecolor="#E64B35", label="Increased (post > pre)"),
        Patch(facecolor="#3C5488", label="Decreased (post < pre)")
    ]
    # 大小图例
    ref_sizes = [1, 5, 10]
    legend_size = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markersize=np.sqrt(s * size_scale / 3.14),
               label=f"|Δ| = {s}%")
        for s in ref_sizes if s <= bubble_plot["abs_delta"].max() * 1.2
    ]
    leg1 = ax3.legend(handles=legend_color, loc="upper left", fontsize=9,
                      title="Direction", title_fontsize=10, frameon=True)
    ax3.add_artist(leg1)
    ax3.legend(handles=legend_size, loc="lower right", fontsize=8,
               title="Bubble Size", title_fontsize=9, frameon=True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig3_path = r"C:\Users\49625\Desktop\大创相关\code\image\bubble_change_pre_post_tissue.png"
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    print(f"✅ 气泡图已保存至: {fig3_path}")
else:
    print("⚠️ 无显著变化的细胞类型（|Δ| ≥ 0.5%），未生成气泡图。")
# 图4 —— 分组柱状图（按组织来源分面，pre vs post并排）
# 取比例前15的细胞类型避免图面过于拥挤
top_celltypes = heatmap_pre.sum(axis=1).sort_values(ascending=False).head(15).index.tolist()
fig4, axes4 = plt.subplots(
    2, 2, figsize=(18, 14), sharey=True
)
axes4_flat = axes4.flatten()
tp_colors_2 = {"pre-treatment": "#F39B7F", "post-treatment": "#8491B4"}
for idx, tis in enumerate(available_tissue):
    if idx >= 4:
        break
    ax = axes4_flat[idx]
    key_pre = f"pre-treatment|{tis}"
    key_post = f"post-treatment|{tis}"
    x_pos = np.arange(len(top_celltypes))
    bar_w = 0.35
    vals_pre = [prop_df.loc[ct, key_pre] if key_pre in prop_df.columns and ct in prop_df.index else 0
                for ct in top_celltypes]
    vals_post = [prop_df.loc[ct, key_post] if key_post in prop_df.columns and ct in prop_df.index else 0
                 for ct in top_celltypes]
    ax.bar(x_pos - bar_w / 2, vals_pre, bar_w,
           label="pre-treatment", color=tp_colors_2["pre-treatment"],
           edgecolor="white", linewidth=0.3)
    ax.bar(x_pos + bar_w / 2, vals_post, bar_w,
           label="post-treatment", color=tp_colors_2["post-treatment"],
           edgecolor="white", linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_celltypes, rotation=50, ha="right", fontsize=8)
    ax.set_title(f"Tissue: {tis}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx % 2 == 0:
        ax.set_ylabel("Proportion (%)", fontsize=11)
    if idx == 0:
        ax.legend(fontsize=10, frameon=True)
fig4.suptitle("Cell Type Proportion: Pre vs Post Treatment by Tissue Origin\n(Top 15 Cell Types)",
              fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig4_path = r"C:\Users\49625\Desktop\大创相关\code\image\facet_grouped_bar_tissue_pre_post.png"
fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
print(f"✅ 分面分组柱状图已保存至: {fig4_path}")
