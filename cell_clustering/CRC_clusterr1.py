import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_optimized_neighbors.h5ad")
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f"leiden_res_{res}")
    n_clusters = adata.obs[f"leiden_res_{res}"].nunique()
    print(f"Resolution {res:.1f} -> {n_clusters} clusters")
from pyclustree import clustree
# 构建 cluster_keys 列表：按分辨率从低到高排列的obs列名
cluster_keys = [f"leiden_res_{res}" for res in resolutions]
# 确认这些列都存在于adata.obs中
existing_keys = [k for k in cluster_keys if k in adata.obs.columns]
print(f"将传入clustree的聚类列 ({len(existing_keys)}个): {existing_keys}")
# 调用clustree，传入adata和cluster_keys列表
fig = clustree(
    adata,
    cluster_keys=existing_keys,       # 必须参数：聚类结果列名列表
    title="Leiden Clustering Resolution Optimization",
    node_colormap="tab20",             # 节点配色方案
    node_size_range=(100, 1000),       # 节点大小范围（细胞数映射）
    edge_width_range=(0.5, 5.0),       # 边宽度范围（转移比例映射）
    edge_weight_threshold=0.0,         # 边权重阈值（低于此值的边不显示）
    show_fraction=True,                # 在边上显示转移比例
    show_cluster_keys=True,            # 显示分辨率标签
    order_clusters=True,               # 自动排列聚类节点位置
)
fig.set_size_inches(22, 18)
fig.savefig(
    r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_clustree.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
print("clustree图已保存")
# 结合UMAP可视化不同分辨率的聚类效果
selected_res = [0.3, 0.5, 0.8, 1.0, 1.5]
fig, axes = plt.subplots(1, len(selected_res), figsize=(6 * len(selected_res), 5))
for i, res in enumerate(selected_res):
    col_name = f"leiden_res_{res}"
    sc.pl.umap(
        adata,
        color=col_name,
        ax=axes[i],
        show=False,
        legend_loc="on data",
        legend_fontsize=6,
        title=f"Leiden res={res}\n({adata.obs[col_name].nunique()} clusters)",
        frameon=False,
    )
plt.tight_layout()
plt.savefig(
    r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_umap_resolutions.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
print("多分辨率UMAP对比图已保存")
# ---- 步骤4：轮廓系数辅助评估 ----
from sklearn.metrics import silhouette_score
if "X_pca_harmony" in adata.obsm:
    embed = adata.obsm["X_pca_harmony"][:, :30]
else:
    embed = adata.obsm["X_pca"][:, :30]
n_subsample = min(50000, adata.n_obs)
np.random.seed(114514)
subsample_idx = np.random.choice(adata.n_obs, n_subsample, replace=False)
embed_sub = embed[subsample_idx]
silhouette_scores = {}
for res in resolutions:
    labels = adata.obs[f"leiden_res_{res}"].values[subsample_idx]
    n_unique = len(set(labels))
    if n_unique < 2:
        continue
    score = silhouette_score(embed_sub, labels, sample_size=min(10000, n_subsample), random_state=42)
    silhouette_scores[res] = score
    print(f"Resolution {res:.1f}: Silhouette Score = {score:.4f}, Clusters = {n_unique}")
# 绘制分辨率优化图
fig, ax1 = plt.subplots(figsize=(10, 5))
res_list = sorted(silhouette_scores.keys())
scores = [silhouette_scores[r] for r in res_list]
n_clusters_list = [adata.obs[f"leiden_res_{r}"].nunique() for r in res_list]
color1 = "#2196F3"
color2 = "#FF5722"
ax1.plot(res_list, scores, "o-", color=color1, linewidth=2, markersize=6, label="Silhouette Score")
ax1.set_xlabel("Resolution", fontsize=12)
ax1.set_ylabel("Silhouette Score", color=color1, fontsize=12)
ax1.tick_params(axis="y", labelcolor=color1)
ax2 = ax1.twinx()
ax2.plot(res_list, n_clusters_list, "s--", color=color2, linewidth=2, markersize=6, label="Number of Clusters")
ax2.set_ylabel("Number of Clusters", color=color2, fontsize=12)
ax2.tick_params(axis="y", labelcolor=color2)
best_res = max(silhouette_scores, key=silhouette_scores.get)
best_score = silhouette_scores[best_res]
ax1.axvline(x=best_res, color="green", linestyle=":", alpha=0.7)
ax1.annotate(
    f"Best: res={best_res}\nScore={best_score:.4f}",
    xy=(best_res, best_score),
    xytext=(best_res + 0.2, best_score),
arrowprops=dict(arrowstyle="->", color="green"),
    fontsize=10, color="green"
)
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.95))
plt.title("Resolution Optimization", fontsize=14)
plt.tight_layout()
plt.savefig(
    r"C:\Users\49625\Desktop\大创相关\code\image\CRC1_resolution_optimization.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
print(f"\n===== 推荐分辨率: {best_res}（Silhouette Score最高: {best_score:.4f}）=====")