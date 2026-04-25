import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import harmonypy as hm
import warnings
import time
warnings.filterwarnings("ignore")
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_merged_neighbors.h5ad")
if adata.obsm["X_pca"].shape[1] < 50:
    print("PCA维度不足50，重新计算PCA（n_comps=50）...")
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    print(f"PCA重新计算完成: {adata.obsm['X_pca'].shape}")
def compute_batch_entropy(adata_obj, use_rep, n_pcs, n_neighbors=30, 
                          batch_col="patient", n_sample=5000, seed=42):
    from scipy.stats import entropy as sp_ent
    import scipy.sparse as sps
    from sklearn.neighbors import NearestNeighbors
    embed = adata_obj.obsm[use_rep][:, :n_pcs]
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(embed)
    dist_matrix = nn.kneighbors_graph(mode="distance")
    batch_labels = adata_obj.obs[batch_col].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[b] for b in batch_labels])
    np.random.seed(seed)
    sample_idx = np.random.choice(adata_obj.n_obs, min(n_sample, adata_obj.n_obs), replace=False)
    max_ent = np.log(n_batches)
    entropies = []
    for idx in sample_idx:
        neighbors = dist_matrix[idx].nonzero()[1]
        if len(neighbors) == 0:
            continue
        nb = batch_indices[neighbors]
        counts = np.bincount(nb, minlength=n_batches)
        props = counts / counts.sum()
        props = props[props > 0]
        entropies.append(sp_ent(props))
    entropies = np.array(entropies)
    norm_ent = entropies / max_ent if max_ent > 0 else entropies
    return np.mean(norm_ent), np.median(norm_ent), np.mean(norm_ent < 0.3)
# 步骤3：参数网格搜索
print("\n" + "=" * 70)
print("【参数网格搜索】")
print("寻找最优 n_pcs × theta 组合")
print("=" * 70)
# 定义搜索空间
pcs_candidates = [30, 35, 40, 50]
theta_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
results = []
# 记录原始（未校正前）的批次熵作为baseline
print("\n计算原始Harmony校正(n_pcs=30, theta=default)的批次熵作为baseline...")
mean_ent_orig, med_ent_orig, low_pct_orig = compute_batch_entropy(
    adata, "X_pca_harmony", n_pcs=30
)
print(f"  原始 Harmony: 均值={mean_ent_orig:.4f}, 中位数={med_ent_orig:.4f}, <0.3比例={low_pct_orig:.2%}")
print(f"\n开始网格搜索（共{len(pcs_candidates)}×{len(theta_candidates)}={len(pcs_candidates)*len(theta_candidates)}个组合）...\n")
for n_pcs in pcs_candidates:
    pca_data = adata.obsm["X_pca"][:, :n_pcs]
    for theta in theta_candidates:
        t0 = time.time()
        print(f"  测试 n_pcs={n_pcs}, theta={theta}...", end=" ", flush=True)
        try:
            harmony_out = hm.run_harmony(
                pca_data,
                adata.obs,
                "patient",
                theta=theta,
                max_iter_harmony=20,
                random_state=42
            )
            # 获取校正后嵌入
            Z = harmony_out.Z_corr
            if Z.shape[0] == n_pcs and Z.shape[1] == adata.n_obs:
                Z = Z.T
            # 临时存入计算熵
            temp_key = f"_temp_harmony_{n_pcs}_{theta}"
            adata.obsm[temp_key] = Z
            
            mean_ent, med_ent, low_pct = compute_batch_entropy(
                adata, temp_key, n_pcs=n_pcs
            )
            elapsed = time.time() - t0
            print(f"熵均值={mean_ent:.4f}, 中位数={med_ent:.4f}, <0.3比例={low_pct:.2%}  ({elapsed:.1f}s)")
            results.append({
                "n_pcs": n_pcs,
                "theta": theta,
                                "mean_entropy": mean_ent,
                "median_entropy": med_ent,
                "low_entropy_pct": low_pct,
                "time_sec": elapsed
            })
            # 清理临时嵌入
            del adata.obsm[temp_key]
        except Exception as e:
            print(f"失败: {e}")
            results.append({
                "n_pcs": n_pcs,
                "theta": theta,
                "mean_entropy": -1,
                "median_entropy": -1,
                "low_entropy_pct": -1,
                "time_sec": -1
            })
df_results = pd.DataFrame(results)
df_valid = df_results[df_results["mean_entropy"] > 0].copy()
print("\n" + "=" * 70)
print("【网格搜索结果汇总】")
print("=" * 70)
# 以表格形式输出
pivot = df_valid.pivot_table(
    index="theta", columns="n_pcs", values="mean_entropy"
)
print("\n批次熵均值矩阵（行=theta, 列=n_pcs）：")
print(pivot.to_string(float_format="%.4f"))
# 找最优
best_idx = df_valid["mean_entropy"].idxmax()
best_row = df_valid.loc[best_idx]
print(f"\n→ 最优组合: n_pcs={int(best_row['n_pcs'])}, theta={best_row['theta']}")
print(f"  批次熵均值: {best_row['mean_entropy']:.4f}")
print(f"  批次熵中位数: {best_row['median_entropy']:.4f}")
print(f"  熵<0.3的比例: {best_row['low_entropy_pct']:.2%}")
print(f"  (对比原始: 均值={mean_ent_orig:.4f})")
# 步骤5：使用最优参数重新校正并保存
print("\n" + "=" * 70)
print("【使用最优参数重新校正并保存】")
print("=" * 70)
best_n_pcs = int(best_row["n_pcs"])
best_theta = best_row["theta"]
print(f"使用 n_pcs={best_n_pcs}, theta={best_theta} 重新运行Harmony...")
pca_best = adata.obsm["X_pca"][:, :best_n_pcs]
harmony_best = hm.run_harmony(
    pca_best,
    adata.obs,
    "patient",
    theta=best_theta,
    max_iter_harmony=30,  # 最终版用更多迭代确保收敛
    random_state=42
)
Z_best = harmony_best.Z_corr
if Z_best.shape[0] == best_n_pcs and Z_best.shape[1] == adata.n_obs:
    Z_best = Z_best.T
adata.obsm["X_pca_harmony_opt"] = Z_best
print(f"✅ 最优Harmony嵌入已保存至 obsm['X_pca_harmony_opt']，维度: {Z_best.shape}")
# 重建邻居图
print("重建邻居图...")
sc.pp.neighbors(
    adata,
    use_rep="X_pca_harmony_opt",
    n_pcs=best_n_pcs,
    n_neighbors=30,
    random_state=0
)
print("✅ 邻居图重建完成")
# 重新计算UMAP
print("重新计算UMAP...")
sc.tl.umap(adata, random_state=42)
print(f"✅ UMAP计算完成")
# 保存
save_path = r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized.h5ad"
adata.write_h5ad(save_path)
print(f"✅ 优化后的数据已保存至: {save_path}")
# 验证
print("\n【验证保存结果】")
adata_check = sc.read_h5ad(save_path)
print(f"  obsm键: {list(adata_check.obsm.keys())}")
print(f"  X_pca_harmony_opt维度: {adata_check.obsm['X_pca_harmony_opt'].shape}")
params = adata_check.uns["neighbors"]["params"]
print(f"  邻居图 use_rep: {params['use_rep']}")
print(f"  邻居图 n_pcs: {params['n_pcs']}")
print(f"  邻居图 n_neighbors: {params['n_neighbors']}")
assert params["use_rep"] == "X_pca_harmony_opt"
assert params["n_pcs"] == best_n_pcs
print("\n✅ 验证通过！后续可直接读取此文件进行多分辨率Leiden聚类。")
print(f"\n提示：网格搜索结果已打印在上方，如需保存为CSV：")
df_results.to_csv(r"C:\\Users\\49625\\Desktop\\大创相关\\code\\data\\harmony_grid_search.csv", index=False)
del adata_check
