import scanpy as sc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
adata = sc.read_h5ad(r"E:\数据集\filtered_h5ad\CRC1合并\CRC1_harmony_optimized_clustered_celltypist_annotated.h5ad")
cluster_col = "leiden_0.5"
print(f"聚类列: {cluster_col}")
print(f"聚类数: {adata.obs[cluster_col].nunique()}")
print(f"各聚类细胞数:\n{adata.obs[cluster_col].value_counts().sort_index()}")
if adata.raw is not None:
    print("  ✔ 检测到 adata.raw，将基于raw进行差异分析")
else:
    print("  ⚠️ 未检测到 adata.raw，将基于当前X矩阵进行差异分析")
sc.tl.rank_genes_groups(
    adata,
    groupby=cluster_col,
    method="wilcoxon",
    n_genes=20,
    use_raw=(adata.raw is not None),
    pts=True,            # 计算表达比例
    key_added="marker_top20"
)
print("  ✔ 差异表达分析完成")
result = adata.uns["marker_top20"]
clusters = sorted(adata.obs[cluster_col].unique(), key=lambda x: int(x) if x.isdigit() else x)
all_rows = []
for cl in clusters:
    genes = result["names"][cl] if isinstance(result["names"], dict) else [result["names"][i][cl] for i in range(20)]
    
    # 兼容structured array格式
    if not isinstance(result["names"], dict):
        genes = [result["names"][cl][i] for i in range(20)] if hasattr(result["names"][cl], '__len__') else []
    
    for rank_idx in range(20):
        try:
            gene = result["names"][rank_idx][cl]
            score = result["scores"][rank_idx][cl]
            lfc = result["logfoldchanges"][rank_idx][cl]
            pval = result["pvals"][rank_idx][cl]
            pval_adj = result["pvals_adj"][rank_idx][cl]
            
            # 提取表达比例
            pct_in = np.nan
            pct_out = np.nan
            if "pts" in result:
                if isinstance(result["pts"], pd.DataFrame):
                    if gene in result["pts"].index:
                        pct_in = result["pts"].loc[gene, cl] if cl in result["pts"].columns else np.nan
                if "pts_rest" in result and isinstance(result["pts_rest"], pd.DataFrame):
                    if gene in result["pts_rest"].index:
                        pct_out = result["pts_rest"].loc[gene, cl] if cl in result["pts_rest"].columns else np.nan
            
            all_rows.append({
                "cluster": cl,
                "rank": rank_idx + 1,
                "gene": gene,
                "scores": round(float(score), 4),
                "logfoldchanges": round(float(lfc), 4),
                "pvals": float(pval),
                "pvals_adj": float(pval_adj),
                "pct_in": round(float(pct_in), 4) if not np.isnan(pct_in) else np.nan,
                "pct_out": round(float(pct_out), 4) if not np.isnan(pct_out) else np.nan
            })
        except Exception as e:
            print(f"  ⚠️ Cluster {cl}, rank {rank_idx+1} 提取异常: {e}")
            continue
marker_df = pd.DataFrame(all_rows)
print(f"  ✔ 共提取 {len(marker_df)} 条记录")
print(f"  ✔ 涵盖 {marker_df['cluster'].nunique()} 个聚类")
csv_path = r"C:\Users\49625\Desktop\大创相关\code\聚类1\CRC1_cluster_markers_top20.csv"
marker_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  ✔ CSV文件已保存至: {csv_path}")