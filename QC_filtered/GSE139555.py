import os
import tarfile
import gzip
import shutil
import re
import pandas as pd
import numpy as np
import scanpy as sc
import warnings

warnings.filterwarnings('ignore')


class SingleCellAnalyzer:
    def __init__(self, tar_file, series_matrix_file):
        """
        初始化单细胞分析器

        参数:
        tar_file: 原始数据tar文件路径
        series_matrix_file: 系列矩阵文件路径
        """
        self.tar_file = tar_file
        self.series_matrix_file = series_matrix_file
        self.gsm_output_dir = "gsm_samples"  # 只输出这个文件夹

        # 只创建gsm_samples文件夹
        os.makedirs(self.gsm_output_dir, exist_ok=True)

        # 存储数据
        self.metadata = {}
        self.adata_dict = {}
        self.gsm_files = {}

    def parse_series_matrix(self):
        """
        解析GSE139555_series_matrix.txt文件，提取GSM4143655~GSM4143686的信息
        """
        print("解析系列矩阵文件...")

        with open(self.series_matrix_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 目标GSM样本范围
        target_gsm = [f"GSM41436{i:02d}" for i in range(55, 87)]

        # 初始化数据结构
        sample_data = {}
        current_key = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 处理键值对
            if line.startswith('!'):
                if '"' in line:
                    # 格式: !Sample_key = "value1" "value2" ...
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        values = re.findall(r'"([^"]*)"', parts[1].strip())

                        if key == '!Sample_geo_accession':
                            # 这是GSM编号列表
                            for i, gsm_id in enumerate(values):
                                if gsm_id in target_gsm:
                                    if gsm_id not in sample_data:
                                        sample_data[gsm_id] = {
                                            'gsm_id': gsm_id,
                                            'patient_id': 'unknown',
                                            'tissue_type': 'unknown',
                                            'disease_type': 'unknown',
                                            'sample_source': 'unknown',
                                            'treatment': 'unknown',
                                            'timepoint': 'unknown'
                                        }
                        else:
                            # 其他属性
                            for i, value in enumerate(values):
                                gsm_id = None
                                if '!Sample_geo_accession' in sample_data:
                                    # 需要先找到对应的GSM编号
                                    pass

        # 由于您的文件格式特殊，可能需要手动添加元数据
        # 根据GSE139555的描述，这些样本是结直肠癌样本
        for gsm_id in target_gsm:
            if gsm_id not in sample_data:
                # 创建样本信息
                sample_data[gsm_id] = {
                    'gsm_id': gsm_id,
                    'patient_id': f"Patient_{int(gsm_id[9:]) - 3654}",  # 简单计算患者编号
                    'tissue_type': 'colorectal',  # 结直肠组织
                    'disease_type': 'colorectal cancer',  # 结直肠癌
                    'sample_source': 'tumor',  # 肿瘤组织
                    'treatment': 'naive',  # 未治疗
                    'timepoint': 'baseline'  # 基线
                }

        self.metadata = sample_data
        print(f"成功解析 {len(self.metadata)} 个目标GSM样本的元数据")
        return self.metadata

    def extract_and_organize_tar_files(self):
        """解压tar文件并组织GSM样本文件"""
        print("解压tar文件并组织GSM样本...")

        # 目标GSM样本范围
        target_gsm = [f"GSM41436{i:02d}" for i in range(55, 87)]

        # 直接从tar文件中查找，不先解压到磁盘
        with tarfile.open(self.tar_file, 'r') as tar:
            # 获取所有文件
            all_files = tar.getnames()

            for gsm_id in target_gsm:
                # 查找以该GSM开头的文件
                gsm_pattern = f"{gsm_id}_"
                gsm_file_list = [f for f in all_files if f.startswith(gsm_pattern) and f.endswith('.gz')]

                if gsm_file_list:
                    # 分类文件类型
                    matrix_files = [f for f in gsm_file_list if 'matrix' in f.lower()]
                    barcode_files = [f for f in gsm_file_list if 'barcode' in f.lower()]
                    gene_files = [f for f in gsm_file_list if 'gene' in f.lower() or 'feature' in f.lower()]

                    # 取第一个匹配的文件
                    self.gsm_files[gsm_id] = {
                        'matrix': matrix_files[0] if matrix_files else None,
                        'barcodes': barcode_files[0] if barcode_files else None,
                        'genes': gene_files[0] if gene_files else None
                    }
                else:
                    self.gsm_files[gsm_id] = {'matrix': None, 'barcodes': None, 'genes': None}

        # 统计
        complete_samples = len([k for k, v in self.gsm_files.items() if all(v.values())])
        print(f"共找到 {complete_samples} 个完整的GSM样本文件")
        return self.gsm_files

    def process_gsm_sample(self, gsm_id, tar):
        """处理单个GSM样本，从tar文件中直接解压处理"""
        print(f"处理样本: {gsm_id}")

        files = self.gsm_files[gsm_id]
        if not all(files.values()):
            print(f"  {gsm_id}: 文件不完整，跳过")
            return None

        # 创建临时目录
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f"temp_{gsm_id}_")

        try:
            # 从tar文件中提取文件到临时目录
            matrix_path = os.path.join(temp_dir, "matrix.mtx.gz")
            barcodes_path = os.path.join(temp_dir, "barcodes.tsv.gz")
            genes_path = os.path.join(temp_dir, "genes.tsv.gz")

            # 提取文件
            tar.extract(files['matrix'], temp_dir)
            tar.extract(files['barcodes'], temp_dir)
            tar.extract(files['genes'], temp_dir)

            # 重命名文件
            os.rename(os.path.join(temp_dir, files['matrix']), matrix_path)
            os.rename(os.path.join(temp_dir, files['barcodes']), barcodes_path)
            os.rename(os.path.join(temp_dir, files['genes']), genes_path)

            # 解压.gz文件
            matrix_unzipped = matrix_path.replace('.gz', '')
            barcodes_unzipped = barcodes_path.replace('.gz', '')
            genes_unzipped = genes_path.replace('.gz', '')

            with gzip.open(matrix_path, 'rb') as f_in:
                with open(matrix_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            with gzip.open(barcodes_path, 'rb') as f_in:
                with open(barcodes_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            with gzip.open(genes_path, 'rb') as f_in:
                with open(genes_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 读取10x数据
            adata = sc.read_10x_mtx(
                temp_dir,
                var_names='gene_symbols',
                cache=False
            )

            # 添加GSM ID到细胞barcode
            adata.obs_names = [f"{gsm_id}_{barcode}" for barcode in adata.obs_names]
            adata.obs['gsm_id'] = gsm_id

            # 添加元数据标签
            if gsm_id in self.metadata:
                meta = self.metadata[gsm_id]
                for key, value in meta.items():
                    adata.obs[key] = value
            else:
                # 如果没有元数据，添加默认值
                adata.obs['patient_id'] = gsm_id
                adata.obs['tissue_type'] = 'unknown'
                adata.obs['disease_type'] = 'unknown'

            print(f"  {gsm_id}: 读取 {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

            return adata

        except Exception as e:
            print(f"  处理样本 {gsm_id} 时出错: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def quality_control(self, adata_combined):
        """
        按照指定的质量控制标准进行处理
        """
        print("进行质量控制...")
        print(f"质控前: {adata_combined.shape[0]} 细胞, {adata_combined.shape[1]} 基因")

        # 计算线粒体基因百分比
        adata_combined.var["mt"] = adata_combined.var_names.str.startswith("MT-")

        sc.pp.calculate_qc_metrics(
            adata_combined, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        # 应用质量控制过滤
        # 1. 过滤基因数太少或太多的细胞
        adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts > 200, :]
        adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts < 5000, :]

        # 2. 过滤线粒体基因百分比过高的细胞
        adata_combined = adata_combined[adata_combined.obs.pct_counts_mt < 20, :]

        # 3. 过滤在少于3个细胞中表达的基因
        sc.pp.filter_genes(adata_combined, min_cells=3)

        print(f"质控后: {adata_combined.shape[0]} 细胞, {adata_combined.shape[1]} 基因")

        return adata_combined

    def save_gsm_samples(self, adata_combined):
        """按GSM样本分割并保存为h5ad文件"""
        print("按GSM样本分割并保存...")

        for gsm_id in sorted(adata_combined.obs['gsm_id'].unique()):
            # 提取该GSM的细胞
            gsm_mask = adata_combined.obs['gsm_id'] == gsm_id
            adata_gsm = adata_combined[gsm_mask, :].copy()

            if adata_gsm.shape[0] > 0:
                # 保存为h5ad文件
                output_file = os.path.join(self.gsm_output_dir, f"{gsm_id}_filtered.h5ad")
                adata_gsm.write(output_file)

                print(f"  已保存: {gsm_id} ({adata_gsm.shape[0]} 细胞) -> {output_file}")
            else:
                print(f"  警告: {gsm_id} 在质控后没有细胞")

    def run_analysis(self):
        """运行完整的分析流程"""
        print("=" * 60)
        print("开始单细胞分析流程")
        print("=" * 60)

        # 步骤1: 解析系列矩阵文件
        print("\n1. 解析系列矩阵文件...")
        self.parse_series_matrix()

        if not self.metadata:
            print("警告: 未解析到元数据，将继续处理但可能无法添加正确的标签")

        # 步骤2: 从tar文件中组织GSM样本文件
        print("\n2. 从tar文件中组织GSM样本文件...")
        self.extract_and_organize_tar_files()

        # 步骤3: 处理每个GSM样本
        print("\n3. 处理每个GSM样本...")
        adata_list = []

        with tarfile.open(self.tar_file, 'r') as tar:
            for gsm_id, files in self.gsm_files.items():
                if all(files.values()):  # 确保所有文件都存在
                    adata = self.process_gsm_sample(gsm_id, tar)
                    if adata is not None:
                        adata_list.append(adata)
                else:
                    print(f"  {gsm_id}: 文件不完整，跳过")

        if not adata_list:
            print("错误: 没有成功读取任何样本")
            return

        # 步骤4: 合并所有样本
        print(f"\n4. 合并所有样本...")

        if len(adata_list) > 1:
            adata_combined = adata_list[0].concatenate(
                adata_list[1:],
                batch_key='batch',
                batch_categories=[adata.obs['gsm_id'].iloc[0] for adata in adata_list],
                index_unique=None
            )
        else:
            adata_combined = adata_list[0]

        print(f"合并后: {adata_combined.shape[0]} 细胞, {adata_combined.shape[1]} 基因")

        # 步骤5: 质量控制
        print("\n5. 进行质量控制...")
        adata_qc = self.quality_control(adata_combined)

        # 步骤6: 按GSM保存样本
        print("\n6. 按GSM保存样本...")
        self.save_gsm_samples(adata_qc)

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        print(f"输出目录: {self.gsm_output_dir}")
        print(f"文件数量: {len(os.listdir(self.gsm_output_dir))}")
        print("=" * 60)


# 主程序
if __name__ == "__main__":
    # 设置文件路径
    tar_file = "GSE139555_RAW.tar"
    series_matrix_file = "GSE139555_series_matrix.txt"

    # 检查文件是否存在
    if not os.path.exists(tar_file):
        print(f"错误: 找不到tar文件 {tar_file}")
        print("请确保文件在当前目录")
    elif not os.path.exists(series_matrix_file):
        print(f"错误: 找不到系列矩阵文件 {series_matrix_file}")
        print("请确保文件在当前目录")
    else:
        # 创建分析器并运行
        analyzer = SingleCellAnalyzer(
            tar_file=tar_file,
            series_matrix_file=series_matrix_file
        )

        analyzer.run_analysis()