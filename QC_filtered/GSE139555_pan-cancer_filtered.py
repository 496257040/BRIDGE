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


# ======================生成标签==========================
def parse_sample_info_from_dirname(dirname):
    info = {  # 标签类型,要包括以下几个，记得标准化命名
        "sample_id": "unknown",
        "patient": "unknown",
        "tissue": "unknown",
        "disease": "CRC",
        "cancer_type": "CRC",
        "treatment_drug": "none",
        "treatment_timepoint": "none",
        "cell_sorting": "none",
    }
    parts = dirname.split("_")
    # 提取样本ID，通常是以GSM开头的部分
    gsm_id = parts[0] if parts[0].startswith("GSM") else "unknown"
    info["sample_id"] = gsm_id
    # 提取组织来源和疾病状态
    tissue_keywords = ["PBMC", "tumor", "LM", "normal"]
    for keyword in tissue_keywords:
        if keyword in dirname:
            info["tissue"] = keyword
            break
    disease_keywords = ["NSCLC", "NET", "UCEC", "CRC", "RCC"]
    for keyword in disease_keywords:
        if keyword in dirname:
            info["disease"] = keyword
            info["cancer_type"] = keyword
            break
    simplified = re.sub(r'^GSM\d+_raw_feature_bc_matrix_', '', dirname)
    simplified = re.sub(r'^GSM\d+_', '', simplified) if simplified == dirname else simplified
    # 提取患者编号，假设格式为P1, P2等，或者P1-P2等
    # 提取一个或多个以短横线连接的完整患者编号（如 COL07 或 COL07-COL12）
    patient_match = re.search(r'((?:Lung\d+)|(?:Endo\d+)|(?:Colon\d+)|(?:Renal\d+)*)', simplified)
    if patient_match:
        info['patient'] = patient_match.group(1)  # 直接获取完整编号

    print(info)
    return info


# ===========================生成gsm样本的详细信息，供标准化标签使用==========================
# 从GEO系列矩阵文件中提取特定GSM样本的详细信息
def extract_sample_info(series_matrix_file, target_gsm):
    desired_attributes = ["Sample_geo_accession", "Sample_characteristics_region", "Sample_characteristics_patient",
                          "Sample_characteristics_phenotype"]

    # 读取文件
    with open(series_matrix_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析文件内容
    sample_data = {}
    current_attribute = None
    gsm_columns = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')

        if line.startswith('!'):
            attr_name = line[1:].split('\t')[0] if '\t' in line else line[1:]

            if attr_name.lower() in ['sample_geo_accession']:
                gsm_columns = parts[1:]

            current_attribute = attr_name

        if current_attribute and len(parts) > 1:
            if current_attribute not in sample_data:
                sample_data[current_attribute] = {}

            for i, gsm in enumerate(gsm_columns):
                if i + 1 < len(parts):
                    sample_data[current_attribute][gsm] = parts[i + 1]

    # 提取目标GSM样本的指定属性信息
    if target_gsm in gsm_columns and sample_data:
        print(f"\n{'=' * 60}")
        print(f"样本 {target_gsm} 的指定属性信息:")
        print('=' * 60)

        result = {}
        for attr, samples_dict in sample_data.items():
            # 只提取我们感兴趣的属性
            if attr in desired_attributes and target_gsm in samples_dict:
                value = sample_data[attr][target_gsm]
                if value == '"region: Tumor"':
                    result[attr] = "tumor"
                elif value == '"phenotype: Lung adenocarcinoma"':
                    result[attr] = "NSCLC"
                elif value == '"region: Normal adjacent tissue"':
                    result[attr] = "normal"
                elif value == '"phenotype: Lung squamous cell carcinoma, nonkeratinizing"':
                    result[attr] = "NSCLC"
                elif value == '"phenotype: Lung squamous cell carcinoma, keratinizing"':
                    result[attr] = "NSCLC"
                elif value == '"phenotype: Large cell neuroendocrine carcinoma"':
                    result[attr] = "NET"
                elif value == '"region: Blood"':
                    result[attr] = "PBMC"
                elif value == '"phenotype: Endometrial adenocarcinoma"':
                    result[attr] = "UCEC"
                elif value == '"phenotype: Colorectal adenocarcinoma"':
                    result[attr] = "CRC"
                elif value == '"phenotype: Renal cell carcinoma, clear cell"':
                    result[attr] = "RCC"
                else:
                    result[attr] = value
        # 保存结果到字符串
        if result:
            # 将结果字典的所有值用下划线连接成字符串
            result_str = "_".join(str(value) for value in result.values() if value)
            result_clean = result_str.replace('"', '').replace('patient: ', '')
            print(f"信息已转换为字符串: {result_clean}")
            return result_clean


class SingleCellAnalyzer:
    def __init__(self, tar_file, sample_info_list):
        """
        初始化单细胞分析器

        参数:
        tar_file: 原始数据tar文件路径
        sample_info_list: 样本信息字典列表
        """
        self.tar_file = tar_file
        self.sample_info_list = sample_info_list
        self.gsm_output_dir = "gsm_samples"  # 只输出这个文件夹

        # 初始化细胞计数字典
        self.cell_counts = {}  # 记录每个样本的细胞数量
        self.original_cell_counts = {}  # 记录原始细胞数量

        # 只创建gsm_samples文件夹
        os.makedirs(self.gsm_output_dir, exist_ok=True)

        # 存储数据
        self.metadata = {}
        self.adata_dict = {}
        self.gsm_files = {}

    def parse_sample_info(self):
        """
        从提供的样本信息列表中解析元数据
        """
        print("解析样本信息列表...")

        for sample_info in self.sample_info_list:
            gsm_id = sample_info.get('sample_id')
            if not gsm_id:
                print(f"警告: 样本信息中没有'sample_id'键: {sample_info}")
                continue

            # 映射字段到内部格式
            self.metadata[gsm_id] = {
                'gsm_id': gsm_id,
                'patient_id': sample_info.get('patient', 'unknown'),
                'tissue_type': sample_info.get('tissue', 'unknown'),
                'disease_type': sample_info.get('disease', 'unknown'),
                'cancer_type': sample_info.get('cancer_type', 'unknown'),
                'sample_source': sample_info.get('tissue', 'unknown'),  # 使用tissue作为样本来源
                'treatment': sample_info.get('treatment_drug', 'none'),
                'timepoint': sample_info.get('treatment_timepoint', 'none'),
                'cell_sorting': sample_info.get('cell_sorting', 'none'),
                'original_info': sample_info  # 保存原始信息以备后用
            }

        print(f"成功解析 {len(self.metadata)} 个样本的元数据")

        # 打印样本信息摘要
        print("\n样本信息摘要:")
        for gsm_id, info in self.metadata.items():
            print(f"  {gsm_id}: {info['patient_id']}, {info['tissue_type']}, {info['disease_type']}")

        return self.metadata

    def generate_sample_summary_table(self):
        """生成样本摘要表格，类似于图片中的格式"""

        summary_data = []

        for gsm_id, metadata in self.metadata.items():
            # 获取细胞数量 - 从cell_counts字典中获取
            cell_count = self.cell_counts.get(gsm_id, 0)

            # 从metadata中获取其他信息
            patient = metadata.get('patient_id', 'unknown')
            tissue = metadata.get('tissue_type', 'unknown')
            disease = metadata.get('disease_type', 'unknown')
            cancer_type = metadata.get('cancer_type', 'unknown')
            treatment_drug = metadata.get('treatment', 'unknown')
            treatment_timepoint = metadata.get('timepoint', 'unknown')
            cell_sorting = metadata.get('cell_sorting', 'unknown')

            # 添加到摘要数据
            summary_data.append({
                'sample_id': gsm_id,
                'cell_count': cell_count,
                'patient': patient,
                'tissue': tissue,
                'disease': disease,
                'cancer_type': cancer_type,
                'treatment_drug': treatment_drug,
                'treatment_timepoint': treatment_timepoint,
                'cell_sorting': cell_sorting
            })

        # 创建DataFrame
        summary_df = pd.DataFrame(summary_data)

        # 按sample_id排序
        summary_df = summary_df.sort_values('sample_id')

        # 保存为CSV文件
        output_file = os.path.join(self.gsm_output_dir, "sample_summary.csv")
        summary_df.to_csv(output_file, index=False)

        # 打印表格摘要
        print("\n" + "=" * 100)
        print("样本摘要表格已生成:")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100)
        print(f"\n表格已保存到:")
        print(f"  CSV格式: {output_file}")

        return summary_df

    def extract_and_organize_tar_files(self):
        """解压tar文件并组织GSM样本文件"""
        print("解压tar文件并组织GSM样本...")

        # 目标GSM样本从元数据中获取
        target_gsm = list(self.metadata.keys())
        print(f"目标GSM样本: {target_gsm}")

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
                    print(f"警告: 未找到GSM样本 {gsm_id} 的文件")

        # 统计
        complete_samples = len([k for k, v in self.gsm_files.items() if all(v.values())])
        incomplete_samples = len([k for k, v in self.gsm_files.items() if not all(v.values())])

        print(f"完整样本: {complete_samples} 个")
        print(f"不完整样本: {incomplete_samples} 个")

        if incomplete_samples > 0:
            print("不完整的样本:")
            for gsm_id, files in self.gsm_files.items():
                if not all(files.values()):
                    missing_files = [k for k, v in files.items() if v is None]
                    print(f"  {gsm_id}: 缺少 {missing_files}")

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

            # 记录原始细胞数量
            self.original_cell_counts[gsm_id] = adata.shape[0]
            print(f"  {gsm_id}: 读取 {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

            # 添加GSM ID到细胞barcode
            adata.obs_names = [f"{gsm_id}_{barcode}" for barcode in adata.obs_names]
            adata.obs['gsm_id'] = gsm_id

            # 添加元数据标签 - 使用提供的字典信息
            if gsm_id in self.metadata:
                meta = self.metadata[gsm_id]

                # 添加所有元数据字段
                for key, value in meta.items():
                    if key != 'original_info':  # 不添加原始信息字典
                        adata.obs[key] = value

                # 打印添加的元数据
                print(
                    f"  {gsm_id}: 添加元数据 - 病人: {meta.get('patient_id')}, 组织: {meta.get('tissue_type')}, 疾病: {meta.get('disease_type')}")
            else:
                print(f"  警告: 没有找到 {gsm_id} 的元数据")
                # 如果没有元数据，添加默认值
                adata.obs['patient_id'] = gsm_id
                adata.obs['tissue_type'] = 'unknown'
                adata.obs['disease_type'] = 'unknown'
                adata.obs['cancer_type'] = 'unknown'
                adata.obs['treatment'] = 'none'
                adata.obs['timepoint'] = 'none'
                adata.obs['cell_sorting'] = 'none'

            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

            return adata

        except Exception as e:
            print(f"  处理样本 {gsm_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
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
        """按GSM样本分割并保存为h5ad文件，并记录质控后的细胞数量"""
        print("按GSM样本分割并保存...")

        for gsm_id in sorted(adata_combined.obs['gsm_id'].unique()):
            # 提取该GSM的细胞
            gsm_mask = adata_combined.obs['gsm_id'] == gsm_id
            adata_gsm = adata_combined[gsm_mask, :].copy()

            if adata_gsm.shape[0] > 0:
                # 记录质控后的细胞数量
                self.cell_counts[gsm_id] = adata_gsm.shape[0]

                # 保存为h5ad文件
                output_file = os.path.join(self.gsm_output_dir, f"{gsm_id}_filtered.h5ad")
                adata_gsm.write(output_file)

                # 获取原始细胞数量
                original_cells = self.original_cell_counts.get(gsm_id, 0)
                remaining_percent = (adata_gsm.shape[0] / original_cells * 100) if original_cells > 0 else 0

                # 添加统计信息
                if gsm_id in self.metadata:
                    meta = self.metadata[gsm_id]
                    print(
                        f"  已保存: {gsm_id} ({adata_gsm.shape[0]} 细胞, 原始{original_cells}, 保留{remaining_percent:.1f}%) - {meta.get('patient_id')}, {meta.get('tissue_type')}, {meta.get('disease_type')}")
                else:
                    print(
                        f"  已保存: {gsm_id} ({adata_gsm.shape[0]} 细胞, 原始{original_cells}, 保留{remaining_percent:.1f}%)")
            else:
                self.cell_counts[gsm_id] = 0
                print(f"  警告: {gsm_id} 在质控后没有细胞")

    def run_analysis(self):
        """运行完整的分析流程"""
        print("=" * 60)
        print("开始单细胞分析流程")
        print("=" * 60)

        # 步骤1: 解析样本信息列表
        print("\n1. 解析样本信息列表...")
        self.parse_sample_info()

        if not self.metadata:
            print("错误: 未解析到样本元数据，无法继续处理")
            return

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

        # 步骤7: 生成样本摘要表格
        print("\n7. 生成样本摘要表格...")
        self.generate_sample_summary_table()

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        print(f"输出目录: {self.gsm_output_dir}")
        if os.path.exists(self.gsm_output_dir):
            print(f"文件数量: {len([f for f in os.listdir(self.gsm_output_dir) if f.endswith('.h5ad')])}")
        print("=" * 60)

        return adata_qc


if __name__ == "__main__":
    # 文件路径
    series_matrix_file = "GSE139555_series_matrix.txt"
    input_dir = r'C:\Users\14584\Desktop\数据贴标签 - 0407\GSE139555'
    # 1. 使用提供的样本信息字典
    sample_info_list1 = []
    for a in range(4143655, 4143687):
        target_gsm = '"' + "GSM" + str(a) + '"'  # 替换为您要查找的GSM编号
        filename = extract_sample_info(series_matrix_file, target_gsm)
        output_info = parse_sample_info_from_dirname(filename)
        sample_info_list1.append(output_info)
    analyzer = SingleCellAnalyzer(
        tar_file="GSE139555_RAW.tar",  # 你的tar文件路径
        sample_info_list=sample_info_list1
    )
    analyzer.run_analysis()
