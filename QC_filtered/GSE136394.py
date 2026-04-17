
import re
import os
import tarfile
import gzip
import shutil
import scanpy as sc
import tempfile
import pandas as pd
print("正在运行...")

#======================生成标签==========================
def parse_sample_info_from_dirname(dirname):
    info = {      #标签类型,要包括以下几个，记得标准化命名
        "sample_id": "unknown",
        "patient": "unknown",
        "tissue": "unknown",
        "disease": "CRC",
        "cancer_type": "CRC",
        "treatment_drug": "TIL therapy or unknown",
        "treatment_timepoint": "none",
        "cell_sorting": "none",
    }
    parts = dirname.split("_")
    # 提取样本ID，通常是以GSM开头的部分
    gsm_id = parts[0] if parts[0].startswith("GSM") else "unknown"
    info["sample_id"] = gsm_id
    # 提取组织来源和疾病状态
    tissue_keywords = ["PBMC", "tumor","LM"]
    for keyword in tissue_keywords:
        if keyword in dirname:
            info["tissue"] = keyword
            break
    simplified = re.sub(r'^GSM\d+_raw_feature_bc_matrix_', '', dirname)
    simplified = re.sub(r'^GSM\d+_', '', simplified) if simplified == dirname else simplified
    # 提取患者编号，假设格式为P1, P2等，或者P1-P2等
    # 提取一个或多个以短横线连接的完整患者编号（如 COL07 或 COL07-COL12）
    patient_match = re.search(r'(\d+)', simplified)  # 使用 \d+ 匹配连续数字
    if patient_match:
        info['patient'] = patient_match.group(1)  # 直接获取数字编号
    tx_match = re.search(r'(post-treatment|pre-treatment)', simplified, re.IGNORECASE)
    # 提取治疗时间点，假设包含pre-Tx或post-Tx等关键词
    if tx_match:
        raw_tx = tx_match.group(1).lower()
        if "post-treatment" in raw_tx:
            info["treatment_timepoint"] = "post-treatment"
        elif "pre-treatment" in raw_tx:
            info["treatment_timepoint"] = "pre-treatment"

    # 编译正则表达式模式（保持不变）
    tg_match = re.compile(
        r'(TIL\s*therapy|unknown)',
        re.IGNORECASE)

    # 在目标字符串（例如 filename）中进行匹配
    match_result = tg_match.search(filename)  # 或使用 match 方法，根据需求
    # 提取治疗药物
    if match_result:
        raw_tg = match_result.group(1)
        if "TIL therapy" in raw_tg:
            info["treatment_drug"] = "TIL therapy"
        elif "unknown" in raw_tg:
            info["treatment_drug"] = "unknown"
    # 提取细胞分选信息
    sort_match = re.search(r'sorted_(.+?)_(PBMC|tumor|LM|Tumor|TUMOR|Normal)', simplified, re.IGNORECASE)
    if sort_match:
        info["cell_sorting"] = sort_match.group(1)
    print(info)
    return info

 #===========================生成gsm样本的详细信息，供标准化标签使用==========================
 #从GEO系列矩阵文件中提取特定GSM样本的详细信息
def extract_sample_info(series_matrix_file, target_gsm):

    desired_attributes = ["Sample_geo_accession"]

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
        result['gsm_id']=target_gsm
        attr1="Patient"
        attr2="disease"
        attr3="tissue"
        attr4="treatment_drug"
        attr5="treatment_timepoint"
        for b in range(4047944,4047948):
            b_gsm='"'+"GSM"+str(b) +'"'
            if target_gsm == b_gsm :
                result[attr1]="4095"
                result[attr2]="CRC"
                result[attr3]="tumor"
                result[attr4]="TIL therapy"
                result[attr5]="pre-treatment"
            else:
                continue
        for b in range(4047952,4047960):
            b_gsm='"'+"GSM"+str(b) +'"'
            if target_gsm == b_gsm :
                result[attr1]="4095"
                result[attr2]="CRC"
                result[attr3]="PBMC"
                result[attr4]="TIL therapy"
                result[attr5]="pre-treatment"
            else:
                continue
        if target_gsm == '"GSM4047948"':
            result[attr1] = "4007"
            result[attr2] = "CRC"
            result[attr3] = "tumor"
            result[attr4] = "unknown"
            result[attr5] = "post-treatment"
        elif target_gsm == '"GSM4047949"':
            result[attr1] = "4069"
            result[attr2] = "CRC"
            result[attr3] = "tumor"
            result[attr4] = "unknown"
            result[attr5] = "post-treatment"
        elif target_gsm == '"GSM4047950"':
            result[attr1] = "4071"
            result[attr2] = "CRC"
            result[attr3] = "tumor"
            result[attr4] = "unknown"
            result[attr5] = "post-treatment"
        elif target_gsm == '"GSM4047951"':
            result[attr1] = "4081"
            result[attr2] = "CRC"
            result[attr3] = "tumor"
            result[attr4] = "unknown"
            result[attr5] = "post-treatment"

        # 保存结果到字符串
        if result:
            # 将结果字典的所有值用下划线连接成字符串
            result_str = "_".join(str(value) for value in result.values() if value)
            result_clean = result_str.replace('"', '')
            print(f"信息已转换为字符串: {result_clean}")
            return result_clean


class SingleCellAnalyzer:
    def __init__(self, tar_file, sample_info_list):
        self.tar_file = tar_file
        self.sample_info_list = sample_info_list
        self.gsm_output_dir = "gsm_samples"
        os.makedirs(self.gsm_output_dir, exist_ok=True)
        self.metadata = {}
        self.gsm_files = {}
        self.cell_counts = {}  # 存储每个样本的细胞数量
        self.sample_summary = []  # 存储样本摘要信息

    def parse_sample_info(self):
        """从提供的样本信息字典列表中解析元数据"""
        target_gsm = [f"GSM40479{i:02d}" for i in range(44, 60)]

        for sample_info in self.sample_info_list:
            gsm_id = sample_info.get('sample_id')
            if not gsm_id or gsm_id not in target_gsm:
                continue

            # 从字典中提取信息
            metadata_entry = {
                'gsm_id': gsm_id,
                'patient_id': sample_info.get('patient', 'unknown'),
                'tissue_type': sample_info.get('tissue', 'unknown'),
                'disease_type': sample_info.get('disease', 'unknown'),
                'cancer_type': sample_info.get('cancer_type', 'unknown'),
                'treatment_drug': sample_info.get('treatment_drug', 'unknown'),
                'treatment_timepoint': sample_info.get('treatment_timepoint', 'unknown'),
                'cell_sorting': sample_info.get('cell_sorting', 'unknown'),
                'title': f"{gsm_id}_{sample_info.get('patient', 'unknown')}_{sample_info.get('tissue', 'unknown')}"
            }

            # 根据tissue字段判断sample_source
            tissue = sample_info.get('tissue', '').lower()
            if 'tumor' in tissue:
                metadata_entry['sample_source'] = 'tumor'
            elif 'normal' in tissue:
                metadata_entry['sample_source'] = 'normal'
            elif 'blood' in tissue or 'pbmc' in tissue.lower():
                metadata_entry['sample_source'] = 'blood'
            elif 'lm' in tissue.lower():  # LM可能表示肝转移
                metadata_entry['sample_source'] = 'liver_metastasis'
            else:
                metadata_entry['sample_source'] = 'unknown'

            self.metadata[gsm_id] = metadata_entry

        # 确保所有目标GSM都有元数据条目
        for gsm_id in target_gsm:
            if gsm_id not in self.metadata:
                print(f"警告: 样本 {gsm_id} 在提供的样本信息列表中未找到")
                self.metadata[gsm_id] = {
                    'gsm_id': gsm_id,
                    'patient_id': 'unknown',
                    'tissue_type': 'unknown',
                    'disease_type': 'unknown',
                    'cancer_type': 'unknown',
                    'treatment_drug': 'unknown',
                    'treatment_timepoint': 'unknown',
                    'cell_sorting': 'unknown',
                    'sample_source': 'unknown',
                    'title': gsm_id
                }

        return self.metadata

    def extract_and_organize_tar_files(self):
        target_gsm = [f"GSM40479{i:02d}" for i in range(44, 60)]

        with tarfile.open(self.tar_file, 'r') as tar:
            all_files = tar.getnames()

            for gsm_id in target_gsm:
                gsm_pattern = f"{gsm_id}_"
                gsm_file_list = [f for f in all_files if f.startswith(gsm_pattern) and f.endswith('.gz')]

                if gsm_file_list:
                    matrix_files = [f for f in gsm_file_list if 'matrix' in f.lower()]
                    barcode_files = [f for f in gsm_file_list if 'barcode' in f.lower()]
                    gene_files = [f for f in gsm_file_list if 'gene' in f.lower() or 'feature' in f.lower()]

                    self.gsm_files[gsm_id] = {
                        'matrix': matrix_files[0] if matrix_files else None,
                        'barcodes': barcode_files[0] if barcode_files else None,
                        'genes': gene_files[0] if gene_files else None
                    }
                else:
                    self.gsm_files[gsm_id] = {'matrix': None, 'barcodes': None, 'genes': None}

        return self.gsm_files

    def process_gsm_sample(self, gsm_id, tar):
        files = self.gsm_files[gsm_id]
        if not all(files.values()):
            print(f"警告: 样本 {gsm_id} 缺少必要的文件")
            return None

        temp_dir = tempfile.mkdtemp(prefix=f"temp_{gsm_id}_")

        try:
            matrix_path = os.path.join(temp_dir, "matrix.mtx.gz")
            barcodes_path = os.path.join(temp_dir, "barcodes.tsv.gz")
            genes_path = os.path.join(temp_dir, "genes.tsv.gz")

            tar.extract(files['matrix'], temp_dir)
            tar.extract(files['barcodes'], temp_dir)
            tar.extract(files['genes'], temp_dir)

            os.rename(os.path.join(temp_dir, files['matrix']), matrix_path)
            os.rename(os.path.join(temp_dir, files['barcodes']), barcodes_path)
            os.rename(os.path.join(temp_dir, files['genes']), genes_path)

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

            adata = sc.read_10x_mtx(
                temp_dir,
                var_names='gene_symbols',
                cache=False
            )

            adata.obs_names = [f"{gsm_id}_{barcode}" for barcode in adata.obs_names]
            adata.obs['gsm_id'] = gsm_id

            if gsm_id in self.metadata:
                meta = self.metadata[gsm_id]
                for key, value in meta.items():
                    adata.obs[key] = value
            else:
                adata.obs['patient_id'] = gsm_id
                adata.obs['tissue_type'] = 'unknown'
                adata.obs['disease_type'] = 'unknown'

            shutil.rmtree(temp_dir, ignore_errors=True)

            return adata

        except Exception as e:
            print(f"处理样本 {gsm_id} 时出错: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def quality_control(self, adata_combined):
        adata_combined.var["mt"] = adata_combined.var_names.str.startswith("MT-")

        sc.pp.calculate_qc_metrics(
            adata_combined, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts > 200, :]
        adata_combined = adata_combined[adata_combined.obs.n_genes_by_counts < 5000, :]
        adata_combined = adata_combined[adata_combined.obs.pct_counts_mt < 20, :]
        sc.pp.filter_genes(adata_combined, min_cells=3)

        return adata_combined

    def save_gsm_samples(self, adata_combined):
        """保存每个GSM样本的独立文件，并记录细胞数量"""
        for gsm_id in sorted(adata_combined.obs['gsm_id'].unique()):
            gsm_mask = adata_combined.obs['gsm_id'] == gsm_id
            adata_gsm = adata_combined[gsm_mask, :].copy()

            if adata_gsm.shape[0] > 0:
                output_file = os.path.join(self.gsm_output_dir, f"{gsm_id}_filtered.h5ad")
                adata_gsm.write(output_file)

                # 记录细胞数量
                cell_count = adata_gsm.shape[0]
                self.cell_counts[gsm_id] = cell_count
                print(f"已保存样本 {gsm_id} 到 {output_file}, 细胞数: {cell_count}")

    def generate_sample_summary_table(self):
        """生成样本摘要表格，类似于图片中的格式"""

        summary_data = []

        for gsm_id, metadata in self.metadata.items():
            # 获取细胞数量
            cell_count = self.cell_counts.get(gsm_id, 0)

            # 从metadata中获取其他信息
            patient = metadata.get('patient_id', 'unknown')
            tissue = metadata.get('tissue_type', 'unknown')
            disease = metadata.get('disease_type', 'unknown')
            cancer_type = metadata.get('cancer_type', 'unknown')
            treatment_drug = metadata.get('treatment_drug', 'unknown')
            treatment_timepoint = metadata.get('treatment_timepoint', 'unknown')
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

        # 保存为Excel文件（如果需要）
        excel_file = os.path.join(self.gsm_output_dir, "sample_summary.xlsx")
        summary_df.to_excel(excel_file, index=False)

        # 保存为TSV文件
        tsv_file = os.path.join(self.gsm_output_dir, "sample_summary.tsv")
        summary_df.to_csv(tsv_file, sep='\t', index=False)

        # 打印表格摘要
        print("\n" + "=" * 100)
        print("样本摘要表格已生成:")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100)
        print(f"\n表格已保存到:")
        print(f"  CSV格式: {output_file}")
        print(f"  Excel格式: {excel_file}")
        print(f"  TSV格式: {tsv_file}")

        return summary_df

    def generate_detailed_report(self, adata_combined):
        """生成详细分析报告"""
        report_file = os.path.join(self.gsm_output_dir, "analysis_report.txt")

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("单细胞RNA-seq数据分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {pd.Timestamp.now()}\n\n")

            f.write("1. 样本统计\n")
            f.write("-" * 40 + "\n")

            # 按样本统计
            sample_stats = adata_combined.obs['gsm_id'].value_counts().sort_index()
            for sample_id, count in sample_stats.items():
                f.write(f"  {sample_id}: {count} 个细胞\n")

            f.write(f"\n  总细胞数: {adata_combined.shape[0]}\n")
            f.write(f"  总基因数: {adata_combined.shape[1]}\n\n")

            f.write("2. 患者统计\n")
            f.write("-" * 40 + "\n")
            patient_stats = adata_combined.obs['patient_id'].value_counts()
            for patient_id, count in patient_stats.items():
                f.write(f"  {patient_id}: {count} 个细胞\n")

            f.write("\n3. 组织类型统计\n")
            f.write("-" * 40 + "\n")
            tissue_stats = adata_combined.obs['tissue_type'].value_counts()
            for tissue_type, count in tissue_stats.items():
                f.write(f"  {tissue_type}: {count} 个细胞\n")

            f.write("\n4. 疾病类型统计\n")
            f.write("-" * 40 + "\n")
            disease_stats = adata_combined.obs['disease_type'].value_counts()
            for disease_type, count in disease_stats.items():
                f.write(f"  {disease_type}: {count} 个细胞\n")

            f.write("\n5. 治疗药物统计\n")
            f.write("-" * 40 + "\n")
            treatment_stats = adata_combined.obs['treatment_drug'].value_counts()
            for treatment, count in treatment_stats.items():
                f.write(f"  {treatment}: {count} 个细胞\n")

        print(f"详细报告已保存到: {report_file}")

    def run_analysis(self):
        print("正在处理...")

        self.parse_sample_info()
        self.extract_and_organize_tar_files()

        adata_list = []

        with tarfile.open(self.tar_file, 'r') as tar:
            for gsm_id, files in self.gsm_files.items():
                if all(files.values()):
                    print(f"正在处理样本 {gsm_id}...")
                    adata = self.process_gsm_sample(gsm_id, tar)
                    if adata is not None:
                        adata_list.append(adata)
                        print(f"样本 {gsm_id} 处理完成，细胞数: {adata.shape[0]}")
                else:
                    print(f"跳过样本 {gsm_id}: 缺少文件")

        if not adata_list:
            print("错误: 没有成功加载任何样本数据")
            return

        print(f"\n成功加载 {len(adata_list)} 个样本")

        if len(adata_list) > 1:
            adata_combined = adata_list[0].concatenate(
                adata_list[1:],
                batch_key='batch',
                batch_categories=[adata.obs['gsm_id'].iloc[0] for adata in adata_list],
                index_unique=None
            )
        else:
            adata_combined = adata_list[0]

        print(f"合并后总细胞数: {adata_combined.shape[0]}, 总基因数: {adata_combined.shape[1]}")

        adata_qc = self.quality_control(adata_combined)
        print(f"质控后细胞数: {adata_qc.shape[0]}, 质控后基因数: {adata_qc.shape[1]}")

        # 保存单个样本文件
        self.save_gsm_samples(adata_qc)

        # 生成样本摘要表格
        summary_df = self.generate_sample_summary_table()

        # 生成详细分析报告
        self.generate_detailed_report(adata_qc)

        # 保存合并后的数据
        combined_file = os.path.join(self.gsm_output_dir, "combined_data_filtered.h5ad")
        adata_qc.write(combined_file)
        print(f"\n合并后的数据已保存到: {combined_file}")

        print("\n处理完成")

if __name__ == "__main__":
    # 文件路径
    series_matrix_file = "GSE136394_series_matrix.txt"
    input_dir=r'C:\Users\14584\Desktop\数据贴标签 - 0407\GSE136394'
    # 1. 使用提供的样本信息字典
    sample_info_list1 = []
    for a in range(4047944,4047960):
        target_gsm='"'+"GSM"+str(a) +'"'   # 替换为您要查找的GSM编号
        filename=extract_sample_info(series_matrix_file,target_gsm)
        output_info=parse_sample_info_from_dirname(filename)
        sample_info_list1.append(output_info)
    analyzer = SingleCellAnalyzer(
        tar_file="GSE136394_RAW.tar",
        sample_info_list=sample_info_list1
    )
    analyzer.run_analysis()