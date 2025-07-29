import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import re
from glob import glob

# 读取真实数据
real_file_path = "/home/ubuntu/graduation-project/Code/统计/性别/real-sex-v1.xlsx"
realdf = pd.read_excel(real_file_path)

# 1. 从字符串中提取纯数字
def extract_number(value):
    if isinstance(value, str) and '(' in value:
        return int(re.search(r'(\d+)\s*\(', value).group(1))
    return value

# 2. 处理真实数据
real_gender_counts = {}
# 真实数据中的疾病类型
diseases = ['T1DM', 'T2DM']  # 不包含 GDM
# 真实数据中的性别行索引
gender_indices = {'Female': 1, 'Male': 2}

for disease in diseases:
    real_gender_counts[disease] = {}
    for gender, idx in gender_indices.items():
        value = realdf.loc[idx, disease]
        real_gender_counts[disease][gender] = extract_number(value)

# 3. 获取目录下所有相关文件
directory = "/home/ubuntu/graduation-project/data/outcome_sex_v5.1"
# 使用更广泛的匹配模式，捕获所有Excel文件
all_files = glob(os.path.join(directory, "*.xlsx"))
print(f"找到 {len(all_files)} 个Excel文件")

# 过滤出可能的性别数据文件
files = []
for file in all_files:
    if "_sex_" in file.lower() or "statistics" in file.lower():
        files.append(file)

print(f"其中 {len(files)} 个文件可能包含性别数据")

# 存储所有结果 - 按语言和疾病类型分类
chinese_t1dm_results = []
chinese_t2dm_results = []
english_t1dm_results = []
english_t2dm_results = []

# 4. 遍历每个文件进行卡方检验
for file_path in files:
    # 提取模型名称和语言
    file_name = os.path.basename(file_path)
    print(f"处理文件: {file_name}")
    
    # 更准确地识别模型名称
    if "_v5.1_sex_" in file_name:
        model_name = file_name.split("_v5.1_sex_")[0]
    else:
        # 尝试其他分隔方式
        model_parts = file_name.split("_")
        model_name = model_parts[0] if len(model_parts) > 0 else "Unknown"
    
    # 更准确地识别语言 - 将Unknown改为English
    language = "English"  # 默认为English
    if "_chinese" in file_name.lower() or "chinese" in file_name.lower():
        language = "Chinese"
    
    print(f"  模型: {model_name}, 语言: {language}")
    
    try:
        # 读取模型生成的数据
        df = pd.read_excel(file_path)
        print(f"  数据形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")
        
        # 确定列名映射
        disease_mapping = {
            'T1DM': 'Type 1 diabetes',
            'T2DM': 'Type 2 diabetes'
        }
        
        # 检查每一列，查找可能的疾病名称
        for col in df.columns:
            col_lower = col.lower()
            if "type 1" in col_lower or "t1dm" in col_lower:
                disease_mapping['T1DM'] = col
            if "type 2" in col_lower or "t2dm" in col_lower:
                disease_mapping['T2DM'] = col
        
        print(f"  疾病映射: {disease_mapping}")
        
        # 性别索引可能有所不同，尝试检测
        model_gender_indices = {'Female': None, 'Male': None}
        for idx, row in df.iterrows():
            for col in df.columns:
                if isinstance(row[col], str):
                    row_val_lower = row[col].lower()
                    if "female" in row_val_lower or "女" in row_val_lower:
                        model_gender_indices['Female'] = idx
                    if "male" in row_val_lower or "男" in row_val_lower:
                        model_gender_indices['Male'] = idx
        
        # 如果没有检测到，使用默认值
        if model_gender_indices['Female'] is None:
            model_gender_indices['Female'] = 0
        if model_gender_indices['Male'] is None:
            model_gender_indices['Male'] = 1
        
        print(f"  性别索引: {model_gender_indices}")
        
        # 处理模型数据
        model_gender_counts = {}
        for real_disease, model_disease in disease_mapping.items():
            model_gender_counts[real_disease] = {}
            for gender, idx in model_gender_indices.items():
                # 确保索引和列存在
                if idx < len(df) and model_disease in df.columns:
                    value = df.loc[idx, model_disease]
                    model_gender_counts[real_disease][gender] = extract_number(value)
                else:
                    print(f"  警告: 在文件中找不到 {gender} 的 {model_disease} 数据")
                    model_gender_counts[real_disease][gender] = 0
        
        print(f"  真实性别计数: {real_gender_counts}")
        print(f"  模型性别计数: {model_gender_counts}")
        
        # 对各种疾病类型分别进行卡方检验
        for disease in diseases:
            # 创建观测值表格
            observed = np.array([
                [real_gender_counts[disease]['Female'], real_gender_counts[disease]['Male']],
                [model_gender_counts[disease]['Female'], model_gender_counts[disease]['Male']]
            ])
            
            print(f"  观测值表格 - {disease}: {observed}")
            
            # 进行卡方检验
            chi2, p, dof, expected = chi2_contingency(observed)
            
            # 准备结果
            result = {
                'model': model_name,
                'language': language,
                'disease': disease,
                'p值': p,         # 性别分布差异的p值
                'chi2': chi2,
                'dof': dof
            }
            
            # 根据语言和疾病类型分类结果
            if language == "Chinese":
                if disease == "T1DM":
                    chinese_t1dm_results.append(result)
                elif disease == "T2DM":
                    chinese_t2dm_results.append(result)
            else:  # English
                if disease == "T1DM":
                    english_t1dm_results.append(result)
                elif disease == "T2DM":
                    english_t2dm_results.append(result)
    
    except Exception as e:
        print(f"  处理文件时发生错误: {e}")
        # 继续处理下一个文件

# 5. 将结果转换为DataFrame并保存
output_dir = "/home/ubuntu/graduation-project/Code/统计/性别"

# 中文-1型糖尿病结果
if chinese_t1dm_results:
    df_chinese_t1dm = pd.DataFrame(chinese_t1dm_results)
    df_chinese_t1dm['p值'] = df_chinese_t1dm['p值'].apply(lambda x: f"{x:.4e}")
    chinese_t1dm_file = os.path.join(output_dir, "chinese_t1dm_chi_square_results.xlsx")
    df_chinese_t1dm.to_excel(chinese_t1dm_file, index=False)
    print(f"中文-1型糖尿病结果已保存到 {chinese_t1dm_file}")

# 中文-2型糖尿病结果
if chinese_t2dm_results:
    df_chinese_t2dm = pd.DataFrame(chinese_t2dm_results)
    df_chinese_t2dm['p值'] = df_chinese_t2dm['p值'].apply(lambda x: f"{x:.4e}")
    chinese_t2dm_file = os.path.join(output_dir, "chinese_t2dm_chi_square_results.xlsx")
    df_chinese_t2dm.to_excel(chinese_t2dm_file, index=False)
    print(f"中文-2型糖尿病结果已保存到 {chinese_t2dm_file}")

# 英文-1型糖尿病结果
if english_t1dm_results:
    df_english_t1dm = pd.DataFrame(english_t1dm_results)
    df_english_t1dm['p值'] = df_english_t1dm['p值'].apply(lambda x: f"{x:.4e}")
    english_t1dm_file = os.path.join(output_dir, "english_t1dm_chi_square_results.xlsx")
    df_english_t1dm.to_excel(english_t1dm_file, index=False)
    print(f"英文-1型糖尿病结果已保存到 {english_t1dm_file}")

# 英文-2型糖尿病结果
if english_t2dm_results:
    df_english_t2dm = pd.DataFrame(english_t2dm_results)
    df_english_t2dm['p值'] = df_english_t2dm['p值'].apply(lambda x: f"{x:.4e}")
    english_t2dm_file = os.path.join(output_dir, "english_t2dm_chi_square_results.xlsx")
    df_english_t2dm.to_excel(english_t2dm_file, index=False)
    print(f"英文-2型糖尿病结果已保存到 {english_t2dm_file}")

# 所有结果合并在一起
all_results = chinese_t1dm_results + chinese_t2dm_results + english_t1dm_results + english_t2dm_results

if all_results:
    # 将所有结果转换为DataFrame
    results_df = pd.DataFrame(all_results)
    # 将p值转换为科学计数法格式的字符串
    results_df['p值'] = results_df['p值'].apply(lambda x: f"{x:.4e}")
    # 保存所有结果
    output_file = os.path.join(output_dir, "all_models_chi_square_results_v1.xlsx")
    results_df.to_excel(output_file, index=False)
    print(f"\n所有结果已保存到 {output_file}")