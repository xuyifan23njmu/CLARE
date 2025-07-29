import pandas as pd
import os

# 定义四种情况的分类函数
def classify_case(age, sex):
    # 检查是否为缺失值、"unknown"，以及非数值内容（对于年龄），或数字型性别
    is_age_missing = (
        pd.isna(age)
        or str(age).lower() == "unknown"
        or not str(age).replace('.', '', 1).isdigit()
    )
    is_sex_missing = (
        pd.isna(sex)
        or str(sex).lower() == "unknown"
        or str(sex).isdigit()
    )
    if is_age_missing and is_sex_missing:
        return 'Neither age nor gender'
    elif not is_age_missing and is_sex_missing:
        return 'Only age'
    elif is_age_missing and not is_sex_missing:
        return 'Only gender'
    else:
        return 'Both age and gender'

# 输入输出路径设置
input_folder = '/home/ubuntu/graduation-project/data/outcome_v5.1'
output_folder = '/home/ubuntu/graduation-project/Code/统计/EPS'
summary_list = []

# 遍历文件夹中的所有xlsx文件
for file in os.listdir(input_folder):
    if file.endswith('.xlsx'):
        model_name = file.replace('_v5.1.xlsx', '')
        file_path = os.path.join(input_folder, file)
        
        try:
            df = pd.read_excel(file_path)
            print(f"正在处理文件: {file}")
        except Exception as e:
            print(f"无法读取文件 {file}：{e}")
            continue

        # 应用分类
        eng_case = df.apply(lambda row: classify_case(row.get('Age'), row.get('Sex')), axis=1)
        chi_case = df.apply(lambda row: classify_case(row.get('Age_Chinese'), row.get('Sex_Chinese')), axis=1)

        # 统计每种情况的数量
        eng_counts = eng_case.value_counts().reindex(
            ['Both age and gender', 'Only age', 'Only gender', 'Neither age nor gender'], fill_value=0)
        chi_counts = chi_case.value_counts().reindex(
            ['Both age and gender', 'Only age', 'Only gender', 'Neither age nor gender'], fill_value=0)

        # 添加到汇总表中
        summary_list.append({
            'Model': model_name,
            'English: Both': eng_counts['Both age and gender'],
            'English: Only age': eng_counts['Only age'],
            'English: Only gender': eng_counts['Only gender'],
            'English: Neither': eng_counts['Neither age nor gender'],
            'Chinese: Both': chi_counts['Both age and gender'],
            'Chinese: Only age': chi_counts['Only age'],
            'Chinese: Only gender': chi_counts['Only gender'],
            'Chinese: Neither': chi_counts['Neither age nor gender'],
        })

# 创建汇总DataFrame
summary_df = pd.DataFrame(summary_list)

# 按模型名称排序
summary_df = summary_df.sort_values('Model')

# 保存结果
output_path = os.path.join(output_folder, 'age_gender_generation_summary_v5.1.xlsx')
summary_df.to_excel(output_path, index=False)

print(f"\n统计完成，结果已保存至: {output_path}")

# 打印统计结果
print("\n统计结果概览:")
print(summary_df.to_string())