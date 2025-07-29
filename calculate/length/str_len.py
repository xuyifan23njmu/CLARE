import pandas as pd
import os
from pathlib import Path

# 设置保存路径
output_directory = "/home/ubuntu/graduation-project/data/outcome_v5.10"
os.makedirs(output_directory, exist_ok=True)

# 指定目录路径
directory = "/home/ubuntu/graduation-project/data/outcome_v5.9"


# 遍历目录中的所有xlsx文件并直接在原文件添加统计信息
for file in Path(directory).glob('*.xlsx'):
    try:
        # 读取Excel文件
        df = pd.read_excel(file)
        
        # 检查所需列是否存在
        required_columns = ['English_a', 'Chinese_a']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: {file.name} missing required columns")
            continue
        
        # 添加每个字段对应的字符长度列（方便后续统计）
        for field in ['English_a', 'Chinese_a']:
            lengths = df[field].astype(str).str.len()
            df[f'{field}_length'] = lengths  # 添加每个单元格的实际长度
        
        # 保存更新后的原始文件
        new_name = file.stem.replace("v5.9", "v5.10") + ".xlsx"
        output_file = os.path.join(output_directory, new_name)
        df.to_excel(output_file, index=False)
        print(f"Processed {file.name} successfully")
            
    except Exception as e:
        print(f"Error processing file {file.name}: {str(e)}")