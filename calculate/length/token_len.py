import pandas as pd
from pathlib import Path

# 设置数据目录
directory = Path("/home/ubuntu/graduation-project/data/outcome_v5.13")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# 用于存储统计结果
summary = []

# 遍历目录中的所有 xlsx 文件
for file in directory.glob("*.xlsx"):
    df = pd.read_excel(file)

    # 跳过缺少必要字段的文件
    if not {"model_name", "English_a_tokens", "Chinese_a_tokens"}.issubset(df.columns):
        continue

    model_name = df["model_name"].iloc[0]
    eng_tokens = df["English_a_tokens"].dropna()
    chi_tokens = df["Chinese_a_tokens"].dropna()

    summary.append({
        "model_name": model_name,
        "english_a_tokens": f"{eng_tokens.mean():.2f} ± {eng_tokens.std():.2f}",
        "chinese_a_tokens": f"{chi_tokens.mean():.2f} ± {chi_tokens.std():.2f}"
    })

# 转为 DataFrame 并排序展示
result_df = pd.DataFrame(summary)

# 提取参数数字进行排序（假设 model_name 中包含如 7B, 14B 等）
def extract_param(model_name):
    import re
    match = re.search(r'(\d+\.?\d*)B', model_name)
    return float(match.group(1)) if match else 0

result_df["model_param"] = result_df["model_name"].apply(extract_param)
result_df = result_df.sort_values(by="model_param", ascending=False).drop(columns=["model_param"])

# 输出或保存
print(result_df)
result_df.to_excel("/home/ubuntu/graduation-project/Code/统计/字符长度/token_length_summary_qwen.xlsx", index=False)