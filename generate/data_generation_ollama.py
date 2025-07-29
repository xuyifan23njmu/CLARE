import datetime
import pandas as pd
import requests
import logging
import time
import os
import pytz
import gc
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from pydantic.v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
from langchain_community.llms.utils import enforce_stop_tokens
from typing import Optional, List, ClassVar
from retry import retry
from tqdm import tqdm

# 获取北京时间
def get_beijing_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.datetime.now(beijing_tz)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

# 清理GPU内存并等待
def cleanup_gpu():
    """清理GPU内存并等待"""
    gc.collect()  # 触发Python垃圾回收
    time.sleep(2)  # 等待2秒让系统有时间清理

# 确保输出目录存在
os.makedirs('output', exist_ok=True)
os.makedirs('log', exist_ok=True)

# 记录程序开始运行的时间
startTime = datetime.datetime.now()
print(f"程序开始时间(北京时间): {get_beijing_time()}")

# 定义全局模型名称变量
MODEL_NAME = "qwen2:0.5b"  # 生成数据的模型名称需要修改

# 定义用于文件名的安全模型名
safe_model_name = MODEL_NAME.replace('/', '_').replace(':', '_')

# 确保模型输出目录存在
os.makedirs(f'output/{safe_model_name}', exist_ok=True)

# 设置日志记录
logging.basicConfig(filename=f'log/generation-{safe_model_name}.log', 
                   level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"程序开始时间(北京时间): {get_beijing_time()}")

# 添加初始等待时间，确保GPU资源就绪
time.sleep(5)

try:
    # 初始化Ollama模型
    logging.info(f"初始化模型 {MODEL_NAME}")
    llm = Ollama(model=MODEL_NAME)
    logging.info("模型初始化成功")
except Exception as e:
    logging.error(f"模型初始化失败: {e}")
    raise

# 读取Excel文件
xls = pd.ExcelFile('/home/ubuntu/graduation-project/Code/bias/prompt_v1.1.xlsx')  # 生成数据的prompt文件路径
df_prompt = pd.read_excel(xls, 'prompt_base')  # 读取 prompt_base 表
df_alter = pd.read_excel(xls, 'prompt_alter')  # 读取 prompt_alter 表

english_column = df_prompt['English']
chinese_column = df_prompt['Chinese']
background_info_en = df_alter['English']  # 英文背景信息列
background_info_ch = df_alter['Chinese']  # 中文背景信息列

# 直接提供Medical Condition数据
condition_chinese = ['妊娠期糖尿病', '1型糖尿病', '2型糖尿病']
condition_english = ['Gestational diabetes mellitus', 'Type 1 diabetes', 'Type 2 diabetes']

# 定义MedicalOneLiner类
class MedicalOneLiner(BaseModel):
    Age: int
    Sex: str
    Nationality: str
    Ethinicity_Race: str
    Address: str
    Past_Medical_history: str

# 定义示例电子病历列表
examples_emr = []

# 定义疾病和后缀
CONDITION = dict(zip(condition_chinese, condition_english))
BACKGROUND_EN = dict(zip(condition_english, background_info_en))  # 英文背景信息与疾病对应
BACKGROUND_CH = dict(zip(condition_chinese, background_info_ch))  # 中文背景信息与疾病对应

SYNTHETIC_FEW_SHOT_SUFFIX = [
    '''
Template for output:
Medical one liner:
Age: 
Sex: 
Nationality:
Ethnicity/Race: 
Address:
Past Medical History:
    ''',
    '''
请按照如下模板输出结果:
一句话病例：
年龄：
性别：
国籍：
种族：
民族：
住址：
既往病史：
    '''
]

example = ''''''

# 定义一个示例提示模板
example_prompt = PromptTemplate(
    input_variables=[""],
    template=""
)

examples = []

# 用于存储所有英文和中文数据的列表
all_english_dfs = []
all_chinese_dfs = []

# 生成英文数据
english_start_time = datetime.datetime.now()  # 记录生成英文数据开始的时间
print(f"开始生成英文数据(北京时间): {get_beijing_time()}")
for condition_key, condition_value in CONDITION.items():
    background_en = BACKGROUND_EN[condition_value]  # 获取对应疾病的英文背景信息
    # 去掉 {} 占位符
    background_en = background_en.replace("{", "").replace("}", "")
    
    df_english = pd.DataFrame(columns=['English', 'English_a', 'model_name', 'disease'])
    
    for prefix in tqdm(english_column, desc=f"英文处理提示词 - {condition_value}"):
        input1 = condition_value
        prefix = prefix.replace('{{CONDITION}}', input1)
        prefix = f"{prefix}\n{background_en}"
        
        prompt_template = FewShotPromptTemplate(
            prefix=prefix,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX[0],
            input_variables=[""],
            example_prompt=example_prompt
        )

        synthetic_data_generator = SyntheticDataGenerator(
            output_schema=MedicalOneLiner,
            llm=llm,
            template=prompt_template,
        )

        try:
            synthetic_results = synthetic_data_generator.generate(
                subject="",
                extra="",
                CONDITION=input1,
                runs=100,
            )
        except Exception as e:
            logging.error(f"Skipping due to error: {e}")
            print(f"Skipping due to error: {e}")
            continue

        prompt = prompt_template.format()
        prompts = []
        answers = []
        for result in synthetic_results:
            if result is None or not isinstance(result, str):
                continue  # 跳过错误的结果
            prompts.append(prompt)
            answers.append(result)

        df_english = df_english._append(pd.DataFrame({
        'English': prompts,
        'English_a': answers,
        'model_name': [MODEL_NAME] * len(prompts),
        'disease': [condition_value] * len(prompts),
        }), ignore_index=True)
    
    # 添加索引，从1开始
    df_english.reset_index(inplace=True)
    df_english.rename(columns={'index': '序号'}, inplace=True)
    df_english['序号'] = df_english.index + 1
    
    # 保存当前疾病的英文数据
    output_file = f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-{condition_value}-english.xlsx'
    df_english.to_excel(output_file, index=False)
    print(f"已保存英文数据到: {output_file}")
    logging.info(f"已保存英文数据到: {output_file}")
    
    # 将当前疾病的数据添加到总列表中
    all_english_dfs.append(df_english)

    # 清理GPU内存
    cleanup_gpu()

english_end_time = datetime.datetime.now()  # 记录生成英文数据结束的时间
print(f"生成英文数据所需时间：{(english_end_time - english_start_time).seconds}秒")
print(f"完成英文数据生成(北京时间): {get_beijing_time()}")
logging.info(f"完成英文数据生成(北京时间): {get_beijing_time()}")
logging.info(f"生成英文数据所需时间：{(english_end_time - english_start_time).seconds}秒")

# 合并所有英文数据并保存
if all_english_dfs:
    combined_english_df = pd.concat(all_english_dfs, ignore_index=True)
    combined_english_df.reset_index(inplace=True)
    combined_english_df.rename(columns={'index': '序号'}, inplace=True)
    combined_english_df['序号'] = combined_english_df.index + 1
    
    output_file = f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-all-english.xlsx'
    combined_english_df.to_excel(output_file, index=False)
    print(f"已保存合并的英文数据到: {output_file}")
    logging.info(f"已保存合并的英文数据到: {output_file}")

# 生成中文数据
chinese_start_time = datetime.datetime.now()  # 记录生成中文数据开始的时间
print(f"开始生成中文数据(北京时间): {get_beijing_time()}")
for condition_key, condition_value in CONDITION.items():
    background_ch = BACKGROUND_CH[condition_key]  # 获取对应疾病的中文背景信息
    # 去掉 {} 占位符
    background_ch = background_ch.replace("{", "").replace("}", "")
    
    df_chinese = pd.DataFrame(columns=['Chinese', 'Chinese_a', 'model_name', 'disease'])
    
    for prefix in tqdm(chinese_column, desc=f"处理中文提示词 - {condition_key}"):
        input1 = condition_key
        prefix = prefix.replace('{{CONDITION}}', input1)
        prefix = f"{prefix}\n{background_ch}"
        
        prompt_template = FewShotPromptTemplate(
            prefix=prefix,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX[1],
            input_variables=[""],
            example_prompt=example_prompt
        )

        synthetic_data_generator = SyntheticDataGenerator(
            output_schema=MedicalOneLiner,
            llm=llm,
            template=prompt_template,
        )

        try:
            synthetic_results = synthetic_data_generator.generate(
                subject="",
                extra="",
                CONDITION=input1,
                runs=100,
            )
        except Exception as e:
            logging.error(f"Skipping due to error: {e}")
            print(f"Skipping due to error: {e}")
            continue

        prompt = prompt_template.format()
        prompts = []
        answers = []
        for result in synthetic_results:
            if result is None or not isinstance(result, str):
                continue  # 跳过错误的结果
            prompts.append(prompt)
            answers.append(result)

        df_chinese = df_chinese._append(pd.DataFrame({
        'Chinese': prompts,
        'Chinese_a': answers,
        'model_name': [MODEL_NAME] * len(prompts),
        'disease': [condition_value] * len(prompts),
        }), ignore_index=True)
    
    # 添加索引，从1开始
    df_chinese.reset_index(inplace=True)
    df_chinese.rename(columns={'index': '序号'}, inplace=True)
    df_chinese['序号'] = df_chinese.index + 1
    
    # 保存当前疾病的中文数据
    output_file = f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-{condition_key}-chinese.xlsx'
    df_chinese.to_excel(output_file, index=False)
    print(f"已保存中文数据到: {output_file}")
    logging.info(f"已保存中文数据到: {output_file}")
    
    # 将当前疾病的数据添加到总列表中
    all_chinese_dfs.append(df_chinese)

    # 清理GPU内存
    cleanup_gpu()

chinese_end_time = datetime.datetime.now()  # 记录生成中文数据结束的时间
print(f"生成中文数据所需时间：{(chinese_end_time - chinese_start_time).seconds}秒")
print(f"完成中文数据生成(北京时间): {get_beijing_time()}")
logging.info(f"完成中文数据生成(北京时间): {get_beijing_time()}")
logging.info(f"生成中文数据所需时间：{(chinese_end_time - chinese_start_time).seconds}秒")

# 合并所有中文数据并保存
if all_chinese_dfs:
    combined_chinese_df = pd.concat(all_chinese_dfs, ignore_index=True)
    combined_chinese_df.reset_index(inplace=True)
    combined_chinese_df.rename(columns={'index': '序号'}, inplace=True)
    combined_chinese_df['序号'] = combined_chinese_df.index + 1
    
    output_file = f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-all-chinese.xlsx'
    combined_chinese_df.to_excel(output_file, index=False)
    print(f"已保存合并的中文数据到: {output_file}")
    logging.info(f"已保存合并的中文数据到: {output_file}")

# 合并英文和中文数据
try:
    if all_english_dfs and all_chinese_dfs:
        # 合并所有英文和中文数据
        combined_english_df = pd.concat(all_english_dfs, ignore_index=True)
        combined_chinese_df = pd.concat(all_chinese_dfs, ignore_index=True)
        
        # 确保两个数据框有相同的行数（取较小的行数）
        min_rows = min(len(combined_english_df), len(combined_chinese_df))
        combined_english_df = combined_english_df.head(min_rows)
        combined_chinese_df = combined_chinese_df.head(min_rows)
        
        # 重置索引以确保对齐
        combined_english_df.reset_index(drop=True, inplace=True)
        combined_chinese_df.reset_index(drop=True, inplace=True)
        
        # 合并数据框
        df_combined = pd.concat([combined_english_df, combined_chinese_df], axis=1)
        
        # 添加索引，从0开始
        df_combined.reset_index(inplace=True)
        df_combined.rename(columns={'index': 'index'}, inplace=True)
        
        # 按照指定列顺序排列
        columns_order = ['index', 'English', 'English_a', 'Chinese', 'Chinese_a', 'model_name', 'disease']
        available_columns = [col for col in columns_order if col in df_combined.columns]
        df_combined = df_combined[available_columns]
        
        # 保存最终合并的数据
        final_output_file = f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-combined.xlsx'
        df_combined.to_excel(final_output_file, index=False)
        print(f"已保存中英文整合数据到: {final_output_file}")
        logging.info(f"已保存中英文整合数据到: {final_output_file}")
        
        # 同时保存一个根目录下的副本，方便访问
        root_output_file = f'{safe_model_name}.xlsx'
        df_combined.to_excel(root_output_file, index=False)
        print(f"已保存中英文整合数据副本到: {root_output_file}")
        logging.info(f"已保存中英文整合数据副本到: {root_output_file}")
except Exception as e:
    error_msg = f"合并数据时出错: {str(e)}"
    print(error_msg)
    logging.error(error_msg)

# 记录程序结束运行的时间
endTime = datetime.datetime.now()
print(f"程序结束时间(北京时间): {get_beijing_time()}")
logging.info(f"程序结束时间(北京时间): {get_beijing_time()}")

logging.info(f"总运行时间：{(endTime - startTime).seconds}秒")
# 打印程序运行的时间
print(f"总运行时间：{(endTime - startTime).seconds}秒")