import datetime
import pandas as pd
import requests
import time
import multiprocessing
import os
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from pydantic.v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from typing import Optional, List, ClassVar
from retry import retry
from tqdm import tqdm

# 记录程序开始运行的时间
startTime = datetime.datetime.now()

# 定义要处理的多个模型名称
MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
]

# 自定义LLM类
class ModelChat(LLM):
    history: ClassVar[List] = []  # 初始化历史记录为空列表，并标记为ClassVar
    api_secret: str = ""  # 初始化api_secret为空字符串
    model_name: str = ""  # 添加model_name属性

    def __init__(self, api_secret: str, model_name: str):  # 构造函数，接受api_secret和model_name参数
        super().__init__()  # 调用父类的构造函数
        self.api_secret = api_secret  # 设置实例的api_secret属性
        self.model_name = model_name  # 设置实例的model_name属性

    @property
    def _llm_type(self) -> str:  # 定义_llm_type属性
        return None  # 返回None

    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt):  # 定义model_completion方法，接受messages参数
        url = "https://api.siliconflow.cn/v1/chat/completions"  # 设置API请求的URL
        headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_secret}"
                    }
        payload = {
                "model": self.model_name,  # 使用实例的model_name属性
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": 512,
                "temperature": 0.9,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1
            }

        response = requests.post(url, json=payload, headers=headers)  # 发送POST请求
        if response.status_code == 200:  # 如果响应状态码为200
            return response.json()["choices"][0]["message"]["content"]  # 返回响应内容中的消息内容
        else:  # 如果响应状态码不是200
            print(f"Error: {response.status_code}, {response.text}")  # 打印错误信息
            raise Exception(f"API request failed with status code {response.status_code}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:  # 定义_call方法，接受prompt和stop参数
        for _ in range(5):  # 重试五次
            try:
                response = self.model_completion(prompt)
                if stop is not None:  # 如果stop参数不为空
                    response = enforce_stop_tokens(response, stop)  # 使用enforce_stop_tokens函数处理响应
                return response  # 返回响应
            except Exception as e:
                print(f"Error: {e}, retrying...")
                time.sleep(5)  # 等待5秒后重试
        print("Skipping due to repeated errors.")
        return None

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:  # 定义__call__方法，接受prompt和stop参数
        return self._call(prompt, stop)  # 调用_call方法并返回结果

# 定义一个函数，处理单个模型的数据生成
def process_model(model_name):
    print(f"开始处理模型: {model_name}")
    model_start_time = datetime.datetime.now()
    
    # 定义用于文件名的安全模型名
    safe_model_name = model_name.replace('/', '_')
    
    # API secret
    api_secret = "sk-biumwxhvgyqwyjtqtkcbjkpfacibnzblbakumlbrzzwrgdmn"
    # 创建一个ModelChat对象
    llm = ModelChat(api_secret, model_name)

    # 读取Excel文件
    xls = pd.ExcelFile('/home/ubuntu/graduation-project/Code/bias/prompt_v1.1.xlsx') 
    df_prompt = pd.read_excel(xls, 'prompt_base')  # 读取 prompt_base 表
    df_alter = pd.read_excel(xls, 'prompt_alter')  # 读取 prompt_alter 表

    english_column = df_prompt['English']
    chinese_column = df_prompt['Chinese']
    background_info_en = df_alter['English']
    background_info_ch = df_alter['Chinese']

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
    
    # 创建输出目录（如果不存在）
    output_dir = '/home/ubuntu/graduation-project/Code/bias/output'
    os.makedirs(output_dir, exist_ok=True)

    # 用于收集所有数据的DataFrame
    all_english_data = pd.DataFrame()
    all_chinese_data = pd.DataFrame()

    # 生成英文数据
    english_start_time = datetime.datetime.now()
    print(f"开始为模型 {model_name} 生成英文数据")
    for condition_key, condition_value in CONDITION.items():
        background_en = BACKGROUND_EN[condition_value]
        # 去掉 {} 占位符
        background_en = background_en.replace("{", "").replace("}", "")

        df_english = pd.DataFrame(columns=['English', 'English_a', 'model_name', 'disease'])
        
        for prefix in tqdm(english_column, desc=f"[{model_name}] 英文处理提示词"):
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
                total_runs = 100
                synthetic_results = []
                for run in tqdm(range(total_runs), desc=f"[{model_name}] 生成第 {len(synthetic_results) + 1}-{total_runs} 次", leave=False):
                    try:
                        result = synthetic_data_generator.generate(
                            subject="",
                            extra="",
                            CONDITION=input1,
                            runs=1,
                        )
                        if result and isinstance(result[0], str):
                            synthetic_results.extend(result)
                    except Exception as e:
                        print(f"第{run+1}次生成失败: {e}")
                        continue
                    
                    # 每10次生成后等待一小段时间，避免请求过于频繁
                    if (run + 1) % 10 == 0:
                        time.sleep(1)
                
                print(f"成功完成prompt生成，共生成 {len(synthetic_results)} 条有效数据")

            except Exception as e:
                print(f"[{model_name}] 跳过出错的请求: {e}")
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
                'model_name': [model_name] * len(prompts),
                'disease': [condition_value] * len(prompts),
            }), ignore_index=True)
            
        # 添加索引，从1开始
        df_english.reset_index(inplace=True)
        df_english.rename(columns={'index': '序号'}, inplace=True)
        df_english['序号'] = df_english.index + 1
        
        # 累加到全部英文数据
        all_english_data = pd.concat([all_english_data, df_english])
        
        # 保存生成的数据到Excel文件
        output_file = f'{output_dir}/{safe_model_name}-{condition_value}-english.xlsx'
        df_english.to_excel(output_file, index=False)
        print(f"[{model_name}] 已保存英文数据到: {output_file}")
    
    english_end_time = datetime.datetime.now()
    print(f"[{model_name}] 生成英文数据所需时间：{(english_end_time - english_start_time).seconds}秒")
    
    # 生成中文数据
    chinese_start_time = datetime.datetime.now()
    print(f"开始为模型 {model_name} 生成中文数据")
    for condition_key, condition_value in CONDITION.items():
        background_ch = BACKGROUND_CH[condition_key]
        # 去掉 {} 占位符
        background_ch = background_ch.replace("{", "").replace("}", "")
        
        df_chinese = pd.DataFrame(columns=['Chinese', 'Chinese_a', 'model_name', 'disease'])
        
        for prefix in tqdm(chinese_column, desc=f"[{model_name}] 处理中文提示词"):
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
                total_runs = 100
                synthetic_results = []
                for run in tqdm(range(total_runs), desc=f"[{model_name}] 生成第 {len(synthetic_results) + 1}-{total_runs} 次", leave=False):
                    try:
                        result = synthetic_data_generator.generate(
                            subject="",
                            extra="",
                            CONDITION=input1,
                            runs=1,
                        )
                        if result and isinstance(result[0], str):
                            synthetic_results.extend(result)
                    except Exception as e:
                        print(f"第{run+1}次生成失败: {e}")
                        continue
                    
                    # 每10次生成后等待一小段时间，避免请求过于频繁
                    if (run + 1) % 10 == 0:
                        time.sleep(1)
                
                print(f"成功完成prompt生成，共生成 {len(synthetic_results)} 条有效数据")

            except Exception as e:
                print(f"[{model_name}] 跳过出错的请求: {e}")
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
                'model_name': [model_name] * len(prompts),
                'disease': [condition_value] * len(prompts),
            }), ignore_index=True)
            
        # 添加索引，从1开始
        df_chinese.reset_index(inplace=True)
        df_chinese.rename(columns={'index': '序号'}, inplace=True)
        df_chinese['序号'] = df_chinese.index + 1
        
        # 累加到全部中文数据
        all_chinese_data = pd.concat([all_chinese_data, df_chinese])
        
        # 保存生成的数据到Excel文件
        output_file = f'{output_dir}/{safe_model_name}-{condition_value}-chinese.xlsx'
        df_chinese.to_excel(output_file, index=False)
        print(f"[{model_name}] 已保存中文数据到: {output_file}")
    
    chinese_end_time = datetime.datetime.now()
    print(f"[{model_name}] 生成中文数据所需时间：{(chinese_end_time - chinese_start_time).seconds}秒")

    # 合并英文和中文数据
    all_english_data.reset_index(drop=True, inplace=True)
    all_chinese_data.reset_index(drop=True, inplace=True)
    
    # 确保两个DataFrame有相同的行数
    min_rows = min(len(all_english_data), len(all_chinese_data))
    all_english_data = all_english_data.iloc[:min_rows]
    all_chinese_data = all_chinese_data.iloc[:min_rows]
    
    df = pd.concat([all_english_data, all_chinese_data], axis=1)

    # 添加索引，从0开始
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'index'}, inplace=True)

    # 按照指定列顺序排列
    columns_order = ['index', 'English', 'English_a', 'Chinese', 'Chinese_a', 'model_name', 'disease']
    df = df[columns_order]

    # 保存最终合并的数据
    final_output_file = f'{output_dir}/{safe_model_name}.xlsx'
    df.to_excel(final_output_file, index=False)
    print(f"[{model_name}] 已保存合并数据到: {final_output_file}")

    model_end_time = datetime.datetime.now()
    print(f"模型 {model_name} 总处理时间: {(model_end_time - model_start_time).seconds}秒")
    return model_name

# 主程序流程
if __name__ == "__main__":
    # 创建进程池
    pool = multiprocessing.Pool(processes=len(MODEL_NAMES))
    
    # 并行处理多个模型
    results = pool.map(process_model, MODEL_NAMES)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 记录程序结束运行的时间
    endTime = datetime.datetime.now()
    
    # 打印程序运行的时间
    print("所有模型处理完成！")
    print("总运行时间是：%ss" % (endTime - startTime).seconds)
    for model_name in results:
        print(f"已完成模型: {model_name}")