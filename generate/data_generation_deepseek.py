import datetime
import pandas as pd
import requests
import time
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from pydantic.v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from typing import Optional, List, ClassVar
from retry import retry
from tqdm import tqdm
import os
# 记录程序开始运行的时间
startTime = datetime.datetime.now()

# 定义全局模型名称变量
MODEL_NAME = "deepseek-r1 " ## 生成数据的模型名称需要修改

# 定义用于文件名的安全模型名
safe_model_name = MODEL_NAME.replace('/', '_')
# 自定义LLM类
class ModelChat(LLM):
    history: ClassVar[List] = []  # 初始化历史记录为空列表，并标记为ClassVar
    api_secret: str = ""  # 初始化api_secret为空字符串

    def __init__(self, api_secret: str):  # 构造函数，接受api_secret参数
        super().__init__()  # 调用父类的构造函数
        self.api_secret = api_secret  # 设置实例的api_secret属性

    @property
    def _llm_type(self) -> str:  # 定义_llm_type属性
        return None  # 返回None

    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt):  # 定义model_completion方法，接受messages参数
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 设置API请求的URL
        headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_secret}"
                    }
        payload = {
                "model": MODEL_NAME,
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

# API secret
# api_secret = "sk-biumwxhvgyqwyjtqtkcbjkpfacibnzblbakumlbrzzwrgdmn"
api_secret = os.getenv("DASHSCOPE_API_KEY")
# 创建一个LlamaChat对象
llm = ModelChat(api_secret)

# 读取Excel文件
xls = pd.ExcelFile('/home/ubuntu/graduation-project/Code/bias/prompt.xlsx') ## 生成数据的prompt文件路径需要修改
df_prompt = pd.read_excel(xls, 'Sheet1')
english_column = df_prompt['English']
chinese_column = df_prompt['Chinese']

# # 读取Medical Condition表
# df_condition = pd.read_excel(xls, 'medical_condition')
# condition_chinese = df_condition['Medical Condition_chinese']
# condition_english = df_condition['Medical Condition_en']

# 直接提供Medical Condition数据
condition_chinese = ['妊娠期糖尿病', '1型糖尿病', '2型糖尿病']
condition_english = ['Gestational diabetes mellitus', 'Type 1 diabetes', 'Type 2 diabetes']

# #读取location表
# df_location = pd.read_excel(xls, 'location')
# location_chinese = df_location['Location_ch']
# location_english = df_location['Location_en']

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

# #定义地点和后缀
# LOCATION = dict(zip(location_chinese, location_english))

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

# 生成英文数据
english_start_time = datetime.datetime.now()  # 记录生成英文数据开始的时间
for condition_key, condition_value in CONDITION.items():
    # for location_key, location_value in LOCATION.items():
        df_english = pd.DataFrame(columns=['English', 'English_a', 'model_name', 'disease'])
        for prefix in tqdm(english_column, desc="英文处理提示词"):
            input1 = condition_value
            # input2 = location_value
            prefix = prefix.replace('{{CONDITION}}', input1)
            # prefix = prefix.replace('[LOCATION]', input2)
            prompt_template = FewShotPromptTemplate(
                prefix=prefix,
                examples=examples,  # 添加examples参数
                suffix=SYNTHETIC_FEW_SHOT_SUFFIX[0],
                input_variables=[""],  # 添加input_variables参数
                example_prompt=example_prompt  # 添加example_prompt参数
            )
            # print(prompt_template)
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
                   # LOCATION=input2,
                    runs=1,
                )
            except Exception as e:
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
            # 'location': [location_value] * len(prompts)
            }), ignore_index=True)
        # 添加索引，从1开始
        df_english.reset_index(inplace=True)
        df_english.rename(columns={'index': '序号'}, inplace=True)
        df_english['序号'] = df_english.index + 1
        # 保存生成的数据到Excel文件
        df_english.to_excel(f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-{condition_value}-english.xlsx', index=False) ## 生成数据的输出路径需要修改
        english_end_time = datetime.datetime.now()  # 记录生成英文数据结束的时间
        print(f"生成数据所需时间：{(english_end_time - english_start_time).seconds}秒")
        
        # 生成中文数据
chinese_start_time = datetime.datetime.now()  # 记录生成中文数据开始的时间
for condition_key, condition_value in CONDITION.items():
    # for location_key, location_value in LOCATION.items():
        df_chinese = pd.DataFrame(columns=['Chinese', 'Chinese_a', 'model_name', 'disease'])
        for prefix in tqdm(chinese_column, desc="处理中文提示词"):
            input1 = condition_key
            # input2 = location_key
            prefix = prefix.replace('{{CONDITION}}', input1)
            # prefix = prefix.replace('[LOCATION]', input2)
            prompt_template = FewShotPromptTemplate(
                prefix=prefix,
                examples=examples,  # 添加examples参数
                suffix=SYNTHETIC_FEW_SHOT_SUFFIX[1],
                input_variables=[""],  # 添加input_variables参数
                example_prompt=example_prompt  # 添加example_prompt参数
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
                    # LOCATION=input2,
                    runs=1,
                )
            except Exception as e:
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
            # 'location': [location_value] * len(prompts)
            }), ignore_index=True)
        # 添加索引，从1开始
        df_chinese.reset_index(inplace=True)
        df_chinese.rename(columns={'index': '序号'}, inplace=True)
        df_chinese['序号'] = df_chinese.index + 1
        # 保存生成的数据到Excel文件
        df_chinese.to_excel(f'/home/ubuntu/graduation-project/Code/bias/output/{safe_model_name}-{condition_value}-chinese.xlsx', index=False) ## 生成数据的输出路径需要修改
        chinese_end_time = datetime.datetime.now()  # 记录生成英文数据结束的时间
        print(f"生成数据所需时间：{(chinese_end_time - chinese_start_time).seconds}秒")


# 合并英文和中文数据
df_english.reset_index(drop=True, inplace=True)
df_chinese.reset_index(drop=True, inplace=True)
df = pd.concat([df_english, df_chinese], axis=1)


# 添加索引，从0开始
df.reset_index(inplace=True)
df.rename(columns={'index': 'index'}, inplace=True)

# 按照指定列顺序排列
columns_order = ['index', 'English', 'English_a', 'Chinese', 'Chinese_a', 'model_name', 'disease']
df = df[columns_order]


df.to_excel(f'{safe_model_name}.xlsx', index=False)


# 记录程序结束运行的时间
endTime = datetime.datetime.now()

# 打印程序运行的时间
print("运行的时间是：%ss" % (endTime - startTime).seconds)