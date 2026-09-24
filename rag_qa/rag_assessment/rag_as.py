# 该脚本用于: 通过RAGAS框架 对RAG系统做评分, 主要考察: 检索部分 和 生成部分.
# 一共有4个指标, 分别是:
#   生成部分: 忠实度(忠诚度), 答案相关性.
#   检索部分: 上下文召回率(召回), 上下文精确率(准确).

# todo 1.导包
from ragas import evaluate          # 导入ragas库的evaluate函数，用于执行RAG评估
from ragas.metrics import (         # 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
    faithfulness,           # 忠实度
    answer_relevancy,       # 答案相关性
    context_precision,      # 上下文精确度
    context_recall          # 上下文召回率
)
from datasets import Dataset        # 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
import json                 # 导入json库，用于加载JSON格式的评估数据集
import os                   # 导入路径处理模块

# 导入OpenAI相关模型(兼容ragas接口) -> 需要你要翻墙, 且注册ChatGPT的API Key -> 课件中用的是这个版本.
# 导入langchain_openai的聊天模型 和 嵌入模型，用于评估时的语义计算和推理
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 金融版使用与主系统一致的 DeepSeek，并使用本地 BGE 模型计算语义相似度.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from base.config import Config


# todo 2. 加载评估数据集 -> 读取预先 生成的包含问题, 答案, 上下文, 真实答案的Json文件.
eval_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_evaluate_data.json')
with open(eval_file, 'r', encoding='utf-8') as f:
    data = json.load(f)                       # data的数据格式: [{'question': '问题', 'answer': '答案', 'context': '上下文', 'ground_truth': '真实答案'}, ...]
print(f'加载的评估样本数量: {len(data)}')        # 30


# todo 3. 转换数据格式 -> 将原始数据转成RAGAS框架所需的数据格式(Dataset)
# 1. 构建符合RAGAS数据格式的字典, key:固定的字段名, value:对应的数据列表
eval_data = {
    'question': [item['question'] for item in data],            # 问题列表, 格式为: ['问题1', '问题2', '问题3'...]
    'answer': [item['answer'] for item in data],                # (大模型生成的)答案列表, 格式为: ['答案1', '答案2', '答案3'...]
    'contexts': [item['context'] for item in data],             # 上下文列表, 格式为: ['上下文1', '上下文2', '上下文3'...]
    'ground_truth': [item['ground_truth'] for item in data]     # 真实答案列表, 格式为: ['真实答案1', '真实答案2', '真实答案3'...]
}
# 2. 将字典转换为Dataset对象.
dataset = Dataset.from_dict(eval_data)
# 3. 打印转换后的数据集.
print(f'转换后的数据集: {dataset}')


# todo 4.配置评估模型 -> 选择用于评估的LLM(大语言模型) 和 嵌入模型(用于计算相似度)
# 版本1: 使用OpenAI模型 -> 需要API Key密钥, 适用于需要高精度评估的场景, 可能需要科学上网.
# 初始化ChatOpenAI模型，指定使用gpt-4模型，并设置OpenAI API密钥
# llm = ChatOpenAI(model="gpt-4", openai_api_key="your_openai_api_key")
# 初始化OpenAI嵌入模型，用于计算语义相似度，设置API密钥
# embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")

# 金融版评估配置: LLM 使用 DeepSeek，嵌入模型使用项目内 BGE-M3 本地权重.
conf = Config()
llm = ChatOpenAI(
    model=conf.LLM_MODEL,
    api_key=conf.LLM_API_KEY,
    base_url=conf.LLM_BASE_URL,
)
embedding_model_dir = os.getenv('FINRAG_MODEL_ROOT') or conf.MODELS_DIR
embeddings = HuggingFaceEmbeddings(
    model_name=os.path.join(embedding_model_dir, 'bge-m3'),
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)


# todo 5. 执行评估 -> 使用RAGAS库的evaluate函数执行评估, 传入数据集、LLM和嵌入模型, 计算各项评估指标得分.
result = evaluate(
    dataset=dataset,            # 转换好的评估数据集.
    metrics=[
        faithfulness,           # 忠实度
        answer_relevancy,       # 答案相关性
        context_precision,      # 上下文精确度
        context_recall          # 上下文召回率
    ],
    llm=llm,                    # 传入配置好的LLM模型
    embeddings=embeddings       # 传入配置好的嵌入模型
)

# todo 6. 打印评估结果 -> 输出评估结果, 包括各项指标得分和评价.
print(f'RAGAS评估结果: {result}')


















