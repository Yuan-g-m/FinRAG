# 该脚本用于: 配置文件管理.

# 导包
import configparser                 # 导入配置文件解析库, 解析INI格式的配置文件.
import os                           # 导入路径操作模块
import json                         # 导入 JSON 解析库，安全解析列表配置
from dotenv import load_dotenv      # 从 .env 文件加载敏感配置


# todo 1.配置文件路径计算.
# 1. 获取当前文件的绝对路径.
current_file_path = os.path.abspath(__file__)
# print(f'current_file_path: {current_file_path}')    # D:\workspace\ai_30_bj\integrated_qa_system\base\config.py

# 2. 获取当前文件所在目录的绝对路径.
current_dir_path = os.path.dirname(current_file_path)
# print(f'current_dir_path: {current_dir_path}')      # D:\workspace\ai_30_bj\integrated_qa_system\base

# 3. 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir_path)
# print(f'project_root: {project_root}')                # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 拼接配置文件(config.ini)的完整路径.
config_file_path = os.path.join(project_root, 'config.ini')
# print(f'config_file_path: {config_file_path}')          # D:\workspace\ai_30_bj\integrated_qa_system\config.ini


# todo 2.配置解析类 -> 封装配置文件读取逻辑, 提供统一的配置参数访问接口.
class Config:
    # todo 2.1 初始化方法: 读取配置文件并解析各服务的配置参数.
    def __init__(self, config_file=config_file_path):
        # # 1. 创建配置文件解析器实例.
        # self.config = configparser.ConfigParser()
        # # 2. 读取配置文件 -> 根据传入的路径加载ini文件内容.
        # self.config.read(config_file, encoding='utf-8')
        #
        # # 3. 解析并存储各服务配置参数.
        # # 3.1 解析MySQL数据库配置.
        # self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        # self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='finance_kg')
        #
        # # 3.2 解析Redis数据库配置.
        # self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # self.REDIS_PORT = self.config.get('redis', 'port', fallback=6379)
        # self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        # self.REDIS_DB = self.config.get('redis', 'db', fallback=0)
        #
        # # 3.3 解析日志配置.
        # # 日志文件路径: 从 [logger]节点的log_file键获取, 默认值为: logs/app.log
        # self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')


        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        # 优先加载项目根目录的 .env，敏感信息不应直接写在 config.ini 中.
        load_dotenv(os.path.join(self.PROJECT_ROOT, '.env'))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.FINANCE_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/finance_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        self.config.read(config_file, encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE',
                                        self.config.get('mysql', 'database', fallback='finance_kg'))
        # MySQL 端口
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.get('mysql', 'port', fallback=3306)))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='finance_rag'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name',
                                                                fallback='finrag_final'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='deepseek-chat'))
        # LLM API 密钥与 OpenAI 兼容接口地址.
        self.LLM_API_KEY = os.getenv('LLM_API_KEY', self.config.get('llm', 'api_key', fallback=''))
        self.LLM_BASE_URL = os.getenv('LLM_BASE_URL', self.config.get('llm', 'base_url',
                                                                      fallback='https://api.deepseek.com'))
        # 历史配置项名保留为兼容别名，旧代码仍可读取.
        self.DASHSCOPE_API_KEY = self.LLM_API_KEY
        self.DASHSCOPE_BASE_URL = self.LLM_BASE_URL

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        # MySQL FAQ BM25 命中阈值
        self.FAQ_THRESHOLD = self.config.getfloat('retrieval', 'faq_threshold', fallback=0.15)

        # 应用配置
        # 有效来源列表
        self.APP_NAME = self.config.get('app', 'app_name', fallback='FinRAG智能金融问答系统')
        self.APP_TAGLINE = self.config.get('app', 'tagline', fallback='基于RAG的金融知识智能问答助手')
        self.VALID_SOURCES = json.loads(
            self.config.get('app', 'valid_sources', fallback='["股票", "基金", "债券", "银行", "保险"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='400-800-8899')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

# 扩展: 创建Config类的实例.
config = Config()

# todo 3. 主程序入口 -> 测试代码.
if __name__ == '__main__':
    # 1. 手动指定配置文件路径 -> 用于测试, 实际使用可忽略.
    # config_file = 'D:/workspace/ai_30_bj/integrated_qa_system/config.ini'      # 绝对路径.
    config_file = '../config.ini'      # 相对路径.

    # 2. 实例化Config类 -> 加载并解析配置文件.
    conf = Config(config_file)
    # 3. 测试: 获取各服务配置参数.
    print(f'mysql: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, {conf.MYSQL_PASSWORD}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, {conf.REDIS_PASSWORD}, {conf.REDIS_DB}')
    print(f'log: {conf.LOG_FILE}')
