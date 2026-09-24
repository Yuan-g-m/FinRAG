# 该文件用于帮助 mysql_qa这个Python包, 管理包内的脚本, 方便其它包的调用.

import os, sys          # os: 操作系统模块, sys: 主要管理系统配置等.


# 1. 记录当前目录.
current_dir = os.path.abspath(__file__)
# print(f'current_dir: {current_dir}')        # D:\workspace\ai_30_bj\integrated_qa_system\mysql_qa\__init__.py

# 2. 记录 mysql_qa目录 -> MySQL版的问答系统的目录.
mysql_qa_path = os.path.dirname(current_dir)
# print(f'mysql_qa_path: {mysql_qa_path}')    # D:\workspace\ai_30_bj\integrated_qa_system\mysql_qa

# 3. 添加 mysql_qa_path路径到 系统路径变量中.
sys.path.insert(0, mysql_qa_path)


# 4. 具体的导包动作
from db.mysql_client import MySQLClient         # MySQL的客户端
from cache.redis_client import RedisClient      # Redis的客户端
from retrieval.bm25_search import BM25Search    # BM25 -> 相似度检索