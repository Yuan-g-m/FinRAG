# 该文件用于帮助 rag_qa这个Python包, 管理包内的脚本, 方便其它包的调用.

import os, sys

# 1. 记录当前目录.
current_dir = os.path.abspath(__file__)
# print(f'current_dir: {current_dir}')        # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\__init__.py

# 2. 记录rag_qa目录 -> Milvus版的问答系统的目录.
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path: {rag_qa_path}')    # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa

# 3. 添加 rag_qa_path路径到 系统路径变量中.
sys.path.insert(0, rag_qa_path)


# 4. 具体的导包动作
from core.prompts import RAGPrompts         # RAG系统的提示语
from core.vector_store import VectorStore   # 向量存储和检索
# 旧版 rag_system 已迁移至 legacy_backup，当前金融版统一使用 new_rag_system.
from core.new_rag_system import RAGSystem   # RAG系统的核心代码 -> 添加历史记录 和 流式输出时选择.
