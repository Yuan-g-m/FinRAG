# 该脚本是: RAG_QA系统的主入口.

import os
import sys
from base.config import config as conf
from base.logger import logger
from rag_qa.core.document_processor import process_documents  # 导入处理文档的函数
from rag_qa.core.vector_store import VectorStore              # 向量存储和检索
from rag_qa.core.new_rag_system import RAGSystem              # RAG_QA系统的核心代码
from openai import OpenAI                                     # 使用 OpenAI 接口


# todo 1.定义主函数main -> 系统入口逻辑, 根据模式执行数据处理或交互式查询.
def main(query_mode=True, directory_path="data"):
    """
    函数功能: 系统主函数, 控制两种运行模式.
    :param query_mode:
        数据处理模式: query_model=False  解析指定目录的文档, 分割为文档块并存入向量数据.
        交互式查询模式: query_model=True 接收用户问题, 调用RAG系统生成答案并展示.
    :param directory_path: 数据处理模式下的文档根目录, 默认是: ./data, 需包含各个金融领域的子目录, 例如: 股票_data, 基金_data
    :return: 无
    """

    # 1. 初始化LLM客户端 -> 创建调用大模型的客户端,生成答案用.
    try:
        client = OpenAI(api_key=conf.LLM_API_KEY, base_url=conf.LLM_BASE_URL)
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败 (请检查 API Key 和 Base URL): {e}")
        # 如果客户端初始化失败，可能无法继续，取决于模式
        if query_mode: # 查询模式下必须要有 LLM
             print("错误：无法初始化语言模型客户端，无法进入查询模式。")
             return
        # 数据处理模式可能不需要 LLM，可以继续，但最好记录错误
        client = None # 标记客户端不可用


    # 2. 定义 LLM 调用函数 (仅在需要时定义和使用) -> 封装 DeepSeek 大模型调用逻辑, 供 RAG 系统使用.
    def call_llm(prompt):
        if not client: # 检查客户端是否可用
            logger.error("LLM 客户端未初始化，无法调用 LLM")
            return f"错误: LLM客户端不可用"
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是 FinRAG 金融智能助手，请基于金融知识严谨回答问题，涉及投资建议时提示仅供参考。"},
                    {"role": "user", "content": prompt},
                ]
                # 可以添加 temperature 等参数
            )
            if completion.choices and completion.choices[0].message:
                 return completion.choices[0].message.content
            else:
                 logger.error("LLM API 调用返回无效响应或空消息")
                 return "错误: LLM返回无效响应"
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            return f"错误: 调用LLM失败 - {e}"


    # 3. 初始化 VectorStore向量数据库 -> 连接Milvus服务, 用于存储和检索向量文档.
    try:
        vector_store = VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME,
        )
    except Exception as e:
        logger.error(f"初始化 VectorStore 失败 (请检查 Milvus 连接配置): {e}")
        print("错误：无法连接到向量数据库，程序无法继续。")
        return


    # 4. 根据模式执行不同操作 -> 数据处理模式: 解析文档并添加到向量数据库.
    if not query_mode:
        # --- 数据处理模式 ---
        logger.info("进入数据处理模式...")
        total_chunks_added = 0
        for source_dir in conf.VALID_SOURCES:
            dir_path = os.path.join(directory_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录: {dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP,
                    )
                    if chunks:
                        vector_store.add_documents(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")
                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
                except Exception as e:
                    logger.error(f"处理目录 {dir_path} 时出错: {e}")
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")
    else:
        # --- 交互式查询模式 ---
        if not client: # 再次检查 LLM 客户端是否必须且可用
            print("错误：查询模式需要语言模型客户端，但初始化失败。")
            return

        logger.info("进入交互式查询模式...")
        try:
            rag_system = RAGSystem(vector_store, call_llm)
        except Exception as e:
             logger.error(f"初始化 RAGSystem 失败: {e}")
             print("错误：无法初始化 RAG 系统，无法进入查询模式。")
             return

        valid_sources = conf.VALID_SOURCES
        print("\n欢迎使用 FinRAG 金融交互式查询系统！")
        print(f"支持的金融领域：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                break

            source_filter_input = input(f"请输入金融领域 ({'/'.join(valid_sources)}) (直接回车默认不过滤)：").strip()
            source_filter = None # 默认不过滤
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了金融领域过滤: {source_filter}")
                else:
                    logger.warning(
                        f"无效的金融领域 '{source_filter_input}'，将不过滤"
                    )
                    print(f"提示：输入的金融领域 '{source_filter_input}' 无效，将不过滤。")


            try:
                print("正在生成答案，请稍候...")
                answer = ''.join(
                    chunk if isinstance(chunk, str) else str(chunk)
                    for chunk in rag_system.generate_answer(query, source_filter=source_filter)
                )
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")


# todo 2. 程序的主入口: 解析命令参数, 决定运行模式 -> 处理处理 / 查询模式
if __name__ == "__main__":
    # 解释: 默认进入查询模式, 若要执行数据处理，可以修改调用方式，例如： main(query_mode=False) 或者通过命令行参数控制

    # 导入argparse模块 -> 命令行参数解析库.
    import argparse
    # 1. 创建参数解析器, 为解析器设置'描述信息', 当用户在终端执行.
    parser = argparse.ArgumentParser(description="FinRAG System Main Entry Point")
    # 2. 添加数据处理模式参数.    --data-processing -> 意思是: 进入数据处理模式.
    parser.add_argument('--data-processing', action='store_true', help='Run in data processing mode instead of query mode.')
    # 3. 添加数据目录参数. --data-dir -> 含义: 指定数据目录.
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to the data directory.')
    # 4. 解析命令行参数.
    args = parser.parse_args()
    # 5. 根据参数调用main函数 -> query_mode为False表示数据处理模式, 否则为查询模式.
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)
