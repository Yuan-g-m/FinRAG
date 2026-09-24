# 该脚本用于: RAG系统的核心逻辑 -> 实现RAG(检索增强生成)系统的核心逻辑, 整合查询分类(意图识别), 策略选择, 文档检索 和 答案生成.
# 注: 该脚本的 generate_answer 方法较之于旧版做了优化, 加入了 history历史对话.


# todo 1.导包
import sys, os                                  # 导入路径处理和系统配置.
import time                                     # 导入 time 模块，用于计算时间

from rag_qa.core.prompts import RAGPrompts                  # 导入RAG相关的提示模板 -> 定义大模型的输入和输出格式.
from rag_qa.core.query_classifier import QueryClassifier    # 导入查询分类器 -> 判断用户问题属于通用知识还是专业咨询,意图识别
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器 -> 专业咨询的情况下用哪种检索策略(例如:直接检索...)
from rag_qa.core.vector_store import VectorStore            # 导入向量数据库对象 -> 存储和检索文档向量.


# todo 2.配置项目路径 -> 确保程序能正确导入其它目录的模块.
# 1. 获取当前文件的目录的绝对路径.
current_dir = os.path.dirname(os.path.abspath(__file__))        # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core
# 2. 获取当前目录的上级目录.
rag_qa_path = os.path.dirname(current_dir)                      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa
# 3. 将rag_qa目录添加到Python的模块搜索路径.
sys.path.insert(0, rag_qa_path)
# 4. 获取项目根目录.
project_root = os.path.dirname(rag_qa_path)                     # D:\workspace\ai_30_bj\integrated_qa_system
# 5. 将项目根目录添加到Python的模块搜索路径.
sys.path.insert(0, project_root)

from base.config import Config                  # 配置文件
from base.logger import logger                  # 日志对象


# 分类器所需模型已由 QueryClassifier 统一加载，这里不再重复加载 BERT，避免启动时重复占用内存。


# 加载项目配置
conf = Config()


# todo 3. 定义RAGSystem类 -> 封装RAG系统的完整流程, 即: 查询分类 -> 策略选择 -> 文档检索 -> 答案生成
class RAGSystem:
    # todo 3.1 初始化方法 -> 创建RAG系统所需的核心组件 -> 向量库, 大模型, 分类器...
    def __init__(self, vector_store, llm):
        """
        函数作用: 初始化RAG系统
        :param vector_store: 向量数据库对象 -> 用于存储和检索文档, 提供相似性搜索功能.
        :param llm: 大语言模型调用函数 -> 接收提示文本, 返回模型生成的回答.
        """
        # 1. 保存向量数据库对象.
        self.vector_store = vector_store
        # 2. 保存大模型调用函数 -> 用于生成答案, 子查询, 假设答案等...
        self.llm = llm
        # 3. 加载RAG提示模板 -> 定义生成答案时的固定格式.
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 4. 初始化查询分类器 -> 用于判断用户查询是'通用知识'还是'专业咨询'
        # 4.1 拼接分类模型的路径.
        models_dir = os.getenv('FINRAG_MODEL_ROOT') or os.path.join(rag_qa_path, 'models')
        if not os.path.isdir(models_dir):
            models_dir = os.path.join(rag_qa_path, 'models')
        classifier_path = os.path.join(models_dir, 'bert_query_classifier_finance')
        # 4.2 创建查询分类器实例.
        self.query_classifier = QueryClassifier(classifier_path)
        # 5. 初始化策略选择器 -> 用于专业咨询的检索策略选择.
        self.strategy_selector = StrategySelector()


    def _collect_llm_response(self, prompt):
        """兼容流式与非流式 LLM 返回值，统一拼接为字符串。"""
        response = self.llm(prompt)
        if isinstance(response, str):
            return response
        try:
            return ''.join(response)
        except TypeError:
            return str(response)


    # todo 3.2 定义私有方法，使用假设文档进行检索（HyDE） -> 生成假设答案, 用假设答案检索相关文档.
    def _stream_llm_response(self, prompt):
        """逐字返回 LLM 输出，供前端实现打字机逐字输出。"""
        response = self.llm(prompt)
        if isinstance(response, str):
            for char in response:
                yield char
            return
        for chunk in response:
            if chunk:
                for char in chunk:
                    yield char

    def _retrieve_with_hyde(self, query, source_filter=None):
        """
        函数作用: 针对于抽象/开放性查询, 先生成假设性答案, 再用假设答案检索文档 -> 解决 抽象查询 直接检索关键词 匹配度低 的问题.
        :param query: 用户的原始查询文本(str).
        :param source_filter: 检索来源过滤条件(str或者None), 例如: 股票 表示只检索与股票相关的文档,  None表示不过滤.
        :return: list[Document], 每个Document对象的page_content属性为文档文本, 用于后续生成答案的上下文.
        """
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #  1.获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #  2.调用大语言模型生成假设答案
        try:
            hypo_answer = self._collect_llm_response(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            # 3.使用假设答案进行检索，并返回检索结果
            #  注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,                    # 检索关键词: 生成的假设答案.
                k=conf.RETRIEVAL_K,             # 使用 K 而非 M
                source_filter=source_filter     # 应用来源过滤条件.
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []


    # todo 3.3 定义私有方法，使用子查询进行检索 -> 将复杂查询拆分为子查询, 分别检索后合并结果.
    def _retrieve_with_subqueries(self, query, source_filter=None):
        """
        函数作用: 针对于多实体/多方面的复杂查询, 拆分为多个简单子查询, 分别检索后合并去重 -> 解决多维度查询检索不全面的问题.
        :param query: 用户的原始查询文本.
        :param source_filter: 检索来源过滤条件(str或者None), 例如: 股票 表示只检索与股票相关的文档,  None表示不过滤.
        :return: list[Document]对象.
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self._collect_llm_response(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            #  初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K, source_filter=source_filter  # 应用来源过滤条件  # 使用 K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: Config.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []


    #  todo 3.4 定义私有方法，使用回溯问题进行检索 -> 将复杂查询简化为基础问题后再检索.
    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self._collect_llm_response(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []


    # todo 3.5 核心方法: 根据检索策略检索文档, 用 合并/筛选最终上下文对象.
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):  # 新增 strategy 参数
        """
        函数作用: 统一入口: 根据指定策略(或自动选择策略)调用对应检索方法, 筛选最终用于生成答案的上下文文档.
        :param query: 用户的原始查询文本 -> str, 传递给策略选择器 和 检索方法.
        :param source_filter: 检索来源过滤条件
        :param strategy: 指定检索策略, 可选: 直接检索, 回溯问题检索...
        :return: list[Document]
        """
        # 1. 如果未指定检索策略，自动选择检索策略.
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 2. 根据策略调用对应的检索方法. -> 获取候选文档列表.
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter=source_filter)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        # 3. 选择最终上下文文档, 截取前conf.CANDIDATE_M个文档(控制上下文长度, 避免LLM输入超限)
        logger.info(f"策略 {strategy} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]      # 截取前 conf.CANDIDATE_M 个文档
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        # 4. 返回最终的上下文文档
        return final_context_docs


    # 优化2
    # todo 3.6 定义方法，生成答案
    def generate_answer(self, query, source_filter=None, history=None):
        """
        函数作用: 根据查询类型选择直接用LLM或者执行RAG流程, 支持流式输出.
        :param query: 用户录入的原始查询文本(字符串形式)
        :param source_filter: 检索来源过滤条件, 仅对'专业咨询'生效.
        :param history: 对话历史(列表, 可选), 包含字典: {'question': 问题, 'answer': 答案}
        :return: 生成器, 逐段返回答案文本(字符串)
        """
        # 记录查询开始时间, 用于计算处理耗时
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 金融领域过滤: {source_filter}")

        # 1. 验证并处理对话历史, 确保格式正确, 只保留最近5轮
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮 -> 即: 限制历史长度, 避免上下文过程.
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 2. 构造历史上下文, 格式化历史记录为字符串.
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 3. 判断查询类型 -> 通过分类器区分通用知识 和 专业咨询.
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        #  4. 如果查询属于“通用知识”类别，则直接使用 LLM 回答, 专业咨询则进行 RAG 检索.
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", history=history_context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )  # 不使用上下文
            try:
                for chunk in self._stream_llm_response(prompt_input):
                    yield chunk
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                yield f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        # 5. 选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        # 6. 检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )  # 传递 strategy

        #  7. 准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #  8. 构造 Prompt，调用大语言模型生成答案
        # 构造提示
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        # 9. 调用大模型生成流式答案.
        try:
            for chunk in self._stream_llm_response(prompt_input):
                yield chunk
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            yield f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        # 记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")



# todo 4.测试代码.
if __name__ == '__main__':
    # 1. 实例化向量数据库.
    vector_store = VectorStore()
    # 2. 定义大语言模型调用函数.
    llm = StrategySelector().call_llm

    # 3. 创建RAGSystem核心类的实例 -> 传入: 向量数据库实例, 大语言模型调用函数.
    rag_system = RAGSystem(vector_store, llm)

    # 4. 测试生成答案: 查询'股票开户需要哪些材料', 过滤条件'股票' -> 只检索股票相关文档.
    # answer = rag_system.generate_answer('股票开户需要哪些材料', source_filter='股票')
    # answer = rag_system.generate_answer('什么是基金定投', source_filter='基金')
    # answer = rag_system.generate_answer('债券收益率受哪些因素影响', source_filter='债券')
    answer = rag_system.generate_answer('股票开户需要哪些材料', source_filter='股票')
    # 5. 打印模型生成的答案 -> 实际部署时可改为: 返回给前端或者存储.
    print(answer)

