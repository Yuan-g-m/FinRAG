# 该脚本用于: 金融FAQ的MySQL检索系统.

# 导包
from db.mysql_client import MySQLClient         # 导入 MySQL 客户端
from cache.redis_client import RedisClient      # 导入 Redis 客户端
from retrieval.bm25_search import BM25Search    # 导入 BM25 搜索
from base import logger     # 导入日志
import time                 # 导入时间库

# todo 1.定义MySQL问答系统类 -> 整合MySQL, Redis, BM25搜索功能, 提供统一的查询接口.
class MySQLQASystem:
    # todo 1.1 初始化操作.
    def __init__(self):
        # 1. 日志记录器.
        self.logger = logger
        # 2. MySQL客户端
        self.mysql_client = MySQLClient()
        # 3. Redis客户端
        self.redis_client = RedisClient()
        # 4. BM25搜索模型
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)


    # todo 1.2 处理用户查询的核心方法 -> 调用BM25搜索, 记录日志与耗时, 并返回答案.
    def query(self, query):
        """
        处理用户查询, 通过BM25搜索从MySQL获取答案, 返回结果并记录日志.
        :param query: 用户输入的查询文本.
        :return: 匹配到的答案 -> 未找到就返回默认提示
        """
        # 1. 记录: 查询开始时间.
        start_time = time.time()
        # 2. 记录INFO日志: 标记当前正在处理的查询内容.
        self.logger.info(f'处理查询: {query}')
        # 3. 调用BM25搜索模型, 获取答案.
        # _表示 忽略是否需要进一步处理, 这里我们只获取答案.
        answer, _ = self.bm25_search.search(query)
        # 4. 分支处理搜索结果: 找到答案就记录, 未找到就返回默认提示.
        if answer:
            self.logger.info(f'MySQL答案: {answer}')
        else:
            self.logger.info('金融FAQ中未找到答案, 需要调用RAG系统')
            answer = '金融FAQ中未找到答案'        # 设置默认提示性答案

        # 5. 计算查询处理耗时, 并记录耗时日志.
        process_time = time.time() - start_time
        self.logger.info(f'查询处理耗时: {process_time:.3f}秒')
        # 6. 返回答案.
        return answer


# todo 2. 主函数 -> 创建问答系统实例, 提供交互式命令界面.
def main():
    # 1. 创建MySQL问答系统实例.
    mysql_qa = MySQLQASystem()
    try:
        # 2. 打印欢迎信息 和 使用说明.
        print('\n欢迎使用金融FAQ检索系统:')
        print('输入查询进行回答, 输入 exit 退出!!!')

        # 3. 循环获取用户输出, 处理查询 或者 退出.
        while True:
            # 3.1 获取用户输入并移除首尾空格.
            query = input('\n请输入查询: ').strip()
            # 3.2 判断是否是exit, 如果是, 就退出.
            if query.lower() == 'exit':
                logger.info('退出MySQL系统')
                print('再见!!!')
                break
            # 3.3 调用问答系统处理查询, 获取答案.
            answer = mysql_qa.query(query)
            print(f'答案: {answer}')

    except Exception as e:
        # 捕获异常并记录错误日志.
        logger.error(f'系统错误: {e}')
    finally:
        # 无论是否发生问题, 都会走这里, 我们关闭: MySQL连接.
        mysql_qa.mysql_client.close()


# todo 3. 测试代码.
if __name__ == '__main__':
    main()
