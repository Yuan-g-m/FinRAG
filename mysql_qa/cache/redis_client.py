# 该脚本用于: Redis缓存技术.

# 导包
import redis
import json
import os, sys

# todo 0.路径相关 -> 如下的设置是为了 <<将项目根目录加入系统路径>>, 绝大多数电脑不设置也可以, 如果你直接导包遇到问题了, 请执行如下设置.
# # 1.获取当前文件所在目录的绝对路径.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir: {current_dir}')
#
# # 2. 获取当前目录的上级目录.
# module_dir = os.path.dirname(current_dir)
# print(f'module_dir: {module_dir}')
#
# # 3. 获取项目根目录的绝对路径.
# project_root = os.path.dirname(module_dir)
# print(f'project_root: {project_root}')
#
# # 4. 将项目根目录加入系统路径, 确保跨目录导入自定义模块(例如: config, logger)...
# sys.path.insert(0, project_root)

# 5. 导入自定义配置 和 日志模块.
from base import Config, logger


# todo 1.定义Redis客户端类 -> 封装Redis连接, 数据存储, 数据获取, 答案查询等功能.
class RedisClient:
    # todo 1.1 初始化方法, 获取Redis连接, 初始化日志, 处理连接异常.
    def __init__(self):
        # 1. 获取日志记录器.
        self.logger = logger

        # 2. 建立Redis连接.
        try:
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,           # Redis数据库主机地址
                port=Config().REDIS_PORT,           # Redis数据库端口号
                password=Config().REDIS_PASSWORD,   # Redis数据库密码
                db=Config().REDIS_DB,               # Redis数据库编号
                decode_responses=True               # 自动将返回的字节数据转换为字符串(避免手动解码)
            )
        except redis.RedisError as e:
            self.logger.error(f'Redis连接异常: {e}')
            raise   # 声明抛出异常.

        # 3. 记录INFO日志
        self.logger.info('Redis连接成功')

    # todo 1.2 存储数据到Redis -> 将Python对象序列化为json字符串存储.
    def set_data(self, key, value):
        try:
            # 1. 存储数据.
            self.client.set(key, json.dumps(value, ensure_ascii=False)) # ensure_ascii: 保证中文等非ASCII字符正常显式.
            # 2. 记录INFO日志.
            self.logger.info(f'Redis存储数据成功: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')


    # todo 1.3 从Redis中获取数据 -> 反序列化为Python对象.
    def get_data(self, key):
        try:
            # 1. 获取数据.
            data = self.client.get(key)
            # 2. 处理结果 -> 反序列化为Python对象, 再返回.
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None     # 保持方法的返回值一致性.


    # todo 1.4 根据查询内容从Redis获取缓存的答案 -> 键格式固定为: "answer:{query}"
    def get_answer(self, query):
        try:
            # 1. 构建键名, 区分不同类型的缓存数据.
            answer = self.client.get(f"answer:{query}")
            # 2. 若存在缓存答案, 记录INFO日志, 并返回内容即可.
            if answer:
                self.logger.info(f'从Redis中获取答案成功: {query}')
                return json.loads(answer)
            # 3.走到这里, 说明缓存答案不存在, 需要从数据库中查询, 返回None
            return None
        except redis.RedisError as e:
            self.logger.error(f'从Redis中获取答案失败: {e}')
            return None


# todo 2.测试代码
if __name__ == '__main__':
    # 1. 实例化RedisClient客户端.
    redCli = RedisClient()

    # 2. 获取Redis中所有的键.
    print(redCli.client.keys("*"))

    # 3. 测试get_data()方法: 获取Redis中存储的答案.
    # print(redCli.get_data('user:1'))            # {'name': 'HangGe', 'age': 18}

    # 4. 测试get_answer方法: 获取Redis中存储的答案.  -> 这个才是后续我们要用的.
    # print(redCli.get_answer('熵的解释'))
