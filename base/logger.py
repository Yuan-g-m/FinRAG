# 自定义的 日志工具类.

# 导包
import logging
import os
from config import Config


# todo 1.计算日志文件的最终保存路径.
# 1. 获取当前脚本文件的绝对路径.
module_path = os.path.abspath(__file__)
# print(f'module_path: {module_path}')        # D:\workspace\ai_30_bj\integrated_qa_system\base\logger.py

# 2. 获取当前脚本所在目录的路径.
base_path = os.path.dirname(module_path)
# print(f'base_path: {base_path}')              # D:\workspace\ai_30_bj\integrated_qa_system\base

# 3. 获取当前脚本所在目录的父目录的路径.
current_path = os.path.dirname(base_path)
# print(f'current_path: {current_path}')          # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 拼接日志文件的完整路径.   上级目录 + logs文件夹 + app.log文件.
# log_file = os.path.join(current_path, 'logs/app.log')
log_file = os.path.join(current_path, Config().LOG_FILE)       # 效果同上.
# print(f'log_file: {log_file}')                  # D:\workspace\ai_30_bj\integrated_qa_system\logs/app.log


# todo 2. 封装日志配置函数 -> 可重复使用的双输出日志.
def setup_logger(log_file=log_file):
    """
    函数功能: 创建并返回1个日志记录器, 支持日志同时输出到控制台和文件, 且避免重复添加处理器.
    :param log_file: 日志文件的保存路径.
    :return: 日志记录器
    """
    # 1. 确保日志目录存在.
    # 参1: 获取日志的所在目录.        参2: 如果目录已存在,不报错, 如果不存在,则创建.
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 2. 创建日志记录器对象 -> 统一管理日志处理器, 并设置日志级别.
    logger = logging.getLogger('FinRAG')
    logger.setLevel(logging.INFO)

    # 7. 为日志记录器添加处理器, 核心: 判断处理器是否存在, 避免多次调用函数导致重复添加.
    if not logger.handlers:

        # 3. 创建控制台处理器 -> 用于将日志输出到控制台.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 4. 创建文件处理器 -> 用于将日志输出到文件.
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 5. 创建日志输出格式, 即: 时间戳 - 日志器名称 - 日志级别 - 日志内容
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 6. 为两个处理器分别绑定日志格式.
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 7. 添加处理器到日志记录器中.
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 8. 返回日志记录器.
    return logger


# todo 3. 初始化日志器 -> 项目启动时自动执行, 其它模块导入此logger日志器即可直接使用.
logger = setup_logger()
