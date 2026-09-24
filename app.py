# 该脚本用于: 基于FastAPI搭建智能问答系统Web服务, 提供Http接口, WebSocket流式问答, 静态资源服务, 会话历史管理, 跨域配置
#           健康探针等完整Web能力, 底层对接集成 RAG + MySQL的问答核心系统.

# todo 0. 导包
# 导入 FastAPI 相关模块，用于构建 API 和 WebSocket  -> 构建Http接口, WebSocket长连接, 异常抛出, 请求参数依赖注入...
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入 FastAPI 响应类型，用于流式响应和文件服务 -> 流式字节输出, 返回本地静态文件.
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
# 导入 CORS 中间件，支持跨域请求 -> 解决前端浏览器访问后端接口产生的CORS跨域拦截问题.
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入 WebSocket 断开异常
from starlette.websockets import WebSocketDisconnect
# 导入系统操作模块，用于文件目录管理
import os
# 导入 Pydantic 模型，用于请求验证
from pydantic import BaseModel, Field
# 导入异步事件循环模块
import asyncio
# 导入 JSON 处理模块
import json
# 导入 UUID 模块，生成唯一会话 ID
import uuid
# 导入类型注解模块
from typing import Optional, List, Dict, Any
# 导入时间模块，记录处理时间
import time
# 导入正则表达式模块，用于匹配日常问候
import re
# 导入优化后的问答系统
from new_main import IntegratedQASystem
from base.logger import logger



# todo 1. FastAPI应用实例初始化 和 全局中间件配置.
# 1. 创建FastAPI应用实例, 设置接口文档标题, 接口整体业务描述, 自动生成Swagger在线文档.
app = FastAPI(title="FinRAG智能金融问答系统API", description="集成MySQL FAQ和Milvus RAG的金融领域智能问答系统")

# 2. 全局注入CORS跨域中间件, 配置前后端跨域访问规则.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有前端域名来源访问, 生产环境需配置具体前端域名. 避免安全风险.
    allow_credentials=False,    # 与 allow_origins=["*"] 同时使用时不能为 True, 否则浏览器会拒绝跨域响应.
    allow_methods=["*"],        # 允许所有 HTTP 方法 -> 放行全部Http请求: GET/POST/PUT/DELETE....
    allow_headers=["*"],        # 允许客户端所有自定义请求头, 如: 鉴权Token...
)

# 3. 创建静态资源存储目录 static,    exist_ok=True目录已存在是不抛出异常.
os.makedirs('static', exist_ok=True)

# 4. 全局实例化问答核心系统.
qa_system = IntegratedQASystem()


def success_response(data=None, errno=0, errmsg="ok"):
    """统一成功响应格式."""
    return {"errno": errno, "errmsg": errmsg, "log_id": str(uuid.uuid4()), "data": data}


def error_response(errno, errmsg, status_code=500):
    """统一失败响应格式."""
    return JSONResponse(
        status_code=status_code,
        content={"errno": errno, "errmsg": errmsg, "log_id": str(uuid.uuid4()), "data": None},
    )


# todo 2. 全局固定问候语 正则规则配置.
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是 FinRAG 金融智能助手，专注于解答金融知识问题，很高兴为您服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是 FinRAG，您的智能金融问答助手，可为您解答股票、基金、债券、银行、保险等领域的问题！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是 FinRAG 金融智能助手，随时为您解答金融领域问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为您解答金融知识相关问题！有什么可以帮您的？"
    }
]


# todo 3. Pydantic请求/响应数据模型定义
# todo 3.1 非流式查询接口入参结构体 -> 接收Post请求Json参数.
# 定义查询请求模型
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)  # 查询内容，必填，限制长度
    source_filter: Optional[str] = None  # 金融领域过滤，可选
    session_id: Optional[str] = None     # 会话 ID，可选

# todo 3.2 非流式查询接口标准返回结构体 -> 统一接口出参格式.
# 定义查询响应模型
class QueryResponse(BaseModel):
    answer: str                 # 答案内容 -> 问答最终回复内容.
    is_streaming: bool          # 是否流式响应 -> 标签当前回复是否需要流式输出, True代表需切换WebSocket接口
    session_id: str             # 会话 ID -> 用于上下文历史关联
    processing_time: float      # 处理时间 -> 本次问答完整处理耗时, 单位: 秒.


# todo 4. 静态资源挂载与页面路由配置.
# 1. 将本地static目录挂载至服务器根路径.
app.mount('/static', StaticFiles(directory='static'), name='static')

# todo根路径 GET接口, 访问首页HTML页面.
@app.get("/")               # 这个/的意思是: http://ip地址:端口号/
async def read_root():
    # 返回static目录下的首页html文件 -> 实现域名根路径直接打开前端页面.
    return FileResponse('static/index.html')


# todo 5. Http RestFul业务接口定义.
# 创建新会话接口
# TODO 类似于java中的@PostMapping
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return success_response({"session_id": session_id})


# todo  5.2 查询历史消息接口
# TODO restful api 规范 。比如某厂规范：internal-api.xxxx.com/{业务线: driver/passanger}/{二级:运营、打车}/xxxx
# TODO 对外的api(openapi) :  /api/v1/fin-rag/history/{session_id}


@app.get("/api/sessions")
async def list_sessions():
    try:
        sessions = qa_system.list_sessions()
        return success_response({"sessions": sessions})
    except Exception as e:
        logger.exception("获取会话列表失败")
        return error_response(50003, f"获取会话列表失败: {str(e)}", status_code=500)


@app.get("/api/archived")
async def list_archived_sessions():
    """返回已归档会话列表，供前端回收站展示."""
    try:
        sessions = qa_system.list_sessions(archived=True)
        return success_response({"sessions": sessions})
    except Exception as e:
        logger.exception("获取归档会话列表失败")
        return error_response(50004, f"获取归档会话列表失败: {str(e)}", status_code=500)


@app.post("/api/archive/{session_id}")
async def archive_session(session_id: str):
    """归档指定会话."""
    success = qa_system.set_session_archived(session_id, True)
    if success:
        return success_response({"status": "success", "message": "会话已归档"})
    return error_response(50005, "归档会话失败", status_code=500)


@app.post("/api/unarchive/{session_id}")
async def unarchive_session(session_id: str):
    """从回收站恢复指定会话."""
    success = qa_system.set_session_archived(session_id, False)
    if success:
        return success_response({"status": "success", "message": "会话已恢复"})
    return error_response(50006, "恢复会话失败", status_code=500)


@app.delete("/api/archive/{session_id}")
async def permanently_delete_session(session_id: str):
    """从回收站永久删除指定会话."""
    success = qa_system.delete_session(session_id)
    if success:
        return success_response({"status": "success", "message": "会话已永久删除"})
    return error_response(50007, "永久删除会话失败", status_code=500)
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = qa_system.get_session_history(session_id)
        return success_response({"session_id": session_id, "history": history})
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        logger.exception("获取历史记录失败")
        return error_response(50001, f"获取历史记录失败: {str(e)}", status_code=500)


# @ todo 5.3 清除历史消息接口 -> 根据会话ID清空历史对话接口.
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    # 清除指定会话的历史记录
    success = qa_system.clear_session_history(session_id)
    if success:
        return success_response({"status": "success", "message": "历史记录已清除"})
    else:
        # 抛出 HTTP 异常
        return error_response(50002, "清除历史记录失败", status_code=500)


# todo 6. 公共工具函数 -> 检测用户输入是否为问候短句, 匹配则返回预设回顾 -: 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格
    for pattern_info in GREETING_PATTERNS:
        # 使用正则匹配，忽略大小写, 兼容中英文大小写混用场景.
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 返回匹配的回复
    return None  # 无匹配返回 None


# todo 7. 同步非主流式问答POST接口, 一次性返回完整回复答案, 不支持分段流式输出.
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间
    if request.source_filter and request.source_filter not in qa_system.config.VALID_SOURCES:
        return error_response(40001, f"无效的金融领域: {request.source_filter}", status_code=400)
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        if session_id:
            try:
                qa_system.update_session_history(session_id, request.query, greeting_response)
            except Exception as e:
                logger.error(f"保存问候历史失败: {e}")        # 返回问候回复
        return success_response({
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        })
    # 执行 BM25 搜索
    answer, need_rag = qa_system.bm25_search.search(request.query)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        return success_response({
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        })
    if session_id:
        try:
            qa_system.update_session_history(session_id, request.query, answer)
        except Exception as e:
            logger.error(f"保存问答历史失败: {e}")    # 返回 MySQL 答案
    return success_response({
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    })


# 流式查询 WebSocket 接口
# WebSocket长连接流式问答接口, 逐Token分段推送答案, 实现: 打字机流失效果.
# todo 8. 这里用的还是fastapi框架，但是这里接受的对象不会转成我们定义的结构体。 使用的是WebSocket
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            loop = asyncio.get_running_loop()  # 用于把同步 LLM 调用投递到线程池, 避免阻塞事件循环
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "error",
                        "errno": 40001,
                        "errmsg": f"无效的金融领域: {source_filter}",
                    })
                break
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
                # 向前端推送思考过程，提升逐字输出前的等待体验
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "thinking",
                        "message": "正在解析问题意图...",
                        "step": 1
                    })
                    await asyncio.sleep(0.15)
                    await websocket.send_json({
                        "type": "thinking",
                        "message": "正在检索金融知识库...",
                        "step": 2
                    })            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                try:
                    # 问候语同样写入历史, 保持与 HTTP /api/query 行为一致
                    await loop.run_in_executor(None, qa_system.update_session_history, session_id, query, greeting_response)
                except Exception as e:
                    logger.error(f"保存问候历史失败: {e}")
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break
            # 调用问答系统，流式处理查询.
            # 将同步生成器放到线程池逐步 next(), 避免同步 LLM 流式调用阻塞 asyncio 事件循环, 支持多会话并发.
            collected_answer = ""
            answer_gen = qa_system.query(query, source_filter=source_filter, session_id=session_id)
            while True:
                try:
                    token, is_complete = await loop.run_in_executor(None, next, answer_gen)
                except StopIteration:
                    break
                collected_answer += token  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        logger.exception(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "errno": 50003,
                "errmsg": f"服务处理异常: {str(e)}",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            logger.warning(f"Error closing WebSocket: {str(e)}")

# todo 9. 健康检查接口 -> 运维监控接口(了解即可)
# TODO k8s: 管理和部署容器， 基于docker(其他虚拟化框架)一个管理工具
# TODO 探针机制(去调用接口知道当前的服务它的状态)： 就绪探针(探测是不是启动)、 存活探针(是不是正常服务)
@app.get("/health")
async def health_check():
    return success_response({"status": "healthy", "service": qa_system.config.APP_NAME})


@app.get("/live")
async def live_check():
    return success_response({"status": "alive"})


@app.get("/ready")
async def ready_check():
    try:
        ready = bool(
            qa_system.mysql_client.connection
            and qa_system.vector_store.client.has_collection(qa_system.vector_store.collection_name)
        )
    except Exception as e:
        # 依赖中间件异常时探针自身不应抛 500, 返回 not_ready 供上层判断.
        logger.warning(f"就绪检查失败: {e}")
        ready = False
    return success_response({"status": "ready" if ready else "not_ready"})

# 获取有效金融领域接口
@app.get("/api/sources")
async def get_sources():
    return success_response({"sources": qa_system.config.VALID_SOURCES})


# 主程序入口
if __name__ == "__main__":
    # springboot = springcore + spirngmvc + tomcat
    # fastapi = springmvc (url -> 方法调用)
    # uvicorn = tomcat (服务容器，负责处理多线程、高并发等)

    import uvicorn      # uvicorn库 -> 异步web服务容器, 用于启动FastAPI应用.
    import os

    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8080
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    # 运行 FastAPI 应用，监听指定的主机和端口
    # reload=False 关闭热重载, 生产环境禁用, 避免性能损耗.
    uvicorn.run("app:app", host=host, port=port, reload=False)
