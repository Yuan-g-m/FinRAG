# FinRAG 金融领域 RAG 问答系统

基于 `MySQL FAQ + Redis 缓存 + Milvus RAG + DeepSeek LLM` 的金融领域智能问答系统，覆盖股票、基金、债券、银行、保险五个主题。前端提供多会话并行对话、流式回答与思考过程展示、会话归档/回收站、明暗主题切换等交互能力。

## 功能特性

- 金融领域意图分类：通用知识直答，专业咨询走 RAG 检索
- 双路问答：MySQL FAQ 快速匹配 + Milvus 向量检索重排
- Redis 缓存与多会话管理
- WebSocket 流式输出（逐字/分块），前端实时渲染思考过程
- 会话归档、回收站恢复与彻底删除
- DeepSeek 兼容接口，可通过 `LLM_BASE_URL` 切换其他兼容服务

## 快速开始

### 1. 安装依赖

```bash
# 推荐 Python 3.10
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填写 MySQL、Redis、Milvus 连接信息与 `LLM_API_KEY`。`.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。

### 3. 启动中间件

```bash
docker compose up -d
```

`docker-compose.yml` 启动 MySQL（`3308`）、Redis（`6379`）、Milvus standalone（`19530`）及其 Etcd / MinIO / Attu 依赖。默认密码为示例值，生产环境请修改。

### 4. 下载模型

模型文件体积较大且涉及本地 junction，`rag_qa/models/` 默认被 `.gitignore` 忽略，需单独下载：

```bash
python scripts/download_models.py
```

模型清单与自定义分类器的说明见下文「模型下载」。

### 5. 导入数据

```bash
python -m rag_qa.main --data-processing --data-dir rag_qa/data
```

### 6. 启动 Web 服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

访问 `http://127.0.0.1:8080/`。

## 环境变量

所有配置优先从 `.env` 读取，未设置时回退到 `config.ini`。`config.ini` 中的密码仅为 `change_me` 占位，实际部署必须通过 `.env` 注入真实值。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MYSQL_HOST` / `MYSQL_PORT` | `localhost` / `3306` | MySQL 地址与端口 |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `root` / 空 | MySQL 账号密码 |
| `MYSQL_DATABASE` | `finance_kg` | FAQ 数据库名 |
| `MYSQL_ROOT_PASSWORD` | `change_me` | 供 `docker-compose.yml` 初始化 MySQL root 密码 |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Redis 地址与端口 |
| `REDIS_PASSWORD` / `REDIS_DB` | 空 / `0` | Redis 密码与库编号 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus 地址与端口 |
| `MILVUS_DATABASE_NAME` | `finance_rag` | Milvus 数据库名 |
| `MILVUS_COLLECTION_NAME` | `finrag_final` | Milvus 集合名 |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `change_me` | 供 `docker-compose.yml` 初始化 MinIO |
| `LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `LLM_API_KEY` | 空 | 必需，LLM API 密钥 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `FINRAG_MODEL_ROOT` | `rag_qa/models` | 模型存放根目录（可指向外部目录） |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Web 服务监听地址与端口 |

## 端口映射

| 服务 | 端口 |
| --- | --- |
| Web 前端 / API | `8080` |
| MySQL | `3308`（容器内 `3306`） |
| Redis | `6379` |
| Milvus | `19530` |
| Attu 控制台 | `30000` |

## 数据格式

- FAQ 数据：`mysql_qa/data/金融知识问答.csv`，三列 `领域,问题,答案`，领域取值为 `股票/基金/债券/银行/保险`。
- 知识库文档：`rag_qa/data/<领域>_data/*.md`，每个领域一个子目录，可放置基础/进阶等 Markdown 文档。
- 导入命令：

```bash
python -m rag_qa.main --data-processing --data-dir rag_qa/data
```

## 模型下载

运行时需要以下模型，统一放在 `rag_qa/models/` 目录下（可用环境变量 `FINRAG_MODEL_ROOT` 指向其他目录）：

| 目录名 | 用途 | 来源 |
| --- | --- | --- |
| `bert-base-chinese` | 查询分类器基座 / 分词器 | HuggingFace `google-bert/bert-base-chinese` |
| `bert_query_classifier_finance` | 金融意图二分类器（自定义训练） | 本地训练产物，见下文 |
| `bge-m3` | 向量召回嵌入模型 | HuggingFace `BAAI/bge-m3` |
| `bge-reranker-large` | 召回结果重排 | HuggingFace `BAAI/bge-reranker-large` |
| `nlp_bert_document-segmentation_chinese-base` | 文档语义切分（约 388MB） | ModelScope `iic/nlp_bert_document-segmentation_chinese-base` / HuggingFace `damo/nlp_bert_document-segmentation_chinese-base` |

`scripts/download_models.py` 会自动下载上述公开模型（除自定义分类器外）。

### 自定义查询分类器

`bert_query_classifier_finance` 是基于 `bert-base-chinese` 微调的二分类模型，非公开权重。首次运行时若该目录不存在，`QueryClassifier` 会退回加载未微调的 `bert-base-chinese` 初始化模型；代码内已内置金融关键词兜底路由，保证金融问题仍进入 RAG。如需恢复完整分类能力，可参考 `rag_qa/core/query_classifier.py` 中的 `train_model()` 使用自有标注数据重新训练，或从原部署环境复制该目录。

## 主要接口

- `GET /health`：健康检查
- `GET /live`：存活探针
- `GET /ready`：就绪探针
- `GET /api/sources`：金融领域列表
- `POST /api/create_session`：创建会话
- `GET /api/sessions`：活跃会话列表
- `GET /api/history/{session_id}`：查询历史
- `DELETE /api/history/{session_id}`：清除历史
- `POST /api/archive/{session_id}`：归档会话
- `GET /api/archived`：归档会话列表（回收站）
- `POST /api/unarchive/{session_id}`：恢复归档会话
- `DELETE /api/archive/{session_id}`：彻底删除会话
- `POST /api/query`：非流式 FAQ 查询（保留用于测试，前端默认走 WebSocket）
- `WebSocket /api/stream`：流式 RAG 查询（前端实际使用的通道）

统一响应格式：

```json
{
  "errno": 0,
  "errmsg": "ok",
  "log_id": "uuid",
  "data": {}
}
```

## 目录结构

- `app.py`：FastAPI Web 服务
- `new_main.py`：MySQL FAQ + RAG 集成主入口
- `mysql_qa/`：FAQ 检索与缓存
- `rag_qa/`：RAG 检索、分类、文档处理与评估
- `rag_qa/data/`：金融知识文档
- `mysql_qa/data/金融知识问答.csv`：FAQ 数据
- `scripts/generate_finance_datasets.py`：示例数据集生成脚本
- `scripts/download_models.py`：模型下载脚本
- `scripts/smoke_test.py`：API 冒烟测试
- `static/`：前端页面
- `tests/`：单元测试

## 测试

```bash
# 启动服务后运行冒烟测试
python scripts/smoke_test.py
```

## 常见问题

- **模型下载失败/较慢**：`bge-m3`、`bge-reranker-large` 等体积较大，建议设置 HuggingFace 镜像后重试 `python scripts/download_models.py`；文档切分模型（约 388MB）走 ModelScope。
- **提示缺少分类器**：`bert_query_classifier_finance` 为自定义训练权重，未下载；系统会自动退化为 `bert-base-chinese` 初始化模型并以金融关键词兜底，不影响基本问答，详见「模型下载」。
- **连不上 Milvus/MySQL/Redis**：先确认 `docker compose up -d` 已成功，再检查 `.env` 的端口与密码是否和容器一致。
- **Windows 中文路径问题**：可用 `FINRAG_MODEL_ROOT` 指向无中文路径的模型目录，避免从 `D:\xxx\项目` 这类路径加载模型。

## 生产部署注意事项

- 修改 `.env` 中所有 `change_me` 占位密码，`docker-compose.yml` 中的默认值仅用于本地开发。
- `app.py` 中 CORS 当前为 `allow_origins=["*"]`，公网部署应改为具体前端域名。
- 前端默认通过 `ws://127.0.0.1:8080/api/stream` 通信，若走 HTTPS/WSS 需相应调整前端连接地址。
- `.env`、`rag_qa/models/`、`legacy_backup/` 均被 `.gitignore` 排除，请勿提交到仓库。

## 许可证

本项目采用 MIT License，详见 `LICENSE`。请另行确认依赖（PyTorch、Transformers 等）与模型权重各自的许可条款允许再分发。
