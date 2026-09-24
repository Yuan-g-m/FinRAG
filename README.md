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

## 许可证

尚未选择开源许可证。上传 GitHub 前建议明确许可证（如 MIT / Apache-2.0），并确认依赖（PyTorch、Transformers 等）与模型权重各自的许可条款允许再分发。

