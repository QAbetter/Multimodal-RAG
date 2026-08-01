# 部署指南：图片 RAG 服务 + RAGFlow 对接

本项目作为 RAGFlow 的**外部知识库**（检索后端），RAGFlow 调用本服务的 `/api/v1/dify/retrieval` 接口获取图片检索结果。本服务复用已有的 CLIP 向量 + 标签 + caption BM25 + RRF 融合检索，不使用 RAGFlow 自带的 CLIP/文本向量。

## 一、服务器部署本服务

### 1. 准备项目文件

将项目拷贝到服务器，确保目录结构完整：

```
book-rag-exe/
├── app/                    # 应用代码
├── scripts/                # 索引脚本
├── tests/                  # 测试
├── data/                   # 数据目录（索引后生成）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，**必须填写**以下项：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `OPENAI_API_KEY` | 智谱 API Key（LLM + Embedding） | `dfec39812bc...` |
| `MINERU_TOKEN` | MinerU API Token（PDF 解析） | `sk-rVuMLvTOC1...` |
| `EXTERNAL_BASE_URL` | **RAGFlow 能访问本服务的地址** | `http://192.168.1.100:8000` |
| `DIFY_API_KEY` | 外部知识库 API Key（自定义，RAGFlow 侧填同样的值） | `ragflow-my-secret-key-2026` |

> **EXTERNAL_BASE_URL 说明**：RAGFlow 拿到检索结果后，需要通过此 URL 访问图片。填 RAGFlow 能访问到的本服务地址（服务器内网 IP 或公网 IP），**不要用 localhost/127.0.0.1**（RAGFlow 在另一个容器里访问不到）。

### 3. Docker 一键启动

```bash
docker compose up -d --build
```

首次构建约 5-10 分钟（装 torch 等依赖），后续启动秒级。

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# 预期返回：{"status":"ok"}

# 检索接口测试（无 API Key 时本地可直接调）
curl -X POST http://localhost:8000/api/v1/dify/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query":"青瓷","retrieval_setting":{"top_k":3,"score_threshold":0.0}}'
# 预期返回：{"records":[{"content":"...","title":"...","score":0.4,"metadata":{...}}]}
```

### 5. 索引数据（按需）

服务启动后，需要先索引图片/PDF 才能检索到数据：

```bash
# 进入容器索引图片
docker exec -it book-rag bash

# 索引散落图片（data/images/raw/ 下的图片）
python scripts/batch_index_images.py --full

# 索引 PDF（data/pdf/ 下的 PDF，会调 MinerU API）
python scripts/batch_index_pdf.py --dir data/pdf
```

索引数据持久化在 `./data` 目录（通过 volume 挂载），重启容器不丢失。

## 二、RAGFlow 侧配置

### 1. 登录 RAGFlow

浏览器打开 RAGFlow 管理界面（通常 `http://<ragflow-server-ip>:80`）。

### 2. 配置外部知识库

在 RAGFlow 的**知识库管理**或**系统设置**中找到**外部知识库 API**配置：

| 配置项 | 填写值 |
|--------|--------|
| **名称** | 自定义，如 `文博图片RAG` |
| **Endpoint** | `http://<本服务IP>:8000/api/v1/dify` |
| **API Key** | `.env` 中 `DIFY_API_KEY` 的值 |

> RAGFlow 会自动在 Endpoint 后拼接 `/retrieval`，完整调用路径为 `http://<本服务IP>:8000/api/v1/dify/retrieval`。

### 3. 测试连通性

在 RAGFlow 的外部知识库配置页面点击**测试**，输入查询词（如"青瓷"），确认能返回检索结果。

### 4. 在应用中使用

在 RAGFlow 的助手/应用中，关联此前配置的外部知识库，即可在问答时调用本服务的图片检索能力。

## 三、架构说明

```
用户提问
  ↓
RAGFlow（编排/评测前端）
  ↓ POST /api/v1/dify/retrieval
  ↓ Headers: Authorization: Bearer <DIFY_API_KEY>
本服务（检索后端）
  ├─ CLIP 向量召回（语义相似）
  ├─ 标签倒排召回（精确匹配）
  ├─ caption BM25 召回（文本匹配）
  └─ RRF 融合 → 返回 records
  ↓
RAGFlow 拿到 records，用 content 喂给 LLM 生成答案
  ↓
返回用户
```

## 四、常见问题

### Q1: RAGFlow 测试外部知识库返回空结果

**原因**：本服务还没索引数据，或 `EXTERNAL_BASE_URL` 配置错误。

**排查**：
```bash
# 检查本服务是否有索引数据
curl http://localhost:8000/image/stats
# 看 registered 和 ready 字段是否 > 0

# 直接调检索接口看是否有结果
curl -X POST http://localhost:8000/api/v1/dify/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query":"青瓷","retrieval_setting":{"top_k":3}}'
```

### Q2: RAGFlow 报 401 未授权

**原因**：`DIFY_API_KEY` 在本服务和 RAGFlow 两侧不一致。

**排查**：检查 `.env` 的 `DIFY_API_KEY` 和 RAGFlow 外部知识库配置中的 API Key 是否完全一致。

### Q3: RAGFlow 拿到结果但图片无法显示

**原因**：`EXTERNAL_BASE_URL` 配置错误，或本服务端口未对 RAGFlow 开放。

**排查**：
```bash
# 在 RAGFlow 容器内测试能否访问本服务
docker exec -it <ragflow-container> curl http://<本服务IP>:8000/health
# 确认 RAGFlow 所在机器能访问本服务的 8000 端口（防火墙/安全组放行）
```

### Q4: 首次启动很慢

**原因**：Chinese-CLIP 模型首次需从 HuggingFace 下载（约 400MB）。docker-compose.yml 已配置 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像，但仍需几分钟。

### Q5: 索引 PDF 时报 MinerU API 错误

**原因**：`MINERU_TOKEN` 未配置或已过期。

**排查**：到 https://mineru.net/apiManage 检查 Token 有效性。
