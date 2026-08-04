# RAGFlow 使用文档

版本：v1.0  
日期：2026-07-30  
适用环境：浪潮 GPU 服务器上的 RAGFlow v0.26.0 部署

## 1. 访问入口

- Web 入口：http://49.232.152.177:33080/
- 部署目录：`/data/ragflow/ragflow/docker`
- 运行方式：Docker Compose，当前使用 CPU profile
- 对外暴露入口：Nginx `33080` 端口
- 内部服务端口：RAGFlow API、MySQL、Redis、MinIO、Elasticsearch 均绑定在 `127.0.0.1`，不要直接对公网开放

> 注意：服务器 SSH 密码、RAGFlow 账号密码、模型 API Key 不应写入共享文档或代码仓库。

## 2. 账号与权限

### 2.1 首次使用

1. 打开 `http://49.232.152.177:33080/`
2. 注册第一个管理员账号
3. 登录后进入右上角头像菜单，完成模型供应商配置

### 2.2 账号管理建议

- 第一个账号建议由管理员注册和保管。
- 后续业务用户使用独立账号，不共用管理员账号。
- 若只给内部团队使用，建议通过 Nginx、云防火墙或 VPN 限制访问来源。
- 若需要对业务系统开放，不建议让业务系统使用 Web 登录态，应使用 RAGFlow API Key。

## 3. 模型配置

RAGFlow 是 RAG 引擎，本身需要接入大模型才能进行问答。至少需要配置：

- Chat model：用于生成回答。
- Embedding model：用于文档向量化和检索。
- Image-to-text model：用于图片、扫描件、复杂 PDF 的视觉理解，按业务需要配置。

配置路径：

1. 登录 RAGFlow。
2. 点击右上角头像。
3. 进入 `Model providers`。
4. 选择模型供应商并填写 API Key。
5. 进入 `System Model Settings`。
6. 设置默认 Chat model、Embedding model 等。

常见选择：

- 如果使用公有云模型：配置对应厂商 API Key。
- 如果使用 OpenAI 兼容接口：选择 `OpenAI-API-Compatible`，填写 base URL、模型名和 API Key。
- 如果后续部署本地模型：可对接 Ollama、Xinference、vLLM、LocalAI 等兼容服务，具体以页面支持项为准。

## 4. 创建知识库 Dataset

Dataset 是 RAGFlow 的知识库，也是后续 Chat 和 Agent 的主要知识来源。

### 4.1 创建 Dataset

1. 点击顶部 `Dataset`。
2. 点击 `Create dataset`。
3. 填写知识库名称，例如 `售后知识库`、`合同制度库`、`产品手册库`。
4. 选择 Embedding model。
5. 选择 Chunk method。
6. 保存。

### 4.2 Chunk method 建议

- PDF、Word、PPT 等格式复杂文档：优先使用默认或面向文档解析的模板。
- FAQ、问答对、标准问题：使用 QA / Manual 类模板更容易控制答案边界。
- 表格、Excel：先用少量样本测试解析效果，再批量导入。
- 代码或配置文件：建议先按目录、模块、文件类型拆分，再导入。

### 4.3 上传文档

1. 进入目标 Dataset。
2. 点击上传文档。
3. 上传 PDF、DOCX、TXT、Markdown、CSV、XLSX、PPTX、图片等文件。
4. 上传后点击解析。
5. 等待状态变为完成。

### 4.4 检查解析质量

解析完成后必须抽查：

- 文档是否被正确分块。
- 表格是否被拆乱。
- 标题、章节、页码是否保留。
- OCR 文本是否有明显乱码。
- 关键业务术语是否被正确识别。

如果解析效果不好：

- 调整 Chunk method。
- 调整 chunk size、overlap 或解析参数。
- 对复杂 PDF 先做 OCR/转文本预处理。
- 对核心文档进行人工 chunk 修正。

## 5. 检索测试

在 Dataset 内做 Retrieval Test，先验证“能不能搜到正确知识”，再接 Chat。

建议测试问题：

- 使用业务真实问题，而不是只问文档标题。
- 每个知识库至少测试 10 到 20 个高频问题。
- 同时测试精确问题和模糊问题。
- 对没有答案的问题，确认系统不会乱答。

重点观察：

- Top chunks 是否来自正确文档。
- 相似度是否过低。
- 是否召回了过多无关内容。
- 引用片段是否足够回答问题。

常用调参方向：

- 答案漏召回：降低 similarity threshold，或提高 top N。
- 无关片段太多：提高 similarity threshold，或降低 top N。
- 关键词型问题效果差：启用或增强关键词检索能力。
- 语义相近但术语不同：优化文档写法或增加同义词说明。

## 6. 创建 Chat 应用

Chat 是面向用户的问答助手，可以绑定一个或多个 Dataset。

### 6.1 创建 Chat

1. 点击顶部 `Chat`。
2. 点击 `Create chat`。
3. 进入 Chat 配置页。
4. 设置名称，例如 `售后问答助手`。
5. 绑定一个或多个 Dataset。
6. 选择 Chat model。
7. 设置 Prompt engine。
8. 保存后开始测试。

### 6.2 Prompt 建议

建议系统提示词：

```text
你是公司内部知识库助手。请优先基于已检索到的知识库内容回答。
如果知识库中没有足够依据，请明确说明“当前知识库未找到可靠依据”，不要编造。
回答时尽量给出来源、章节、文件名或引用片段。
```

### 6.3 Empty response 建议

如果希望答案严格受知识库约束，设置 Empty response：

```text
当前知识库未找到可靠依据，请联系业务负责人确认。
```

如果不设置 Empty response，模型可能在检索不到内容时自行发挥，不适合制度、合同、财务、客服标准答案等场景。

## 7. API 接入

业务系统后续建议直接通过 API 调用 RAGFlow，不依赖 Web 登录态。

### 7.1 获取 API Key

1. 登录 RAGFlow。
2. 进入用户设置或 API Key 管理入口。
3. 创建 API Key。
4. 复制后交给业务系统后端保存。

不要把 API Key 写到前端代码、浏览器本地存储或公开仓库。

### 7.2 公网 API Base URL

当前通过 Nginx 统一入口访问：

```text
http://49.232.152.177:33080
```

### 7.3 通用 Chat Completions 示例

适用于直接对话，也可指定 `chat_id` 使用某个 Chat 应用配置。

```bash
curl --request POST \
  --url 'http://49.232.152.177:33080/api/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <YOUR_RAGFLOW_API_KEY>' \
  --data-binary '{
    "chat_id": "<CHAT_ID>",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "请根据知识库说明这个产品的保修政策"
      }
    ]
  }'
```

如果不传 `chat_id`，请求会使用租户默认聊天模型，不一定会使用指定知识库；业务接入建议明确传 `chat_id`。

### 7.4 OpenAI 兼容接口示例

适合已有 OpenAI SDK 的业务系统迁移。

```bash
curl --request POST \
  --url 'http://49.232.152.177:33080/api/v1/chats_openai/<CHAT_ID>/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <YOUR_RAGFLOW_API_KEY>' \
  --data '{
    "model": "model",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "请总结知识库中的交付验收流程"
      }
    ]
  }'
```

### 7.5 上传文档 API 示例

```bash
curl --request POST \
  --url 'http://49.232.152.177:33080/api/v1/datasets/<DATASET_ID>/documents' \
  --header 'Authorization: Bearer <YOUR_RAGFLOW_API_KEY>' \
  --form 'file=@./test1.txt' \
  --form 'file=@./test2.pdf'
```

### 7.6 触发文档解析 API 示例

```bash
curl --request POST \
  --url 'http://49.232.152.177:33080/api/v1/datasets/<DATASET_ID>/chunks' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <YOUR_RAGFLOW_API_KEY>' \
  --data '{
    "document_ids": ["<DOCUMENT_ID_1>", "<DOCUMENT_ID_2>"]
  }'
```

## 8. 推荐业务落地流程

### 阶段 1：管理员初始化

1. 注册管理员账号。
2. 配置模型供应商。
3. 设置系统默认模型。
4. 创建第一个测试 Dataset。
5. 上传 5 到 10 份代表性文档。
6. 跑通解析、检索测试和 Chat。

### 阶段 2：业务知识库建设

1. 按业务域拆分 Dataset，例如售后、合同、产品、制度。
2. 每个 Dataset 先导入小样本。
3. 抽查 chunk 质量。
4. 调整解析策略。
5. 批量导入正式文档。
6. 建立文档更新责任人。

### 阶段 3：业务系统接入

1. 为每个业务场景创建独立 Chat。
2. 记录每个 Chat 的 `chat_id`。
3. 为业务系统创建 API Key。
4. 后端调用 `/api/v1/chat/completions`。
5. 前端展示答案、引用来源和兜底提示。
6. 增加日志，记录用户问题、命中文档、回答状态和失败原因。

## 9. 运维命令

以下命令在服务器上执行。

### 9.1 查看服务状态

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml ps
```

### 9.2 查看 RAGFlow 日志

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml logs -f --tail=200 ragflow-cpu
```

### 9.3 查看全部服务日志

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml logs -f --tail=200
```

### 9.4 重启 RAGFlow 应用

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml restart ragflow-cpu
```

### 9.5 重启整套 RAGFlow

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml up -d
```

### 9.6 停止服务

```bash
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml down
```

不要随意执行：

```bash
sudo docker compose -f docker-compose.yml down -v
```

`-v` 会删除数据卷，可能导致 MySQL、MinIO、Elasticsearch 数据丢失。

### 9.7 检查本机入口

```bash
curl -I http://127.0.0.1:9388/
curl -I http://127.0.0.1:33080/
```

### 9.8 检查公网入口

```bash
curl -I http://49.232.152.177:33080/
```

### 9.9 检查关键内核参数

```bash
sysctl vm.max_map_count
```

期望值：

```text
vm.max_map_count = 262144
```

## 10. 常见问题

### 10.1 页面打不开

检查顺序：

1. 确认服务器是否能访问。
2. 检查 Nginx 是否监听 `33080`。
3. 检查 RAGFlow 容器是否运行。
4. 查看 `ragflow-cpu` 日志。

命令：

```bash
sudo ss -ltnp | grep 33080
cd /data/ragflow/ragflow/docker
sudo docker compose -f docker-compose.yml ps
sudo docker compose -f docker-compose.yml logs --tail=200 ragflow-cpu
```

### 10.2 登录后提示模型不可用

通常是模型未配置或 API Key 错误：

1. 检查 `Model providers`。
2. 检查 API Key。
3. 检查 `System Model Settings`。
4. 检查模型供应商余额、限流和网络。

### 10.3 文档解析失败

处理方式：

1. 查看文档格式是否受支持。
2. 用小文件测试。
3. 对扫描 PDF 先 OCR。
4. 换 Chunk method。
5. 查看 RAGFlow 日志。

### 10.4 问答乱答

处理方式：

1. 设置 Empty response。
2. 收紧 Prompt，要求只基于知识库回答。
3. 提高 similarity threshold。
4. 减少 top N。
5. 检查是否绑定了错误 Dataset。

### 10.5 检索不到答案

处理方式：

1. 在 Dataset 内做 Retrieval Test。
2. 确认文档解析完成。
3. 降低 similarity threshold。
4. 增加 top N。
5. 优化文档标题、关键词和同义词。

### 10.6 上传文件过大

当前 Nginx 已配置较大的上传限制。若仍失败，检查：

- RAGFlow `.env` 中上传大小限制。
- 宿主 Nginx `client_max_body_size`。
- RAGFlow 容器内 Nginx 配置。
- 浏览器或代理超时。

## 11. 安全建议

- 不要公网开放 MySQL、Redis、MinIO、Elasticsearch。
- 不要把 API Key 放到前端。
- 每个业务系统使用独立 API Key。
- 定期轮换 API Key。
- 管理员账号启用强密码。
- 如果公网访问，仅允许可信 IP 或加 VPN。
- 生产使用建议配置 HTTPS。
- 定期备份 MySQL、MinIO 和 Elasticsearch 数据卷。

## 12. 备份建议

RAGFlow 的关键数据包括：

- MySQL：用户、配置、Dataset、Chat、任务记录等元数据。
- MinIO：上传文件和对象存储内容。
- Elasticsearch：文档索引、chunk 检索数据。
- Docker 配置：`/data/ragflow/ragflow/docker/.env`、`service_conf.yaml.template`、Nginx 配置。

建议：

- 每天自动备份 MySQL。
- 每天备份 MinIO 数据卷或对象数据。
- 定期快照 Elasticsearch 数据卷。
- 每次改 `.env` 和 Nginx 配置前先备份。
- 恢复演练至少做一次，不要只做备份不验证。

## 13. 后续扩展方向

- 接入企业统一登录或访问网关。
- 为不同业务线拆分 Dataset 和 Chat。
- 增加 API 调用日志和问答质检。
- 接入本地大模型，降低公有云模型成本。
- 修复服务器 NVIDIA 驱动后启用 GPU profile。
- 对核心知识库建立更新、审核、下线流程。

## 14. 官方参考

- RAGFlow Quickstart：https://ragflow.net/docs
- RAGFlow Dataset 配置：https://ragflow.net/docs/configure_knowledge_base
- RAGFlow Docker 配置：https://ragflow.net/docs/configurations
- RAGFlow HTTP API：https://ragflow.com.cn/docs/http_api_reference
- RAGFlow Python API：https://ragflow.net/docs/python_api_reference
