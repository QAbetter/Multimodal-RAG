"""
进程级互斥锁：保护 JSON 持久化的读-改-写操作，避免并发竞态。

适用场景：FastAPI 同步路由在 Starlette 线程池中执行（默认 40 线程），
多个并发请求会对 images.json / tag_index.json 产生 TOCTOU 竞态
（读 → 改 → 写之间被其他线程插入，导致后写覆盖先写）。

设计要点：
- 进程内互斥：threading.Lock 保证单进程多线程安全，覆盖 uvicorn 单 worker 部署
- 两把独立锁：注册表和标签索引数据无交叉，用独立锁避免不必要的互斥
- 不覆盖多进程：gunicorn -w N 多 worker 场景需用文件锁（filelock）或换 SQLite，
  当前项目用 uvicorn 单 worker 部署，进程级锁足够

使用约定：
- 所有"读-改-写"images.json 的操作用 registry_lock
- 所有"读-改-写"tag_index.json 的操作用 tag_index_lock
- 纯只读操作（如 load_registered_images 仅读一次）可不加锁，但若读后立即改写则整个块加锁
"""
from __future__ import annotations

import threading

# 保护 images.json 注册表的读-改-写
registry_lock = threading.Lock()

# 保护 tag_index.json 标签倒排索引的读-改-写
tag_index_lock = threading.Lock()
