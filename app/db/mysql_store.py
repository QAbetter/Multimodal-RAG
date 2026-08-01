"""
MySQL 持久化存储层：实现 LangChain BaseStore 协议，供 ParentDocumentRetriever 的
docstore 使用，替代原 bRAG Notebook 中的 InMemoryByteStore（重启即丢数据）。

表结构（book_parent_chunks）：
    doc_id VARCHAR(64) PRIMARY KEY  -- 与子块向量 metadata 中的 doc_id 一对一对应
    book_id VARCHAR(64)
    chapter_title VARCHAR(255) NULL
    content LONGTEXT
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator, Optional, Sequence

import pymysql
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from pymysql.cursors import DictCursor

from app.core.config import get_settings

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS book_parent_chunks (
    doc_id VARCHAR(64) PRIMARY KEY,
    book_id VARCHAR(64) NOT NULL,
    chapter_title VARCHAR(255) NULL,
    content LONGTEXT NOT NULL,
    INDEX idx_book_id (book_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


@lru_cache
def _get_or_create_connection() -> pymysql.connections.Connection:
    settings = get_settings()
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=DictCursor,
    )
    with conn.cursor() as cursor:
        cursor.execute(_CREATE_TABLE_SQL)
    return conn


def get_mysql_connection() -> pymysql.connections.Connection:
    """返回可用的 MySQL 连接：复用缓存的单例，若已断开（超时/网络问题）则自动重连。"""
    conn = _get_or_create_connection()
    conn.ping(reconnect=True)
    return conn


class MySQLDocStore(BaseStore[str, Document]):
    """实现 LangChain BaseStore[str, Document] 协议，底层用 MySQL 持久化父块。"""

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        if not keys:
            return []
        conn = get_mysql_connection()
        placeholders = ",".join(["%s"] * len(keys))
        sql = f"SELECT * FROM book_parent_chunks WHERE doc_id IN ({placeholders})"
        with conn.cursor() as cursor:
            cursor.execute(sql, tuple(keys))
            rows = {row["doc_id"]: row for row in cursor.fetchall()}

        results: list[Optional[Document]] = []
        for key in keys:
            row = rows.get(key)
            results.append(_row_to_document(row) if row else None)
        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        if not key_value_pairs:
            return
        conn = get_mysql_connection()
        sql = """
        INSERT INTO book_parent_chunks (doc_id, book_id, chapter_title, content)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            book_id=VALUES(book_id), chapter_title=VALUES(chapter_title), content=VALUES(content)
        """
        rows = []
        for doc_id, doc in key_value_pairs:
            meta = doc.metadata
            rows.append(
                (
                    doc_id,
                    meta["book_id"],
                    meta.get("chapter_title"),
                    doc.page_content,
                )
            )
        with conn.cursor() as cursor:
            cursor.executemany(sql, rows)

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        conn = get_mysql_connection()
        placeholders = ",".join(["%s"] * len(keys))
        sql = f"DELETE FROM book_parent_chunks WHERE doc_id IN ({placeholders})"
        with conn.cursor() as cursor:
            cursor.execute(sql, tuple(keys))

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        conn = get_mysql_connection()
        sql = "SELECT doc_id FROM book_parent_chunks"
        params: tuple = ()
        if prefix:
            sql += " WHERE doc_id LIKE %s"
            params = (f"{prefix}%",)
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                yield row["doc_id"]

    def delete_by_book(self, book_id: str) -> None:
        """按 book_id 批量删除，供重新索引前清理旧数据使用。"""
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM book_parent_chunks WHERE book_id = %s", (book_id,))


def _row_to_document(row: dict) -> Document:
    return Document(
        page_content=row["content"],
        metadata={
            "book_id": row["book_id"],
            "chapter_title": row["chapter_title"],
        },
    )


@lru_cache
def get_doc_store() -> MySQLDocStore:
    return MySQLDocStore()
