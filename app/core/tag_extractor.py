"""
用 GLM-4V 多模态 LLM 提取文物图片的结构化元数据 + 标签 + 描述，失败时降级为空。

对应书籍 RAG 中 query_rewriter.py 的模式：调用 LLM 做信息抽取，失败降级不中断主流程。
区别在于这里用多模态模型（glm-4v），输入是图片而非纯文本。

设计要点：
- 复用 get_llm() 单例模式，但用独立的 image_tag_llm_model（glm-4v）配置项，
  与书籍问答的 llm_model（glm-4-flash）隔离，避免互相影响。
- 一次 GLM-4V 调用同时产出：结构化字段（朝代/材质/器型...）+ tags + caption，
  避免多次调用浪费 token 和时间。
- 失败降级：标签提取是增强项而非必需项，失败时返回空对象，索引继续走纯向量检索。
  这与 rerank.py 中 Flashrank 失败降级为原始排序是同一思路。
- 图片用 base64 编码内联到消息中，不依赖图片公网 URL（本地图片无法用 URL 访问）。
- 文博字段：category_top 限定枚举值，其他字段不确定时由模型填空字符串。
"""
from __future__ import annotations

import base64
import json
import logging
import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 文博藏品著录专用 Prompt：一次调用产出结构化字段 + tags + caption
_RELIC_PROMPT = """你是资深文博藏品著录专家，请对输入的文物图片进行标准化识别与打标。
严格输出 JSON 格式，禁止多余解释、禁止换行乱码、禁止臆造无依据信息，不确定字段填空字符串。

输出字段要求：
1. caption_standard: 标准文物著录描述（客观、形制、纹饰、材质、工艺、完整状态，50字）
2. caption_public: 大众科普通俗描述（简洁易懂、讲清用途与看点，20字）
3. category_top: 一级文物分类【陶瓷器、青铜器、玉器、书画、金银器、石刻、漆器、织绣、杂项】选其一，无法判断填杂项
4. category_sub: 二级具体器型名称
5. dynasty: 年代/朝代/文化
6. material: 材质/质地
7. color_feature: 色彩、釉色、沁色特征
8. craft: 核心工艺技法
9. pattern_theme: 纹饰题材，数组形式，无纹饰返回空数组 []
10. function_usage: 器物原始功用
11. relic_condition: 完残状态
12. tags: 3-{max_count} 个核心检索标签（用于标签检索，覆盖类别/材质/年代/纹饰/工艺关键词）

仅返回纯JSON，无任何多余内容。示例：
{{"caption_standard":"北宋汝窑天青釉碗，敞口浅腹，圈足，通体施天青釉，釉面开片，足部露胎","caption_public":"宋代汝窑青瓷碗，天青色釉面温润如玉","category_top":"陶瓷器","category_sub":"碗","dynasty":"北宋","material":"瓷","color_feature":"天青色","craft":"施釉","pattern_theme":["开片纹"],"function_usage":"食器","relic_condition":"完整","tags":["汝窑","青瓷","碗","北宋","天青釉"]}}"""


class RelicMetadata(TypedDict):
    """文博藏品结构化元数据（GLM-4V 一次产出的全部字段）。

    与 ImageMetadata 的文博字段一一对应，extract_relic_metadata 返回此结构，
    由 image_indexer 写入 ImageMetadata 的对应字段。
    """

    caption_standard: str
    caption_public: str
    category_top: str
    category_sub: str
    dynasty: str
    material: str
    color_feature: str
    craft: str
    pattern_theme: list[str]
    function_usage: str
    relic_condition: str
    tags: list[str]


def _empty_metadata() -> RelicMetadata:
    """返回空元数据（降级时使用，所有字段为空值）。"""
    return RelicMetadata(
        caption_standard="",
        caption_public="",
        category_top="",
        category_sub="",
        dynasty="",
        material="",
        color_feature="",
        craft="",
        pattern_theme=[],
        function_usage="",
        relic_condition="",
        tags=[],
    )


@lru_cache
def get_tag_llm() -> ChatOpenAI:
    """标签提取专用 LLM 单例（glm-4v 多模态模型）。

    与 retriever.py 的 get_llm() 分离：图片标签提取需要多模态能力，
    用 glm-4v；书籍问答用 glm-4-flash。两者复用同一个智谱 API Key 和 base_url。
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.image_tag_llm_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )


def _encode_image_base64(file_path: str) -> str:
    """读取图片文件并编码为 base64 字符串。"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _parse_json_response(content: str) -> RelicMetadata:
    """解析 GLM-4V 返回的 JSON 内容，容错处理。

    GLM-4V 偶尔会在 JSON 外包一层 ```json ... ``` 或附加解释文字，
    这里用正则提取第一个 JSON 对象，解析失败时降级为空元数据。
    """
    # 去掉可能的 ```json ... ``` 包裹
    content = content.strip()
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if not json_match:
        logger.warning("GLM-4V 返回内容未匹配到 JSON: %s", content[:200])
        return _empty_metadata()

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("GLM-4V JSON 解析失败: %s", json_match.group()[:200])
        return _empty_metadata()

    # 规范化：所有字段兜底为空值，pattern_theme/tags 确保是 list
    result = _empty_metadata()
    for key in result:
        if key in data:
            val = data[key]
            if key in ("pattern_theme", "tags"):
                result[key] = val if isinstance(val, list) else []
            else:
                result[key] = str(val).strip() if val else ""
    return result


def extract_relic_metadata(file_path: str) -> RelicMetadata:
    """调用 GLM-4V 提取文物图片的结构化元数据 + 标签，失败时降级为空对象。

    file_path 为可直接打开的图片路径（绝对路径或相对项目根目录的完整路径），
    调用方（image_indexer）已拼好 image_storage_dir 前缀，这里不再重复拼接。

    一次调用同时产出：
    - 结构化字段：朝代/材质/器型/纹饰/工艺...（用于 metadata 精确过滤）
    - tags：核心检索标签（用于标签倒排索引）
    - caption_standard/caption_public：标准著录描述 + 科普描述（用于文本检索）

    失败降级：返回空对象（所有字段为空值），索引继续走纯向量检索，不阻塞主流程。
    """
    settings = get_settings()
    try:
        image_b64 = _encode_image_base64(file_path)
        llm = get_tag_llm()

        message = HumanMessage(content=[
            {"type": "text", "text": _RELIC_PROMPT.format(max_count=settings.image_tag_max_count)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ])
        response = llm.invoke([message])
        metadata = _parse_json_response(response.content)

        # tags 截断到配置的最大数量
        max_count = settings.image_tag_max_count
        metadata["tags"] = metadata["tags"][:max_count]
        return metadata
    except Exception:
        logger.exception("文物元数据提取失败，降级为空对象: %s", file_path)
        return _empty_metadata()


def extract_tags(file_path: str) -> list[str]:
    """调用 GLM-4V 提取图片标签（向后兼容接口，内部调用 extract_relic_metadata）。

    新代码应直接调用 extract_relic_metadata 一次拿到全部字段，避免重复调用 GLM-4V。
    此函数保留是为了兼容旧的调用点（如 image_indexer.index_image 的单张索引流程），
    仅返回 tags 字段。
    """
    return extract_relic_metadata(file_path)["tags"]
