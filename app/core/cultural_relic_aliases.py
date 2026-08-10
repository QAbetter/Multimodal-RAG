"""文博结构化字段的同义词表与 query 解析。

本模块解决"结构化字段并入标签路检索"的两个问题：
1. 写入端归一化：GLM-4V 可能输出"唐"/"唐代"/"唐朝"等不同写法，需归一化为标准值
2. 查询端解析：用户 query 是自然语言（如"唐代的青铜剑"），需要解析出结构化标签
   （["朝代:唐代", "材质:青铜", "二级分类:剑"]）才能精确命中标签倒排索引

设计要点：
- 纯规则匹配，不调 LLM（延迟低、稳定、可调试）
- 同义词表覆盖常见的朝代/材质/器型/工艺写法差异
- 解析出的标签格式为 "命名空间:标准值"，与索引写入端一致
- 命名空间对应 ImageMetadata 的字段名，确保写入和查询的 key 完全一致
"""
from __future__ import annotations


# ===========================================================================
# 同义词表：query 中的各种写法 → 标准值
# ===========================================================================

# 朝代同义词（query 中的写法 → 标准值，标准值与 GLM-4V 著录习惯对齐）
_DYNASTY_ALIASES: dict[str, list[str]] = {
    "商": ["商", "商代", "商朝", "殷商"],
    "周": ["周", "周代", "周朝", "西周", "东周"],
    "春秋": ["春秋", "春秋时期", "春秋时代"],
    "战国": ["战国", "战国时期", "战国时代"],
    "秦": ["秦", "秦代", "秦朝"],
    "汉": ["汉", "汉代", "汉朝", "西汉", "东汉", "两汉"],
    "魏晋": ["魏晋", "魏晋南北朝", "三国", "两晋"],
    "南北朝": ["南北朝", "北朝", "南朝"],
    "隋": ["隋", "隋代", "隋朝"],
    "唐": ["唐", "唐代", "唐朝", "大唐"],
    "五代": ["五代", "五代十国", "五代时期"],
    "宋": ["宋", "宋代", "宋朝", "大宋", "北宋", "南宋", "两宋"],
    "辽": ["辽", "辽代", "辽朝"],
    "金": ["金", "金代", "金朝"],
    "元": ["元", "元代", "元朝", "蒙古"],
    "明": ["明", "明代", "明朝", "大明"],
    "清": ["清", "清代", "清朝", "大清", "清晚期", "清中期", "清早期"],
    "民国": ["民国", "中华民国", "民国时期"],
}

# 材质同义词
_MATERIAL_ALIASES: dict[str, list[str]] = {
    "青铜": ["青铜", "铜器", "铜质", "青铜器"],
    "黄铜": ["黄铜"],
    "白铜": ["白铜"],
    "青瓷": ["青瓷"],
    "白瓷": ["白瓷"],
    "青花瓷": ["青花瓷", "青花"],
    "粉彩瓷": ["粉彩瓷", "粉彩"],
    "彩瓷": ["彩瓷", "彩陶"],
    "陶": ["陶", "陶器", "陶质", "灰陶", "红陶", "黑陶"],
    "瓷": ["瓷", "瓷器", "瓷质"],
    "玉": ["玉", "玉石", "玉器", "和田玉", "青玉", "白玉", "碧玉"],
    "金": ["金", "黄金", "金质", "金器"],
    "银": ["银", "白银", "银质", "银器"],
    "铁": ["铁", "铁质", "铁器"],
    "漆": ["漆", "漆器", "大漆", "生漆"],
    "绢": ["绢", "绢本", "绢质"],
    "纸": ["纸", "纸本", "纸质"],
    "石": ["石", "石器", "石质", "石刻"],
    "木": ["木", "木器", "木质", "木胎"],
    "骨": ["骨", "骨器", "骨质", "骨角"],
    "琉璃": ["琉璃", "玻璃"],
}

# 器型/二级分类同义词
_WARE_TYPE_ALIASES: dict[str, list[str]] = {
    "剑": ["剑", "宝剑", "青铜剑"],
    "刀": ["刀", "刀具", "佩刀"],
    "矛": ["矛", "长矛"],
    "鼎": ["鼎", "青铜鼎"],
    "爵": ["爵", "青铜爵"],
    "尊": ["尊", "青铜尊"],
    "壶": ["壶", "青铜壶", "瓷壶", "茶壶", "酒壶"],
    "瓶": ["瓶", "花瓶", "瓷瓶", "梅瓶", "玉壶春瓶"],
    "罐": ["罐", "瓷罐", "陶罐", "盖罐"],
    "碗": ["碗", "瓷碗", "陶碗", "茶碗"],
    "盘": ["盘", "瓷盘", "陶盘", "碟"],
    "杯": ["杯", "瓷杯", "茶杯", "酒杯"],
    "佛像": ["佛像", "佛", "造像", "神像"],
    "壁画": ["壁画", "壁画片段"],
    "碑": ["碑", "碑刻", "墓碑", "石碑"],
    "镜": ["镜", "铜镜", "青铜镜"],
    "佩": ["佩", "玉佩", "佩饰"],
    "饰": ["饰", "饰件", "装饰"],
}

# 工艺同义词
_CRAFT_ALIASES: dict[str, list[str]] = {
    "范铸法": ["范铸法", "范铸", "块范法"],
    "失蜡法": ["失蜡法", "失蜡", "熔模铸造"],
    "刻花": ["刻花", "刻花工艺"],
    "划花": ["划花", "划花工艺"],
    "印花": ["印花", "印花工艺"],
    "贴花": ["贴花", "贴花工艺"],
    "镂空": ["镂空", "镂雕", "透雕"],
    "浮雕": ["浮雕", "浮雕工艺"],
    "圆雕": ["圆雕", "圆雕工艺"],
    "掐丝": ["掐丝", "掐丝珐琅"],
    "镶嵌": ["镶嵌", "镶嵌工艺"],
    "彩绘": ["彩绘", "彩绘工艺"],
    "施釉": ["施釉", "上釉"],
}

# 色彩同义词
_COLOR_ALIASES: dict[str, list[str]] = {
    "青绿": ["青绿", "青绿色"],
    "青黄": ["青黄", "青黄色"],
    "青灰": ["青灰", "青灰色"],
    "白釉": ["白釉", "釉色白"],
    "青釉": ["青釉", "釉色青"],
    "黄釉": ["黄釉", "釉色黄"],
    "黑釉": ["黑釉", "釉色黑"],
    "红斑": ["红斑", "红色斑"],
    "沁色": ["沁色", "鸡骨白", "水银沁"],
}


# ===========================================================================
# 命名空间映射：ImageMetadata 字段 → 标签前缀
# ===========================================================================

# 每个结构化字段对应的标签命名空间前缀
# 写入端：_structured_fields_to_tags 用这个表生成 "命名空间:值" 格式标签
# 查询端：parse_structured_tags 用这个表确定生成的标签前缀
FIELD_NAMESPACE: dict[str, str] = {
    "dynasty": "朝代",
    "material": "材质",
    "category_sub": "二级分类",
    "craft": "工艺",
    "function_usage": "功用",
    "relic_condition": "完残状态",
    "color_feature": "色彩",
}


# ===========================================================================
# 写入端：结构化字段 → 命名空间标签
# ===========================================================================

def normalize_dynasty(dynasty: str | None) -> str | None:
    """归一化朝代值，如 '清康熙' → '清'，'清晚期' → '清'，'唐代' → '唐'。

    写入端调用：把 GLM-4/importer 产出的原始 dynasty 值归一化为标准值，
    确保与查询端 parse_structured_tags 解析出的标签一致。
    原始值仍保留在 caption 中供 BM25 精确匹配（如"清康熙"作为关键词命中）。

    遍历 _DYNASTY_ALIASES，若原始值包含某别名则归一化为对应标准值。
    未匹配的返回原值（如"红山文化"等非朝代值保持不变）。
    """
    if not dynasty or not dynasty.strip():
        return dynasty
    value = dynasty.strip()
    for canonical, aliases in _DYNASTY_ALIASES.items():
        if any(alias in value for alias in aliases):
            return canonical
    return value


def structured_fields_to_tags(
    dynasty: str | None = None,
    material: str | None = None,
    category_sub: str | None = None,
    craft: str | None = None,
    function_usage: str | None = None,
    relic_condition: str | None = None,
    color_feature: str | None = None,
) -> list[str]:
    """把文博结构化字段转为 "命名空间:值" 的标签列表。

    用于索引写入：在 image_indexer 的 _apply_relic_metadata 中调用，
    把结构化字段追加到 tags 一起写入 tag_store。

    空值跳过，不生成标签。

    dynasty 会先经 normalize_dynasty 归一化（如"清康熙"→"清"），
    确保写入端标签与查询端 parse_structured_tags 解析出的标准值一致。

    示例：
        structured_fields_to_tags(dynasty="清康熙", material="青铜", craft="范铸法")
        → ["朝代:清", "材质:青铜", "工艺:范铸法"]
    """
    fields = {
        "dynasty": normalize_dynasty(dynasty) if dynasty else None,
        "material": material,
        "category_sub": category_sub,
        "craft": craft,
        "function_usage": function_usage,
        "relic_condition": relic_condition,
        "color_feature": color_feature,
    }
    tags: list[str] = []
    for field, value in fields.items():
        if value and value.strip():
            prefix = FIELD_NAMESPACE[field]
            tags.append(f"{prefix}:{value.strip()}")
    return tags


# ===========================================================================
# 查询端：自然语言 query → 结构化标签
# ===========================================================================

# 同义词表分组：命名空间 → {标准值 → [别名]}
_ALIAS_TABLES: dict[str, dict[str, list[str]]] = {
    "朝代": _DYNASTY_ALIASES,
    "材质": _MATERIAL_ALIASES,
    "二级分类": _WARE_TYPE_ALIASES,
    "工艺": _CRAFT_ALIASES,
    "色彩": _COLOR_ALIASES,
}


def parse_structured_tags(query: str) -> list[str]:
    """从自然语言 query 解析出结构化标签，用于标签路召回。

    纯规则匹配，遍历同义词表，发现 query 含某别名即生成对应标准值标签。
    不调 LLM，延迟在 1ms 以内。

    示例：
        parse_structured_tags("唐代的青铜剑")
        → ["朝代:唐", "材质:青铜", "二级分类:剑"]

        parse_structured_tags("宋代的青瓷碗")
        → ["朝代:宋", "材质:青瓷", "二级分类:碗"]

        parse_structured_tags("这件器物什么样")
        → []  # 无结构化信息，返回空列表

    返回的标签格式与索引写入端 structured_fields_to_tags 完全一致，
    确保 search_by_tags 能精确命中。
    """
    if not query:
        return []

    tags: list[str] = []
    for namespace, alias_table in _ALIAS_TABLES.items():
        for canonical, aliases in alias_table.items():
            if any(alias in query for alias in aliases):
                tags.append(f"{namespace}:{canonical}")
    return tags
