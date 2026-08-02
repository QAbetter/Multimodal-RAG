"""
文博结构化字段标签化的单元测试：验证同义词表、写入端转换、查询端解析的一致性。

测试覆盖：
1. structured_fields_to_tags：结构化字段 → "命名空间:值" 标签
2. parse_structured_tags：自然语言 query → 结构化标签
3. 写入端与查询端一致性：同义词归一化后能精确命中
4. 边界情况：空值、无结构化信息的 query

运行：
    .venv\\Scripts\\python.exe -m pytest tests/test_cultural_relic_aliases.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.cultural_relic_aliases import (
    parse_structured_tags,
    structured_fields_to_tags,
)


# ===========================================================================
# 1. 写入端：structured_fields_to_tags
# ===========================================================================

class TestStructuredFieldsToTags:
    """测试 structured_fields_to_tags：把字段值转为命名空间标签。"""

    def test_single_field(self):
        """单字段转换。"""
        tags = structured_fields_to_tags(dynasty="唐")
        assert tags == ["朝代:唐"]

    def test_multiple_fields(self):
        """多字段同时转换。"""
        tags = structured_fields_to_tags(
            dynasty="唐",
            material="青铜",
            category_sub="剑",
            craft="范铸法",
        )
        assert "朝代:唐" in tags
        assert "材质:青铜" in tags
        assert "二级分类:剑" in tags
        assert "工艺:范铸法" in tags
        assert len(tags) == 4

    def test_none_values_skipped(self):
        """None 值不生成标签。"""
        tags = structured_fields_to_tags(dynasty="唐", material=None, craft="")
        assert tags == ["朝代:唐"]

    def test_all_none_returns_empty(self):
        """全空返回空列表。"""
        assert structured_fields_to_tags() == []

    def test_whitespace_stripped(self):
        """值的前后空白被 strip。"""
        tags = structured_fields_to_tags(dynasty="  唐  ")
        assert tags == ["朝代:唐"]

    def test_all_fields_covered(self):
        """所有 7 个字段都能正确转换。"""
        tags = structured_fields_to_tags(
            dynasty="宋",
            material="青瓷",
            category_sub="碗",
            craft="刻花",
            function_usage="食器",
            relic_condition="完整",
            color_feature="青釉",
        )
        assert len(tags) == 7
        assert "朝代:宋" in tags
        assert "材质:青瓷" in tags
        assert "二级分类:碗" in tags
        assert "工艺:刻花" in tags
        assert "功用:食器" in tags
        assert "完残状态:完整" in tags
        assert "色彩:青釉" in tags


# ===========================================================================
# 2. 查询端：parse_structured_tags
# ===========================================================================

class TestParseStructuredTags:
    """测试 parse_structured_tags：从 query 解析结构化标签。"""

    def test_dynasty_alias(self):
        """朝代同义词：多种写法都归一化为标准值。"""
        # "唐代" → 朝代:唐
        assert "朝代:唐" in parse_structured_tags("唐代的青铜剑")
        # "唐朝" → 朝代:唐
        assert "朝代:唐" in parse_structured_tags("唐朝的瓷器")
        # "大唐" → 朝代:唐
        assert "朝代:唐" in parse_structured_tags("大唐风格")
        # "北宋" → 朝代:宋
        assert "朝代:宋" in parse_structured_tags("北宋的青瓷")

    def test_material_alias(self):
        """材质同义词。"""
        assert "材质:青铜" in parse_structured_tags("青铜剑")
        assert "材质:青铜" in parse_structured_tags("铜器鉴定")
        assert "材质:青瓷" in parse_structured_tags("青瓷碗")
        assert "材质:玉" in parse_structured_tags("玉佩")

    def test_ware_type_alias(self):
        """器型同义词。"""
        assert "二级分类:剑" in parse_structured_tags("青铜剑")
        assert "二级分类:碗" in parse_structured_tags("青瓷碗")
        assert "二级分类:瓶" in parse_structured_tags("梅瓶")

    def test_craft_alias(self):
        """工艺同义词。"""
        assert "工艺:范铸法" in parse_structured_tags("范铸法制作的铜器")
        assert "工艺:刻花" in parse_structured_tags("刻花工艺的瓷器")

    def test_color_alias(self):
        """色彩同义词。"""
        assert "色彩:青绿" in parse_structured_tags("青绿色的铜器")

    def test_multi_field_query(self):
        """复合 query：一次解析出多个字段。"""
        tags = parse_structured_tags("唐代的青铜剑")
        assert "朝代:唐" in tags
        assert "材质:青铜" in tags
        assert "二级分类:剑" in tags

    def test_no_structured_info(self):
        """无结构化信息的 query 返回空列表。"""
        assert parse_structured_tags("这件器物什么样") == []
        assert parse_structured_tags("好看的图片") == []

    def test_empty_query(self):
        """空 query 返回空列表。"""
        assert parse_structured_tags("") == []
        assert parse_structured_tags(None) == []  # type: ignore[arg-type]

    def test_overlapping_dynasty_names(self):
        """含子串的朝代名匹配行为：子串匹配会同时命中多个朝代标签。

        "后唐" 含 "唐" 子串，会命中 "朝代:唐"。
        这是子串匹配的已知行为，由 RRF 融合的多标签命中来缓解（命中越多分越高）。
        实际检索中 "后唐" 相关图片也会被 "朝代:唐" 标签召回，靠向量路和 caption 路做区分。
        """
        tags = parse_structured_tags("后唐时期的器物")
        # "后唐" 含 "唐"，命中朝代:唐
        assert "朝代:唐" in tags

    def test_southern_northern_dynasties(self):
        """南北朝系列朝代。"""
        assert "朝代:南北朝" in parse_structured_tags("南北朝的佛像")
        assert "朝代:南北朝" in parse_structured_tags("北朝的石窟")


# ===========================================================================
# 3. 写入端与查询端一致性（核心：索引的 key 和查询的 key 完全一致）
# ===========================================================================

class TestWriteReadConsistency:
    """验证写入端和查询端生成的标签格式完全一致，确保 search_by_tags 能精确命中。"""

    def test_dynasty_consistency(self):
        """朝代字段写入后能被 query 精确命中。"""
        # 写入端：GLM-4V 提取 dynasty="唐" → 写入标签 "朝代:唐"
        write_tags = structured_fields_to_tags(dynasty="唐")
        # 查询端：用户搜"唐代的青铜剑" → 解析出 "朝代:唐"
        query_tags = parse_structured_tags("唐代的青铜剑")
        # 写入的标签必须能在查询标签中找到
        assert "朝代:唐" in write_tags
        assert "朝代:唐" in query_tags

    def test_material_consistency(self):
        """材质字段一致性。"""
        write_tags = structured_fields_to_tags(material="青铜")
        query_tags = parse_structured_tags("青铜器")
        assert "材质:青铜" in write_tags
        assert "材质:青铜" in query_tags

    def test_ware_type_consistency(self):
        """器型字段一致性。"""
        write_tags = structured_fields_to_tags(category_sub="剑")
        query_tags = parse_structured_tags("青铜剑")
        assert "二级分类:剑" in write_tags
        assert "二级分类:剑" in query_tags

    def test_full_relic_consistency(self):
        """完整文博元数据的一致性：写入唐代青铜剑，查询唐代青铜剑能全部命中。"""
        # 写入端：索引一张唐代青铜剑图片
        write_tags = structured_fields_to_tags(
            dynasty="唐",
            material="青铜",
            category_sub="剑",
            craft="范铸法",
        )
        # 查询端：用户搜"唐代的范铸法青铜剑"
        query_tags = parse_structured_tags("唐代的范铸法青铜剑")
        # 所有写入的标签都应该能被查询端命中
        for tag in write_tags:
            assert tag in query_tags, f"写入的标签 {tag} 未被查询端命中"

    def test_synonym_normalization(self):
        """同义词归一化：不同写法的 query 解析出相同的标签。"""
        # "唐代" 和 "唐朝" 应该解析出相同的 "朝代:唐"
        tags1 = parse_structured_tags("唐代的器物")
        tags2 = parse_structured_tags("唐朝的器物")
        tags3 = parse_structured_tags("大唐的器物")
        assert "朝代:唐" in tags1
        assert "朝代:唐" in tags2
        assert "朝代:唐" in tags3
