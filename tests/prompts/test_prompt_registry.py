"""PromptRegistry 单元测试"""
import pytest
from src.prompts.registry import PromptRegistry


class TestPromptRegistry:
    def test_register_first_version(self, registry):
        entry = registry.register(
            name="test.template",
            template="Hello {name}!",
            description="测试模板",
        )
        assert entry.name == "test.template"
        assert len(entry.versions) == 1
        assert entry.versions[0].version == "v1.0"
        assert entry.versions[0].status == "active"
        assert len(entry.versions[0].hash) == 12

    def test_new_version_increments_major(self, registry):
        registry.register("test.t", "v1 content")
        v2 = registry.new_version("test.t", "v2 content", changelog="更新内容")
        assert v2.version == "v2.0"
        assert v2.status == "active"
        v1 = registry.get("test.t", pin="v1.0")
        assert v1.status == "deprecated"

    def test_new_version_skip_on_identical(self, registry):
        registry.register("test.t", "same content")
        v2 = registry.new_version("test.t", "same content")
        assert v2 is None  # 无新版本，跳过

    def test_new_version_skip_on_identical_whitespace(self, registry):
        registry.register("test.t", "same content")
        v2 = registry.new_version("test.t", "same content  ")
        assert v2 is None  # strip 后一致，跳过

    def test_get_active_by_default(self, registry):
        registry.register("test.t", "v1")
        registry.new_version("test.t", "v2")
        active = registry.get("test.t")
        assert active.version == "v2.0"

    def test_get_by_version_pin(self, registry):
        registry.register("test.t", "v1")
        registry.new_version("test.t", "v2")
        v1 = registry.get("test.t", pin="v1.0")
        assert v1.version == "v1.0"
        assert "v1" in v1.template

    def test_get_by_hash(self, registry):
        registry.register("test.t", "content")
        v1 = registry.get("test.t")
        by_hash = registry.get("test.t", pin_hash=v1.hash)
        assert by_hash.template == "content"

    def test_get_unknown_name_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_unknown_version_returns_none(self, registry):
        registry.register("test.t", "content")
        assert registry.get("test.t", pin="v99.0") is None

    def test_render_returns_rendered_prompt(self, registry):
        registry.register("test.greeting", "Hello {name}!",
                          variables=[{"name": "name", "required": True}])
        rendered = registry.render("test.greeting", {"name": "World"})
        assert rendered.text == "Hello World!"
        assert rendered.template_hash is not None
        assert rendered.template_version == "v1.0"

    def test_render_with_pin_hash(self, registry):
        registry.register("test.t", "version {x}")
        v1 = registry.get("test.t")
        registry.new_version("test.t", "VERSION {x}")
        rendered = registry.render("test.t", {"x": "1"}, pin_hash=v1.hash)
        assert rendered.text == "version 1"

    def test_render_unknown_name_raises(self, registry):
        with pytest.raises(KeyError):
            registry.render("unknown", strict=True)

    def test_list_versions_ordered(self, registry):
        registry.register("test.t", "v1")
        registry.new_version("test.t", "v2")
        registry.new_version("test.t", "v3")
        versions = registry.list_versions("test.t")
        assert [v.version for v in versions] == ["v3.0", "v2.0", "v1.0"]

    def test_list_templates_filters_by_group(self, registry):
        registry.register("t1", "c1", group="writing")
        registry.register("t2", "c2", group="review")
        registry.register("t3", "c3", group="writing")
        writing = registry.list_templates(group="writing")
        assert len(writing) == 2
        assert all(e.group == "writing" for e in writing)

    def test_set_status(self, registry):
        registry.register("test.t", "v1")
        registry.new_version("test.t", "v2")
        registry.set_status("test.t", "v2.0", "archived")
        v2 = registry.get("test.t", pin="v2.0")
        assert v2.status == "archived"

    def test_set_status_reactivates_other(self, registry):
        registry.register("test.t", "v1")
        registry.new_version("test.t", "v2")
        registry.set_status("test.t", "v1.0", "active")
        v1 = registry.get("test.t", pin="v1.0")
        v2 = registry.get("test.t", pin="v2.0")
        assert v1.status == "active"
        assert v2.status == "deprecated"

    def test_diff(self, registry):
        registry.register("test.t", "line1\nline2\nline3")
        registry.new_version("test.t", "line1\nline2 modified\nline3\nline4")
        diff = registry.diff("test.t", "v1.0", "v2.0")
        assert diff.lines_added >= 1
        assert diff.lines_removed >= 1
        assert "line2" in diff.template_diff

    def test_diff_same_version(self, registry):
        registry.register("test.t", "content")
        diff = registry.diff("test.t", "v1.0", "v1.0")
        assert diff.lines_added == 0
        assert diff.lines_removed == 0
