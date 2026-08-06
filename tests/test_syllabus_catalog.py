from pathlib import Path

from syllabus_catalog import BUILTIN_SYLLABUSES
from syllabus_ui import hierarchy_to_tree_nodes, replace_tree_nodes


def _leaves(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        yield from node


def test_chinese_and_usmle_syllabuses_coexist():
    assert list(BUILTIN_SYLLABUSES) == [
        "USMLE Step 1 (English)",
        "Chinese Medical Licensing Examination (Chinese)",
    ]

    english = list(_leaves(BUILTIN_SYLLABUSES["USMLE Step 1 (English)"]))
    chinese = list(
        _leaves(BUILTIN_SYLLABUSES["Chinese Medical Licensing Examination (Chinese)"])
    )

    assert len(english) >= 100
    assert len(chinese) >= 100
    assert not any("\u4e00" <= char <= "\u9fff" for topic in english for char in topic)
    assert any("\u4e00" <= char <= "\u9fff" for topic in chinese for char in topic)


def test_ui_exposes_builtin_syllabus_selector():
    source = (Path(__file__).parents[1] / "multi_agent_dev.py").read_text(encoding="utf-8")

    assert "list(BUILTIN_SYLLABUSES)," in source
    assert 'label="Built-in Syllabus"' in source
    assert "switch_builtin_syllabus" in source
    assert "Reset Current Syllabus" in source
    assert 'initial_syllabus_name = APP_STATE.get("active_builtin_syllabus"' in source
    assert "BUILTIN_SYLLABUSES[initial_syllabus_name]" in source


def test_tree_replacement_updates_nicegui_props_and_refreshes_component():
    class FakeTree:
        def __init__(self):
            self.props = {"nodes": [{"label": "old"}], "ticked": ["old"]}
            self.update_count = 0

        def update(self):
            self.update_count += 1

    tree = FakeTree()
    new_nodes = [{"label": "new"}]

    replace_tree_nodes(tree, new_nodes)

    assert tree.props["nodes"] == new_nodes
    assert tree.props["ticked"] == []
    assert tree.update_count == 1


def test_tree_conversion_includes_list_items_as_selectable_leaves():
    nodes = hierarchy_to_tree_nodes(
        {"System": {"Topic Group": ["Specific Topic A", "Specific Topic B"]}}
    )

    topic_group = nodes[0]["children"][0]
    assert [child["label"] for child in topic_group["children"]] == [
        "Specific Topic A",
        "Specific Topic B",
    ]
    assert all("children" not in child for child in topic_group["children"])
