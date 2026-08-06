"""Small UI helpers for updating NiceGUI syllabus trees."""

from typing import Any


def replace_tree_nodes(tree: Any, nodes: list[dict]) -> None:
    tree.props["nodes"] = nodes
    tree.props["ticked"] = []
    tree.update()
