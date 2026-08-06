"""Small UI helpers for updating NiceGUI syllabus trees."""

from typing import Any


def hierarchy_to_tree_nodes(
    hierarchical_data: dict, parent_key: str = ""
) -> list[dict]:
    nodes = []
    for key, value in hierarchical_data.items():
        node_id = f"{parent_key}/{key}" if parent_key else key
        node = {"id": node_id, "label": key}
        if isinstance(value, dict):
            node["children"] = hierarchy_to_tree_nodes(value, node_id)
        elif isinstance(value, list):
            node["children"] = [
                {"id": f"{node_id}/{item}", "label": str(item)} for item in value
            ]
        nodes.append(node)
    return nodes


def replace_tree_nodes(tree: Any, nodes: list[dict]) -> None:
    tree.props["nodes"] = nodes
    tree.props["ticked"] = []
    tree.update()
