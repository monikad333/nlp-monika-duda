from __future__ import annotations

import os
import re
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from lab04.config import PLOTS_DIR

_RELATION_KEYWORDS = {
    "founder": ["founder", "founded", "zalozyl", "zalozycielem", "posiada"],
    "worked_for": ["works for", "ceo", "dyrektor", "pracuje"],
    "located_in": ["located in", "w mieście", "siedziba", "based in"],
    "belongs_to": ["belongs to", "nalezy do", "czesc"],
}


def _guess_relation(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 40) : end + 40].lower()
    for relation, keywords in _RELATION_KEYWORDS.items():
        if any(keyword in window for keyword in keywords):
            return relation
    return "related_to"


def build_entity_relations(text: str, entities: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    sorted_entities = sorted(entities, key=lambda entity: entity["start"])

    for index in range(len(sorted_entities) - 1):
        current = sorted_entities[index]
        next_entity = sorted_entities[index + 1]

        same_sentence = not re.search(r"[.!?]", text[current["end"] : next_entity["start"]])
        if not same_sentence:
            continue

        relation = _guess_relation(text, current["end"], next_entity["start"])
        relations.append((current["text"], relation, next_entity["text"]))

    return relations


def save_knowledge_graph(relations: list[tuple[str, str, str]], plots_dir: str = PLOTS_DIR) -> str | None:
    if not relations:
        return None

    graph = nx.DiGraph()
    for source, relation, target in relations:
        graph.add_edge(source, target, label=relation)

    fig, axis = plt.subplots(figsize=(8, 6))
    layout = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, layout, ax=axis, node_color="#4e79a7", node_size=1800)
    nx.draw_networkx_labels(graph, layout, ax=axis, font_size=8, font_color="white")
    nx.draw_networkx_edges(graph, layout, ax=axis, arrows=True)

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, ax=axis, font_size=7)

    axis.set_title("Knowledge graph")
    axis.axis("off")

    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(plots_dir, f"knowledge_graph_{timestamp}.png")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
