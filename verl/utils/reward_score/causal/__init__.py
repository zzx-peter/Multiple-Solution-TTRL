# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rule-based reward for HypoSpace causal graph discovery."""

import re
from collections.abc import Iterable
from typing import Any, Optional


def _canonical_edges(edges: Iterable[Iterable[str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(str(src), str(dst)) for src, dst in edges}))


def _canonical_graph(graph: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return _canonical_edges(graph.get("edges", []))


def _graph_key(edges: tuple[tuple[str, str], ...]) -> str:
    if not edges:
        return "No edges"
    return ",".join(f"{src}->{dst}" for src, dst in edges)


def _is_acyclic(edges: Iterable[tuple[str, str]], nodes: Iterable[str]) -> bool:
    graph = {node: [] for node in nodes}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        graph.setdefault(dst, [])

    visiting = set()
    visited = set()

    def visit(node: str) -> bool:
        if node in visited:
            return True
        if node in visiting:
            return False
        visiting.add(node)
        for child in graph.get(node, []):
            if not visit(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in graph)


def _extract_graph_line(solution_str: str) -> Optional[str]:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if answer_matches:
        solution_str = answer_matches[-1]

    graph_matches = re.findall(r"(?im)\bgraph\s*:\s*(.+)$", solution_str)
    if graph_matches:
        return graph_matches[-1].strip()

    lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _parse_prediction(
    solution_str: str,
    nodes: Optional[Iterable[str]] = None,
    max_edges: Optional[int] = None,
) -> Optional[tuple[tuple[str, str], ...]]:
    line = _extract_graph_line(solution_str)
    if not line:
        return None

    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()

    line = line.replace("→", "->").replace("-->", "->").replace("=>", "->").rstrip(" .;")
    if re.search(r"\b(no\s+edges?|empty|none|null)\b", line, re.IGNORECASE):
        return ()

    allowed_nodes = set(nodes or [])
    edges = []
    for part in [piece.strip() for piece in line.split(",") if piece.strip()]:
        match = re.fullmatch(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)", part)
        if not match:
            return None
        src, dst = match.group(1), match.group(2)
        if src == dst:
            return None
        if allowed_nodes and (src not in allowed_nodes or dst not in allowed_nodes):
            return None
        edges.append((src, dst))

    canonical = tuple(sorted(dict.fromkeys(edges)))
    if max_edges is not None and len(canonical) > max_edges:
        return None
    if not _is_acyclic(canonical, allowed_nodes or {node for edge in canonical for node in edge}):
        return None
    return canonical


def _get_ground_truth_list(ground_truth: Any) -> list[dict[str, Any]]:
    if isinstance(ground_truth, dict):
        return ground_truth.get("ground_truth_list", [])
    if isinstance(ground_truth, list):
        return ground_truth
    return []


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Return single-sample accuracy plus fields used for HypoSpace recovery."""
    ground_truth_list = _get_ground_truth_list(ground_truth)
    nodes = ground_truth.get("nodes") if isinstance(ground_truth, dict) else None
    max_edges = ground_truth.get("max_edges") if isinstance(ground_truth, dict) else None
    valid_graphs = {_canonical_graph(graph) for graph in ground_truth_list}
    n_ground_truths = len(valid_graphs)

    prediction = _parse_prediction(solution_str, nodes=nodes, max_edges=max_edges)
    if prediction is None:
        return {
            "score": format_score,
            "acc": format_score,
            "causal_parse_success": 0.0,
            "causal_valid": 0.0,
            "causal_pred": "",
            "causal_n_ground_truths": n_ground_truths,
        }

    valid_score = score if prediction in valid_graphs else format_score
    return {
        "score": valid_score,
        "acc": valid_score,
        "causal_parse_success": 1.0,
        "causal_valid": float(prediction in valid_graphs),
        "causal_pred": _graph_key(prediction),
        "causal_n_ground_truths": n_ground_truths,
    }


__all__ = ["compute_score"]
