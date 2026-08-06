# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rule-based reward for HypoSpace 3D structure discovery."""

import json
import re
from typing import Any, Optional

Structure = tuple[tuple[tuple[int, ...], ...], ...]


def _canonical_structure(structure: Any) -> Optional[Structure]:
    layers = structure.get("layers") if isinstance(structure, dict) else structure
    if not isinstance(layers, (list, tuple)) or not layers:
        return None

    canonical_layers = []
    for layer in layers:
        if not isinstance(layer, (list, tuple)) or not layer:
            return None
        canonical_rows = []
        for row in layer:
            if not isinstance(row, (list, tuple)) or not row:
                return None
            try:
                canonical_row = tuple(int(value) for value in row)
            except (TypeError, ValueError):
                return None
            if any(value not in (0, 1) for value in canonical_row):
                return None
            canonical_rows.append(canonical_row)
        canonical_layers.append(tuple(canonical_rows))
    return tuple(canonical_layers)


def _structure_key(structure: Structure) -> str:
    """Return a compact, stable identity for a parsed 3D structure."""
    return json.dumps(structure, separators=(",", ":"))


def _structure_from_key(key: str) -> Optional[Structure]:
    try:
        return _canonical_structure(json.loads(key))
    except (json.JSONDecodeError, TypeError):
        return None


def _format_structure(structure: Structure) -> str:
    lines = ["Structure:"]
    for layer_index, layer in enumerate(structure, start=1):
        lines.append(f"Layer {layer_index}:")
        lines.extend(" ".join(str(value) for value in row) for row in layer)
    return "\n".join(lines)


def _extract_structure_section(solution_str: str) -> Optional[str]:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if answer_matches:
        solution_str = answer_matches[-1]

    matches = list(re.finditer(r"(?im)^\s*Structure\s*:\s*$", solution_str))
    if not matches:
        return None
    return solution_str[matches[-1].end() :]


def _parse_row(line: str, grid_size: int) -> Optional[tuple[int, ...]]:
    line = line.strip().strip("`").strip()
    if line.startswith("[") and line.endswith("]"):
        line = line[1:-1]
    if re.search(r"[^01,\s]", line):
        return None
    values = re.findall(r"[01]", line)
    if len(values) != grid_size:
        return None
    return tuple(int(value) for value in values)


def _is_supported_structure(structure: Structure, grid_size: int, max_height: Optional[int]) -> bool:
    if not structure or (max_height is not None and len(structure) > max_height):
        return False
    if any(len(layer) != grid_size for layer in structure):
        return False
    if any(len(row) != grid_size for layer in structure for row in layer):
        return False
    if not any(value for row in structure[0] for value in row):
        return False
    if not any(value for row in structure[-1] for value in row):
        return False

    for layer_index in range(1, len(structure)):
        lower_layer = structure[layer_index - 1]
        layer = structure[layer_index]
        for row_index in range(grid_size):
            for column_index in range(grid_size):
                if layer[row_index][column_index] and not lower_layer[row_index][column_index]:
                    return False
    return True


def _parse_prediction(
    solution_str: str,
    grid_size: int = 3,
    max_height: Optional[int] = None,
) -> Optional[Structure]:
    section = _extract_structure_section(solution_str)
    if section is None:
        return None

    lines = section.splitlines()
    layers = []
    expected_layer_number = 1
    index = 0
    while index < len(lines):
        line = lines[index].strip().strip("`").strip()
        if not line:
            index += 1
            continue

        layer_match = re.fullmatch(r"Layer\s+(\d+)\s*:", line, flags=re.IGNORECASE)
        if layer_match is None:
            if layers:
                break
            return None
        if int(layer_match.group(1)) != expected_layer_number:
            return None

        rows = []
        index += 1
        while index < len(lines) and len(rows) < grid_size:
            if not lines[index].strip():
                index += 1
                continue
            row = _parse_row(lines[index], grid_size)
            if row is None:
                return None
            rows.append(row)
            index += 1
        if len(rows) != grid_size:
            return None
        layers.append(tuple(rows))
        expected_layer_number += 1

    structure = tuple(layers)
    if not _is_supported_structure(structure, grid_size=grid_size, max_height=max_height):
        return None
    return structure


def _get_ground_truth_list(ground_truth: Any) -> list[dict[str, Any]]:
    if isinstance(ground_truth, dict):
        return ground_truth.get("ground_truth_list", [])
    if isinstance(ground_truth, list):
        return ground_truth
    return []


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Return single-sample accuracy plus fields used for HypoSpace recovery."""
    ground_truth_list = _get_ground_truth_list(ground_truth)
    grid_size = int(ground_truth.get("grid_size", 3)) if isinstance(ground_truth, dict) else 3
    max_height = ground_truth.get("max_height") if isinstance(ground_truth, dict) else None
    if max_height is not None:
        max_height = int(max_height)

    valid_structures = {
        structure
        for item in ground_truth_list
        if (structure := _canonical_structure(item)) is not None
    }
    n_ground_truths = len(valid_structures)
    prediction = _parse_prediction(solution_str, grid_size=grid_size, max_height=max_height)
    if prediction is None:
        return {
            "score": format_score,
            "acc": format_score,
            "3d_parse_success": 0.0,
            "3d_valid": 0.0,
            "3d_pred": "",
            "3d_n_ground_truths": n_ground_truths,
        }

    valid_score = score if prediction in valid_structures else format_score
    return {
        "score": valid_score,
        "acc": valid_score,
        "3d_parse_success": 1.0,
        "3d_valid": float(prediction in valid_structures),
        "3d_pred": _structure_key(prediction),
        "3d_n_ground_truths": n_ground_truths,
    }


__all__ = ["compute_score"]
