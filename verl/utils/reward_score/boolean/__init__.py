# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rule-based reward for HypoSpace Boolean expression discovery."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

Expression = tuple[Any, ...]
_BINARY_OPERATORS = {"AND", "OR"}
_TOKEN_PATTERN = re.compile(r"\s*(AND\b|OR\b|NOT\b|[A-Za-z_][A-Za-z0-9_]*|[()])", re.IGNORECASE)


def _canonical_expression(expression: Expression) -> Expression:
    operator = expression[0]
    if operator == "VAR":
        return expression
    if operator == "NOT":
        return ("NOT", _canonical_expression(expression[1]))

    children = []
    for child in expression[1:]:
        child = _canonical_expression(child)
        if child[0] == operator:
            children.extend(child[1:])
        else:
            children.append(child)
    children = sorted(set(children), key=_expression_key)
    if len(children) == 1:
        return children[0]
    return (operator, *children)


def _expression_key(expression: Expression) -> str:
    """Return a compact identity after the configured structural symmetries."""
    return json.dumps(expression, separators=(",", ":"))


def _expression_from_key(key: str) -> Optional[Expression]:
    try:
        value = json.loads(key)
    except (json.JSONDecodeError, TypeError):
        return None

    def convert(item: Any) -> Optional[Expression]:
        if not isinstance(item, list) or not item:
            return None
        operator = item[0]
        if operator == "VAR" and len(item) == 2 and isinstance(item[1], str):
            return ("VAR", item[1])
        if operator == "NOT" and len(item) == 2:
            child = convert(item[1])
            return ("NOT", child) if child is not None else None
        if operator in _BINARY_OPERATORS and len(item) >= 3:
            children = [convert(child) for child in item[1:]]
            if any(child is None for child in children):
                return None
            return (operator, *children)
        return None

    expression = convert(value)
    if expression is None:
        return None
    expression = _canonical_expression(expression)
    return expression if _expression_key(expression) == key else None


def _tokenize(formula: str) -> Optional[list[str]]:
    tokens = []
    position = 0
    while position < len(formula):
        match = _TOKEN_PATTERN.match(formula, position)
        if match is None:
            if formula[position:].strip():
                return None
            break
        token = match.group(1)
        tokens.append(token.upper() if token.upper() in {"AND", "OR", "NOT"} else token)
        position = match.end()
    return tokens


class _ExpressionParser:
    def __init__(self, tokens: list[str], variables: set[str], operators: set[str]):
        self.tokens = tokens
        self.variables = variables
        self.operators = operators
        self.position = 0

    def parse(self) -> Optional[Expression]:
        expression = self._parse_or()
        if expression is None or self.position != len(self.tokens):
            return None
        return expression

    def _accept(self, token: str) -> bool:
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False

    def _parse_or(self) -> Optional[Expression]:
        left = self._parse_and()
        if left is None:
            return None
        children = [left]
        while self._accept("OR"):
            if "OR" not in self.operators:
                return None
            right = self._parse_and()
            if right is None:
                return None
            children.append(right)
        return ("OR", *children) if len(children) > 1 else left

    def _parse_and(self) -> Optional[Expression]:
        left = self._parse_unary()
        if left is None:
            return None
        children = [left]
        while self._accept("AND"):
            if "AND" not in self.operators:
                return None
            right = self._parse_unary()
            if right is None:
                return None
            children.append(right)
        return ("AND", *children) if len(children) > 1 else left

    def _parse_unary(self) -> Optional[Expression]:
        if self._accept("NOT"):
            if "NOT" not in self.operators:
                return None
            child = self._parse_unary()
            return ("NOT", child) if child is not None else None
        if self._accept("("):
            expression = self._parse_or()
            return expression if expression is not None and self._accept(")") else None
        if self.position >= len(self.tokens):
            return None
        variable = self.tokens[self.position]
        if variable not in self.variables:
            return None
        self.position += 1
        return ("VAR", variable)


def _expression_depth(expression: Expression) -> int:
    if expression[0] == "VAR":
        return 0
    return 1 + max(_expression_depth(child) for child in expression[1:])


def _parse_formula(
    formula: str,
    variables: list[str] | tuple[str, ...] = ("x", "y"),
    operators: list[str] | tuple[str, ...] = ("AND", "OR", "NOT"),
    max_depth: Optional[int] = None,
) -> Optional[Expression]:
    tokens = _tokenize(formula)
    if not tokens:
        return None
    expression = _ExpressionParser(tokens, set(variables), {operator.upper() for operator in operators}).parse()
    if expression is None or (max_depth is not None and _expression_depth(expression) > max_depth):
        return None
    return _canonical_expression(expression)


def _format_expression(expression: Expression, parent_precedence: int = 0) -> str:
    operator = expression[0]
    if operator == "VAR":
        return expression[1]

    precedence = {"OR": 1, "AND": 2, "NOT": 3}[operator]
    if operator == "NOT":
        child = expression[1]
        child_text = _format_expression(child, precedence)
        text = f"NOT {child_text}"
    else:
        text = f" {operator} ".join(_format_expression(child, precedence) for child in expression[1:])
    return f"({text})" if precedence < parent_precedence else text


def _extract_expression_line(solution_str: str) -> Optional[str]:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if answer_matches:
        solution_str = answer_matches[-1]

    expression_matches = re.findall(r"(?im)^\s*Expression\s*:\s*(.+?)\s*$", solution_str)
    if expression_matches:
        return expression_matches[-1].strip()
    lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _parse_prediction(
    solution_str: str,
    variables: list[str] | tuple[str, ...] = ("x", "y"),
    operators: list[str] | tuple[str, ...] = ("AND", "OR", "NOT"),
    max_depth: Optional[int] = None,
) -> Optional[Expression]:
    formula = _extract_expression_line(solution_str)
    if not formula:
        return None
    formula = formula.strip().strip("`").strip().rstrip(".;").strip()
    return _parse_formula(formula, variables=variables, operators=operators, max_depth=max_depth)


def _get_ground_truth_list(ground_truth: Any) -> list[Any]:
    if isinstance(ground_truth, dict):
        return ground_truth.get("ground_truth_list", [])
    if isinstance(ground_truth, list):
        return ground_truth
    return []


def _ground_truth_expression(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        formula = item.get("formula")
        return str(formula) if formula is not None else None
    return str(item) if isinstance(item, str) else None


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Return structural-expression accuracy and mode-recovery fields."""
    ground_truth_list = _get_ground_truth_list(ground_truth)
    variables = ground_truth.get("variables", ["x", "y"]) if isinstance(ground_truth, dict) else ["x", "y"]
    operators = (
        ground_truth.get("operators", ["AND", "OR", "NOT"])
        if isinstance(ground_truth, dict)
        else ["AND", "OR", "NOT"]
    )
    max_depth = ground_truth.get("max_depth") if isinstance(ground_truth, dict) else None
    if max_depth is not None:
        max_depth = int(max_depth)

    valid_expressions = set()
    for item in ground_truth_list:
        formula = _ground_truth_expression(item)
        if formula is None:
            continue
        expression = _parse_formula(formula, variables=variables, operators=operators, max_depth=max_depth)
        if expression is not None:
            valid_expressions.add(expression)

    prediction = _parse_prediction(
        solution_str,
        variables=variables,
        operators=operators,
        max_depth=max_depth,
    )
    if prediction is None:
        return {
            "score": format_score,
            "acc": format_score,
            "boolean_parse_success": 0.0,
            "boolean_valid": 0.0,
            "boolean_pred": "",
            "boolean_n_ground_truths": len(valid_expressions),
        }

    valid_score = score if prediction in valid_expressions else format_score
    return {
        "score": valid_score,
        "acc": valid_score,
        "boolean_parse_success": 1.0,
        "boolean_valid": float(prediction in valid_expressions),
        "boolean_pred": _expression_key(prediction),
        "boolean_n_ground_truths": len(valid_expressions),
    }


__all__ = ["compute_score"]
