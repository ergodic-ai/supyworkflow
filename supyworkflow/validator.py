"""Script validation — catches errors before execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("supyworkflow")


@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    error_type: str = ""
    line: int | None = None


def validate_script(source: str) -> ValidationResult:
    """Validate a workflow script without executing it.

    Checks:
    1. Syntax (compile check)
    2. Basic structural issues detectable via AST

    Returns ValidationResult with error details if invalid.
    """
    # 1. Compile check — catches syntax errors
    try:
        compile(source, "<validation>", "exec")
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            error=f"SyntaxError: {e.msg}",
            error_type="syntax",
            line=e.lineno,
        )

    # 2. Check for common Pydantic forward reference issues
    # If a BaseModel class references another class defined later, it will fail at runtime
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Already caught above, shouldn't happen
        return ValidationResult(valid=True)

    # Collect class names and their definition order
    class_names: list[str] = []
    class_bases: dict[str, list[str]] = {}
    class_field_types: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
            # Check base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
            class_bases[node.name] = bases

            # Check field type annotations for forward references
            field_types = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.annotation:
                    _collect_type_names(item.annotation, field_types)
            class_field_types[node.name] = field_types

    # Check if any class references another class defined later
    for i, cls_name in enumerate(class_names):
        later_classes = set(class_names[i + 1:])
        field_refs = set(class_field_types.get(cls_name, []))
        forward_refs = field_refs & later_classes
        if forward_refs:
            return ValidationResult(
                valid=False,
                error=f"Class '{cls_name}' references {forward_refs} which is defined later. "
                      f"Move {forward_refs} above '{cls_name}' or merge them into one model.",
                error_type="forward_reference",
            )

    return ValidationResult(valid=True)


def _collect_type_names(node: object, names: list[str]) -> None:
    """Recursively collect type annotation names from an AST node."""
    import ast
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Subscript):
        _collect_type_names(node.value, names)
        _collect_type_names(node.slice, names)
    elif isinstance(node, ast.Attribute):
        pass  # e.g., typing.List — skip
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            _collect_type_names(elt, names)
