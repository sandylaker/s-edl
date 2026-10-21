"""Semantic equivalence verification for mathematical expressions."""
from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def parse_math_expression(text: str) -> tuple[object | None, bool]:
    """Parse a mathematical expression into SymPy form.

    Args:
        text: String containing a mathematical expression (e.g., "42", "3/4", "\\boxed{42}").

    Returns:
        Tuple of (parsed_expression, is_parseable).
        If parsing fails, returns (None, False).
    """
    try:
        # Wrap in LaTeX environment if not already
        if not any(delim in text for delim in ['$', '\\[', '\\boxed']):
            text = f'${text}$'
        parsed = parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        # parsed can be None or empty list [] for unparseable input
        return (parsed, True) if parsed else (None, False)
    except Exception:
        return (None, False)


def are_semantically_equivalent(expr1: str, expr2: str) -> bool:
    """Check if two mathematical expressions are semantically equivalent.

    Args:
        expr1: First expression string.
        expr2: Second expression string.

    Returns:
        True if expressions are semantically equivalent, False otherwise.
        Returns False if either expression is unparseable.
    """
    parsed1, ok1 = parse_math_expression(expr1)
    parsed2, ok2 = parse_math_expression(expr2)

    if not ok1 or not ok2:
        return False

    try:
        return verify(parsed1, parsed2)
    except Exception:
        return False
