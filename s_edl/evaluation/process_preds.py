import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...}, handling nested braces.

    Uses the last \\boxed{} since CoT models often refine their answer.

    Args:
        text: Text containing a \\boxed{} expression.

    Returns:
        The content inside the last \\boxed{}, or None if not found.
    """
    # Find all \boxed{ occurrences and use the last one
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None

    # Use the last match
    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    pos = start

    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        return text[start:pos - 1].strip()
    return None


def extract_last_number(text: str) -> str | None:
    """Extract the last number from text.

    Matches integers, decimals, and negative numbers.

    Args:
        text: Text containing numbers.

    Returns:
        The last number as a string, or None if no number found.
    """
    # Match integers, decimals, and negative numbers
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]
    return None


def extract_prediction(text: str) -> str:
    """Extract prediction from model output.

    First tries to extract from \\boxed{}, then falls back to last number.

    Args:
        text: Model-generated text.

    Returns:
        Extracted prediction, or empty string if extraction fails.
    """
    result = extract_boxed_answer(text)
    if result is not None:
        return result

    result = extract_last_number(text)
    if result is not None:
        return result

    return ''
