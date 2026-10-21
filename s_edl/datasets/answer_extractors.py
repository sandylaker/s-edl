"""Dataset-specific answer extraction functions."""
import re


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the final answer after '####' delimiter.

    Args:
        answer: Raw GSM8K answer string.

    Returns:
        The extracted final answer as a string.
    """
    match = re.search(r'####\s*(.+?)\s*$', answer)
    if match:
        return match.group(1).strip()
    return answer.strip()


def extract_gsm8k_reasoning(answer: str) -> str:
    """Extract reasoning steps before '####' delimiter.

    Args:
        answer: Raw GSM8K answer string.

    Returns:
        The extracted reasoning steps as a string.
    """
    return re.sub(r'\n*####.+$', '', answer).strip()


def extract_math_reasoning(answer: str) -> str:
    """Return solution with the last \\boxed{...} unwrapped (content preserved).

    Replaces the last ``\\boxed{content}`` with just ``content`` so that the
    formatted CoT completion does not contain a duplicate ``\\boxed{}`` block
    before the explicit "My answer is: \\boxed{answer}" line.

    Args:
        answer: Raw MATH solution string.

    Returns:
        The solution with the last \\boxed{} replaced by its plain content.
    """
    matches = list(re.finditer(r'\\boxed\s*\{', answer))
    if not matches:
        return answer.strip()

    last_match = matches[-1]
    depth = 1
    pos = last_match.end()
    while pos < len(answer) and depth > 0:
        if answer[pos] == '{':
            depth += 1
        elif answer[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        content = answer[last_match.end():pos - 1]
        return (answer[:last_match.start()] + content + answer[pos:]).strip()
    return answer.strip()


def extract_webinstruct_reasoning(answer: str) -> str:
    r"""Return solution with the last \\boxed{...} unwrapped and 'Final answer:' line stripped.

    First unwraps the last ``\boxed{content}`` (via :func:`extract_math_reasoning`),
    then removes the trailing ``Final answer: ...`` line to avoid redundancy
    with the ``My answer is: \boxed{...}`` suffix added by ``format_answer``.

    Args:
        answer: Raw CoT solution string (e.g. ChemistryQA or FormulaReasoning training data).

    Returns:
        The solution with the last \\boxed{} unwrapped and 'Final answer:' line removed.
    """
    reasoning = extract_math_reasoning(answer)
    return re.sub(r'\n*\s*Final answer:.*$', '', reasoning).strip()
