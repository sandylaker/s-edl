"""Unit tests for tools.misc.generate_stem_cot."""
import pytest

from tools.misc.generate_stem_cot import (
    SampledAnswer,
    build_output_row,
    choose_best_candidate,
    parse_args,
    parse_equivalent_indices,
)


def test_cli_num_answers_per_question_defaults_and_rejects_num_samples() -> None:
    """CLI should accept --num-answers-per-question and reject --num-samples."""
    args = parse_args(['--num-answers-per-question', '3'])
    assert args.num_answers_per_question == 3

    # Default should be 2
    args = parse_args([])
    assert args.num_answers_per_question == 2

    with pytest.raises(SystemExit):
        parse_args(['--num-samples', '2'])


def test_default_preset_is_chemistryqa() -> None:
    """Default preset should resolve to the ChemistryQA dataset and output path."""
    args = parse_args([])
    assert args.dataset_preset == 'chemistryqa'
    assert args.dataset == 'avaliev/ChemistryQA'
    assert args.output == 'data/chemistryqa_cot.jsonl'


def test_only_kept_presets_are_available() -> None:
    """Only the ChemistryQA and FormulaReasoning presets should be selectable."""
    args = parse_args(['--dataset-preset', 'formulareasoning'])
    assert args.dataset == 'cat-overflow/FormulaReasoning'

    for removed in ('webinstruct', 'drop'):
        with pytest.raises(SystemExit):
            parse_args(['--dataset-preset', removed])


def test_parse_equivalent_indices_listwise() -> None:
    """Listwise parser should parse valid integer lists."""
    text = 'Some reasoning\nEquivalent indices: [2, 0, 2]'
    parsed = parse_equivalent_indices(text, num_candidates=3)
    assert parsed == [2, 0]


def test_parse_equivalent_indices_malformed_returns_none() -> None:
    """Malformed listwise output should return None."""
    parsed = parse_equivalent_indices('Equivalent indices: not-a-list', num_candidates=3)
    assert parsed is None


def test_shortest_correct_selection_with_tie_break() -> None:
    """Best candidate should be shortest correct; ties break by lower index."""
    candidates = [
        SampledAnswer(
            index=0,
            raw_response='r0',
            formal_solution='s0',
            final_answer='a0',
            formal_tokens=12,
        ),
        SampledAnswer(
            index=1,
            raw_response='r1',
            formal_solution='s1',
            final_answer='a1',
            formal_tokens=7,
        ),
        SampledAnswer(
            index=2,
            raw_response='r2',
            formal_solution='s2',
            final_answer='a2',
            formal_tokens=7,
        ),
    ]
    best = choose_best_candidate(candidates, correct_indices=[0, 1, 2])
    assert best is not None
    assert best.index == 1


def test_output_row_schema_is_exact() -> None:
    """Output row should contain exactly id, question, answer, and solution."""
    row = {
        'id': 1,
        'question': 'What is 1+1?',
        'answer': '2',
        'extra': 'ignored',
    }
    output = build_output_row(row, 'Final answer: \\boxed{2}')
    assert list(output.keys()) == ['id', 'question', 'answer', 'solution']
    assert output['solution'] == 'Final answer: \\boxed{2}'


def test_output_row_includes_hint_when_given() -> None:
    """The optional hint field should be appended only when provided."""
    row = {'id': 1, 'question': 'What is 1+1?', 'answer': '2'}
    output = build_output_row(row, 'Final answer: \\boxed{2}', hint='Addition.')
    assert list(output.keys()) == ['id', 'question', 'answer', 'solution', 'hint']
    assert output['hint'] == 'Addition.'
