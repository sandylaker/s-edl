"""Tests for s_edl.datasets.math_prompts module."""
import json
import re
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from s_edl.datasets import BaseMathPromptBuilder, GSM8KPromptBuilder, MATHPromptBuilder
from s_edl.datasets.answer_extractors import extract_math_reasoning


class TestGSM8KPromptBuilder:
    """Tests for GSM8KPromptBuilder class."""

    def test_unsupported_prompt_type_raises(self) -> None:
        """Test that prompt types other than 'concise' and 'cot' are rejected."""
        with pytest.raises(ValueError, match="prompt_type must be 'concise' or 'cot'"):
            GSM8KPromptBuilder(prompt_type='sentence')

    def test_extract_final_answer(self) -> None:
        """Test extraction of final answer from GSM8K format."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        # Standard format
        answer = 'Some reasoning steps.\n#### 42'
        assert builder.extract_final_answer(answer) == '42'

        # With extra whitespace
        answer = 'Reasoning.\n####   123  '
        assert builder.extract_final_answer(answer) == '123'

        # No delimiter - returns stripped answer
        answer = 'Just an answer'
        assert builder.extract_final_answer(answer) == 'Just an answer'

    def test_extract_reasoning(self) -> None:
        """Test extraction of reasoning from GSM8K format."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        answer = 'Step 1: Do this.\nStep 2: Do that.\n#### 42'
        assert builder.extract_reasoning(answer) == 'Step 1: Do this.\nStep 2: Do that.'

    def test_format_answer_concise(self) -> None:
        """Test answer formatting for concise prompt type."""
        builder = GSM8KPromptBuilder(prompt_type='concise')

        answer = 'Some reasoning.\n#### 42'
        formatted = builder.format_answer(answer)
        assert formatted == '\\boxed{42}'

    def test_format_answer_cot(self) -> None:
        """Test answer formatting for CoT prompt type."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        answer = 'Step 1: Calculate.\n#### 42'
        formatted = builder.format_answer(answer)
        assert 'Reasoning: Step 1: Calculate.' in formatted
        assert formatted.endswith('My answer is: \\boxed{42}')

    def test_build_messages_zero_shot(self, qwen3_tokenizer: AutoTokenizer) -> None:
        """Test building messages with zero few-shot examples."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer=None,
            few_shot_examples=[],
        )

        assert len(messages) == 2  # system + user
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert 'Question: What is 2 + 2?' in messages[1]['content']
        assert 'Examples:' not in messages[1]['content']

    def test_build_messages_with_few_shot(self, qwen3_tokenizer: AutoTokenizer) -> None:
        """Test building messages with few-shot examples."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        few_shot = [
            ('What is 1 + 1?', 'One plus one equals two.\n#### 2'),
            ('What is 3 - 1?', 'Three minus one equals two.\n#### 2'),
        ]

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer=None,
            few_shot_examples=few_shot,
        )

        assert len(messages) == 2  # system + user
        user_content = messages[1]['content']
        assert 'Examples:' in user_content
        assert 'Question: What is 1 + 1?' in user_content
        assert 'Question: What is 3 - 1?' in user_content
        assert 'Question: What is 2 + 2?' in user_content

    def test_build_messages_with_answer(self, qwen3_tokenizer: AutoTokenizer) -> None:
        """Test building messages with assistant answer."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer='Two plus two equals four.\n#### 4',
            few_shot_examples=[],
        )

        assert len(messages) == 3  # system + user + assistant
        assert messages[2]['role'] == 'assistant'
        assert 'Reasoning:' in messages[2]['content']
        assert '\\boxed{4}' in messages[2]['content']


class TestGSM8KPrepareDatasets:
    """Tests for prepare_datasets method with real GSM8K data."""

    @pytest.fixture
    def output_dir(self) -> Path:
        """Get the output directory for saving test examples."""
        path = Path(__file__).parent / '.pytest_cache' / 'prompt_examples'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_train_dataset_builds_successfully(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that training dataset builds without errors."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),  # Use subset for speed
            tokenizer=qwen3_tokenizer,
        )

        assert 'train' in result
        train = result['train']

        # Check required fields exist
        assert 'full_prompt' in train.column_names
        assert 'prefill_prompt' in train.column_names
        assert 'completion_prompt' in train.column_names
        assert 'final_answer' in train.column_names

        # Check all examples have non-empty prompts
        for i in range(len(train)):
            assert len(train[i]['full_prompt']) > 0
            assert len(train[i]['prefill_prompt']) > 0
            assert len(train[i]['completion_prompt']) > 0
            assert len(train[i]['final_answer']) > 0

    def test_train_and_test_datasets_build_successfully(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
    ) -> None:
        """Test that both train and test datasets build without errors."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=qwen3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(5)),
        )

        assert 'train' in result
        assert 'test' in result

        test = result['test']
        assert 'prefill_prompt' in test.column_names
        assert 'completion_prompt' in test.column_names
        assert 'final_answer' in test.column_names

        # Test split should have no completion prompt
        for i in range(len(test)):
            assert test[i]['completion_prompt'] is None

    def test_zero_shot_prompts(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test zero-shot prompts and save examples to file."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(3)),
        )

        train = result['train']
        test = result['test']

        # Verify zero-shot: no "Examples:" in prompts
        for i in range(len(train)):
            assert 'Examples:' not in train[i]['full_prompt']
            assert 'Examples:' not in train[i]['prefill_prompt']

        for i in range(len(test)):
            assert 'Examples:' not in test[i]['prefill_prompt']

        # Save 4 examples each to file
        examples = {
            'train': [
                {
                    'full_prompt': train[i]['full_prompt'],
                    'prefill_prompt': train[i]['prefill_prompt'],
                    'completion_prompt': train[i]['completion_prompt'],
                    'final_answer': train[i]['final_answer'],
                } for i in range(min(4, len(train)))
            ],
            'test': [
                {
                    'prefill_prompt': test[i]['prefill_prompt'],
                    'completion_prompt': test[i]['completion_prompt'],
                    'final_answer': test[i]['final_answer'],
                } for i in range(min(4, len(test)))
            ],
        }

        with open(output_dir / 'zero_shot_cot.json', 'w') as f:
            json.dump(examples, f, indent=2)

    def test_four_shot_prompts(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test four-shot prompts and save examples to file."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        # Use enough training examples to have 4 few-shot
        train_subset = gsm8k_train_dataset.select(range(20))
        test_subset = gsm8k_test_dataset.select(range(4))

        result = builder.prepare_datasets(
            train_dataset=train_subset,
            tokenizer=qwen3_tokenizer,
            test_dataset=test_subset,
        )

        train = result['train']
        test = result['test']

        # Test dataset should always have 4 few-shot examples
        for i in range(len(test)):
            # Count "Question:" occurrences - should be 5 (4 few-shot + 1 actual)
            question_count = test[i]['prefill_prompt'].count('Question:')
            assert question_count == 5, f"Expected 5 questions (4 few-shot + 1 actual), got {question_count}"
            assert 'Examples:' in test[i]['prefill_prompt']

        # Save 4 examples each to file
        examples = {
            'train': [
                {
                    'full_prompt': train[i]['full_prompt'],
                    'prefill_prompt': train[i]['prefill_prompt'],
                    'completion_prompt': train[i]['completion_prompt'],
                    'final_answer': train[i]['final_answer'],
                } for i in range(4)
            ],
            'test': [
                {
                    'prefill_prompt': test[i]['prefill_prompt'],
                    'completion_prompt': test[i]['completion_prompt'],
                    'final_answer': test[i]['final_answer'],
                } for i in range(min(4, len(test)))
            ],
        }

        with open(output_dir / 'four_shot_cot.json', 'w') as f:
            json.dump(examples, f, indent=2)

    def test_concise_prompts(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test concise prompt type and save examples."""
        builder = GSM8KPromptBuilder(prompt_type='concise', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(20)),
            tokenizer=qwen3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(4)),
        )

        train = result['train']
        test = result['test']

        # Concise prompts should have \boxed{} but no "Reasoning:"
        assert '\\boxed{' in train[0]['full_prompt']
        assert 'Reasoning:' not in train[0]['full_prompt']

        # Save 4 examples each to file
        examples = {
            'train': [
                {
                    'full_prompt': train[i]['full_prompt'],
                    'prefill_prompt': train[i]['prefill_prompt'],
                    'completion_prompt': train[i]['completion_prompt'],
                    'final_answer': train[i]['final_answer'],
                } for i in range(4)
            ],
            'test': [
                {
                    'prefill_prompt': test[i]['prefill_prompt'],
                    'completion_prompt': test[i]['completion_prompt'],
                    'final_answer': test[i]['final_answer'],
                } for i in range(min(4, len(test)))
            ],
        }

        with open(output_dir / 'four_shot_concise.json', 'w') as f:
            json.dump(examples, f, indent=2)

    def test_zero_shot_concise_prompts(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test zero-shot concise prompts and save examples."""
        builder = GSM8KPromptBuilder(prompt_type='concise', max_few_shot=0, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(4)),
        )

        train = result['train']
        test = result['test']

        # Zero-shot concise: no "Examples:", no "Reasoning:", has \boxed{}
        for i in range(len(train)):
            assert 'Examples:' not in train[i]['full_prompt']
            assert 'Reasoning:' not in train[i]['full_prompt']
            assert '\\boxed{' in train[i]['full_prompt']

        for i in range(len(test)):
            assert 'Examples:' not in test[i]['prefill_prompt']

        # Save 4 examples each to file
        examples = {
            'train': [
                {
                    'full_prompt': train[i]['full_prompt'],
                    'prefill_prompt': train[i]['prefill_prompt'],
                    'completion_prompt': train[i]['completion_prompt'],
                    'final_answer': train[i]['final_answer'],
                } for i in range(min(4, len(train)))
            ],
            'test': [
                {
                    'prefill_prompt': test[i]['prefill_prompt'],
                    'completion_prompt': test[i]['completion_prompt'],
                    'final_answer': test[i]['final_answer'],
                } for i in range(min(4, len(test)))
            ],
        }

        with open(output_dir / 'zero_shot_concise.json', 'w') as f:
            json.dump(examples, f, indent=2)

    def test_test_fewshot_comes_from_train(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
    ) -> None:
        """Test that few-shot examples in test dataset come from training set."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        train_subset = gsm8k_train_dataset.select(range(20))
        test_subset = gsm8k_test_dataset.select(range(3))

        result = builder.prepare_datasets(
            train_dataset=train_subset,
            tokenizer=qwen3_tokenizer,
            test_dataset=test_subset,
        )

        test = result['test']

        # Get all training questions
        train_questions = set(train_subset['question'])

        # For each test example, check that few-shot questions are from training
        for i in range(len(test)):
            prefill = test[i]['prefill_prompt']

            # The test question should NOT be in training set
            test_question = test_subset[i]['question']
            assert test_question not in train_questions

            # Extract questions from prefill (between "Question: " and "\nAnswer:")
            # The last question is the actual test question
            questions_in_prefill = re.findall(r'Question: (.+?)(?=\nAnswer:)', prefill, re.DOTALL)

            # All but the last question should be from training set
            few_shot_questions = questions_in_prefill[:-1]
            for fs_q in few_shot_questions:
                assert fs_q.strip() in train_questions, f"Few-shot question not in training: {fs_q[:50]}..."

    def test_full_prompt_extends_prefill_prompt(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that full_prompt == prefill_prompt + completion_prompt."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=qwen3_tokenizer,
        )

        train = result['train']

        for i in range(len(train)):
            full = train[i]['full_prompt']
            prefill = train[i]['prefill_prompt']
            completion = train[i]['completion_prompt']

            assert prefill.endswith('<|im_start|>assistant\n')
            assert completion is not None
            assert full == prefill + completion

    def test_prefill_prompt_consistency_train_vs_test(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
    ) -> None:
        """Test that prefill prompts have consistent format between train and test."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=qwen3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(5)),
        )

        train = result['train']
        test = result['test']

        # Both should end with the same generation prompt suffix
        expected_suffix = '<|im_start|>assistant\n'

        for i in range(len(train)):
            assert train[i]['prefill_prompt'].endswith(expected_suffix)

        for i in range(len(test)):
            assert test[i]['prefill_prompt'].endswith(expected_suffix)

    def test_train_fewshot_excludes_self(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that training few-shot examples exclude the current example."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        train_subset = gsm8k_train_dataset.select(range(10))

        result = builder.prepare_datasets(
            train_dataset=train_subset,
            tokenizer=qwen3_tokenizer,
        )

        train = result['train']

        for i in range(len(train)):
            current_question = train_subset[i]['question']
            prefill = train[i]['prefill_prompt']

            # Extract all questions from prefill
            questions_in_prefill = re.findall(r'Question: (.+?)(?=\nAnswer:)', prefill, re.DOTALL)

            # The current question should appear exactly once (as the last/actual question)
            # But it should NOT appear in the few-shot examples (all but last)
            few_shot_questions = [q.strip() for q in questions_in_prefill[:-1]]
            assert current_question not in few_shot_questions, \
                f"Example {i}'s question appears in its own few-shot examples"

    def test_full_prompt_ends_without_im_end(
        self,
        qwen3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that full_prompt ends with <|endoftext|> directly, not <|im_end|><|endoftext|>.

        For Qwen3-Base, we skip <|im_end|> for the last assistant message because:
        1. Qwen3-Base hasn't seen <|im_end|> during pretraining
        2. Embedding/lm_head are frozen during LoRA SFT
        3. Including <|im_end|> causes garbage token generation at inference time

        See chat_templates.py for details.
        """
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
        )

        train = result['train']

        for i in range(len(train)):
            full = train[i]['full_prompt']

            # Should end with <|endoftext|>
            assert full.endswith('<|endoftext|>'), \
                f"Example {i} should end with <|endoftext|>, got: ...{full[-50:]}"

            # Should NOT have <|im_end|> right before <|endoftext|>
            assert '<|im_end|><|endoftext|>' not in full, \
                f"Example {i} should not have <|im_end|><|endoftext|>, this causes garbage tokens"

            # Should NOT have <|im_end|> followed by newline then <|endoftext|>
            assert '<|im_end|>\n<|endoftext|>' not in full, \
                f"Example {i} should not have <|im_end|> before <|endoftext|>"


# ---------------------------------------------------------------------------
# extract_math_reasoning unit tests
# ---------------------------------------------------------------------------


class TestExtractMathReasoning:
    """Tests for the extract_math_reasoning function."""

    def test_single_boxed_at_end(self) -> None:
        """Last \\boxed{} is unwrapped; its content is preserved inline."""
        answer = r'We compute $1 + 2 = \boxed{3}$.'
        result = extract_math_reasoning(answer)
        assert r'\boxed{' not in result
        assert '3' in result

    def test_no_boxed_returns_stripped(self) -> None:
        """Returns stripped answer unchanged when no \\boxed{} present."""
        assert extract_math_reasoning('  The answer is 42.  ') == 'The answer is 42.'

    def test_nested_braces_unwrapped_correctly(self) -> None:
        """Nested braces inside \\boxed{} are handled via depth tracking."""
        answer = r'The probability is $\boxed{\frac{1}{4}}$.'
        result = extract_math_reasoning(answer)
        assert r'\boxed{' not in result
        assert r'\frac{1}{4}' in result

    def test_multiple_boxed_only_last_unwrapped(self) -> None:
        """Only the last \\boxed{} is unwrapped; earlier ones are preserved."""
        answer = r'Step one: $\boxed{2}$. Final answer: $\boxed{3}$.'
        result = extract_math_reasoning(answer)
        assert result.count(r'\boxed{') == 1
        assert r'\boxed{2}' in result
        assert r'\boxed{3}' not in result
        assert '3' in result

    def test_preserves_text_surrounding_boxed(self) -> None:
        """Text surrounding the unwrapped \\boxed{} is preserved."""
        answer = r'So $x = \boxed{42}$ is our answer.'
        result = extract_math_reasoning(answer)
        assert 'So $x = ' in result
        assert '42' in result
        assert 'is our answer.' in result

    def test_content_with_equals_sign(self) -> None:
        """Content such as 'x=2' inside \\boxed{} is preserved."""
        answer = r'We find $\boxed{x=2}$.'
        result = extract_math_reasoning(answer)
        assert 'x=2' in result
        assert r'\boxed{' not in result


# ---------------------------------------------------------------------------
# MATHPromptBuilder unit tests (no tokenizer required)
# ---------------------------------------------------------------------------


class TestMATHPromptBuilderUnit:
    """Unit tests for MATHPromptBuilder that do not require a tokenizer."""

    def test_is_subclass_of_base(self) -> None:
        assert issubclass(MATHPromptBuilder, BaseMathPromptBuilder)

    def test_extract_final_answer_from_boxed(self) -> None:
        builder = MATHPromptBuilder(prompt_type='cot')
        assert builder.extract_final_answer(r'We get $x = \boxed{\frac{1}{4}}$.') == r'\frac{1}{4}'

    def test_extract_final_answer_no_boxed_fallback(self) -> None:
        """Falls back to the stripped answer string when no \\boxed{} present."""
        builder = MATHPromptBuilder(prompt_type='cot')
        assert builder.extract_final_answer('  42  ') == '42'

    def test_extract_reasoning_unwraps_last_boxed(self) -> None:
        builder = MATHPromptBuilder(prompt_type='cot')
        result = builder.extract_reasoning(r'We compute $x = \boxed{42}$.')
        assert r'\boxed{42}' not in result
        assert '42' in result

    def test_format_answer_cot_has_reasoning_and_answer(self) -> None:
        builder = MATHPromptBuilder(prompt_type='cot')
        formatted = builder.format_answer(r'We get $\boxed{3}$.')
        assert 'Reasoning:' in formatted
        assert 'My answer is:' in formatted
        assert r'\boxed{3}' in formatted

    def test_format_answer_cot_no_duplicate_boxed(self) -> None:
        """CoT formatted answer contains exactly one \\boxed{} when solution has one."""
        builder = MATHPromptBuilder(prompt_type='cot')
        answer = r'We solve to find $x = \frac{1}{4}$. The answer is $\boxed{\frac{1}{4}}$.'
        formatted = builder.format_answer(answer)
        assert formatted.count(r'\boxed{') == 1
        assert r'\boxed{\frac{1}{4}}' in formatted

    def test_format_answer_concise(self) -> None:
        builder = MATHPromptBuilder(prompt_type='concise')
        assert builder.format_answer(r'We get $\boxed{42}$.') == r'\boxed{42}'

    def test_default_question_field_is_problem(self) -> None:
        import inspect
        sig = inspect.signature(MATHPromptBuilder.prepare_datasets)
        assert sig.parameters['question_field'].default == 'problem'

    def test_default_answer_field_is_solution(self) -> None:
        import inspect
        sig = inspect.signature(MATHPromptBuilder.prepare_datasets)
        assert sig.parameters['answer_field'].default == 'solution'

    def test_build_messages_zero_shot_no_answer(self) -> None:
        builder = MATHPromptBuilder(prompt_type='cot')
        messages = builder.build_messages(
            question=r'What is $1 + 1$?',
            answer=None,
            few_shot_examples=[],
        )
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert r'What is $1 + 1$?' in messages[1]['content']
        assert 'Examples:' not in messages[1]['content']

    def test_build_messages_with_answer_adds_assistant_turn(self) -> None:
        builder = MATHPromptBuilder(prompt_type='cot')
        messages = builder.build_messages(
            question=r'What is $1 + 1$?',
            answer=r'We get $\boxed{2}$.',
            few_shot_examples=[],
        )
        assert len(messages) == 3
        assert messages[2]['role'] == 'assistant'
        assert 'Reasoning:' in messages[2]['content']
        assert r'\boxed{2}' in messages[2]['content']


# ---------------------------------------------------------------------------
# MATHPromptBuilder integration tests (require tokenizer + real MATH JSONL)
# ---------------------------------------------------------------------------


class TestMATHPrepareDatasets:
    """Integration tests for MATHPromptBuilder.prepare_datasets using the local MATH JSONL."""

    @pytest.fixture
    def output_dir(self) -> Path:
        path = Path(__file__).parent / '.pytest_cache' / 'prompt_examples'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_train_builds_successfully(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """Training dataset builds without errors on a real MATH subset."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(10)),
            tokenizer=qwen3_tokenizer,
        )
        assert 'train' in result
        train = result['train']
        for col in ('full_prompt', 'prefill_prompt', 'completion_prompt', 'final_answer'):
            assert col in train.column_names
        for i in range(len(train)):
            assert len(train[i]['full_prompt']) > 0
            assert len(train[i]['final_answer']) > 0

    def test_train_and_test_build_successfully(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """Both train and test datasets build without errors."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        subset = math_train_dataset.select(range(10))
        result = builder.prepare_datasets(
            train_dataset=subset,
            tokenizer=qwen3_tokenizer,
            test_dataset=subset.select(range(5)),
        )
        assert 'train' in result
        assert 'test' in result
        test = result['test']
        for col in ('question', 'prefill_prompt', 'final_answer'):
            assert col in test.column_names
        for i in range(len(test)):
            assert test[i]['completion_prompt'] is None

    def test_question_field_normalized_in_test(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """Test dataset emits 'question' column (normalized from 'problem' field)."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        subset = math_train_dataset.select(range(5))
        result = builder.prepare_datasets(
            train_dataset=subset,
            tokenizer=qwen3_tokenizer,
            test_dataset=subset,
        )
        test = result['test']
        assert 'question' in test.column_names
        for i in range(len(test)):
            assert test[i]['question'] == subset[i]['problem']

    def test_final_answer_extracted_correctly(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """Final answers are extracted from \\boxed{} in the MATH solutions."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
        )
        train = result['train']
        # First two examples have known answers from the local JSONL
        assert train[0]['final_answer'] == r'\frac{1}{4}'
        assert train[1]['final_answer'] == '3003'

    def test_cot_completion_no_duplicate_boxed(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """The specific \\boxed{final_answer} appears exactly once in the CoT completion.

        After extract_math_reasoning unwraps the last \\boxed{} in the solution, the only \\boxed{final_answer} in the
        completion should be the one added by format_answer in 'My answer is: \\boxed{final_answer}'.
        """
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(10)),
            tokenizer=qwen3_tokenizer,
        )
        train = result['train']
        for i in range(len(train)):
            completion = train[i]['completion_prompt']
            final_answer = train[i]['final_answer']
            expected_boxed = f'\\boxed{{{final_answer}}}'
            count = completion.count(expected_boxed)
            assert count == 1, (
                f'Example {i}: {expected_boxed!r} appears {count} times in completion; '
                f'expected exactly 1 (in "My answer is: ...")'
            )

    def test_zero_shot_no_examples_in_prompt(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """Zero-shot prompts do not contain 'Examples:'."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
        )
        for i in range(len(result['train'])):
            assert 'Examples:' not in result['train'][i]['prefill_prompt']

    def test_full_prompt_extends_prefill_prompt(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """full_prompt == prefill_prompt + completion_prompt for every example."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
        )
        for i, example in enumerate(result['train']):
            assert example['prefill_prompt'].endswith('<|im_start|>assistant\n')
            assert example['full_prompt'] == example['prefill_prompt'] + example['completion_prompt']

    def test_full_prompt_ends_without_im_end(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
    ) -> None:
        """full_prompt ends with <|endoftext|> directly, not <|im_end|><|endoftext|>."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        result = builder.prepare_datasets(
            train_dataset=math_train_dataset.select(range(5)),
            tokenizer=qwen3_tokenizer,
        )
        for i, example in enumerate(result['train']):
            full = example['full_prompt']
            assert full.endswith('<|endoftext|>'), \
                f'Example {i} should end with <|endoftext|>, got: ...{full[-50:]}'
            assert '<|im_end|><|endoftext|>' not in full
            assert '<|im_end|>\n<|endoftext|>' not in full

    def test_save_zero_shot_cot_examples(
        self,
        qwen3_tokenizer: AutoTokenizer,
        math_train_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Build zero-shot CoT examples and save to file for manual inspection."""
        builder = MATHPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)
        subset = math_train_dataset.select(range(10))
        result = builder.prepare_datasets(
            train_dataset=subset,
            tokenizer=qwen3_tokenizer,
            test_dataset=subset.select(range(5)),
        )
        train = result['train']
        test = result['test']
        examples = {
            'train': [
                {
                    'full_prompt': train[i]['full_prompt'],
                    'prefill_prompt': train[i]['prefill_prompt'],
                    'completion_prompt': train[i]['completion_prompt'],
                    'final_answer': train[i]['final_answer'],
                } for i in range(min(4, len(train)))
            ],
            'test': [
                {
                    'question': test[i]['question'],
                    'prefill_prompt': test[i]['prefill_prompt'],
                    'final_answer': test[i]['final_answer'],
                } for i in range(min(4, len(test)))
            ],
        }
        with open(output_dir / 'math_zero_shot_cot.json', 'w') as f:
            json.dump(examples, f, indent=2)
