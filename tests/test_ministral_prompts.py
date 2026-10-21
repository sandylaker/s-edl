"""Tests for GSM8K prompt rendering with the Ministral-3-8B-Base custom chat template."""
import json
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from s_edl.datasets import GSM8KPromptBuilder


class TestMinistralChatTemplate:
    """Tests for the Mistral-style chat template rendering."""

    def test_zero_shot_template_structure(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that zero-shot messages render with [INST]/[/INST] format."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer='Two plus two is four.\n#### 4',
            few_shot_examples=[],
        )

        full = ministral3_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        assert full.startswith('[INST] ')
        assert ' [/INST] ' in full
        assert '<|im_start|>' not in full
        assert '<|im_end|>' not in full

    def test_prefill_ends_with_space(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that prefill prompt ends with a trailing space after [/INST]."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer=None,
            few_shot_examples=[],
        )

        prefill = ministral3_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        assert prefill.endswith('[/INST] ')

    def test_full_prompt_starts_with_prefill(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test the critical invariant: full_prompt starts with prefill_prompt."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages_full = builder.build_messages(
            question='What is 2 + 2?',
            answer='Two plus two is four.\n#### 4',
            few_shot_examples=[],
        )
        messages_prefill = messages_full[:-1]

        full = ministral3_tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ) + ministral3_tokenizer.eos_token

        prefill = ministral3_tokenizer.apply_chat_template(
            messages_prefill,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        assert full.startswith(prefill), (
            f'full_prompt does not start with prefill_prompt.\n'
            f'full[:100]={full[:100]!r}\nprefill[:100]={prefill[:100]!r}'
        )

        completion = full[len(prefill):]
        assert completion.endswith('</s>')
        assert '\\boxed{4}' in completion

    def test_no_double_bos(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that template does not include BOS (tokenizer adds it during tokenization)."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer=None,
            few_shot_examples=[],
        )

        rendered = ministral3_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Template output should NOT contain <s> -- tokenizer's add_bos_token handles it
        assert not rendered.startswith('<s>')
        assert '<s>' not in rendered

    def test_system_message_inlined(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that system message is inlined into the first [INST] block."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer=None,
            few_shot_examples=[],
        )

        rendered = ministral3_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # System message should appear between [INST] and the question, separated by \n\n
        assert rendered.startswith('[INST] ')
        inst_content = rendered[len('[INST] '):]
        assert '\n\n' in inst_content
        system_part = inst_content.split('\n\n')[0]
        assert 'Solve the following problem' in system_part

    def test_few_shot_rendering(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that few-shot examples render correctly within Mistral format."""
        builder = GSM8KPromptBuilder(prompt_type='cot')

        few_shot = [
            ('What is 1 + 1?', 'One plus one is two.\n#### 2'),
            ('What is 3 - 1?', 'Three minus one is two.\n#### 2'),
        ]

        messages_full = builder.build_messages(
            question='What is 2 + 2?',
            answer='Two plus two is four.\n#### 4',
            few_shot_examples=few_shot,
        )

        full = ministral3_tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ) + ministral3_tokenizer.eos_token

        prefill = ministral3_tokenizer.apply_chat_template(
            messages_full[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Still a single [INST]...[/INST] block (few-shot is in user content)
        assert full.count('[INST]') == 1
        assert full.count('[/INST]') == 1

        # Few-shot examples appear in the prompt
        assert 'Examples:' in full
        assert 'Question: What is 1 + 1?' in full
        assert 'Question: What is 3 - 1?' in full
        assert 'Question: What is 2 + 2?' in full

        # Invariant holds
        assert full.startswith(prefill)

    def test_concise_prompt_type(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test concise prompt type renders correctly with Mistral template."""
        builder = GSM8KPromptBuilder(prompt_type='concise')

        messages = builder.build_messages(
            question='What is 2 + 2?',
            answer='Some reasoning.\n#### 4',
            few_shot_examples=[],
        )

        full = ministral3_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ) + ministral3_tokenizer.eos_token

        assert '\\boxed{4}' in full
        assert 'Reasoning:' not in full
        assert full.endswith('</s>')

    def test_eos_token_is_correct(self, ministral3_tokenizer: AutoTokenizer) -> None:
        """Test that the Ministral EOS token is </s>."""
        assert ministral3_tokenizer.eos_token == '</s>'


class TestMinistralPrepareDatasets:
    """Integration tests for prepare_datasets with Ministral tokenizer and real GSM8K data."""

    @pytest.fixture
    def output_dir(self) -> Path:
        """Get the output directory for saving test examples."""
        path = Path(__file__).parent / '.pytest_cache' / 'prompt_examples'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_train_dataset_builds_successfully(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that training dataset builds without errors."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=ministral3_tokenizer,
        )

        assert 'train' in result
        train = result['train']

        assert 'full_prompt' in train.column_names
        assert 'prefill_prompt' in train.column_names
        assert 'completion_prompt' in train.column_names
        assert 'final_answer' in train.column_names

        for i in range(len(train)):
            assert len(train[i]['full_prompt']) > 0
            assert len(train[i]['prefill_prompt']) > 0
            assert len(train[i]['completion_prompt']) > 0
            assert len(train[i]['final_answer']) > 0

    def test_full_prompt_extends_prefill(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
    ) -> None:
        """Test that full_prompt == prefill_prompt + completion_prompt for all examples."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=ministral3_tokenizer,
        )

        train = result['train']

        for i in range(len(train)):
            full = train[i]['full_prompt']
            prefill = train[i]['prefill_prompt']
            completion = train[i]['completion_prompt']

            assert prefill.endswith('[/INST] ')
            assert completion is not None
            assert full == prefill + completion

    def test_train_and_test_datasets(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
    ) -> None:
        """Test that both train and test datasets build correctly."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=ministral3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(5)),
        )

        assert 'train' in result
        assert 'test' in result

        test = result['test']
        for i in range(len(test)):
            assert test[i]['completion_prompt'] is None
            assert test[i]['prefill_prompt'].endswith('[/INST] ')

    def test_prefill_uses_mistral_format(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
    ) -> None:
        """Test that all prefill prompts use [INST]/[/INST] and not ChatML."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(10)),
            tokenizer=ministral3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(5)),
        )

        for split_name in ('train', 'test'):
            ds = result[split_name]
            for i in range(len(ds)):
                prefill = ds[i]['prefill_prompt']
                assert '[INST]' in prefill
                assert '[/INST]' in prefill
                assert '<|im_start|>' not in prefill
                assert '<|im_end|>' not in prefill

    def test_zero_shot_prompts_and_save(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test zero-shot prompts and save examples for manual inspection."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=0, seed=42)

        result = builder.prepare_datasets(
            train_dataset=gsm8k_train_dataset.select(range(5)),
            tokenizer=ministral3_tokenizer,
            test_dataset=gsm8k_test_dataset.select(range(3)),
        )

        train = result['train']
        test = result['test']

        for i in range(len(train)):
            assert 'Examples:' not in train[i]['full_prompt']

        for i in range(len(test)):
            assert 'Examples:' not in test[i]['prefill_prompt']

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

        with open(output_dir / 'ministral_zero_shot_cot.json', 'w') as f:
            json.dump(examples, f, indent=2)

    def test_four_shot_prompts_and_save(
        self,
        ministral3_tokenizer: AutoTokenizer,
        gsm8k_train_dataset: Dataset,
        gsm8k_test_dataset: Dataset,
        output_dir: Path,
    ) -> None:
        """Test four-shot prompts and save examples for manual inspection."""
        builder = GSM8KPromptBuilder(prompt_type='cot', max_few_shot=4, seed=42)

        train_subset = gsm8k_train_dataset.select(range(20))
        test_subset = gsm8k_test_dataset.select(range(4))

        result = builder.prepare_datasets(
            train_dataset=train_subset,
            tokenizer=ministral3_tokenizer,
            test_dataset=test_subset,
        )

        train = result['train']
        test = result['test']

        # Test dataset should always have 4 few-shot examples
        for i in range(len(test)):
            question_count = test[i]['prefill_prompt'].count('Question:')
            assert question_count == 5, f"Expected 5 questions (4 few-shot + 1 actual), got {question_count}"
            assert 'Examples:' in test[i]['prefill_prompt']

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

        with open(output_dir / 'ministral_four_shot_cot.json', 'w') as f:
            json.dump(examples, f, indent=2)
