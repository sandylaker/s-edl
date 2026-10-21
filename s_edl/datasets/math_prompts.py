import random
from typing import Any

from datasets import Dataset

from ..evaluation.process_preds import extract_boxed_answer
from .answer_extractors import (
    extract_gsm8k_final_answer,
    extract_gsm8k_reasoning,
    extract_math_reasoning,
    extract_webinstruct_reasoning,
)
from .registry import PROMPT_BUILDERS


class BaseMathPromptBuilder:
    """Base prompt builder for math datasets.

    Supports two prompt types:
    - "concise": Model outputs only the final answer in \\boxed{}.
    - "cot": Model reasons step-by-step then outputs answer in \\boxed{}.

    Uses single-user-message format for few-shot examples to avoid multiple
    <think></think> blocks when using models like Qwen3 with enable_thinking=False.

    Format (take Qwen3-Base as example):
        <|im_start|>system
        instruction<|im_end|>
        <|im_start|>user
        Examples:

        Question: fs_question_1
        Answer: fs_answer_1

        Question: fs_question_2
        Answer: fs_answer_2

        Question: actual_question
        Answer:<|im_end|>
        <|im_start|>assistant
        response<|endoftext|>

    Note: For Qwen3-Base, the last assistant message skips <|im_end|> and ends
    directly with <|endoftext|>. If we include <|im_end|>, the training prompt
    ends with <|im_end|><|endoftext|>, causing the SFT model to produce garbage
    tokens after \\boxed{} at inference time. This happens because Qwen3-Base
    hasn't seen <|im_end|> during pretraining, and the embedding/lm_head are
    frozen during LoRA SFT, so the model doesn't learn the "stop-generation"
    signal for <|im_end|>. See chat_templates.py for details.

    Subclasses must implement:
    - extract_final_answer(): Extract final answer from raw answer string
    - extract_reasoning(): Extract reasoning steps from raw answer string
    """

    SYSTEM_CONCISE = (
        'Solve the following problem. '
        'Provide only the final answer wrapped in \\boxed{}. '
        'Do not show any reasoning or intermediate steps.')

    SYSTEM_COT = (
        'Solve the following problem. '
        "First, reason through the problem step by step after 'Reasoning:'. "
        "Then, provide your final answer after 'My answer is:' wrapped in \\boxed{}.")

    def __init__(
        self,
        prompt_type: str,
        max_few_shot: int = 4,
        seed: int = 42,
        with_hint: bool = False,
        hint_field: str = 'hint',
    ) -> None:
        """Initialize the prompt builder.

        Args:
            prompt_type: "concise" or "cot".
            max_few_shot: Maximum number of few-shot examples.
                - Training: K ~ Uniform[0, max_few_shot]
                - Test: K = max_few_shot (fixed)
            seed: Random seed for reproducibility.
            with_hint: If True, append hint text after the main question.
                Hints are only added to the main question, not to few-shot examples.
            hint_field: Name of the dataset field containing hint text.
        """
        if prompt_type not in ('concise', 'cot'):
            raise ValueError(f"prompt_type must be 'concise' or 'cot', got {prompt_type}")

        self.prompt_type = prompt_type
        self.max_few_shot = max_few_shot
        self.seed = seed
        self.with_hint = with_hint
        self.hint_field = hint_field

    @property
    def system_message(self) -> str:
        """Get the system message for the current prompt type."""
        if self.prompt_type == 'concise':
            return self.SYSTEM_CONCISE
        return self.SYSTEM_COT

    def _format_question_with_hint(self, question: str, hint: str | None) -> str:
        """Append hint text to the question if ``with_hint`` is enabled.

        Args:
            question: The original question text.
            hint: Hint text from the dataset, or None if unavailable.

        Returns:
            The question string, optionally with appended hints.
        """
        if self.with_hint and hint:
            return (f'{question}\n\n'
                    'The following hints may be useful.\n'
                    f'Hints:\n{hint}')
        return question

    def extract_final_answer(self, answer: str) -> str:
        """Extract the final answer from raw answer string.

        Args:
            answer: Raw answer string.

        Returns:
            The extracted final answer as a string.
        """
        raise NotImplementedError('Subclasses must implement extract_final_answer()')

    def extract_reasoning(self, answer: str) -> str:
        """Extract reasoning steps from raw answer string.

        Args:
            answer: Raw answer string.

        Returns:
            The extracted reasoning steps as a string.
        """
        raise NotImplementedError('Subclasses must implement extract_reasoning()')

    def format_answer(self, answer: str) -> str:
        """Format the answer based on prompt type.

        Args:
            answer: Raw answer string.

        Returns:
            Formatted answer string (without thinking tokens - tokenizer adds them).
        """
        final_answer = self.extract_final_answer(answer)

        if self.prompt_type == 'concise':
            return f"\\boxed{{{final_answer}}}"

        reasoning = self.extract_reasoning(answer)
        return f"Reasoning: {reasoning}\nMy answer is: \\boxed{{{final_answer}}}"

    def _format_few_shot_block(self, few_shot_examples: list[tuple[str, str]]) -> str:
        """Format few-shot examples as a single text block.

        Args:
            few_shot_examples: List of (question, raw_answer) tuples.

        Returns:
            Formatted few-shot examples as a single string, or empty string if no examples.
        """
        if not few_shot_examples:
            return ''

        lines = ['Examples:', '']
        for q, a in few_shot_examples:
            lines.append(f"Question: {q}")
            lines.append(f"Answer: {self.format_answer(a)}")
            lines.append('')  # blank line between examples

        return '\n'.join(lines)

    def build_messages(
        self,
        question: str,
        answer: str | None,
        few_shot_examples: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        """Build chat messages with few-shot as single user message.

        All few-shot examples are concatenated in a single user message,
        followed by the actual question. This avoids multiple <think></think>
        blocks when using Qwen3 with enable_thinking=False.

        Args:
            question: The question text.
            answer: Raw answer string, or None for inference-only messages.
            few_shot_examples: List of (question, raw_answer) tuples.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        messages = [{'role': 'system', 'content': self.system_message}]

        # Build user content with few-shot examples and question
        few_shot_block = self._format_few_shot_block(few_shot_examples)
        if few_shot_block:
            user_content = f"{few_shot_block}\nQuestion: {question}\nAnswer:"
        else:
            user_content = f"Question: {question}\nAnswer:"

        messages.append({'role': 'user', 'content': user_content})

        # Add assistant response if answer is provided
        if answer is not None:
            messages.append({
                'role': 'assistant',
                'content': self.format_answer(answer),
            })

        return messages

    def prepare_datasets(
        self,
        train_dataset: Dataset,
        tokenizer: Any,
        test_dataset: Dataset | None = None,
        question_field: str = 'question',
        answer_field: str | dict[str, str] = 'answer',
    ) -> dict[str, Dataset]:
        """Prepare train and test datasets with prompts.

        For training:
        - K ~ Uniform[0, max_few_shot] (random few-shot count)
        - Dataset fields: question_field, answer_field, full_prompt, prefill_prompt,
          completion_prompt, final_answer

        For test:
        - K = max_few_shot (fixed few-shot count)
        - Dataset fields: question_field, answer_field, prefill_prompt, completion_prompt,
          final_answer

        Few-shot examples are always sampled from train_dataset.

        Note:
            ``enable_thinking=False`` is passed to ``apply_chat_template()`` calls
            for clarity and as a safeguard. With our custom chat templates (see
            ``chat_templates.py``), this is a no-op since they contain no thinking
            logic. It would only take effect if the default HuggingFace template
            were used instead.

        Args:
            train_dataset: Training dataset.
            tokenizer: Tokenizer for applying chat template.
            test_dataset: Optional test dataset.
            question_field: Name of the question field.
            answer_field: Name of the answer field, or a dict mapping
                ``'train'``/``'test'`` to different field names when the
                train and test datasets use different column names.

        Returns:
            Dict with "train" and optionally "test" datasets.
        """
        if isinstance(answer_field, dict):
            train_answer_field = answer_field['train']
            test_answer_field = answer_field['test']
        else:
            train_answer_field = answer_field
            test_answer_field = answer_field

        rng = random.Random(self.seed)
        n_train = len(train_dataset)

        # Pre-compute K values and few-shot indices for training (random K)
        train_k_values = [rng.randint(0, self.max_few_shot) for _ in range(n_train)]
        train_fs_indices = []
        for idx in range(n_train):
            k = train_k_values[idx]
            if k == 0:
                train_fs_indices.append([])
            else:
                available = list(range(idx)) + list(range(idx + 1, n_train))
                sampled = rng.sample(available, min(k, len(available)))
                train_fs_indices.append(sampled)

        def format_train_example(example: dict, idx: int) -> dict:
            """Format a training example with full_prompt and prefill_prompt."""
            fs_indices = train_fs_indices[idx]
            fs_examples = [
                (train_dataset[i][question_field], train_dataset[i][train_answer_field]) for i in fs_indices
            ]

            # Build messages with assistant response
            question = self._format_question_with_hint(example[question_field], example.get(self.hint_field))
            messages_full = self.build_messages(
                question=question,
                answer=example[train_answer_field],
                few_shot_examples=fs_examples,
            )

            # Messages without assistant response (for prefill)
            messages_prefill = messages_full[:-1]

            # Apply chat template and append EOS token explicitly
            full_prompt = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            ) + tokenizer.eos_token

            prefill_prompt = tokenizer.apply_chat_template(
                messages_prefill,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            if not full_prompt.startswith(prefill_prompt):
                raise ValueError('full_prompt does not start with prefill_prompt')

            completion_prompt = full_prompt[len(prefill_prompt):]

            final_answer = self.extract_final_answer(example[train_answer_field])

            return {
                'full_prompt': full_prompt,
                'prefill_prompt': prefill_prompt,
                'completion_prompt': completion_prompt,
                'final_answer': final_answer,
            }

        # Format training dataset
        formatted_train = train_dataset.map(
            format_train_example,
            with_indices=True,
            desc='Building training prompts',
        )

        result = {'train': formatted_train}

        # Format test dataset if provided
        if test_dataset is not None:
            n_test = len(test_dataset)

            # Pre-compute few-shot indices for test (fixed K = max_few_shot)
            # Use a separate RNG for test to ensure reproducibility
            test_rng = random.Random(self.seed + 1)
            test_fs_indices = []
            for _ in range(n_test):
                k = self.max_few_shot
                if k == 0:
                    test_fs_indices.append([])
                else:
                    sampled = test_rng.sample(range(n_train), min(k, n_train))
                    test_fs_indices.append(sampled)

            def format_test_example(example: dict, idx: int) -> dict:
                """Format a test example with prefill_prompt only."""
                fs_indices = test_fs_indices[idx]
                fs_examples = [
                    (train_dataset[i][question_field], train_dataset[i][train_answer_field]) for i in fs_indices
                ]

                # Build messages without assistant response
                question = self._format_question_with_hint(example[question_field], example.get(self.hint_field))
                messages_prefill = self.build_messages(
                    question=question,
                    answer=None,  # No answer for test
                    few_shot_examples=fs_examples,
                )

                prefill_prompt = tokenizer.apply_chat_template(
                    messages_prefill,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                final_answer = self.extract_final_answer(example[test_answer_field])

                return {
                    'question': example[question_field],
                    'prefill_prompt': prefill_prompt,
                    'completion_prompt': None,
                    'final_answer': final_answer,
                }

            formatted_test = test_dataset.map(
                format_test_example,
                with_indices=True,
                desc='Building test prompts',
            )

            result['test'] = formatted_test

        return result


@PROMPT_BUILDERS.register_module()
class GSM8KPromptBuilder(BaseMathPromptBuilder):
    """Prompt builder for GSM8K dataset.

    GSM8K answer format: "reasoning steps...\\n#### <final_answer>"
    """

    def extract_final_answer(self, answer: str) -> str:
        """Extract the final answer after '####' delimiter.

        Args:
            answer: Raw GSM8K answer string.

        Returns:
            The extracted final answer as a string.
        """
        return extract_gsm8k_final_answer(answer)

    def extract_reasoning(self, answer: str) -> str:
        """Extract reasoning steps before '####' delimiter.

        Args:
            answer: Raw GSM8K answer string.

        Returns:
            The extracted reasoning steps as a string.
        """
        return extract_gsm8k_reasoning(answer)


@PROMPT_BUILDERS.register_module()
class MATHPromptBuilder(BaseMathPromptBuilder):
    """Prompt builder for the MATH dataset.

    MATH answer format: step-by-step solution ending with \\boxed{<final_answer>}.

    ``extract_reasoning`` unwraps the last ``\\boxed{content}`` to plain
    ``content`` so that the CoT completion avoids a duplicate ``\\boxed{}``
    before the explicit "My answer is: \\boxed{answer}" line.

    The default field names match the MATH dataset schema:
    ``question_field='problem'``, ``answer_field='solution'``.
    """

    def prepare_datasets(
        self,
        train_dataset,
        tokenizer: Any,
        test_dataset=None,
        question_field: str = 'problem',
        answer_field: str | dict[str, str] = 'solution',
    ) -> dict:
        """Prepare datasets using MATH-specific field name defaults."""
        return super().prepare_datasets(
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            question_field=question_field,
            answer_field=answer_field,
        )

    def extract_final_answer(self, answer: str) -> str:
        """Extract the content of the last \\boxed{...} in the solution.

        Args:
            answer: Raw MATH solution string.

        Returns:
            The extracted final answer as a string.
        """
        result = extract_boxed_answer(answer)
        return result if result is not None else answer.strip()

    def extract_reasoning(self, answer: str) -> str:
        """Return the solution with the last \\boxed{...} unwrapped.

        Replaces the last ``\\boxed{content}`` with plain ``content`` to avoid
        a duplicate ``\\boxed{}`` in the formatted CoT completion.

        Args:
            answer: Raw MATH solution string.

        Returns:
            The solution with the last \\boxed{} replaced by its plain content.
        """
        return extract_math_reasoning(answer)


@PROMPT_BUILDERS.register_module()
class ChemistryQABuilder(BaseMathPromptBuilder):
    """Prompt builder for ChemistryQA dataset.

    Training data comes from a CoT JSONL file where ``solution`` contains
    step-by-step reasoning ending with ``\\boxed{<final_answer>}``.  Test data comes from the HuggingFace
    ``avaliev/ChemistryQA`` dataset where ``answer`` is a plain-text string
    (no ``\\boxed{}`` wrapping).

    Default ``answer_field`` maps ``'train'`` → ``'solution'`` and
    ``'test'`` → ``'answer'`` to accommodate this difference.
    """

    def prepare_datasets(
        self,
        train_dataset,
        tokenizer: Any,
        test_dataset=None,
        question_field: str = 'question',
        answer_field: str | dict[str, str] = {
            'train': 'solution', 'test': 'answer'
        },
    ) -> dict:
        """Prepare datasets using ChemistryQA-specific field name defaults."""
        return super().prepare_datasets(
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            question_field=question_field,
            answer_field=answer_field,
        )

    def extract_final_answer(self, answer: str) -> str:
        """Extract the final answer from a solution or plain-text answer.

        Tries ``\\boxed{...}`` extraction first (for CoT training data),
        falls back to the raw string (for HuggingFace test data).

        Args:
            answer: Solution string (with ``\\boxed{}``) or plain-text answer.

        Returns:
            The extracted final answer as a string.
        """
        result = extract_boxed_answer(answer)
        return result if result is not None else answer.strip()

    def extract_reasoning(self, answer: str) -> str:
        """Return the solution with \\boxed{...} unwrapped and 'Final answer:' stripped.

        Args:
            answer: Raw solution string.

        Returns:
            The solution with \\boxed{} unwrapped and trailing 'Final answer:' line removed.
        """
        return extract_webinstruct_reasoning(answer)


@PROMPT_BUILDERS.register_module()
class FormulaReasoningPromptBuilder(BaseMathPromptBuilder):
    """Prompt builder for the FormulaReasoning dataset (physics with units).

    Training data comes from a CoT JSONL file where ``solution`` contains
    step-by-step reasoning ending with ``\\boxed{<value with unit>}``
    (e.g. ``\\boxed{140 kg}``).  Test data uses the ``answer`` field which
    is a plain string with units (no ``\\boxed{}`` wrapping).

    Default ``answer_field`` maps ``'train'`` → ``'solution'`` and
    ``'test'`` → ``'answer'`` to accommodate this difference.

    Uses system prompts that instruct the model to include units in the
    ``\\boxed{}`` answer when applicable.
    """

    SYSTEM_CONCISE = (
        'Solve the following problem. '
        'Provide only the final answer with units wrapped in \\\\boxed{}. '
        'Do not show any reasoning or intermediate steps.')

    SYSTEM_COT = (
        'Solve the following problem. '
        "First, reason through the problem step by step after 'Reasoning:'. "
        "Then, provide your final answer with units after 'My answer is:' "
        'wrapped in \\\\boxed{}.')

    def prepare_datasets(
        self,
        train_dataset,
        tokenizer: Any,
        test_dataset=None,
        question_field: str = 'question',
        answer_field: str | dict[str, str] = {
            'train': 'solution', 'test': 'answer'
        },
    ) -> dict:
        """Prepare datasets using FormulaReasoning-specific field name defaults."""
        return super().prepare_datasets(
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            question_field=question_field,
            answer_field=answer_field,
        )

    def extract_final_answer(self, answer: str) -> str:
        """Extract the final answer from a solution or plain-text answer.

        Tries ``\\boxed{...}`` extraction first (for CoT training data),
        falls back to the raw string (for plain-text test data with units).

        Args:
            answer: Solution string (with ``\\boxed{}``) or plain-text answer.

        Returns:
            The extracted final answer as a string.
        """
        result = extract_boxed_answer(answer)
        return result if result is not None else answer.strip()

    def extract_reasoning(self, answer: str) -> str:
        """Return the solution with \\boxed{...} unwrapped and 'Final answer:' stripped.

        Args:
            answer: Raw solution string.

        Returns:
            The solution with \\boxed{} unwrapped and trailing 'Final answer:' line removed.
        """
        return extract_webinstruct_reasoning(answer)
