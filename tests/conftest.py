"""Pytest fixtures for s_edl tests."""
from pathlib import Path

import pytest
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from s_edl.utils.chat_templates import CHAT_TEMPLATES


@pytest.fixture(scope='session')
def qwen3_tokenizer() -> AutoTokenizer:
    """Load the Qwen3-8B-Base tokenizer with custom chat template."""
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base')
    tokenizer.chat_template = CHAT_TEMPLATES['Qwen/Qwen3-8B-Base']
    return tokenizer


@pytest.fixture(scope='session')
def ministral3_tokenizer() -> AutoTokenizer:
    """Load the Ministral-3-8B-Base-2512 tokenizer with custom chat template."""
    tokenizer = AutoTokenizer.from_pretrained('Aratako/Ministral-3-8B-Base-2512-TextOnly')
    tokenizer.chat_template = CHAT_TEMPLATES['Aratako/Ministral-3-8B-Base-2512-TextOnly']
    return tokenizer


@pytest.fixture(scope='session')
def gsm8k_train_dataset() -> Dataset:
    """Load the GSM8K training dataset."""
    return load_dataset('openai/gsm8k', 'main', split='train')


@pytest.fixture(scope='session')
def gsm8k_test_dataset() -> Dataset:
    """Load the GSM8K test dataset."""
    return load_dataset('openai/gsm8k', 'main', split='test')


@pytest.fixture(scope='session')
def math_train_dataset() -> Dataset:
    """Load the MATH training dataset from local JSONL."""
    path = Path(__file__).parent.parent / 'data' / 'math_original_cot_train.jsonl'
    return Dataset.from_json(str(path))
