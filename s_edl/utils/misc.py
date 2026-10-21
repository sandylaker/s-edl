import logging
import os.path as osp
from collections.abc import Mapping
from typing import TypeAlias

import torch
import wandb
from datasets import Dataset

from .logging import get_dist_rank

Device: TypeAlias = str | torch.device


def log_training_examples(
    train_dataset: Dataset,
    output_dir: str,
    logger: logging.Logger,
    num_examples_dict: Mapping[str, int] | None = None,
) -> None:
    """Log and save training examples for sanity check.

    Only runs on rank 0 in distributed environment.

    Args:
        train_dataset: The prepared training dataset.
        output_dir: Directory to save examples file.
        logger: Logger instance.
        num_examples_dict: Per-destination example limits. Supported keys are
            ``terminal``, ``file``, and ``wandb``. Missing keys fall back to
            defaults: ``terminal=1``, ``file=4``, ``wandb=4``.
    """
    if get_dist_rank() != 0:
        return

    default_num_examples_dict = {
        'terminal': 1,
        'file': 4,
        'wandb': 4,
    }
    merged_num_examples_dict = default_num_examples_dict.copy()
    if num_examples_dict is not None:
        invalid_keys = set(num_examples_dict) - set(default_num_examples_dict)
        if invalid_keys:
            raise ValueError(f'Unsupported num_examples_dict keys: {sorted(invalid_keys)}')
        merged_num_examples_dict.update(num_examples_dict)

    for name, count in merged_num_examples_dict.items():
        if not isinstance(count, int):
            raise TypeError(f"num_examples_dict['{name}'] must be int, got {type(count).__name__}")
        if count < 0:
            raise ValueError(f"num_examples_dict['{name}'] must be >= 0, got {count}")
        merged_num_examples_dict[name] = min(count, len(train_dataset))

    terminal_num_examples = merged_num_examples_dict['terminal']
    file_num_examples = merged_num_examples_dict['file']
    wandb_num_examples = merged_num_examples_dict['wandb']

    examples_content = []
    for i in range(terminal_num_examples):
        example = train_dataset[i]
        logger.info(f"[Example {i + 1}]")
        logger.info(f"full_prompt:\n{example['full_prompt']}")
        logger.info('-' * 60)
        logger.info(f"prefill_prompt:\n{example['prefill_prompt']}")
        logger.info('-' * 60)
        logger.info(f"completion_prompt:\n{example['completion_prompt']}")
        logger.info('-' * 60)
        logger.info(f"final_answer: {example['final_answer']}")
        logger.info('=' * 60)

    for i in range(file_num_examples):
        example = train_dataset[i]
        examples_content.append(f"=== Example {i + 1} ===")
        examples_content.append(f"--- full_prompt ---\n{example['full_prompt']}\n")
        examples_content.append(f"--- prefill_prompt ---\n{example['prefill_prompt']}\n")
        examples_content.append(f"--- completion_prompt ---\n{example['completion_prompt']}\n")
        examples_content.append(f"--- final_answer ---\n{example['final_answer']}\n")

    if wandb.run is not None:
        for i in range(wandb_num_examples):
            example = train_dataset[i]
            wandb.log(
                {
                    f'example_{i + 1}/full_prompt': wandb.Html(f"<pre>{example['full_prompt']}</pre>"),
                    f'example_{i + 1}/prefill_prompt': wandb.Html(f"<pre>{example['prefill_prompt']}</pre>"),
                    f'example_{i + 1}/completion_prompt': wandb.Html(f"<pre>{example['completion_prompt']}</pre>"),
                    f'example_{i + 1}/final_answer': example['final_answer'],
                })

    with open(osp.join(output_dir, 'training_examples.txt'), 'w') as f:
        f.write('\n'.join(examples_content))
