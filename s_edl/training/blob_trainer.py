"""BLoB trainer for open-ended SFT with Bayesian LoRA KL regularization."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import SGD
from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback, TrainingArguments, get_scheduler

from s_edl.models.load_blob_model import compute_blob_kl_divergence, set_blob_sampling
from s_edl.utils import setup_logger

logger = setup_logger('s_edl')


@dataclass
class BLoBConfig(TrainingArguments):
    """TrainingArguments extended with BLoB settings."""

    bayes_train_n_samples: int = field(default=1, metadata={'help': 'Number of BLoB MC samples during training.'})
    bayes_eps: float = field(default=0.05, metadata={'help': 'BLoB epsilon for rho initialization.'})
    bayes_beta: float = field(default=0.2, metadata={'help': 'Prior std for BLoB KL term.'})
    bayes_kllr: float = field(default=0.02, metadata={'help': 'Learning rate for KL optimizer.'})
    bayes_kl_reweighting: bool = field(
        default=False,
        metadata={'help': 'Whether to use cyclical KL reweighting from official BLoB.'},
    )
    blob_add_special_tokens: bool = field(
        default=False,
        metadata={'help': 'Whether to add special tokens when tokenizing full_prompt.'},
    )

    max_length: int | None = field(default=None, metadata={'help': 'Max sequence length for tokenization.'})
    pad_to_multiple_of: int | None = field(
        default=None, metadata={'help': 'Pad sequences to a multiple of this value (if set).'})
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Keep full_prompt for custom collator.'},
    )


@dataclass
class BLoBDataCollator:
    """Collator for full-prompt SFT labels."""

    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: bool = False
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        full_prompts = [f['full_prompt'] for f in features]
        enc = self.tokenizer(
            full_prompts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = enc['input_ids'].clone()
        if 'attention_mask' in enc:
            labels[enc['attention_mask'] == 0] = -100

        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc.get('attention_mask'),
            'labels': labels,
        }


class _BLoBKLCallback(TrainerCallback):
    """Applies KL-only optimization step after the main optimizer step."""

    def __init__(self, trainer: 'BLoBTrainer') -> None:
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.prepare_blob_optimizer_for_training()

        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is None:
            train_dataloader = self.trainer.get_train_dataloader()

        try:
            n_batches = len(train_dataloader)
        except TypeError:
            n_batches = max(1, int(args.max_steps))
        grad_acc = max(1, int(args.gradient_accumulation_steps))
        self.trainer._blob_M = max(1, math.ceil(n_batches / grad_acc))
        self.trainer._blob_kl_cycle_step = 1
        logger.info(f'BLoB KL setup: M={self.trainer._blob_M}, kl_reweighting={self.trainer.bayes_kl_reweighting}')
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.trainer.model.training:
            return control
        if self.trainer.blob_optimizer is None:
            return control
        if getattr(self.trainer.accelerator, 'optimizer_step_was_skipped', False):
            return control

        self.trainer.model.zero_grad(set_to_none=True)
        kl_raw = self.trainer.compute_blob_kl_divergence(self.trainer.model)
        if not kl_raw.requires_grad:
            return control

        pi = self.trainer._compute_kl_pi()
        kl_loss = kl_raw * pi
        self.trainer.accelerator.backward(kl_loss)
        self.trainer.blob_optimizer.step()
        if self.trainer.blob_scheduler is not None:
            self.trainer.blob_scheduler.step()
        self.trainer.blob_optimizer.zero_grad(set_to_none=True)

        kl_value = float(kl_loss.detach().float().item())
        self.trainer._last_blob_kl_loss = kl_value
        self.trainer._last_blob_pi = float(pi)
        if self.trainer.is_world_process_zero():
            total_loss = self.trainer._last_blob_sft_loss + kl_value
            self.trainer.log(
                {
                    'train/blob_sft_loss': self.trainer._last_blob_sft_loss,
                    'train/blob_kl_loss': kl_value,
                    'train/blob_pi': float(pi),
                    'train/blob_total_loss': total_loss,
                })
        return control


class BLoBTrainer(Trainer):
    """Trainer for Bayesian LoRA by Backpropagation on open-ended SFT."""

    def __init__(
        self,
        model: nn.Module,
        args: BLoBConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset=None,
        eval_dataset=None,
        data_collator: BLoBDataCollator | None = None,
        **kwargs,
    ) -> None:
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token

        if data_collator is None:
            logger.info(
                f"Tokenizer settings: add_special_tokens={args.blob_add_special_tokens}, "
                f"max_length={args.max_length}, pad_to_multiple_of={args.pad_to_multiple_of}")
            data_collator = BLoBDataCollator(
                tokenizer=processing_class,
                add_special_tokens=args.blob_add_special_tokens,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )

        self.processing_class = processing_class
        self.model_accepts_loss_kwargs = False

        self.bayes_train_n_samples = int(args.bayes_train_n_samples)
        self.bayes_kl_reweighting = bool(args.bayes_kl_reweighting)
        self.bayes_kllr = float(args.bayes_kllr)
        if self.bayes_train_n_samples <= 0:
            raise ValueError('bayes_train_n_samples must be > 0.')
        if self.bayes_kllr <= 0:
            raise ValueError('bayes_kllr must be > 0.')

        self._blob_M = 1
        self._blob_kl_cycle_step = 1
        self._last_blob_sft_loss = 0.0
        self._last_blob_kl_loss = 0.0
        self._last_blob_pi = 1.0

        self.blob_optimizer: SGD | None = None
        self.blob_scheduler = None
        self._blob_optimizer_prepared = False
        self._blob_callback = _BLoBKLCallback(self)
        self.add_callback(self._blob_callback)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.blob_optimizer = SGD(params, lr=self.bayes_kllr)
        warmup_steps = self.args.get_warmup_steps(num_training_steps)
        self.blob_scheduler = get_scheduler(
            name='linear',
            optimizer=self.blob_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        self._blob_optimizer_prepared = False

    def prepare_blob_optimizer_for_training(self) -> None:
        if self.blob_optimizer is None:
            return
        if self._blob_optimizer_prepared:
            return
        if self.is_deepspeed_enabled:
            # Keep behavior explicit: current implementation is not deepspeed-specific.
            logger.warning('BLoB KL optimizer is not integrated with deepspeed optimizer partitioning.')
            self._blob_optimizer_prepared = True
            return

        self.blob_optimizer = self.accelerator.prepare(self.blob_optimizer)
        self._blob_optimizer_prepared = True

    def _compute_kl_pi(self) -> float:
        M = max(1, int(self._blob_M))
        if not self.bayes_kl_reweighting:
            return 1.0 / M

        i = self._blob_kl_cycle_step % M
        if i == 0:
            i = M
        self._blob_kl_cycle_step += 1

        # Stable form of: 2^i / (2^(M+1) - 1)
        # = 2^(i-(M+1)) / (1 - 2^(-(M+1))).
        # This avoids overflow for large M.
        numerator = math.ldexp(1.0, i - (M + 1))
        denominator = 1.0 - math.ldexp(1.0, -(M + 1))
        if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
            return 1.0 / M
        pi = numerator / denominator
        if not math.isfinite(pi):
            return 1.0 / M
        return pi

    def compute_blob_kl_divergence(self, model: nn.Module) -> torch.Tensor:
        return compute_blob_kl_divergence(model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'],
            'use_cache': False,
        }
        if inputs.get('attention_mask') is not None:
            model_inputs['attention_mask'] = inputs['attention_mask']

        if model.training:
            set_blob_sampling(model, True)
            losses = []
            outputs = None
            for _ in range(self.bayes_train_n_samples):
                outputs = model(**model_inputs)
                losses.append(outputs.loss)
            sft_loss = torch.stack(losses).mean()
        else:
            set_blob_sampling(model, False)
            outputs = model(**model_inputs)
            sft_loss = outputs.loss
            set_blob_sampling(model, True)

        self._last_blob_sft_loss = float(sft_loss.detach().float().item())
        if return_outputs:
            return sft_loss, outputs
        return sft_loss
