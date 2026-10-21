"""RCFT trainer for SFT + calibration-aware fine-tuning on open-ended QA."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback, TrainerState, TrainingArguments

from s_edl.utils import setup_logger

logger = setup_logger('s_edl')


@dataclass
class RCFTConfig(TrainingArguments):
    """TrainingArguments extended with RCFT settings."""

    rcft_alpha: float = field(default=0.5, metadata={'help': 'Mixing factor between SFT and RCFT losses.'})
    rcft_calibration_scale: float = field(default=1e-3, metadata={'help': 'Scale for RCFT calibration loss.'})
    rcft_num_bins: int = field(default=10, metadata={'help': 'Number of confidence bins for RCFT E-step.'})
    rcft_eps: float = field(default=1e-6, metadata={'help': 'Numerical epsilon for RCFT stability.'})
    rcft_token_chunk_size: int = field(default=256, metadata={'help': 'Chunk size for active tokens.'})
    rcft_vocab_chunk_size: int = field(default=4096, metadata={'help': 'Chunk size for vocab dimension.'})
    rcft_add_special_tokens: bool = field(
        default=False, metadata={'help': 'Whether to add special tokens when tokenizing full_prompt.'})

    max_length: int | None = field(default=None, metadata={'help': 'Max sequence length for tokenization.'})
    pad_to_multiple_of: int | None = field(
        default=None, metadata={'help': 'Pad sequences to a multiple of this value (if set).'})
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Keep custom fields like full_prompt/prefill_prompt/completion_prompt.'},
    )


@dataclass
class RCFTDataCollator:
    """Collator that keeps SFT labels and adds completion-token mask for RCFT."""

    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: bool = False
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def _completion_len_after_truncation(self, full_prompt: str, prefill_prompt: str, effective_len: int) -> int:
        full_ids = self.tokenizer(full_prompt, add_special_tokens=self.add_special_tokens)['input_ids']
        prefill_ids = self.tokenizer(prefill_prompt, add_special_tokens=self.add_special_tokens)['input_ids']
        full_len = len(full_ids)
        prefill_len = len(prefill_ids)
        completion_len = max(0, full_len - prefill_len)

        if self.max_length is None or full_len <= self.max_length:
            return min(completion_len, effective_len)

        truncation_side = getattr(self.tokenizer, 'truncation_side', 'right')
        if truncation_side == 'right':
            retained_completion = max(0, self.max_length - prefill_len)
            retained_completion = min(retained_completion, completion_len)
            return min(retained_completion, effective_len)

        # Left truncation: remove tokens from sequence front.
        dropped = full_len - self.max_length
        if dropped < prefill_len:
            retained_completion = completion_len
        else:
            dropped_completion = dropped - prefill_len
            retained_completion = max(0, completion_len - dropped_completion)
        return min(retained_completion, effective_len)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        full_prompts = [f['full_prompt'] for f in features]
        prefill_prompts = [f['prefill_prompt'] for f in features]
        completion_prompts = [f.get('completion_prompt') for f in features]

        enc = self.tokenizer(
            full_prompts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = enc['input_ids']
        attention_mask = enc.get('attention_mask')

        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        rcft_completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        seq_len = input_ids.size(1)
        padding_side = getattr(self.tokenizer, 'padding_side', 'right')

        for i, (full_prompt, prefill_prompt) in enumerate(zip(full_prompts, prefill_prompts)):
            if attention_mask is not None:
                effective_len = int(attention_mask[i].sum().item())
            else:
                effective_len = int(seq_len)
            completion_len = self._completion_len_after_truncation(
                full_prompt=full_prompt,
                prefill_prompt=prefill_prompt,
                effective_len=effective_len,
            )
            if completion_len <= 0:
                continue

            token_start = seq_len - effective_len if padding_side == 'left' else 0
            token_end = token_start + effective_len
            completion_start = max(token_start, token_end - completion_len)
            rcft_completion_mask[i, completion_start:token_end] = True

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'rcft_completion_mask': rcft_completion_mask,
            'prefill_prompt': prefill_prompts,
            'completion_prompt': completion_prompts,
        }


class _RCFTEpochCallback(TrainerCallback):
    """Refresh RCFT E-step statistics at each epoch start."""

    def __init__(self, trainer: 'RCFTTrainer') -> None:
        self.trainer = trainer

    def on_epoch_begin(self, args, state: TrainerState, control, **kwargs):
        train_dataloader = kwargs.get('train_dataloader')
        self.trainer._refresh_bin_stats_for_epoch(state=state, train_dataloader=train_dataloader)
        return control


class RCFTTrainer(Trainer):
    """Trainer that mixes SFT loss with RCFT calibration loss.

    To extend RCFT from MCQA to open-ended question answering, we made the
    following extensions:
    - E-step statistics are refreshed at every epoch start (paper-style), not frozen once before training.
    - Calibration is applied on completion-token positions across sequence generation.
    - Calibration uses exact all-vocab probabilities (no top-k approximation) with chunked computation.
    - RCFT calibration loss is averaged over active completion tokens.
    - SFT loss uses full-prompt labels, aligned with this project's existing training setup.
    """

    def __init__(
        self,
        model: nn.Module,
        args: RCFTConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset=None,
        eval_dataset=None,
        data_collator: RCFTDataCollator | None = None,
        **kwargs,
    ) -> None:
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token

        if data_collator is None:
            logger.info(
                f"Tokenizer settings: add_special_tokens={args.rcft_add_special_tokens}, "
                f"max_length={args.max_length}, pad_to_multiple_of={args.pad_to_multiple_of}")
            data_collator = RCFTDataCollator(
                tokenizer=processing_class,
                add_special_tokens=args.rcft_add_special_tokens,
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

        try:
            rcft_alpha = float(args.rcft_alpha)
            rcft_calibration_scale = float(args.rcft_calibration_scale)
            rcft_eps = float(args.rcft_eps)
            rcft_num_bins = int(args.rcft_num_bins)
            rcft_token_chunk_size = int(args.rcft_token_chunk_size)
            rcft_vocab_chunk_size = int(args.rcft_vocab_chunk_size)
        except (TypeError, ValueError) as err:
            raise ValueError('RCFT numeric arguments must be valid numbers.') from err

        if rcft_num_bins <= 0:
            raise ValueError('rcft_num_bins must be > 0.')
        if rcft_token_chunk_size <= 0:
            raise ValueError('rcft_token_chunk_size must be > 0.')
        if rcft_vocab_chunk_size <= 0:
            raise ValueError('rcft_vocab_chunk_size must be > 0.')
        if not 0.0 <= rcft_alpha <= 1.0:
            raise ValueError('rcft_alpha must be in [0, 1].')

        self.processing_class = processing_class
        self.rcft_alpha = rcft_alpha
        self.rcft_calibration_scale = rcft_calibration_scale
        self.rcft_num_bins = rcft_num_bins
        self.rcft_eps = rcft_eps
        self.rcft_token_chunk_size = rcft_token_chunk_size
        self.rcft_vocab_chunk_size = rcft_vocab_chunk_size

        self._bin_upper_boundaries = torch.linspace(
            1.0 / self.rcft_num_bins,
            1.0,
            self.rcft_num_bins,
            dtype=torch.float32,
            device=self.args.device,
        )
        self._bin_accuracy = torch.zeros(self.rcft_num_bins, dtype=torch.float32, device=self.args.device)
        self._bin_total_count = torch.zeros(self.rcft_num_bins, dtype=torch.float32, device=self.args.device)
        self._rcft_nonempty_bins = 0
        self._last_refreshed_epoch: int | None = None

        self.model_accepts_loss_kwargs = False
        self.add_callback(_RCFTEpochCallback(self))

    def train(self, *args, **kwargs):
        self._last_refreshed_epoch = None
        return super().train(*args, **kwargs)

    def _resolve_epoch_index(self, state: TrainerState) -> int:
        if state.epoch is None:
            return 0
        return max(0, int(math.floor(float(state.epoch) + 1e-8)))

    def _refresh_bin_stats_for_epoch(self, state: TrainerState, train_dataloader=None) -> None:
        epoch_idx = self._resolve_epoch_index(state)
        if self._last_refreshed_epoch == epoch_idx:
            return

        dataloader = train_dataloader if train_dataloader is not None else self.get_train_dataloader()
        if hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch_idx)
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch_idx)

        total_batches: int | None = None
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        estep_log_interval = max(1, int(getattr(self.args, 'logging_steps', 1) * 50))

        if self.is_world_process_zero():
            if total_batches is not None:
                logger.info(
                    f'RCFT E-step start for epoch={epoch_idx}: '
                    f'batches={total_batches}, log_interval={estep_log_interval}')
            else:
                logger.info(
                    f'RCFT E-step start for epoch={epoch_idx}: '
                    f'batches=unknown, log_interval={estep_log_interval}')

        bin_correct = torch.zeros(self.rcft_num_bins, dtype=torch.float32, device=self.args.device)
        bin_total = torch.zeros(self.rcft_num_bins, dtype=torch.float32, device=self.args.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, start=1):
                inputs = self._prepare_inputs(batch)
                model_inputs = {'input_ids': inputs['input_ids'], 'use_cache': False}
                if inputs.get('attention_mask') is not None:
                    model_inputs['attention_mask'] = inputs['attention_mask']

                logits = self.model(**model_inputs).logits
                self._accumulate_bin_stats(
                    logits=logits,
                    labels=inputs['labels'],
                    completion_mask=inputs['rcft_completion_mask'],
                    bin_correct=bin_correct,
                    bin_total=bin_total,
                )

                should_log = (
                    batch_idx == 1 or batch_idx % estep_log_interval == 0
                    or (total_batches is not None and batch_idx == total_batches))
                if should_log and self.is_world_process_zero():
                    processed_tokens = int(bin_total.sum().item())
                    if total_batches is None:
                        logger.info(
                            f'RCFT E-step epoch={epoch_idx} progress: '
                            f'batch={batch_idx}, collected_tokens={processed_tokens}')
                    else:
                        progress_pct = 100.0 * batch_idx / max(1, total_batches)
                        logger.info(
                            f'RCFT E-step epoch={epoch_idx} progress: '
                            f'{batch_idx}/{total_batches} ({progress_pct:.1f}%), '
                            f'collected_tokens={processed_tokens}')

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bin_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_total, op=dist.ReduceOp.SUM)

        self._bin_total_count = bin_total
        nonempty = bin_total > 0
        self._bin_accuracy = torch.where(nonempty, bin_correct / bin_total.clamp(min=1.0), torch.zeros_like(bin_total))
        self._rcft_nonempty_bins = int(nonempty.sum().item())
        self._last_refreshed_epoch = epoch_idx

        if was_training:
            self.model.train()

        if self.is_world_process_zero():
            logger.info(
                f"RCFT E-step refreshed for epoch={epoch_idx}: nonempty_bins={self._rcft_nonempty_bins}/"
                f"{self.rcft_num_bins}, total_tokens={float(bin_total.sum().item()):.0f}")

    def _accumulate_bin_stats(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
        bin_correct: torch.Tensor,
        bin_total: torch.Tensor,
    ) -> None:
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        shifted_mask = completion_mask[:, 1:].bool()
        active_mask = shifted_mask & shifted_labels.ne(-100)
        if not active_mask.any():
            return

        vocab_size = shifted_logits.size(-1)
        logits_flat = shifted_logits.reshape(-1, vocab_size)
        labels_flat = shifted_labels.reshape(-1)
        active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).squeeze(-1)
        boundaries = self._bin_upper_boundaries.to(logits.device)

        for start in range(0, active_indices.numel(), self.rcft_token_chunk_size):
            idx = active_indices[start:start + self.rcft_token_chunk_size]
            logits_chunk = logits_flat.index_select(0, idx).float()
            labels_chunk = labels_flat.index_select(0, idx)

            logsumexp = torch.logsumexp(logits_chunk, dim=-1)
            max_logits, pred_idx = logits_chunk.max(dim=-1)
            max_prob = torch.exp(max_logits - logsumexp)
            bin_idx = torch.bucketize(max_prob, boundaries, right=False).clamp(max=self.rcft_num_bins - 1)

            ones = torch.ones_like(max_prob, dtype=torch.float32)
            correct = pred_idx.eq(labels_chunk).to(torch.float32)
            bin_total.scatter_add_(0, bin_idx, ones)
            bin_correct.scatter_add_(0, bin_idx, correct)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {'input_ids': inputs['input_ids'], 'labels': inputs['labels'], 'use_cache': False}
        if inputs.get('attention_mask') is not None:
            model_inputs['attention_mask'] = inputs['attention_mask']
        outputs = model(**model_inputs)
        sft_loss = outputs.loss

        if (not self.model.training or self.rcft_alpha <= 0.0 or self.rcft_calibration_scale <= 0.0
                or self._bin_accuracy is None):
            rcft_loss = sft_loss.new_zeros(())
            active_tokens = 0
            total_loss = sft_loss
        else:
            rcft_loss, active_tokens = self._compute_rcft_loss(
                logits=outputs.logits,
                labels=inputs['labels'],
                completion_mask=inputs['rcft_completion_mask'],
            )
            total_loss = (1.0 - self.rcft_alpha) * sft_loss
            total_loss = total_loss + self.rcft_alpha * self.rcft_calibration_scale * rcft_loss

            if self.is_world_process_zero():
                self.log(
                    {
                        'train/sft_loss': float(sft_loss.detach().float().item()),
                        'train/rcft_loss': float(rcft_loss.detach().float().item()),
                        'train/rcft_total_loss': float(total_loss.detach().float().item()),
                        'train/rcft_active_tokens': float(active_tokens),
                        'train/rcft_alpha': float(self.rcft_alpha),
                        'train/rcft_calibration_scale': float(self.rcft_calibration_scale),
                        'train/rcft_nonempty_bins': float(self._rcft_nonempty_bins),
                    })

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_rcft_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        shifted_mask = completion_mask[:, 1:].bool()
        active_mask = shifted_mask & shifted_labels.ne(-100)
        if not active_mask.any():
            return logits.new_zeros(()), 0

        vocab_size = shifted_logits.size(-1)
        logits_flat = shifted_logits.reshape(-1, vocab_size)
        active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).squeeze(-1)
        total_loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)

        for start in range(0, active_indices.numel(), self.rcft_token_chunk_size):
            idx = active_indices[start:start + self.rcft_token_chunk_size]
            logits_chunk = logits_flat.index_select(0, idx).float()
            total_loss_sum = total_loss_sum + self._compute_rcft_chunk_loss_sum(logits_chunk)

        active_tokens = int(active_indices.numel())
        mean_loss = total_loss_sum / max(1, active_tokens)
        return mean_loss, active_tokens

    def _compute_rcft_chunk_loss_sum(self, logits_chunk: torch.Tensor) -> torch.Tensor:
        if logits_chunk.numel() == 0:
            return logits_chunk.new_zeros(())

        device = logits_chunk.device
        dtype = torch.float32
        vocab_size = logits_chunk.size(-1)
        logsumexp = torch.logsumexp(logits_chunk, dim=-1)

        if vocab_size >= 2:
            top2_logits, top2_idx = torch.topk(logits_chunk, k=2, dim=-1)
            max_logits = top2_logits[:, 0]
            second_logits = top2_logits[:, 1]
            pred_idx = top2_idx[:, 0]
            second_prob = torch.exp(second_logits - logsumexp)
        else:
            max_logits, pred_idx = logits_chunk.max(dim=-1)
            second_prob = torch.zeros_like(max_logits)

        max_prob = torch.exp(max_logits - logsumexp)
        boundaries = self._bin_upper_boundaries.to(device)
        bin_idx = torch.bucketize(max_prob, boundaries, right=False).clamp(max=self.rcft_num_bins - 1)
        q = self._bin_accuracy.to(device).index_select(0, bin_idx).to(dtype)

        max_loss = torch.square(max_prob - q)
        if vocab_size <= 1:
            return max_loss.sum()

        v_minus_one = float(vocab_size - 1)
        one_minus_q = torch.clamp(1.0 - q, min=self.rcft_eps)
        safe_second_prob = torch.clamp(second_prob, min=self.rcft_eps)
        log_const = math.log(v_minus_one)
        gamma = log_const / (safe_second_prob * one_minus_q)
        q_is_one = q >= (1.0 - self.rcft_eps)
        gamma = torch.where(q_is_one, torch.ones_like(gamma), gamma)

        s_p2 = torch.zeros_like(max_prob, dtype=dtype)
        s_pt = torch.zeros_like(max_prob, dtype=dtype)
        s_t2 = torch.zeros_like(max_prob, dtype=dtype)
        s_p = torch.zeros_like(max_prob, dtype=dtype)
        s_t = torch.zeros_like(max_prob, dtype=dtype)

        for v_start in range(0, vocab_size, self.rcft_vocab_chunk_size):
            v_end = min(v_start + self.rcft_vocab_chunk_size, vocab_size)
            logits_slice = logits_chunk[:, v_start:v_end]
            probs = torch.exp(logits_slice - logsumexp.unsqueeze(-1))

            token_ids = torch.arange(v_start, v_end, device=device)
            is_max = pred_idx.unsqueeze(-1).eq(token_ids.unsqueeze(0))
            probs_nonmax = probs.masked_fill(is_max, 0.0)
            tanh_nonmax = torch.tanh(gamma.unsqueeze(-1) * probs_nonmax)

            s_p += probs_nonmax.sum(dim=-1)
            s_p2 += torch.square(probs_nonmax).sum(dim=-1)
            s_t += tanh_nonmax.sum(dim=-1)
            s_t2 += torch.square(tanh_nonmax).sum(dim=-1)
            s_pt += (probs_nonmax * tanh_nonmax).sum(dim=-1)

        alpha = (1.0 - q) / torch.clamp(s_t + v_minus_one, min=self.rcft_eps)
        alpha = torch.where(q_is_one, torch.zeros_like(alpha), alpha)
        beta = alpha

        nonmax_loss = s_p2
        nonmax_loss = nonmax_loss - 2.0 * alpha * s_pt
        nonmax_loss = nonmax_loss + torch.square(alpha) * s_t2
        nonmax_loss = nonmax_loss - 2.0 * beta * s_p
        nonmax_loss = nonmax_loss + 2.0 * alpha * beta * s_t
        nonmax_loss = nonmax_loss + v_minus_one * torch.square(beta)

        token_loss = max_loss + nonmax_loss
        return token_loss.sum()
