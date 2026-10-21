"""Custom Trainer for SFT + self-evidenced EDL loss with vLLM rollouts."""

import inspect
import os
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
from peft import get_peft_model_state_dict
from requests.adapters import HTTPAdapter
from safetensors.torch import save_file
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments
from trl.trainer.utils import entropy_from_logits, pad, selective_log_softmax
from urllib3.util.retry import Retry

from s_edl.evaluation import ClusterClassifier, cluster_predictions_with_ground_truth, extract_prediction
from s_edl.utils import setup_logger

logger = setup_logger('s_edl')


@dataclass
class EDLConfig(TrainingArguments):
    """TrainingArguments extended with EDL + vLLM rollout settings."""

    # EDL loss config
    edl_beta: float = field(default=1.0, metadata={'help': 'Tempering beta >= 1.'})
    edl_weight: float = field(default=1.0, metadata={'help': 'Final EDL loss weight (gamma).'})
    edl_weight_start_step: int | float = field(
        default=0,
        metadata={'help': 'Step (int) or fraction (0-1 float) of total steps to start ramping EDL weight.'},
    )
    edl_weight_warmup_steps: int | float = field(
        default=0,
        metadata={'help': 'Warmup steps (int) or fraction (0-1 float) of total steps to ramp EDL weight.'},
    )
    edl_gt_pseudo_logprob_penalty: float | None = field(
        default=4.605,
        metadata={
            'help':
            'Subtract this value from GT logprob when GT class is missing to create near-zero evidence. '
            'Set to None or <=0 to disable.'
        },
    )
    # Rollout config
    edl_logprob_batch_size: int | None = field(
        default=None,
        metadata={'help': 'Chunk size for logprob computation; None uses full batch.'},
    )
    edl_add_special_tokens: bool = field(
        default=False,
        metadata={'help': 'Whether to add special tokens when tokenizing full_prompt.'},
    )
    edl_sampling: dict[str, Any] | None = field(
        default=None,
        metadata={
            'help': (
                'Sampling dict (for vLLM rollouts) with required keys: n, temperature, top_p, max_tokens. '
                'Additional vLLM/OpenAI-compatible fields (e.g., logprobs, stop) are passed through.')
        },
    )

    # Tokenization config
    max_length: int | None = field(
        default=None,
        metadata={'help': 'Max sequence length for tokenization in EDLDataCollator.'},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={'help': 'Pad sequences to a multiple of this value (if set).'},
    )

    # vLLM API config
    vllm_base_url: str | None = field(default=None, metadata={'help': 'Base URL for vLLM server.'})
    vllm_api_mode: str = field(
        default='openai',
        metadata={'help': "vLLM API mode: 'openai' for /v1/completions."},
    )
    vllm_generate_endpoint: str = field(
        default='/v1/completions', metadata={'help': 'Endpoint for generation requests.'})
    vllm_load_lora_endpoint: str = field(
        default='/v1/load_lora_adapter', metadata={'help': 'Endpoint for loading LoRA adapters.'})
    vllm_unload_lora_endpoint: str = field(
        default='/v1/unload_lora_adapter', metadata={'help': 'Endpoint for unloading LoRA adapters.'})
    vllm_timeout: float = field(default=240.0, metadata={'help': 'HTTP timeout for vLLM calls.'})
    vllm_force_reload_lora: bool = field(
        default=True, metadata={'help': 'Force unload+reload LoRA on sync to ensure latest weights are used.'})
    # Importance sampling correction
    vllm_importance_sampling: bool = field(default=True, metadata={'help': 'Enable IS correction.'})
    vllm_importance_sampling_mode: str = field(
        default='sequence_truncate',
        metadata={'help': 'IS mode: token_truncate/token_mask/sequence_truncate/sequence_mask.'},
    )
    vllm_importance_sampling_cap: float = field(default=2.0, metadata={'help': 'Cap for IS ratio truncation/masking.'})

    # LoRA sync to vLLM
    lora_sync: bool = field(default=True, metadata={'help': 'Whether to sync LoRA to vLLM every step.'})
    lora_adapter_dir: str = field(default='/dev/shm/s_edl_adapter', metadata={'help': 'Adapter dir in shm.'})
    lora_name: str = field(default='edl_adapter', metadata={'help': 'LoRA name in vLLM.'})
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Keep custom fields like full_prompt/prefill_prompt for EDLTrainer.'},
    )


@dataclass
class EDLDataCollator:
    """Collator that keeps raw prompts while building LM inputs."""

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

        batch = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc.get('attention_mask'),
            'labels': labels,
            'prefill_prompt': [f['prefill_prompt'] for f in features],
            'completion_prompt': [f.get('completion_prompt') for f in features],
            'final_answer': [f['final_answer'] for f in features],
        }
        return batch


class EDLTrainer(Trainer):
    """Trainer that mixes SFT loss with self-evidenced EDL loss.

    Notes:
    - vLLM may apply `generation_config.json` by default; start the server with
      `--generation-config vllm` to ensure request-time sampling params are honored.
    - Runtime LoRA reload requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
    - Prefix caching must be disabled (`--no-enable-prefix-caching`) since the
      standard vLLM OpenAI server does not support cache invalidation on LoRA reload.
      Without this, stale KV states may be reused after weight updates.
    """

    def __init__(
        self,
        model: nn.Module,
        args: EDLConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset=None,
        eval_dataset=None,
        data_collator: EDLDataCollator | None = None,
        cluster_classifier: ClusterClassifier | None = None,
        **kwargs,
    ) -> None:
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token

        if data_collator is None:
            logger.info(
                f"Tokenizer settings: add_special_tokens={args.edl_add_special_tokens}, "
                f"max_length={args.max_length}, pad_to_multiple_of={args.pad_to_multiple_of}")
            data_collator = EDLDataCollator(
                tokenizer=processing_class,
                add_special_tokens=args.edl_add_special_tokens,
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
        self.pad_token_id = processing_class.pad_token_id

        # EDL config
        self.edl_beta = args.edl_beta
        if self.edl_beta < 1.0:
            raise ValueError('edl_beta must be >= 1.0 for self-evidence weighting.')
        self.edl_weight = args.edl_weight
        self.edl_weight_start_step = args.edl_weight_start_step
        self.edl_weight_warmup_steps = args.edl_weight_warmup_steps
        # Cached absolute step (int) when EDL weight starts.
        # Resolved lazily from `edl_weight_start_step` (int steps or float fraction).
        self._edl_weight_start_step_resolved: int | None = None
        # Cached absolute warmup steps (int) for EDL weight ramp-up.
        # Resolved lazily from `edl_weight_warmup_steps` (int steps or float fraction).
        self._edl_weight_warmup_steps_resolved: int | None = None
        if args.lora_sync and not args.lora_name:
            raise ValueError('lora_name must be set when lora_sync=True.')
        if not args.edl_sampling:
            raise ValueError('edl_sampling must be set in training config.')
        required_sampling_keys = {'n', 'temperature', 'top_p', 'max_tokens', 'logprobs'}
        missing_sampling = required_sampling_keys - set(args.edl_sampling.keys())
        if missing_sampling:
            raise ValueError(f"edl_sampling missing required keys: {sorted(missing_sampling)}")
        sampling = dict(args.edl_sampling)
        self.edl_sampling = sampling
        logger.info(f"EDL sampling: {self.edl_sampling}")
        self.edl_logprob_batch_size = args.edl_logprob_batch_size
        self.edl_gt_pseudo_logprob_penalty = args.edl_gt_pseudo_logprob_penalty
        # vLLM config
        self.vllm_base_url = args.vllm_base_url
        self.vllm_api_mode = args.vllm_api_mode
        self.vllm_generate_endpoint = args.vllm_generate_endpoint
        self.vllm_load_lora_endpoint = args.vllm_load_lora_endpoint
        self.vllm_unload_lora_endpoint = args.vllm_unload_lora_endpoint
        self.vllm_timeout = args.vllm_timeout
        self.vllm_force_reload_lora = args.vllm_force_reload_lora

        self.vllm_importance_sampling = args.vllm_importance_sampling
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        if self.vllm_importance_sampling and self.edl_sampling.get('logprobs', 0) <= 0:
            raise ValueError('edl_sampling.logprobs must be > 0 when vllm_importance_sampling is enabled.')

        self.lora_sync = args.lora_sync
        self.lora_adapter_dir = args.lora_adapter_dir
        self.lora_name = args.lora_name
        self._adapter_initialized = False
        self._last_lora_sync_step = -1
        self._last_edl_stats: dict[str, float] = {}
        self._warned_logprob_mismatch = False

        self._model_kwarg_keys = set(inspect.signature(self.model.forward).parameters.keys())

        self._vllm_session = None
        if self.vllm_base_url is not None:
            self._init_vllm_session()
            if self.lora_sync and self.is_world_process_zero():
                self._write_adapter_once()
                self._load_adapter_into_vllm()
                logger.info(f"Initial LoRA sync complete: adapter='{self.lora_name}', path={self.lora_adapter_dir}")

        # Optional LLM judge classifier for non-math clustering
        self._cluster_classifier = cluster_classifier

    # ---------------------------
    # vLLM HTTP helpers
    # ---------------------------
    def _init_vllm_session(self) -> None:
        session = requests.Session()
        if HTTPAdapter is not None and Retry is not None:
            retry_strategy = Retry(
                total=5,
                connect=5,
                read=5,
                status=3,
                status_forcelist=[500, 502, 503],
                backoff_factor=2,
                allowed_methods=['POST', 'GET'],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
        self._vllm_session = session

    def _vllm_url(self, endpoint: str) -> str:
        if self.vllm_base_url is None:
            raise ValueError('vLLM base URL is not set.')
        return self.vllm_base_url.rstrip('/') + '/' + endpoint.lstrip('/')

    def _vllm_post(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._vllm_session is None:
            self._init_vllm_session()
        url = self._vllm_url(endpoint)
        response = self._vllm_session.post(url, json=payload, timeout=self.vllm_timeout)
        response.raise_for_status()
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            if endpoint in {self.vllm_load_lora_endpoint, self.vllm_unload_lora_endpoint}:
                return {'message': response.text}
            raise ValueError(f"Non-JSON response from vLLM {url}: {response.text[:200]!r}") from exc

    # ---------------------------
    # LoRA sync to vLLM
    # ---------------------------
    def _write_adapter_once(self) -> None:
        if self._adapter_initialized:
            return
        os.makedirs(self.lora_adapter_dir, exist_ok=True)
        # Write config + initial adapter weights once.
        model_to_save = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        model_to_save.save_pretrained(self.lora_adapter_dir, safe_serialization=True)
        self._adapter_initialized = True

    def _write_adapter_weights(self) -> None:
        model_to_save = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        state = get_peft_model_state_dict(model_to_save)
        tmp = os.path.join(self.lora_adapter_dir, 'adapter_model.safetensors.tmp')
        final = os.path.join(self.lora_adapter_dir, 'adapter_model.safetensors')
        save_file(state, tmp)
        os.replace(tmp, final)

    def _load_adapter_into_vllm(self) -> None:
        if self.vllm_base_url is None:
            return
        payload = {
            'lora_name': self.lora_name,
            'lora_path': self.lora_adapter_dir,
            'load_inplace': True,
        }
        self._vllm_post(self.vllm_load_lora_endpoint, payload)

    def _unload_adapter_from_vllm(self) -> None:
        if self.vllm_base_url is None:
            return
        payload = {
            'lora_name': self.lora_name,
        }
        self._vllm_post(self.vllm_unload_lora_endpoint, payload)

    def _sync_lora_to_vllm(self, step: int | None = None) -> None:
        if not self.lora_sync or self.vllm_base_url is None:
            return
        if not self.is_world_process_zero():
            return
        if not self._adapter_initialized:
            self._write_adapter_once()
        self._write_adapter_weights()
        if self.vllm_force_reload_lora:
            try:
                self._unload_adapter_from_vllm()
            except Exception as exc:
                logger.warning(f"Failed to unload LoRA adapter: {exc}")
        self._load_adapter_into_vllm()

    # ---------------------------
    # Trainer overrides
    # ---------------------------
    def _maybe_sync_lora_before_rollout(self) -> None:
        """Sync LoRA weights to vLLM before generating rollouts if needed.

        Called at the start of compute_loss to ensure vLLM has the latest weights before generating rollouts.
        """
        step = self.state.global_step if self.state is not None else 0
        edl_weight = self._get_edl_weight(step)
        should_sync = (
            self.lora_sync and self.vllm_base_url is not None and edl_weight > 0.0
            and step != self._last_lora_sync_step and step > 0)
        if should_sync:
            self._sync_lora_to_vllm(step)
            self._last_lora_sync_step = step

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # SFT loss
        model_inputs = {'input_ids': inputs['input_ids'], 'labels': inputs['labels']}
        if inputs.get('attention_mask') is not None:
            model_inputs['attention_mask'] = inputs['attention_mask']
        model_inputs['use_cache'] = False
        outputs = model(**model_inputs)
        sft_loss = outputs.loss

        step = self.state.global_step if self.state is not None else 0
        edl_weight = self._get_edl_weight(step)
        if not self.model.training or edl_weight <= 0.0 or self.edl_sampling['n'] <= 0:
            total_loss = sft_loss
        else:
            # Sync LoRA weights to vLLM before generating rollouts
            self._maybe_sync_lora_before_rollout()
            edl_loss = self._compute_edl_loss(inputs)
            total_loss = sft_loss + edl_weight * edl_loss
            if self.is_world_process_zero():
                self.log(
                    {
                        'train/sft_loss': sft_loss.detach().float().item(),
                        'train/edl_loss': edl_loss.detach().float().item(),
                        'train/edl_weight': float(edl_weight),
                        **self._last_edl_stats,
                    })

        if return_outputs:
            return total_loss, outputs
        return total_loss

    # ---------------------------
    # EDL helpers
    # ---------------------------
    def _resolve_edl_step_value(self, value: int | float, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an int or float, not bool.")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            total_steps = self.state.max_steps if self.state is not None else None
            if total_steps is None:
                raise ValueError(f"{name} as fraction requires Trainer state.max_steps.")
            if total_steps <= 0:
                raise ValueError('Trainer max_steps must be > 0 when using fractional EDL steps.')
            return int(value * total_steps)
        raise ValueError(f"{name} must be int steps or float fraction in [0, 1].")

    def _get_edl_weight(self, step: int) -> float:
        if self._edl_weight_start_step_resolved is None or self._edl_weight_warmup_steps_resolved is None:
            self._edl_weight_start_step_resolved = self._resolve_edl_step_value(
                self.edl_weight_start_step, 'edl_weight_start_step')
            self._edl_weight_warmup_steps_resolved = self._resolve_edl_step_value(
                self.edl_weight_warmup_steps, 'edl_weight_warmup_steps')
            logger.info(
                f"Resolved EDL weight schedule: start_step={self._edl_weight_start_step_resolved}, "
                f"warmup_steps={self._edl_weight_warmup_steps_resolved}")
        start_step = self._edl_weight_start_step_resolved
        warmup_steps = self._edl_weight_warmup_steps_resolved
        if step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return self.edl_weight
        progress = (step - start_step) / max(1, warmup_steps)
        return self.edl_weight * max(0.0, min(1.0, progress))

    def _compute_edl_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        prefill_prompts: list[str] = inputs['prefill_prompt']
        completion_prompts: list[str | None] = inputs['completion_prompt']
        final_answers: list[str] = inputs['final_answer']

        rollouts, rollout_logprobs, rollout_token_ids = self._vllm_generate(prefill_prompts)

        # Tokenize prefill once per sample.
        prompt_ids_list = [self.processing_class(p, add_special_tokens=False)['input_ids'] for p in prefill_prompts]

        # Flatten rollouts across batch for logprob computation
        flat_prompt_ids: list[list[int]] = []
        flat_completion_ids: list[list[int]] = []
        flat_rollout_logprobs: list[list[float]] = []
        flat_sample_idx: list[int] = []
        rollout_texts_per_sample: list[list[str]] = []
        disable_is = False
        empty_completion_count = 0

        for sample_idx, sample_rollouts in enumerate(rollouts):
            rollout_texts_per_sample.append(sample_rollouts)
            for r_idx, text in enumerate(sample_rollouts):
                flat_prompt_ids.append(prompt_ids_list[sample_idx])
                if rollout_token_ids is not None:
                    completion_ids = rollout_token_ids[sample_idx][r_idx]
                else:
                    completion_ids = self.processing_class(text, add_special_tokens=False)['input_ids']
                if len(completion_ids) == 0:
                    empty_completion_count += 1
                flat_completion_ids.append(completion_ids)
                if rollout_logprobs is not None:
                    lp = rollout_logprobs[sample_idx][r_idx]
                    if len(lp) != len(completion_ids):
                        min_len = min(len(lp), len(completion_ids))
                        if not self._warned_logprob_mismatch:
                            logger.warning(
                                f"vLLM logprobs length mismatch for sample {sample_idx}: "
                                f"{len(lp)} != {len(completion_ids)}; truncating to {min_len}.")
                            self._warned_logprob_mismatch = True
                        if min_len == 0:
                            disable_is = True
                        else:
                            completion_ids = completion_ids[:min_len]
                            flat_completion_ids[-1] = completion_ids
                            lp = lp[:min_len]
                            flat_rollout_logprobs.append(lp)
                    else:
                        flat_rollout_logprobs.append(lp)
                flat_sample_idx.append(sample_idx)

        if not flat_prompt_ids:
            raise ValueError('No rollouts returned from vLLM; cannot compute EDL loss.')
        if empty_completion_count > 0:
            raise ValueError(
                f"Encountered {empty_completion_count} empty completion(s) from vLLM; "
                'cannot compute logprobs without tokens.')
        if disable_is:
            if self.vllm_importance_sampling:
                raise ValueError('vLLM logprobs length mismatch detected; cannot apply importance sampling.')
            rollout_logprobs = None
            flat_rollout_logprobs = []

        train_logps, completion_mask, train_entropies = self._compute_completion_logps(
            flat_prompt_ids, flat_completion_ids, compute_entropy=True)
        # Cast to float32 BEFORE summing to ensure gradient accumulation is in float32
        train_seq_logps = (train_logps.float() * completion_mask).sum(dim=-1)

        if rollout_logprobs is not None:
            # Pad vLLM logprobs to match the padded completion length for mask-aligned IS correction.
            vllm_logps_tensor = self._pad_rollout_logprobs(flat_rollout_logprobs, completion_mask.size(1))
        elif self.vllm_importance_sampling:
            raise ValueError('vLLM logprobs missing; cannot apply importance sampling.')

        # Compute importance sampling ratio (optional)
        if self.vllm_importance_sampling and rollout_logprobs is not None:
            is_ratio = self._importance_sampling_ratio(
                train_logps=train_logps,
                vllm_logps=vllm_logps_tensor,
                completion_mask=completion_mask,
            )
            is_ratio_max = float(is_ratio.max().item())
        else:
            is_ratio = None
            is_ratio_max = None

        # Compute per-rollout log weights for self-evidence
        device = train_logps.device
        completion_lengths = completion_mask.sum(dim=-1)  # [total_flat_rollouts]
        log_weights = (self.edl_beta - 1.0) * train_seq_logps
        if is_ratio is not None:
            log_weights = log_weights + torch.log(is_ratio.clamp(min=1e-20))

        # Build loss per sample
        total_loss = torch.tensor(0.0, device=device)
        num_clusters_stats = []
        alpha_strength_stats = []
        alpha_max_stats = []
        gt_prob_stats: list[float] = []
        gt_freq_ratio_stats: list[float] = []
        log_u_total_entries = 0
        log_u_clamped_entries = 0
        num_samples = len(prefill_prompts)

        # Group rollouts by sample in order
        per_sample_log_weights: list[list[torch.Tensor]] = [[] for _ in range(num_samples)]
        for idx, sample_idx in enumerate(flat_sample_idx):
            per_sample_log_weights[sample_idx].append(log_weights[idx])

        per_sample_clusters: list[Any] = [None] * num_samples
        per_sample_correct_idx: list[int] = [0] * num_samples
        missing_gt_indices: list[int] = []

        for sample_idx in range(num_samples):
            sample_rollouts = rollout_texts_per_sample[sample_idx]
            if not sample_rollouts:
                continue
            # Extract final predictions for semantic clustering
            preds = [extract_prediction(t) for t in sample_rollouts]
            clusters, correct_idx = cluster_predictions_with_ground_truth(
                preds, final_answers[sample_idx], classifier=self._cluster_classifier)
            per_sample_clusters[sample_idx] = clusters
            per_sample_correct_idx[sample_idx] = correct_idx
            if not clusters[correct_idx].indices:
                missing_gt_indices.append(sample_idx)

        # gt_log_weights is manually set only when:
        # 1) GT cluster is missing from rollouts, 2) completion_prompt exists and is non-empty,
        # 3) edl_gt_pseudo_logprob_penalty is enabled (>= 0).
        # gt_log_weights[sample_idx] holds a pseudo log-weight for the GT cluster when no
        # rollout matched GT. Entries stay None for samples where GT was found in rollouts.
        gt_log_weights: list[torch.Tensor | None] = [None] * num_samples
        penalty = self.edl_gt_pseudo_logprob_penalty
        if missing_gt_indices:
            if penalty is None or penalty < 0:
                raise ValueError(
                    'GT cluster missing from rollouts, but edl_gt_pseudo_logprob_penalty is disabled. '
                    'Set edl_gt_pseudo_logprob_penalty >= 0 to enable pseudo GT logprob.')
            # missing_pairs: subset of missing_gt_indices that have a valid completion_prompt.
            # Each entry is (sample_idx, comp_ids) where comp_ids is the tokenized GT completion.
            # We need this separate list because not every missing-GT sample has a usable
            # completion_prompt, and _compute_completion_logps requires a contiguous batch.
            missing_pairs: list[tuple[int, list[int]]] = []
            for sample_idx in missing_gt_indices:
                completion_prompt = completion_prompts[sample_idx]
                if completion_prompt:
                    comp_ids = self.processing_class(completion_prompt, add_special_tokens=False)['input_ids']
                    if comp_ids:
                        missing_pairs.append((sample_idx, comp_ids))

            if not missing_pairs:
                raise ValueError(
                    'GT cluster missing but completion_prompt is empty; cannot compute pseudo GT logprob.')

            # Teacher-force GT completions in a single batched forward pass.
            gt_prompt_ids = [prompt_ids_list[idx] for idx, _ in missing_pairs]
            gt_completion_ids = [comp_ids for _, comp_ids in missing_pairs]
            gt_logps, gt_mask = self._compute_completion_logps(gt_prompt_ids, gt_completion_ids)
            gt_seq_logps = (gt_logps * gt_mask).sum(dim=-1).float()
            # Map batch results back: `offset` is the index into the batched gt_seq_logps,
            # while `sample_idx` is the original index into the num_samples-sized arrays.
            for offset, (sample_idx, _) in enumerate(missing_pairs):
                logp = gt_seq_logps[offset] - float(penalty)
                gt_log_weights[sample_idx] = (self.edl_beta - 1.0) * logp

        for sample_idx in range(num_samples):
            sample_rollouts = rollout_texts_per_sample[sample_idx]
            if not sample_rollouts:
                raise ValueError('No rollouts for sample; cannot compute EDL loss.')
            clusters = per_sample_clusters[sample_idx]
            if clusters is None:
                raise ValueError('Missing clusters for sample; cannot compute EDL loss.')
            correct_idx = per_sample_correct_idx[sample_idx]
            gt_weight = gt_log_weights[sample_idx]

            # Compute log U_k via log-sum-exp over rollout weights in each cluster
            if not per_sample_log_weights[sample_idx]:
                raise ValueError('Missing per-sample log weights; cannot compute EDL loss.')
            sample_log_weights = torch.stack(per_sample_log_weights[sample_idx])
            log_U = []
            for cluster_idx, cluster in enumerate(clusters):
                # For each cluster: aggregate rollout log-weights; GT cluster uses rollout mass if present,
                # otherwise uses the pseudo GT logprob.
                base_log_u = None
                if cluster.indices:
                    base_log_u = torch.logsumexp(sample_log_weights[cluster.indices], dim=0)
                if cluster_idx == correct_idx:
                    assert (base_log_u is None) != (gt_weight is None), \
                        'GT cluster must have exactly one of rollout mass or pseudo GT weight.'
                    log_U.append(base_log_u if base_log_u is not None else gt_weight)
                else:
                    assert base_log_u is not None, 'Non-GT clusters should always have rollouts.'
                    log_U.append(base_log_u)
            log_U = torch.stack(log_U)
            # Clamp log_U spread to prevent extreme probability ratios in softmax.
            # With max spread of 5, probability ratios are bounded to exp(5) ≈ 148x.
            log_U = log_U - log_U.max()
            log_u_total_entries += log_U.numel()
            log_u_clamped_entries += int((log_U < -5.0).sum().item())
            log_U = torch.clamp(log_U, min=-5.0)
            probs = torch.softmax(log_U, dim=0)
            gt_prob_stats.append(float(probs[correct_idx].item()))
            gt_freq_ratio_stats.append(len(clusters[correct_idx].indices) / len(sample_rollouts))

            n_effective = len(sample_rollouts) + (1 if gt_weight is not None else 0)
            evidences = probs * float(n_effective)
            sample_loss = self._edl_loss_terms(evidences=evidences, correct_idx=correct_idx)
            total_loss = total_loss + sample_loss
            with torch.no_grad():
                alphas = evidences + 1.0
                num_clusters_stats.append(float(len(clusters)))
                alpha_strength_stats.append(float(alphas.sum().item()))
                alpha_max_stats.append(float(alphas.max().item()))

        if num_clusters_stats:
            mask_float = completion_mask.float()
            mask_sum = mask_float.sum().clamp(min=1.0)
            entropy_mean = (train_entropies * mask_float).sum() / mask_sum
            entropy_max = train_entropies.masked_fill(completion_mask == 0, float('-inf')).max()
            self._last_edl_stats = {
                'train/edl_num_clusters': float(sum(num_clusters_stats) / len(num_clusters_stats)),
                'train/edl_alpha_strength': float(sum(alpha_strength_stats) / len(alpha_strength_stats)),
                'train/edl_alpha_max': float(sum(alpha_max_stats) / len(alpha_max_stats)),
                'train/edl_gt_prob_mean': float(sum(gt_prob_stats) / len(gt_prob_stats)),
                'train/edl_gt_freq_ratio': float(sum(gt_freq_ratio_stats) / len(gt_freq_ratio_stats)),
                'train/edl_gt_miss_rate': float(len(missing_gt_indices) / num_samples),
                'train/edl_completion_len_mean': float(completion_lengths.float().mean().item()),
                'train/edl_completion_len_max': float(completion_lengths.max().item()),
                'train/edl_seq_logps_mean': float(train_seq_logps.mean().item()),
                'train/edl_log_weight_mean': float(log_weights.mean().item()),
                'train/edl_log_weight_std': float(log_weights.std(correction=0).item()),
                'train/edl_log_u_clamp_rate': float(log_u_clamped_entries / max(1, log_u_total_entries)),
                'train/edl_entropy_mean': float(entropy_mean.item()),
                'train/edl_entropy_max': float(entropy_max.item()),
            }
            if is_ratio_max is not None:
                self._last_edl_stats['train/edl_is_ratio_max'] = is_ratio_max
        else:
            self._last_edl_stats = {}

        return total_loss / max(1, num_samples)

    def _edl_loss_terms(
        self,
        evidences: torch.Tensor,
        correct_idx: int,
    ) -> torch.Tensor:
        alphas = evidences + 1.0
        strength = torch.sum(alphas)
        probabilities = alphas / strength
        labels = torch.zeros_like(probabilities)
        labels[correct_idx] = 1.0
        error = (labels - probabilities)**2
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)
        return torch.sum(error + variance)

    # ---------------------------
    # Logprob utilities
    # ---------------------------
    def _compute_completion_logps(
        self,
        prompt_ids_list: list[list[int]],
        completion_ids_list: list[list[int]],
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.args.device
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side='left')
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side='left')

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side='right')
        completion_mask = pad(completion_mask, padding_value=0, padding_side='right')

        if completion_ids.size(1) == 0:
            empty = torch.zeros((completion_ids.size(0), 0), device=device)
            if compute_entropy:
                return empty, empty, empty
            return empty, empty

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        result = self._get_per_token_logps(
            self.model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=self.edl_logprob_batch_size,
            compute_entropy=compute_entropy,
        )
        if compute_entropy:
            logps, entropies = result
            return logps, completion_mask, entropies
        return result, completion_mask

    def _get_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int | None = None,
        compute_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies: list[torch.Tensor] = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start:start + batch_size]
            attention_mask_batch = attention_mask[start:start + batch_size]
            model_inputs = {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch}
            if 'logits_to_keep' in self._model_kwarg_keys:
                model_inputs['logits_to_keep'] = logits_to_keep + 1
            model_inputs['use_cache'] = False
            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]  # exclude last
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.edl_sampling['temperature']
            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(entropy_from_logits(logits))
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)
        logps_cat = torch.cat(all_logps, dim=0)
        if compute_entropy:
            return logps_cat, torch.cat(all_entropies, dim=0)
        return logps_cat

    def _pad_rollout_logprobs(self, logprobs: list[list[float]], max_len: int) -> torch.Tensor:
        device = self.args.device
        tensors = [torch.tensor(lp, device=device, dtype=torch.float32) for lp in logprobs]
        if not tensors:
            return torch.zeros((0, max_len), device=device)
        return pad(tensors, padding_value=0.0, padding_side='right')

    def _importance_sampling_ratio(
        self,
        train_logps: torch.Tensor,
        vllm_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        diff = (train_logps - vllm_logps) * completion_mask
        mode = self.vllm_importance_sampling_mode
        cap = self.vllm_importance_sampling_cap
        log_cap = torch.tensor(cap, dtype=diff.dtype, device=diff.device).log()

        if mode.startswith('token'):
            # Clamp diff before exp to avoid overflow/underflow
            if mode == 'token_truncate':
                diff_clamped = torch.clamp(diff, max=log_cap)
                ratio = torch.exp(diff_clamped)
            elif mode == 'token_mask':
                ratio = torch.exp(diff)
                ratio = ratio.masked_fill(ratio > cap, value=0.0)
            else:
                raise ValueError(f"Unknown IS mode: {mode}")
            log_ratio = torch.log(ratio.clamp(min=1e-20)) * completion_mask
            return torch.exp(log_ratio.sum(dim=-1))

        if mode.startswith('sequence'):
            per_seq_diff = diff.sum(dim=-1)
            # Clamp exponent before exp to avoid overflow
            if mode == 'sequence_truncate':
                per_seq_diff_clamped = torch.clamp(per_seq_diff, max=log_cap)
                ratio = torch.exp(per_seq_diff_clamped)
            elif mode == 'sequence_mask':
                ratio = torch.exp(per_seq_diff)
                ratio = ratio.masked_fill(ratio > cap, value=0.0)
            else:
                raise ValueError(f"Unknown IS mode: {mode}")
            return ratio

        raise ValueError(f"Unknown IS mode: {mode}")

    # ---------------------------
    # vLLM rollout generation
    # ---------------------------
    def _vllm_generate(
            self, prompts: list[str]
    ) -> tuple[list[list[str]], list[list[list[float]]] | None, list[list[list[int]]] | None]:
        """Generate rollout completions from a vLLM server via the OpenAI-compatible API.

        Sends prompts to vLLM and collects completions along with optional per-token
        log-probabilities and token IDs. Logprobs and token_ids follow all-or-nothing
        semantics: if any single choice is missing logprobs (or token IDs), the entire
        collection is discarded and returned as None. This avoids partial data that
        would be unusable for importance sampling.

        When logprobs/token_ids are None, the caller falls back to re-tokenizing
        completion texts with the training model's tokenizer and computing log-probs
        via a forward pass, without importance sampling correction.

        Args:
            prompts: List of prefill prompt strings to generate completions for.

        Returns:
            A tuple of ``(rollouts, logprobs, token_ids)`` where:

            - **rollouts**: ``list[list[str]]`` — Completion texts grouped by prompt.
              ``rollouts[i][j]`` is the j-th completion for the i-th prompt.
            - **logprobs**: ``list[list[list[float]]] | None`` — Per-token log-probabilities
              grouped by prompt and completion, or None if any choice was missing logprobs.
              ``logprobs[i][j][k]`` is the log-prob of the k-th token in the j-th completion
              of the i-th prompt.
            - **token_ids**: ``list[list[list[int]]] | None`` — Token IDs from vLLM's
              tokenizer grouped by prompt and completion, or None if unavailable. Used to
              verify tokenization alignment with the training model for importance sampling.

        Raises:
            ValueError: If ``vllm_base_url`` is not set or ``vllm_api_mode`` is not
                ``'openai'``.
        """
        if self.vllm_base_url is None:
            raise ValueError('vLLM base URL must be set for rollouts.')
        if self.vllm_api_mode != 'openai':
            raise ValueError(f"Unsupported vLLM API mode: {self.vllm_api_mode}")

        payload = {
            'model': self.lora_name,
            'prompt': prompts if len(prompts) > 1 else prompts[0],
            'add_special_tokens': False,
            'return_token_ids': True,
        }
        payload.update(self.edl_sampling)
        data = self._vllm_post(self.vllm_generate_endpoint, payload)
        choices = data.get('choices', [])

        # Group completions per prompt
        num_prompts = len(prompts)
        rollouts: list[list[str]] = [[] for _ in range(num_prompts)]
        logprobs: list[list[list[float]]] | None = [[] for _ in range(num_prompts)]
        token_ids: list[list[list[int]]] | None = [[] for _ in range(num_prompts)]

        for idx, choice in enumerate(choices):
            prompt_idx = choice.get('prompt_index')
            if prompt_idx is None:
                # Assume contiguous ordering: n completions per prompt
                prompt_idx = idx // self.edl_sampling['n']
            text = choice.get('text', '')
            rollouts[prompt_idx].append(text)

            # Skip logprob parsing once we've already given up (a prior choice was missing data).
            if logprobs is not None:
                # vLLM returns logprobs in two possible formats:
                #   dict: {"token_logprobs": [...], "token_ids": [...]}
                #   list: bare list of per-token logprobs (no token_ids embedded)
                lp = choice.get('logprobs')
                token_logprobs = None
                token_ids_list = None
                if isinstance(lp, dict):
                    token_logprobs = lp.get('token_logprobs')
                    token_ids_list = lp.get('token_ids')
                elif isinstance(lp, list):
                    token_logprobs = lp
                # Fallback: token_ids may be a top-level field instead of nested in logprobs.
                if token_ids_list is None:
                    token_ids_list = choice.get('token_ids')
                if token_logprobs is None:
                    # This choice has no usable logprobs — disable logprobs and token_ids
                    # for the entire batch (all-or-nothing: partial logprobs are not useful).
                    logprobs = None
                    token_ids = None
                else:
                    logprobs[prompt_idx].append(token_logprobs)
                    if token_ids is not None:
                        if token_ids_list is None:
                            # token_ids unavailable for this choice — disable for entire batch.
                            token_ids = None
                        else:
                            token_ids[prompt_idx].append(token_ids_list)

        # If any prompt ended up with zero logprob entries, discard all logprobs (and token_ids).
        if logprobs is not None and any(len(x) == 0 for x in logprobs):
            logprobs = None
            token_ids = None
        # Similarly, discard token_ids if any prompt has none (logprobs can still be used without them).
        if token_ids is not None and any(len(x) == 0 for x in token_ids):
            token_ids = None
        if self.vllm_importance_sampling and logprobs is not None and token_ids is None:
            logger.warning('vLLM did not return token_ids; importance sampling may fail if tokenization mismatches.')
        return rollouts, logprobs, token_ids
