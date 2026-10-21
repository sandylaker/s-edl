"""GRPO-style pure RL trainer for self-evidenced EDL rewards."""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate.utils import gather_object
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import nanmax, nanmin, pad, use_adapter

from s_edl.evaluation import ClusterClassifier, cluster_predictions_with_ground_truth, extract_prediction
from s_edl.utils import setup_logger

logger = setup_logger('s_edl')


def _zero_reward_func(completions: list[Any], **kwargs) -> list[float]:
    """Dummy reward function required by GRPOTrainer initialization."""
    return [0.0] * len(completions)


@dataclass
class NewEDLConfig(GRPOConfig):
    """GRPOConfig with pure-RL EDL reward settings.

    Attributes:
        edl_beta: Tempering beta for log evidence, where sampled log evidence is
            ``(edl_beta - 1) * mean_completion_logp``. At ``edl_beta == 1``, sampled rows receive uniform
            histogram evidence.
        edl_gt_pseudo_logprob_penalty: Penalty subtracted from the old-policy mean token logprob of a teacher-forced
            GT completion before constructing missing-GT pseudo evidence.
        edl_log_u_min: Floor for normalized class log evidence ``log_U - max(log_U)``. This bounds class probability
            ratios before computing training rewards.
        edl_reward_scale: Multiplicative scale applied to detached EDL rewards/advantages.
        edl_advantage_clip: Optional symmetric clamp on final EDL advantages. ``None`` disables this extra clamp.
        edl_aux_gt_enabled: Whether to add a teacher-forced auxiliary GT PPO term when no sampled completion lands in
            the GT semantic class.
        edl_aux_gt_loss_weight: Scalar weight on the auxiliary GT PPO loss contribution.
        edl_aux_gt_normalization: How to normalize the auxiliary GT branch. ``"row_matched"`` gives each active
            teacher-forced row the same outer-batch coefficient as one sampled rollout row. ``"active_mean"``
            reproduces the legacy behavior that averages only over active auxiliary rows.
        edl_missing_gt_strategy: How sampled rollout rows are handled when no rollout lands in the GT class.
            ``"pseudo"`` computes sampled rewards after inserting pseudo-GT evidence. ``"skip"`` sets sampled
            rewards to zero. The independently controlled auxiliary GT branch may still be active in either mode.
    """

    edl_beta: float = field(default=1.2, metadata={'help': 'Tempering beta >= 1.'})
    edl_gt_pseudo_logprob_penalty: float = field(
        default=4.605,
        metadata={'help': 'Penalty subtracted from GT mean token logprob when the GT class is absent from rollouts.'},
    )
    edl_log_u_min: float = field(
        default=-5.0,
        metadata={'help': 'Minimum normalized class log evidence used for reward computation.'},
    )
    edl_reward_scale: float = field(default=1.0, metadata={'help': 'Scale applied to EDL rewards.'})
    edl_advantage_clip: float | None = field(
        default=None,
        metadata={'help': 'Optional absolute clamp for EDL advantages. Disabled when None.'},
    )
    edl_aux_gt_enabled: bool = field(
        default=True,
        metadata={'help': 'Add an auxiliary teacher-forced GT PPO term when GT is absent from rollouts.'},
    )
    edl_aux_gt_loss_weight: float = field(
        default=1.0,
        metadata={'help': 'Loss weight for the auxiliary missing-GT teacher-forced PPO term.'},
    )
    edl_aux_gt_normalization: str = field(
        default='row_matched',
        metadata={
            'help':
            'Auxiliary GT normalization. "row_matched" multiplies the mean over active teacher-forced GT '
            'rows by their fraction of rollout rows, so each auxiliary row has the same outer-batch coefficient as '
            'one sampled row. Legacy "active_mean" uses the mean over active auxiliary rows directly, regardless of '
            'how few rows are active.'
        },
    )
    edl_missing_gt_strategy: str = field(
        default='pseudo',
        metadata={
            'help': 'Missing-GT sampled-row handling: "pseudo" uses pseudo evidence, "skip" zeros sampled rewards.'
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.edl_beta < 1.0:
            raise ValueError('edl_beta must be >= 1.0.')
        if self.edl_log_u_min > 0.0:
            raise ValueError('edl_log_u_min must be <= 0.')
        if self.edl_advantage_clip is not None and self.edl_advantage_clip <= 0.0:
            raise ValueError('edl_advantage_clip must be positive when set.')
        if self.edl_aux_gt_loss_weight < 0.0:
            raise ValueError('edl_aux_gt_loss_weight must be >= 0.')
        if self.edl_aux_gt_normalization not in {'row_matched', 'active_mean'}:
            raise ValueError('edl_aux_gt_normalization must be either "row_matched" or "active_mean".')
        if self.edl_missing_gt_strategy not in {'pseudo', 'skip'}:
            raise ValueError('edl_missing_gt_strategy must be either "pseudo" or "skip".')


@dataclass
class EDLGroupReward:
    """EDL reward outputs for one prompt group.

    Attributes:
        rewards: Clamped training rewards for sampled completions, shape ``(G,)``.
        raw_rewards: Unclamped diagnostic rewards for sampled completions, shape ``(G,)``.
        correct_mask: Whether each sampled completion belongs to the GT class, shape ``(G,)``.
        clamped_completion_mask: Whether each sampled completion's class was affected by ``log_u_min``, shape ``(G,)``.
        pseudo_gt_reward: Clamped reward for the pseudo-GT completion when GT is missing, scalar or ``None``.
        pseudo_gt_prob: Normalized pseudo-GT class probability when GT is missing, scalar or ``None``.
        num_clusters: Number of semantic classes for this prompt group.
        gt_freq_ratio: Fraction of sampled completions assigned to the GT class.
        clamped_class_count: Number of semantic classes floored by ``log_u_min``.
        num_classes: Number of semantic classes, including a GT-only class when GT is missing.
    """

    rewards: torch.Tensor
    raw_rewards: torch.Tensor
    correct_mask: torch.Tensor
    clamped_completion_mask: torch.Tensor
    pseudo_gt_reward: torch.Tensor | None
    pseudo_gt_prob: torch.Tensor | None
    num_clusters: int
    gt_freq_ratio: float
    clamped_class_count: int
    num_classes: int


def compute_edl_group_rewards(
    sample_log_weights: torch.Tensor,
    cluster_indices: list[list[int]],
    correct_idx: int,
    *,
    gt_log_weight: torch.Tensor | None = None,
    log_u_min: float = -5.0,
    reward_scale: float = 1.0,
) -> EDLGroupReward:
    """Compute detached EDL rewards for one prompt's sampled completions.

    Args:
        sample_log_weights: Log self-evidence weights for sampled completions, shape ``(G,)``.
        cluster_indices: Per-class indices into ``sample_log_weights``. Every non-GT class must be non-empty.
        correct_idx: Index of the GT semantic class.
        gt_log_weight: Optional pseudo-GT log evidence when the GT class is missing, scalar tensor.
        log_u_min: Minimum normalized class log evidence for clamped rewards.
        reward_scale: Scalar multiplier applied to rewards.

    Returns:
        EDLGroupReward containing sampled rewards of shape ``(G,)`` and scalar diagnostics.
    """
    if sample_log_weights.dim() != 1:
        raise ValueError('sample_log_weights must be a 1D tensor.')
    if not cluster_indices:
        raise ValueError('cluster_indices must be non-empty.')
    if correct_idx < 0 or correct_idx >= len(cluster_indices):
        raise ValueError('correct_idx is out of range.')

    device = sample_log_weights.device
    dtype = sample_log_weights.dtype
    num_samples = sample_log_weights.numel()
    num_classes = len(cluster_indices)

    # Build per-sample and per-class views of the same clustering. ``sample_class_idx`` maps each sampled completion
    # back to its semantic class, while ``class_log_u`` stores one aggregated log evidence scalar per class.
    sample_class_idx = torch.empty(num_samples, dtype=torch.long, device=device)
    class_log_u = []

    # The GT class is represented either by sampled completions or by one pseudo-GT evidence value, never both.
    gt_has_samples = len(cluster_indices[correct_idx]) > 0
    if gt_has_samples and gt_log_weight is not None:
        raise ValueError('gt_log_weight should only be set when the GT class has no sampled completions.')
    if not gt_has_samples and gt_log_weight is None:
        raise ValueError('gt_log_weight is required when the GT class has no sampled completions.')

    # ``cluster_indices`` must be a partition of the sampled completions: every sample index appears exactly once.
    seen_indices: set[int] = set()
    for indices in cluster_indices:
        for sample_idx in indices:
            if sample_idx < 0 or sample_idx >= num_samples:
                raise ValueError('cluster_indices contain an out-of-range sample index.')
            if sample_idx in seen_indices:
                raise ValueError('cluster_indices contain a duplicate sample index.')
            seen_indices.add(sample_idx)
    if len(seen_indices) != num_samples:
        raise ValueError('cluster_indices must cover every sampled completion exactly once.')

    # Aggregate sample-level evidence into class-level evidence. Non-GT classes must come from at least one sampled
    # completion; the missing GT class is the only class allowed to use the pseudo-GT evidence scalar.
    for class_idx, indices in enumerate(cluster_indices):
        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            sample_class_idx[index_tensor] = class_idx
            class_log_u.append(torch.logsumexp(sample_log_weights[index_tensor], dim=0))
        elif class_idx == correct_idx and gt_log_weight is not None:
            class_log_u.append(gt_log_weight.to(device=device, dtype=dtype))
        else:
            raise ValueError('Non-GT classes must contain sampled completions.')

    class_log_u = torch.stack(class_log_u)

    # Shift all class log evidence by the same constant before clamping. This keeps softmax probabilities unchanged
    # while making ``log_u_min`` a relative floor below the strongest class.
    normalized_log_u = class_log_u - class_log_u.max()
    clamped_class_mask = normalized_log_u < float(log_u_min)
    clamped_log_u = torch.clamp(normalized_log_u, min=float(log_u_min))

    def rewards_from_class_log_u(
            class_log_u_for_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # Convert class log evidence into class probabilities and the corresponding Dirichlet parameters.
        probs = torch.softmax(class_log_u_for_probs, dim=0)
        n_effective = float(num_samples + (0 if gt_log_weight is None else 1))
        alphas = probs * n_effective + 1.0
        strength = n_effective + float(num_classes)
        labels = torch.zeros(num_classes, dtype=dtype, device=device)
        labels[correct_idx] = 1.0

        # ``d`` is the EDL loss derivative with respect to each class probability, and ``d_bar`` is the softmax
        # expectation term that appears when converting that derivative to log-evidence coordinates.
        d = n_effective * (2.0 * alphas - 2.0 * labels * (strength + 1.0) + 1.0) / (strength * (strength + 1.0))
        d_bar = torch.sum(probs * d)

        rewards = torch.empty(num_samples, dtype=dtype, device=device)
        for class_idx, indices in enumerate(cluster_indices):
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)

            # Split each class-level gradient among the sampled completions in that class according to their
            # within-class evidence weights. Rewards are negative gradients in log-evidence coordinates.
            within_class_weights = torch.softmax(sample_log_weights[index_tensor], dim=0)
            # probs[class] * (u_i / U_class) recovers the paper's normalized sample weight w_i = u_i / U_total.
            sample_weights = probs[class_idx] * within_class_weights
            rewards[index_tensor] = -sample_weights * (d[class_idx] - d_bar) * reward_scale

        pseudo_reward = None
        if gt_log_weight is not None:
            # The pseudo-GT row is not a sampled completion, but it receives the same negative-gradient advantage
            # used later by the auxiliary teacher-forced loss.
            pseudo_reward = -probs[correct_idx] * (d[correct_idx] - d_bar) * reward_scale
        return rewards, pseudo_reward, probs

    raw_rewards, _, _ = rewards_from_class_log_u(normalized_log_u)
    rewards, pseudo_gt_reward, probs = rewards_from_class_log_u(clamped_log_u)

    correct_mask = sample_class_idx == correct_idx
    clamped_completion_mask = clamped_class_mask[sample_class_idx]
    pseudo_gt_prob = probs[correct_idx] if gt_log_weight is not None else None

    return EDLGroupReward(
        rewards=rewards,
        raw_rewards=raw_rewards,
        correct_mask=correct_mask,
        clamped_completion_mask=clamped_completion_mask,
        pseudo_gt_reward=pseudo_gt_reward,
        pseudo_gt_prob=pseudo_gt_prob,
        num_clusters=num_classes,
        gt_freq_ratio=float(len(cluster_indices[correct_idx]) / max(1, num_samples)),
        clamped_class_count=int(clamped_class_mask.sum().item()),
        num_classes=num_classes,
    )


class NewEDLTrainer(GRPOTrainer):
    """Pure RL trainer that converts EDL gradients into GRPO/PPO advantages."""

    _tag_names = ['trl', 'grpo', 'new-edl']
    _name = 'NewEDL'

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: NewEDLConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_funcs=None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config=None,
        tools: list[Callable] | None = None,
        rollout_func=None,
        cluster_classifier: ClusterClassifier | None = None,
    ) -> None:
        if args is None:
            args = NewEDLConfig('new-edl')
        reward_funcs = reward_funcs or _zero_reward_func
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func,
        )
        self.edl_beta = self.args.edl_beta
        self.edl_gt_pseudo_logprob_penalty = self.args.edl_gt_pseudo_logprob_penalty
        self.edl_log_u_min = self.args.edl_log_u_min
        self.edl_reward_scale = self.args.edl_reward_scale
        self.edl_advantage_clip = self.args.edl_advantage_clip
        self.edl_aux_gt_enabled = self.args.edl_aux_gt_enabled
        self.edl_aux_gt_loss_weight = self.args.edl_aux_gt_loss_weight
        self.edl_aux_gt_normalization = self.args.edl_aux_gt_normalization
        self.edl_missing_gt_strategy = self.args.edl_missing_gt_strategy
        self._cluster_classifier = cluster_classifier

    def _tokenize_prompt_completion(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize prompt/completion text into padded model inputs.

        Returns:
            ``prompt_ids`` and ``prompt_mask`` with shape ``(B, P)``, plus ``completion_ids`` and
            ``completion_mask`` with shape ``(B, C)``. Prompts are left-padded and completions are right-padded.
        """
        device = self.accelerator.device
        prompt_ids = [
            torch.tensor(self.processing_class(prompt, add_special_tokens=False)['input_ids'], device=device)
            for prompt in prompts
        ]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side='left')
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side='left')

        completion_ids = [
            torch.tensor(self.processing_class(completion, add_special_tokens=False)['input_ids'], device=device)
            for completion in completions
        ]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side='right')
        completion_mask = pad(completion_mask, padding_value=0, padding_side='right')
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _compute_text_completion_logps(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        model: nn.Module | None = None,
        no_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-force completions and compute old/current per-token logprobs.

        Returns:
            ``prompt_ids``/``prompt_mask`` with shape ``(B, P)``, ``completion_ids``/``completion_mask`` with shape
            ``(B, C)``, and ``per_token_logps`` with shape ``(B, C)``.
        """
        if not prompts:
            device = self.accelerator.device
            empty_ids = torch.empty((0, 0), dtype=torch.long, device=device)
            empty_logps = torch.empty((0, 0), dtype=torch.float32, device=device)
            return empty_ids, empty_ids, empty_ids, empty_ids, empty_logps

        prompt_ids, prompt_mask, completion_ids, completion_mask = self._tokenize_prompt_completion(
            prompts, completions)
        if completion_ids.size(1) == 0:
            raise ValueError('Completion is empty; cannot compute logprobs.')
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size)
        target_model = model or self.model

        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            per_token_logps, _ = self._get_per_token_logps_and_entropies(
                target_model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
            )
        return prompt_ids, prompt_mask, completion_ids, completion_mask, per_token_logps

    def _mean_logps(self, per_token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reduce per-token completion logprobs to mean token logprobs, shape ``(B,)``."""
        return (per_token_logps.float() * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)

    def _compute_global_edl_rewards(
        self,
        *,
        prompts: list[Any],
        completions_text: list[str],
        final_answers: list[str],
        completion_prompts: list[str | None],
        mean_logps: torch.Tensor,
        num_generations: int,
        mode: str,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
        """Cluster all gathered completions and compute EDL rewards.

        Args:
            prompts: Gathered prompts, length ``B_global * G``.
            completions_text: Gathered decoded sampled completions, length ``B_global * G``.
            final_answers: Gathered GT answers, length ``B_global * G``.
            completion_prompts: Gathered teacher-forced GT completions, length ``B_global * G``.
            mean_logps: Old-policy mean token logprobs for sampled completions, shape ``(B_global * G,)``.
            num_generations: Number of sampled completions per prompt group, ``G``.
            mode: ``"train"`` or ``"eval"`` for missing-GT error handling.

        Returns:
            Detached sampled rewards with shape ``(B_global * G,)``, scalar metrics, and sparse auxiliary-GT info
            keyed by global prompt-group index.
        """
        device = mean_logps.device
        dtype = mean_logps.dtype
        total_completions = len(completions_text)
        if total_completions % num_generations != 0:
            raise ValueError(
                f'Number of completions ({total_completions}) must be divisible by '
                f'num_generations ({num_generations}).')

        # 1. Semantic clustering: build one class partition per prompt group and identify the GT class.
        group_infos: list[dict[str, Any]] = []
        missing_group_indices: list[int] = []
        for group_idx, start in enumerate(range(0, total_completions, num_generations)):
            end = start + num_generations
            sample_texts = completions_text[start:end]
            preds = [extract_prediction(text) for text in sample_texts]
            clusters, correct_idx = cluster_predictions_with_ground_truth(
                preds,
                final_answers[start],
                classifier=self._cluster_classifier,
            )
            missing_gt = len(clusters[correct_idx].indices) == 0
            if missing_gt:
                missing_group_indices.append(group_idx)
            group_infos.append(
                {
                    'start': start,
                    'end': end,
                    'clusters': clusters,
                    'correct_idx': correct_idx,
                    'missing_gt': missing_gt,
                })

        # 2. Missing-GT handling: teacher-force GT completions only for groups that need pseudo evidence.
        pseudo_gt_mean_logps: dict[int, torch.Tensor] = {}
        needs_pseudo_gt_scores = self.edl_missing_gt_strategy == 'pseudo' or self.edl_aux_gt_enabled
        if missing_group_indices and needs_pseudo_gt_scores:
            gt_prompts = []
            gt_completions = []
            valid_missing_groups = []
            for group_idx in missing_group_indices:
                start = group_infos[group_idx]['start']
                completion_prompt = completion_prompts[start]
                if not completion_prompt:
                    if mode == 'train':
                        raise ValueError('GT class missing but completion_prompt is empty; cannot compute pseudo GT.')
                    continue
                gt_prompts.append(prompts[start])
                gt_completions.append(completion_prompt)
                valid_missing_groups.append(group_idx)

            # In eval, missing-GT groups can lack completion_prompt; only score pseudo GTs that were collected.
            if gt_prompts:
                _, _, _, gt_completion_mask, gt_per_token_logps = self._compute_text_completion_logps(
                    gt_prompts,
                    gt_completions,
                    no_grad=True,
                )
                gt_mean_logps = self._mean_logps(gt_per_token_logps, gt_completion_mask)
                for offset, group_idx in enumerate(valid_missing_groups):
                    pseudo_gt_mean_logps[group_idx] = gt_mean_logps[offset]

        all_rewards = torch.empty(total_completions, dtype=dtype, device=device)
        raw_rewards = torch.empty_like(all_rewards)
        correct_mask = torch.zeros(total_completions, dtype=torch.bool, device=device)
        clamped_completion_mask = torch.zeros(total_completions, dtype=torch.bool, device=device)
        aux_gt_by_group: dict[int, dict[str, torch.Tensor]] = {}

        num_clusters_stats: list[float] = []
        gt_freq_ratio_stats: list[float] = []
        pseudo_gt_reward_stats: list[float] = []
        pseudo_gt_logp_stats: list[float] = []
        pseudo_gt_log_weight_stats: list[float] = []
        pseudo_gt_prob_stats: list[float] = []
        pseudo_gt_sampled_reward_sum_stats: list[float] = []
        pseudo_gt_conservation_error_stats: list[float] = []
        pseudo_gt_active_groups = 0
        skipped_missing_gt_groups = 0
        clamped_class_total = 0
        class_total = 0

        # 3. Convert class evidence into detached rewards and accumulate diagnostics.
        for group_idx, info in enumerate(group_infos):
            start = info['start']
            end = info['end']
            group_logps = mean_logps[start:end]
            # Rewards are EDL advantages in log-evidence coordinates, so do not apply another `(edl_beta - 1)` factor.
            # At beta=1 this becomes histogram-based EDL over samples.
            sample_log_weights = (self.edl_beta - 1.0) * group_logps
            gt_log_weight = None
            skip_sampled_rewards = info['missing_gt'] and self.edl_missing_gt_strategy == 'skip'
            if info['missing_gt'] and group_idx in pseudo_gt_mean_logps:
                pseudo_logp = pseudo_gt_mean_logps[group_idx]
                gt_log_weight = (self.edl_beta - 1.0) * (pseudo_logp - float(self.edl_gt_pseudo_logprob_penalty))
                pseudo_gt_logp_stats.append(float(pseudo_logp.detach().float().item()))
                pseudo_gt_log_weight_stats.append(float(gt_log_weight.detach().float().item()))
            elif info['missing_gt']:
                if mode == 'train' and needs_pseudo_gt_scores:
                    raise ValueError('GT class missing but completion_prompt is empty; cannot compute pseudo GT.')
                all_rewards[start:end] = 0.0
                raw_rewards[start:end] = 0.0
                skipped_missing_gt_groups += 1
                num_clusters_stats.append(float(len(info['clusters'])))
                gt_freq_ratio_stats.append(0.0)
                class_total += len(info['clusters'])
                continue

            reward_info = compute_edl_group_rewards(
                sample_log_weights,
                [cluster.indices for cluster in info['clusters']],
                info['correct_idx'],
                gt_log_weight=gt_log_weight,
                log_u_min=self.edl_log_u_min,
                reward_scale=self.edl_reward_scale,
            )

            if skip_sampled_rewards:
                all_rewards[start:end] = 0.0
                raw_rewards[start:end] = 0.0
                skipped_missing_gt_groups += 1
            else:
                all_rewards[start:end] = reward_info.rewards
                raw_rewards[start:end] = reward_info.raw_rewards
                correct_mask[start:end] = reward_info.correct_mask
                clamped_completion_mask[start:end] = reward_info.clamped_completion_mask
            num_clusters_stats.append(float(reward_info.num_clusters))
            gt_freq_ratio_stats.append(float(reward_info.gt_freq_ratio))
            clamped_class_total += reward_info.clamped_class_count
            class_total += reward_info.num_classes

            if gt_log_weight is not None and reward_info.pseudo_gt_reward is not None:
                pseudo_gt_active_groups += 1
                pseudo_gt_reward = reward_info.pseudo_gt_reward.detach()
                pseudo_gt_reward_stats.append(float(pseudo_gt_reward.float().item()))
                pseudo_gt_prob_stats.append(float(reward_info.pseudo_gt_prob.detach().float().item()))
                sampled_reward_sum = reward_info.rewards.detach().sum()
                pseudo_gt_sampled_reward_sum_stats.append(float(sampled_reward_sum.float().item()))
                conservation_error = torch.abs(sampled_reward_sum + pseudo_gt_reward)
                pseudo_gt_conservation_error_stats.append(float(conservation_error.float().item()))
                if self.edl_aux_gt_enabled:
                    aux_gt_by_group[group_idx] = {
                        'advantage': pseudo_gt_reward.to(device=device, dtype=dtype),
                        'mean_logp': pseudo_gt_mean_logps[group_idx].detach(),
                        'log_weight': gt_log_weight.detach(),
                    }

        # 4. Optional advantage clipping is separate from log_U clamping and only affects the final PPO advantages.
        unclipped_rewards = all_rewards.clone()
        advantage_clip_rate = None
        if self.edl_advantage_clip is not None:
            clip = float(self.edl_advantage_clip)
            clipped_rewards = torch.clamp(all_rewards, min=-clip, max=clip)
            advantage_clip_rate = (clipped_rewards != all_rewards).float().mean()
            all_rewards = clipped_rewards

        reward_eps = 1e-12
        correct_negative = correct_mask & (all_rewards < -reward_eps)
        incorrect_positive = (~correct_mask) & (all_rewards > reward_eps)
        clamp_delta = torch.abs(unclipped_rewards - raw_rewards)
        raw_sign = torch.sign(raw_rewards)
        clamped_sign = torch.sign(unclipped_rewards)
        sign_flip = raw_sign != clamped_sign

        # 5. Metrics compare training rewards, raw unclamped diagnostics, correctness signs, and missing-GT behavior.
        metrics: dict[str, Any] = {
            'edl/reward_mean': all_rewards.float().mean(),
            'edl/reward_min': all_rewards.float().min(),
            'edl/reward_max': all_rewards.float().max(),
            'edl/num_clusters': sum(num_clusters_stats) / max(1, len(num_clusters_stats)),
            'edl/gt_freq_ratio': sum(gt_freq_ratio_stats) / max(1, len(gt_freq_ratio_stats)),
            'edl/gt_miss_rate': len(missing_group_indices) / max(1, len(group_infos)),
            'edl/reward_correct_below_zero_rate': correct_negative.float().mean(),
            'edl/reward_incorrect_above_zero_rate': incorrect_positive.float().mean(),
            'edl/reward_correctness_sign_mismatch_rate': (correct_negative | incorrect_positive).float().mean(),
            'edl/log_u_clamped_class_rate': clamped_class_total / max(1, class_total),
            'edl/log_u_clamped_completion_rate': clamped_completion_mask.float().mean(),
            'edl/clamp_reward_delta_abs': clamp_delta.float().mean(),
            'edl/clamp_reward_delta_abs_max': clamp_delta.float().max(),
            'edl/clamp_reward_sign_flip_rate': sign_flip.float().mean(),
            'edl/pseudo_gt_active_rate': pseudo_gt_active_groups / max(1, len(group_infos)),
            'edl/aux_gt_active_rate': len(aux_gt_by_group) / max(1, len(group_infos)),
            'edl/missing_gt_skipped_rate': skipped_missing_gt_groups / max(1, len(group_infos)),
        }
        if pseudo_gt_reward_stats:
            pseudo_abs = [abs(value) for value in pseudo_gt_reward_stats]
            metrics.update(
                {
                    'edl/pseudo_gt_reward':
                    sum(pseudo_gt_reward_stats) / len(pseudo_gt_reward_stats),
                    'edl/pseudo_gt_reward_abs_max':
                    max(pseudo_abs),
                    'edl/pseudo_gt_logp':
                    sum(pseudo_gt_logp_stats) / len(pseudo_gt_logp_stats),
                    'edl/pseudo_gt_log_weight':
                    sum(pseudo_gt_log_weight_stats) / len(pseudo_gt_log_weight_stats),
                    'edl/pseudo_gt_prob':
                    sum(pseudo_gt_prob_stats) / len(pseudo_gt_prob_stats),
                    'edl/pseudo_gt_sampled_reward_sum':
                    (sum(pseudo_gt_sampled_reward_sum_stats) / len(pseudo_gt_sampled_reward_sum_stats)),
                    'edl/pseudo_gt_conservation_error_abs_max':
                    max(pseudo_gt_conservation_error_stats),
                })
        if advantage_clip_rate is not None:
            metrics['edl/advantage_clip_rate'] = advantage_clip_rate

        return all_rewards.detach(), metrics, aux_gt_by_group

    def _prepare_local_aux_gt_inputs(
        self,
        *,
        inputs: list[dict[str, Any]],
        prompts: list[Any],
        aux_gt_by_group: dict[int, dict[str, torch.Tensor]],
        num_generations: int,
        local_process_offset: int,
    ) -> dict[str, torch.Tensor] | None:
        """Build batch-aligned tensors for the auxiliary missing-GT PPO branch.

        The normal GRPO tensors contain only sampled rollout rows. When a prompt group misses the GT class, we also
        need to optimize one teacher-forced GT completion, but inserting it into the rollout batch would break the
        fixed ``num_generations`` grouping. Instead, we create parallel auxiliary tensors whose first dimension matches
        the local rollout batch and use ``aux_gt_mask`` to mark which rows are real aux examples.

        Example with three prompt groups and ``num_generations = 2``:
        rows ``0,1`` are group 0, rows ``2,3`` are group 1, and rows ``4,5`` are group 2. If only groups 1 and 2 miss
        GT, this method stores their teacher-forced GT completions in carrier slots ``2`` and ``4``:
        ``aux_gt_mask = [False, False, True, False, True, False]``. False rows are padding/zeros and are ignored by
        ``_compute_aux_gt_loss``; True rows are selected as the compact aux batch.

        Returned tensors all have local rollout-batch leading dimension ``B_local`` so TRL's GRPO buffering and
        gradient-accumulation slicing keep the aux data attached to the same chunk as its prompt group:
        ``aux_gt_mask`` and ``aux_gt_advantages`` have shape ``(B_local,)``; prompt tensors have shape
        ``(B_local, P_aux)``; completion/logprob tensors have shape ``(B_local, C_aux)``.
        """
        if not self.edl_aux_gt_enabled or not aux_gt_by_group:
            return None

        # 1. Scan the local rollout rows and collect one teacher-forced GT example per missing-GT prompt group. The aux
        # row uses the group's first rollout row as a carrier slot, e.g. group 1 with G=2 uses local row 2.
        aux_prompts = []
        aux_completions = []
        aux_advantages = []
        aux_local_indices = []
        for local_idx, example in enumerate(inputs):
            global_idx = local_process_offset + local_idx
            if global_idx % num_generations != 0:
                continue
            group_idx = global_idx // num_generations
            aux_info = aux_gt_by_group.get(group_idx)
            completion_prompt = example.get('completion_prompt')
            if aux_info is None or not completion_prompt:
                continue
            aux_prompts.append(prompts[local_idx])
            aux_completions.append(completion_prompt)
            aux_advantages.append(aux_info['advantage'])
            aux_local_indices.append(local_idx)

        if not aux_prompts:
            return None

        # 2. Score the compact teacher-forced aux batch under the old policy. The returned per-token logprobs are the
        # denominator of the aux PPO/GRPO importance ratio.
        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            old_per_token_logps,
        ) = self._compute_text_completion_logps(
            aux_prompts,
            aux_completions,
            no_grad=True,
        )
        ref_per_token_logps = None
        if self.beta != 0.0:
            # 3. If GRPO KL is enabled, score the same compact aux batch under the reference policy.
            if self.ref_model is not None:
                _, _, _, _, ref_per_token_logps = self._compute_text_completion_logps(
                    aux_prompts,
                    aux_completions,
                    model=self.ref_model,
                    no_grad=True,
                )
            else:
                # This mirrors GRPOTrainer's PEFT path: a named "ref" adapter may provide the frozen reference.
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                peft_config = getattr(unwrapped_model, 'peft_config', None)
                if peft_config is not None:
                    with use_adapter(unwrapped_model, adapter_name='ref' if 'ref' in peft_config else None):
                        _, _, _, _, ref_per_token_logps = self._compute_text_completion_logps(
                            aux_prompts,
                            aux_completions,
                            no_grad=True,
                        )

        # 4. Allocate full-size carrier tensors. They must have the same leading dimension as the rollout batch so TRL
        # can slice them alongside prompt_ids/completion_ids/advantages without losing alignment.
        batch_size = len(inputs)
        aux_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.accelerator.device)
        full_prompt_ids = torch.full(
            (batch_size, prompt_ids.size(1)),
            self.pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        full_prompt_mask = torch.zeros(
            (batch_size, prompt_mask.size(1)),
            dtype=prompt_mask.dtype,
            device=prompt_mask.device,
        )
        full_completion_ids = torch.full(
            (batch_size, completion_ids.size(1)),
            self.pad_token_id,
            dtype=completion_ids.dtype,
            device=completion_ids.device,
        )
        full_completion_mask = torch.zeros(
            (batch_size, completion_mask.size(1)),
            dtype=completion_mask.dtype,
            device=completion_mask.device,
        )
        full_old_per_token_logps = torch.zeros(
            (batch_size, old_per_token_logps.size(1)),
            dtype=old_per_token_logps.dtype,
            device=old_per_token_logps.device,
        )
        full_ref_per_token_logps = None
        if ref_per_token_logps is not None:
            full_ref_per_token_logps = torch.zeros(
                (batch_size, ref_per_token_logps.size(1)),
                dtype=ref_per_token_logps.dtype,
                device=ref_per_token_logps.device,
            )
        full_advantages = torch.zeros(batch_size, dtype=old_per_token_logps.dtype, device=self.accelerator.device)

        # 5. Scatter compact aux rows into their carrier slots. Rows where aux_gt_mask stays False are placeholders;
        # _compute_aux_gt_loss drops them by boolean indexing before computing any aux loss.
        active_indices = torch.tensor(aux_local_indices, dtype=torch.long, device=self.accelerator.device)
        aux_mask[active_indices] = True
        full_prompt_ids[active_indices] = prompt_ids
        full_prompt_mask[active_indices] = prompt_mask
        full_completion_ids[active_indices] = completion_ids
        full_completion_mask[active_indices] = completion_mask
        full_old_per_token_logps[active_indices] = old_per_token_logps
        if full_ref_per_token_logps is not None:
            full_ref_per_token_logps[active_indices] = ref_per_token_logps
        full_advantages[active_indices] = torch.stack(aux_advantages).to(
            device=self.accelerator.device,
            dtype=old_per_token_logps.dtype,
        )
        aux_inputs = {
            'aux_gt_mask': aux_mask,
            'aux_gt_prompt_ids': full_prompt_ids,
            'aux_gt_prompt_mask': full_prompt_mask,
            'aux_gt_completion_ids': full_completion_ids,
            'aux_gt_completion_mask': full_completion_mask,
            'aux_gt_old_per_token_logps': full_old_per_token_logps,
            'aux_gt_advantages': full_advantages,
        }
        if full_ref_per_token_logps is not None:
            aux_inputs['aux_gt_ref_per_token_logps'] = full_ref_per_token_logps
        return aux_inputs

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = 'train' if self.model.training else 'eval'

        prompts = [x['prompt'] for x in inputs]

        if self.environments:
            for prompt, environment, reset_kwargs in zip(prompts, self.environments, inputs, strict=True):
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                prompt[-1]['content'] += observation

        if 'images' in inputs[0]:
            images = [example.get('images') for example in inputs]
        elif 'image' in inputs[0]:
            images = [[example.get('image')] if example.get('image') is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>},
        # {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    'Multimodal training requires conversational prompts. It looks like the dataset contains '
                    'non-conversational inputs, likely because a chat template was applied before passing the dataset '
                    'to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat '
                    'template internally.')
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self.pad_token_id,
            padding_side='left',
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        prompt_mask = pad(
            prompt_mask,
            padding_value=0,
            padding_side='left',
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self.pad_token_id,
            padding_side='right',
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side='right', pad_to_multiple_of=self.pad_to_multiple_of)
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side='right',
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side='right', pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            tool_mask = None

        # === EDL Part Start === #
        # `completion_mask` has two jobs in vanilla GRPO: token existence and loss masking. When
        # mask_truncated_completions=True, GRPO turns truncated rows into all-zero loss rows below. EDL evidence should
        # still be computed from the actual sampled tokens; otherwise an all-zero mask would make the mean logprob look
        # like 0.0, which is artificially strong evidence compared with real negative logprobs.
        edl_completion_mask = completion_mask.clone()
        edl_tool_mask = tool_mask.clone() if tool_mask is not None else None
        # === EDL Part End === #

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # === EDL Part Start === #
        # Use the pre-loss-masking EDL attention mask for old-policy logprobs. The standard attention/loss mask may
        # intentionally drop truncated completions, but the reward calculation needs their real sampled-token evidence.
        edl_attention_mask = torch.cat([prompt_mask, edl_completion_mask], dim=1)
        # === EDL Part End === #

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {'prompt': prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs)['prompt']
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors='pt')
            prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ['input_ids', 'attention_mask']}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if 'token_type_ids' in forward_kwargs:
            token_type_ids = forward_kwargs['token_type_ids']
            forward_kwargs['token_type_ids'] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1)

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # === EDL Part Start === #
            # EDL rewards always need old-policy logprobs, even when base GRPO could skip them.
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                edl_attention_mask,
                logits_to_keep,
                batch_size,
                num_images=num_images,
                **forward_kwargs,
            )
            # === EDL Part End === #

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ['sequence_mask', 'sequence_truncate']
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ['sequence_truncate', 'token_truncate']:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap)
                elif self.vllm_importance_sampling_mode in ['sequence_mask', 'token_mask']:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0)
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. "
                        "Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name='ref' if 'ref' in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # === EDL Part Start === #
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        local_loss_mask = edl_completion_mask if edl_tool_mask is None else edl_completion_mask * edl_tool_mask
        mean_logps = self._mean_logps(old_per_token_logps, local_loss_mask)

        all_mean_logps = self.accelerator.gather(mean_logps)
        all_prompts = gather_object(copy.deepcopy(prompts))
        all_completions_text = gather_object(completions_text)
        all_final_answers = gather_object([example['final_answer'] for example in inputs])
        all_completion_prompts = gather_object([example.get('completion_prompt') for example in inputs])

        all_rewards, edl_metrics, aux_gt_by_group = self._compute_global_edl_rewards(
            prompts=all_prompts,
            completions_text=all_completions_text,
            final_answers=all_final_answers,
            completion_prompts=all_completion_prompts,
            mean_logps=all_mean_logps,
            num_generations=num_generations,
            mode=mode,
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = all_rewards[process_slice]
        aux_gt_inputs = self._prepare_local_aux_gt_inputs(
            inputs=inputs,
            prompts=prompts,
            aux_gt_by_group=aux_gt_by_group,
            num_generations=num_generations,
            local_process_offset=process_slice.start,
        )

        for key, value in edl_metrics.items():
            if torch.is_tensor(value):
                value = self.accelerator.gather(value.detach()).float().mean().item()
            self._metrics[mode][key].append(float(value))

        self._metrics[mode]['reward'].append(all_rewards.float().mean().item())
        self._metrics[mode]['reward_std'].append(all_rewards.float().std().item())
        grouped_rewards = all_rewards.view(-1, num_generations)
        grouped_std = grouped_rewards.std(dim=1) if num_generations > 1 else torch.zeros(
            grouped_rewards.size(0), device=device)
        self._metrics[mode]['frac_reward_zero_std'].append(
            torch.isclose(grouped_std, torch.zeros_like(grouped_std)).float().mean().item())

        self._logs['prompt'].extend(gather_object(prompts_text))
        self._logs['completion'].extend(gather_object(completions_text))
        self._logs['rewards']['edl_reward'].extend(all_rewards.tolist())
        self._logs['advantages'].extend(all_rewards.tolist())

        if images is not None:
            self._logs['images'].extend(gather_object(images))
        # === EDL Part End === #

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]['sampling/sampling_logp_difference/mean'].append(
                self.accelerator.gather(mean_delta).mean().item())
            self._metrics[mode]['sampling/sampling_logp_difference/max'].append(
                self.accelerator.gather(max_delta).max().item())
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device))
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device))
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device))
            self._metrics[mode]['sampling/importance_sampling_ratio/min'].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item())
            self._metrics[mode]['sampling/importance_sampling_ratio/mean'].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item())
            self._metrics[mode]['sampling/importance_sampling_ratio/max'].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item())

        output = {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'advantages': advantages,
            'num_items_in_batch': num_items_in_batch,
            'old_per_token_logps': old_per_token_logps,
        }
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output['importance_sampling_ratio'] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output['sampling_per_token_logps'] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output['ref_per_token_logps'] = ref_per_token_logps
        if 'pixel_values' in forward_kwargs:
            output['pixel_values'] = forward_kwargs['pixel_values']
        if 'image_grid_thw' in forward_kwargs:
            output['image_grid_thw'] = forward_kwargs['image_grid_thw']
        if 'pixel_attention_mask' in forward_kwargs:
            output['pixel_attention_mask'] = forward_kwargs['pixel_attention_mask']
        if 'image_sizes' in forward_kwargs:
            output['image_sizes'] = forward_kwargs['image_sizes']
        if 'token_type_ids' in forward_kwargs:
            output['token_type_ids'] = forward_kwargs['token_type_ids']
        if images is not None:
            output['num_images'] = num_images
        if tool_mask is not None:
            output['tool_mask'] = tool_mask
        if aux_gt_inputs is not None:
            output.update(aux_gt_inputs)
        return output

    def _compute_aux_gt_loss(self, model, aux_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the teacher-forced auxiliary GT loss for active missing-GT rows.

        Pseudo-GT completions are not inserted into the main generated-completion batch because that would mutate
        GRPO's fixed ``num_generations`` grouping and mix teacher-forced rows with sampler-produced rows. Instead,
        this branch replays the active GRPO loss variant on the teacher-forced rows and then applies
        ``edl_aux_gt_loss_weight`` in ``_compute_loss``.

        Args:
            model: Current policy model.
            aux_inputs: Batch-aligned tensors from ``_prepare_local_aux_gt_inputs``. Active rows are selected by
                ``aux_gt_mask`` of shape ``(B,)``; selected token tensors have shape ``(B_active, T_aux)``.

        Returns:
            Scalar auxiliary loss, already normalized for gradient accumulation.
        """
        aux_mask = aux_inputs['aux_gt_mask'].bool()
        prompt_ids = aux_inputs['aux_gt_prompt_ids'][aux_mask]
        prompt_mask = aux_inputs['aux_gt_prompt_mask'][aux_mask]
        completion_ids = aux_inputs['aux_gt_completion_ids'][aux_mask]
        completion_mask = aux_inputs['aux_gt_completion_mask'][aux_mask]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask
        compute_entropy = self.top_entropy_quantile < 1.0

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=compute_entropy,
        )
        old_per_token_logps = aux_inputs['aux_gt_old_per_token_logps'][aux_mask]
        advantages = aux_inputs['aux_gt_advantages'][aux_mask].unsqueeze(1)

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == 'sequence':
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f'Unknown importance sampling level: {self.importance_sampling_level}. '
                "Possible values are 'token' and 'sequence'.")
        coef_1 = torch.exp(log_importance_weights)

        if self.beta != 0.0:
            ref_per_token_logps = aux_inputs.get('aux_gt_ref_per_token_logps')
            if ref_per_token_logps is None:
                raise ValueError('aux_gt_ref_per_token_logps is required when beta != 0.')
            ref_per_token_logps = ref_per_token_logps[aux_mask]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=old_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        if compute_entropy:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'luspo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        elif self.loss_type == 'sapo':
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == 'vespo':
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=None,
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # Pseudo-GT rows are teacher-forced under the local model, not sampled by vLLM, so their sampler correction is
        # the identity even when sampled rollouts use vLLM importance-sampling correction.
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = 'train' if self.model.training else 'eval'
        grad_accum_normalizer = self.current_gradient_accumulation_steps if mode == 'train' else 1.0
        if self.loss_type in ['grpo', 'sapo']:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / grad_accum_normalizer
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / grad_accum_normalizer
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / grad_accum_normalizer
        elif self.loss_type in ['cispo', 'dapo', 'vespo']:
            aux_normalizer = aux_inputs.get('num_items_in_batch')
            if aux_normalizer is None:
                normalizer = mask.sum().clamp(min=1.0)
            else:
                normalizer = torch.as_tensor(aux_normalizer, dtype=per_token_loss.dtype, device=per_token_loss.device)
                normalizer = (normalizer / self.accelerator.num_processes).clamp(min=1.0)
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == 'luspo':
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            loss = loss / grad_accum_normalizer
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
        return loss

    def _normalize_aux_gt_loss(
        self,
        active_mean_aux_loss: torch.Tensor,
        aux_gt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured outer-batch normalization to the active auxiliary loss."""
        aux_row_fraction = aux_gt_mask.float().mean()
        if self.edl_aux_gt_normalization == 'row_matched':
            return active_mean_aux_loss * aux_row_fraction, aux_row_fraction
        return active_mean_aux_loss, aux_row_fraction

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)

        # === EDL Part Start === #
        aux_gt_mask = inputs.get('aux_gt_mask')
        has_aux_gt = aux_gt_mask is not None and aux_gt_mask.bool().any()
        if has_aux_gt and self.edl_aux_gt_enabled and self.edl_aux_gt_loss_weight > 0.0:
            # Keep pseudo-GT optimization as an auxiliary term instead of adding synthetic rows to the generated batch:
            # generated rows keep GRPO's fixed grouping/normalizers, while this branch mirrors the configured
            # loss_type for the teacher-forced GT rows and remains independently weightable.
            active_mean_aux_loss = self._compute_aux_gt_loss(model, inputs)
            normalized_aux_loss, aux_row_fraction = self._normalize_aux_gt_loss(
                active_mean_aux_loss,
                aux_gt_mask,
            )
            scaled_aux_loss = normalized_aux_loss * float(self.edl_aux_gt_loss_weight)
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['edl/aux_gt_row_fraction'].append(
                self.accelerator.gather(aux_row_fraction.detach()).float().mean().item())
            self._metrics[mode]['edl/aux_gt_loss_active_mean'].append(
                self.accelerator.gather(active_mean_aux_loss.detach()).float().mean().item())
            self._metrics[mode]['edl/aux_gt_loss'].append(
                self.accelerator.gather(scaled_aux_loss.detach()).float().mean().item())
            loss = loss + scaled_aux_loss
        # === EDL Part End === #

        return loss
