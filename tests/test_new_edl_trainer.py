"""Unit tests for NewEDL reward computation."""

import copy
import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from s_edl.evaluation import cluster_predictions_with_ground_truth, extract_prediction
from s_edl.training.new_edl_trainer import NewEDLConfig, NewEDLTrainer, compute_edl_group_rewards


def _make_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        '<pad>': 0,
        '<eos>': 1,
        '<unk>': 2,
        'Solve': 3,
        'one': 4,
        'Reasoning': 5,
        ':': 6,
        'My': 7,
        'answer': 8,
        'is': 9,
        '1': 10,
        '2': 11,
        '.': 12,
    }
    raw_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    raw_tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token='<pad>',
        eos_token='<eos>',
        bos_token='<eos>',
        unk_token='<unk>',
    )


def _make_tiny_trainer(tmp_path) -> NewEDLTrainer:
    tokenizer = _make_tiny_tokenizer()
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ))
    dataset = Dataset.from_list(
        [{
            'prompt': 'Solve one',
            'completion_prompt': ' Reasoning : My answer is 1',
            'final_answer': '1',
        }])
    args = NewEDLConfig(
        output_dir=str(tmp_path / 'new_edl'),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        generation_batch_size=2,
        num_generations=2,
        max_completion_length=2,
        use_vllm=False,
        report_to=[],
        save_strategy='no',
        logging_strategy='no',
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    return NewEDLTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )


def _edl_loss_from_log_weights(
    sample_log_weights: torch.Tensor,
    cluster_indices: list[list[int]],
    correct_idx: int,
    gt_log_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    class_log_u = []
    for class_idx, indices in enumerate(cluster_indices):
        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.long, device=sample_log_weights.device)
            class_log_u.append(torch.logsumexp(sample_log_weights[index_tensor], dim=0))
        elif class_idx == correct_idx and gt_log_weight is not None:
            class_log_u.append(gt_log_weight.to(device=sample_log_weights.device, dtype=sample_log_weights.dtype))
        else:
            raise ValueError('Non-GT classes must contain sampled completions.')

    class_log_u = torch.stack(class_log_u)
    probs = torch.softmax(class_log_u - class_log_u.max(), dim=0)
    n_effective = float(sample_log_weights.numel() + (0 if gt_log_weight is None else 1))
    evidences = probs * n_effective
    alphas = evidences + 1.0
    strength = torch.sum(alphas)
    probabilities = alphas / strength
    labels = torch.zeros_like(probabilities)
    labels[correct_idx] = 1.0
    error = (labels - probabilities)**2
    variance = probabilities * (1.0 - probabilities) / (strength + 1.0)
    return torch.sum(error + variance)


def _make_aux_loss_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        'aux_gt_mask': torch.tensor([True, False, True], device=device),
        'aux_gt_prompt_ids': torch.tensor([[3, 4], [3, 0], [3, 4]], device=device),
        'aux_gt_prompt_mask': torch.tensor([[1, 1], [1, 0], [1, 1]], device=device),
        'aux_gt_completion_ids': torch.tensor([[5, 6, 0], [10, 0, 0], [10, 11, 12]], device=device),
        'aux_gt_completion_mask': torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], device=device),
        'aux_gt_old_per_token_logps': torch.zeros((3, 3), device=device),
        'aux_gt_advantages': torch.tensor([0.5, 0.0, -0.25], device=device),
    }


def _manual_completion_logps(
    trainer: NewEDLTrainer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    target_model = model or trainer.model
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    with torch.no_grad():
        logits = target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = logits[:, :-1, :]
    logits = logits[:, -completion_ids.size(1):, :] / trainer.temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)


def _expected_missing_gt_group_reward(
    trainer: NewEDLTrainer,
    *,
    prompt: str,
    completions_text: list[str],
    final_answer: str,
    completion_prompt: str,
    mean_logps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = [extract_prediction(text) for text in completions_text]
    clusters, correct_idx = cluster_predictions_with_ground_truth(preds, final_answer)
    _, _, _, gt_completion_mask, gt_per_token_logps = trainer._compute_text_completion_logps(
        [prompt],
        [completion_prompt],
        no_grad=True,
    )
    pseudo_mean_logp = trainer._mean_logps(gt_per_token_logps, gt_completion_mask)[0]
    gt_log_weight = (trainer.edl_beta - 1.0) * (pseudo_mean_logp - float(trainer.edl_gt_pseudo_logprob_penalty))
    expected = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps,
        cluster_indices=[cluster.indices for cluster in clusters],
        correct_idx=correct_idx,
        gt_log_weight=gt_log_weight,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )
    assert expected.pseudo_gt_reward is not None
    assert expected.pseudo_gt_prob is not None
    return expected.rewards, expected.pseudo_gt_reward, expected.pseudo_gt_prob, pseudo_mean_logp, gt_log_weight


def test_new_edl_config_defaults_are_valid(tmp_path) -> None:
    args = NewEDLConfig(
        output_dir=str(tmp_path / 'new_edl'),
        per_device_train_batch_size=2,
        num_generations=2,
        generation_batch_size=2,
        bf16=False,
        fp16=False,
        report_to=[],
        save_strategy='no',
    )

    assert args.edl_beta == 1.2
    assert args.edl_aux_gt_enabled
    assert args.edl_aux_gt_loss_weight == 1.0
    assert args.edl_aux_gt_normalization == 'row_matched'
    assert args.edl_missing_gt_strategy == 'pseudo'


def test_new_edl_config_rejects_invalid_missing_gt_strategy(tmp_path) -> None:
    with pytest.raises(ValueError, match='edl_missing_gt_strategy'):
        NewEDLConfig(
            output_dir=str(tmp_path / 'new_edl'),
            per_device_train_batch_size=2,
            num_generations=2,
            generation_batch_size=2,
            bf16=False,
            fp16=False,
            report_to=[],
            save_strategy='no',
            edl_missing_gt_strategy='drop',
        )


def test_new_edl_config_rejects_invalid_aux_gt_normalization(tmp_path) -> None:
    with pytest.raises(ValueError, match='edl_aux_gt_normalization'):
        NewEDLConfig(
            output_dir=str(tmp_path / 'new_edl'),
            per_device_train_batch_size=2,
            num_generations=2,
            generation_batch_size=2,
            bf16=False,
            fp16=False,
            report_to=[],
            save_strategy='no',
            edl_aux_gt_normalization='group_std',
        )


def test_edl_rewards_push_correct_and_overconfident_wrong_classes() -> None:
    log_weights = torch.tensor([0.0, -0.2, -0.3], dtype=torch.float32)
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0], [1, 2]],
        correct_idx=0,
        log_u_min=-5.0,
    )

    assert result.rewards[0] > 0
    assert torch.all(result.rewards[1:] < 0)
    assert result.correct_mask.tolist() == [True, False, False]
    assert result.gt_freq_ratio == 1 / 3


def test_edl_rewards_match_negative_edl_loss_gradients() -> None:
    log_weights = torch.tensor([0.0, -0.4, -1.2, 0.3], dtype=torch.float64, requires_grad=True)
    cluster_indices = [[0, 2], [1], [3]]
    correct_idx = 1

    loss = _edl_loss_from_log_weights(log_weights, cluster_indices, correct_idx)
    loss.backward()
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights.detach(),
        cluster_indices=cluster_indices,
        correct_idx=correct_idx,
        log_u_min=-100.0,
    )

    assert torch.allclose(result.rewards, -log_weights.grad, atol=1e-10, rtol=1e-10)


def test_edl_rewards_match_missing_gt_pseudo_gradient() -> None:
    log_weights = torch.tensor([-0.3, -0.7], dtype=torch.float64, requires_grad=True)
    gt_log_weight = torch.tensor(-1.4, dtype=torch.float64, requires_grad=True)
    cluster_indices = [[0, 1], []]
    correct_idx = 1

    loss = _edl_loss_from_log_weights(log_weights, cluster_indices, correct_idx, gt_log_weight=gt_log_weight)
    loss.backward()
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights.detach(),
        cluster_indices=cluster_indices,
        correct_idx=correct_idx,
        gt_log_weight=gt_log_weight.detach(),
        log_u_min=-100.0,
    )

    assert torch.allclose(result.rewards, -log_weights.grad, atol=1e-10, rtol=1e-10)
    assert result.pseudo_gt_reward is not None
    assert torch.allclose(result.pseudo_gt_reward, -gt_log_weight.grad, atol=1e-10, rtol=1e-10)


def test_edl_rewards_are_zero_when_all_samples_are_correct() -> None:
    log_weights = torch.tensor([0.0, -0.5, -1.0], dtype=torch.float32)
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0, 1, 2]],
        correct_idx=0,
        log_u_min=-5.0,
    )

    assert torch.allclose(result.rewards, torch.zeros_like(result.rewards), atol=1e-6)
    assert result.pseudo_gt_reward is None


def test_edl_missing_gt_pseudo_reward_is_positive() -> None:
    log_weights = torch.tensor([0.0, -0.1], dtype=torch.float32)
    gt_log_weight = torch.tensor(-2.0, dtype=torch.float32)
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0, 1], []],
        correct_idx=1,
        gt_log_weight=gt_log_weight,
        log_u_min=-5.0,
    )

    assert result.pseudo_gt_reward is not None
    assert result.pseudo_gt_reward > 0
    assert result.pseudo_gt_prob is not None
    assert 0 < result.pseudo_gt_prob < 1
    assert torch.all(result.rewards < 0)


def test_edl_clamp_diagnostics_detect_changed_rewards() -> None:
    log_weights = torch.tensor([0.0, -20.0], dtype=torch.float32)
    result = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=-5.0,
    )

    assert result.clamped_class_count == 1
    assert result.clamped_completion_mask.tolist() == [False, True]
    delta = torch.abs(result.rewards - result.raw_rewards)
    assert delta.max() > 0


def test_edl_reward_scale_multiplies_rewards() -> None:
    log_weights = torch.tensor([0.0, -0.2, -0.4], dtype=torch.float32)
    unscaled = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0], [1, 2]],
        correct_idx=0,
        reward_scale=1.0,
    )
    scaled = compute_edl_group_rewards(
        sample_log_weights=log_weights,
        cluster_indices=[[0], [1, 2]],
        correct_idx=0,
        reward_scale=2.5,
    )

    assert torch.allclose(scaled.rewards, 2.5 * unscaled.rewards)


def test_edl_reward_validation_errors() -> None:
    with pytest.raises(ValueError, match='1D tensor'):
        compute_edl_group_rewards(torch.zeros((1, 2)), [[0]], correct_idx=0)
    with pytest.raises(ValueError, match='non-empty'):
        compute_edl_group_rewards(torch.zeros(1), [], correct_idx=0)
    with pytest.raises(ValueError, match='out of range'):
        compute_edl_group_rewards(torch.zeros(1), [[0]], correct_idx=1)
    with pytest.raises(ValueError, match='out-of-range sample index'):
        compute_edl_group_rewards(torch.zeros(1), [[1]], correct_idx=0)
    with pytest.raises(ValueError, match='duplicate sample index'):
        compute_edl_group_rewards(torch.zeros(2), [[0, 0], [1]], correct_idx=0)
    with pytest.raises(ValueError, match='cover every sampled completion'):
        compute_edl_group_rewards(torch.zeros(2), [[0]], correct_idx=0)
    with pytest.raises(ValueError, match='should only be set'):
        compute_edl_group_rewards(torch.zeros(1), [[0]], correct_idx=0, gt_log_weight=torch.tensor(0.0))
    with pytest.raises(ValueError, match='required'):
        compute_edl_group_rewards(torch.zeros(1), [[0], []], correct_idx=1)
    with pytest.raises(ValueError, match='Non-GT classes'):
        compute_edl_group_rewards(torch.zeros(1), [[0], [], []], correct_idx=1, gt_log_weight=torch.tensor(0.0))


def test_tokenize_prompt_completion_shapes_and_padding(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)

    prompt_ids, prompt_mask, completion_ids, completion_mask = trainer._tokenize_prompt_completion(
        prompts=['Solve one', 'Solve'],
        completions=['Reasoning : My answer is 1', '1'],
    )

    assert prompt_ids.shape == prompt_mask.shape == (2, 2)
    assert completion_ids.shape == completion_mask.shape == (2, 6)
    assert prompt_mask.tolist() == [[1, 1], [0, 1]]
    assert completion_mask[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert completion_mask[1].tolist() == [1, 0, 0, 0, 0, 0]


def test_compute_text_completion_logps_matches_manual_teacher_forcing(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()

    prompt_ids, prompt_mask, completion_ids, completion_mask, logps = trainer._compute_text_completion_logps(
        prompts=['Solve one', 'Solve'],
        completions=['Reasoning : My answer is 1', '1'],
    )
    expected_logps = _manual_completion_logps(
        trainer,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
    )

    assert prompt_ids.shape == prompt_mask.shape == (2, 2)
    assert completion_ids.shape == completion_mask.shape == logps.shape
    assert torch.allclose(logps, expected_logps, atol=1e-6)


def test_compute_text_completion_logps_uses_explicit_model_argument(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    ref_model = copy.deepcopy(trainer.model)
    ref_model.eval()
    with torch.no_grad():
        target_token_id = trainer.processing_class.convert_tokens_to_ids('1')
        ref_model.lm_head.weight[target_token_id, 0].add_(5.0)

    prompt_ids, prompt_mask, completion_ids, completion_mask, logps = trainer._compute_text_completion_logps(
        prompts=['Solve one'],
        completions=['Reasoning : My answer is 1'],
        model=ref_model,
    )
    expected_logps = _manual_completion_logps(
        trainer,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        model=ref_model,
    )
    policy_logps = _manual_completion_logps(
        trainer,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
    )

    assert torch.allclose(logps, expected_logps, atol=1e-6)
    assert not torch.allclose(logps, policy_logps, atol=1e-6)


def test_compute_text_completion_logps_empty_batch(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)

    tensors = trainer._compute_text_completion_logps(prompts=[], completions=[])

    assert all(tensor.shape == (0, 0) for tensor in tensors)


def test_compute_text_completion_logps_empty_completion_raises(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)

    with pytest.raises(ValueError, match='Completion is empty'):
        trainer._compute_text_completion_logps(prompts=['Solve one'], completions=[''])


def test_mean_logps_uses_token_existence_mask_not_loss_mask(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    per_token_logps = torch.tensor([[-2.0, -4.0]], device=trainer.accelerator.device)
    sampled_token_mask = torch.tensor([[1, 1]], device=trainer.accelerator.device)
    zeroed_loss_mask = torch.tensor([[0, 0]], device=trainer.accelerator.device)

    assert torch.allclose(trainer._mean_logps(per_token_logps, sampled_token_mask), torch.tensor([-3.0]))
    assert torch.allclose(trainer._mean_logps(per_token_logps, zeroed_loss_mask), torch.tensor([0.0]))


def test_mean_logps_average_valid_completion_tokens(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    per_token_logps = torch.tensor([[-2.0, -4.0, -100.0], [-6.0, -100.0, -100.0]], device=trainer.accelerator.device)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], device=trainer.accelerator.device)

    assert torch.allclose(trainer._mean_logps(per_token_logps, mask), torch.tensor([-3.0, -6.0]))


def test_compute_global_edl_rewards_matches_group_helper(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 1', 'My answer is 2'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    expected = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps,
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )

    assert torch.allclose(rewards, expected.rewards)
    assert metrics['edl/gt_freq_ratio'] == 0.5
    assert metrics['edl/gt_miss_rate'] == 0.0
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_beta_one_uses_histogram_rewards(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.edl_beta = 1.0
    low_high_logps = torch.tensor([-100.0, -0.1], dtype=torch.float32, device=trainer.accelerator.device)
    high_low_logps = torch.flip(low_high_logps, dims=[0])

    rewards_a, _, _ = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 1', 'My answer is 2'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=low_high_logps,
        num_generations=2,
        mode='train',
    )
    rewards_b, _, _ = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 1', 'My answer is 2'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=high_low_logps,
        num_generations=2,
        mode='train',
    )
    expected = compute_edl_group_rewards(
        sample_log_weights=torch.zeros(2, dtype=torch.float32, device=trainer.accelerator.device),
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )

    assert torch.allclose(rewards_a, expected.rewards)
    assert torch.allclose(rewards_b, expected.rewards)
    assert not torch.allclose(rewards_a, torch.zeros_like(rewards_a))


def test_compute_global_edl_rewards_clips_advantages_and_reports_rate(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.edl_advantage_clip = 0.05
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 1', 'My answer is 2'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    unclipped = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps,
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    ).rewards
    expected = torch.clamp(unclipped, min=-trainer.edl_advantage_clip, max=trainer.edl_advantage_clip)

    assert torch.allclose(rewards, expected)
    assert torch.allclose(metrics['edl/reward_mean'], expected.float().mean())
    assert torch.allclose(metrics['edl/reward_min'], expected.float().min())
    assert torch.allclose(metrics['edl/reward_max'], expected.float().max())
    assert torch.allclose(metrics['edl/advantage_clip_rate'], torch.tensor(1.0, device=rewards.device))
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_missing_gt_eval_without_completion_prompt(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=mean_logps,
        num_generations=2,
        mode='eval',
    )

    assert torch.allclose(rewards, torch.zeros_like(rewards))
    assert metrics['edl/gt_miss_rate'] == 1.0
    assert metrics['edl/aux_gt_active_rate'] == 0.0
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_missing_gt_train_requires_completion_prompt(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)

    with pytest.raises(ValueError, match='completion_prompt is empty'):
        trainer._compute_global_edl_rewards(
            prompts=['Solve one', 'Solve one'],
            completions_text=['My answer is 2', 'My answer is 3'],
            final_answers=['1', '1'],
            completion_prompts=[None, None],
            mean_logps=mean_logps,
            num_generations=2,
            mode='train',
        )


def test_compute_global_edl_rewards_skip_missing_gt_train_without_completion_prompt(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.edl_missing_gt_strategy = 'skip'
    trainer.edl_aux_gt_enabled = False
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answers=['1', '1'],
        completion_prompts=[None, None],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )

    assert torch.allclose(rewards, torch.zeros_like(rewards))
    assert metrics['edl/gt_miss_rate'] == 1.0
    assert metrics['edl/missing_gt_skipped_rate'] == 1.0
    assert metrics['edl/aux_gt_active_rate'] == 0.0
    assert 'edl/pseudo_gt_reward' not in metrics
    assert 'edl/pseudo_gt_logp' not in metrics
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_skip_missing_gt_keeps_present_groups(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.edl_missing_gt_strategy = 'skip'
    trainer.edl_aux_gt_enabled = False
    mean_logps = torch.tensor([-0.2, -1.0, -0.3, -0.8], dtype=torch.float32, device=trainer.accelerator.device)

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one'] * 4,
        completions_text=['My answer is 1', 'My answer is 2', 'My answer is 2', 'My answer is 3'],
        final_answers=['1'] * 4,
        completion_prompts=[None, None, 'Reasoning : My answer is 1', 'Reasoning : My answer is 1'],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    expected_present = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps[:2],
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )

    assert torch.allclose(rewards[:2], expected_present.rewards)
    assert torch.allclose(rewards[2:], torch.zeros_like(rewards[2:]))
    assert metrics['edl/gt_miss_rate'] == 0.5
    assert metrics['edl/missing_gt_skipped_rate'] == 0.5
    assert metrics['edl/aux_gt_active_rate'] == 0.0
    assert 'edl/pseudo_gt_reward' not in metrics
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_pseudo_without_aux_updates_only_sampled_rows(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.edl_missing_gt_strategy = 'pseudo'
    trainer.edl_aux_gt_enabled = False
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)
    completion_prompt = 'Reasoning : My answer is 1'

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answers=['1', '1'],
        completion_prompts=[completion_prompt, completion_prompt],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    expected_rewards, expected_pseudo_reward, _, _, _ = _expected_missing_gt_group_reward(
        trainer,
        prompt='Solve one',
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answer='1',
        completion_prompt=completion_prompt,
        mean_logps=mean_logps,
    )

    assert torch.allclose(rewards, expected_rewards)
    assert metrics['edl/pseudo_gt_active_rate'] == 1.0
    assert metrics['edl/aux_gt_active_rate'] == 0.0
    assert metrics['edl/missing_gt_skipped_rate'] == 0.0
    assert metrics['edl/pseudo_gt_conservation_error_abs_max'] < 1e-6
    assert torch.allclose(torch.tensor(metrics['edl/pseudo_gt_reward']), expected_pseudo_reward.cpu())
    assert aux_gt_by_group == {}


def test_compute_global_edl_rewards_skip_with_aux_updates_only_teacher_forced_gt(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.edl_missing_gt_strategy = 'skip'
    trainer.edl_aux_gt_enabled = True
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)
    completion_prompt = 'Reasoning : My answer is 1'

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answers=['1', '1'],
        completion_prompts=[completion_prompt, completion_prompt],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    _, expected_pseudo_reward, _, _, _ = _expected_missing_gt_group_reward(
        trainer,
        prompt='Solve one',
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answer='1',
        completion_prompt=completion_prompt,
        mean_logps=mean_logps,
    )

    assert torch.allclose(rewards, torch.zeros_like(rewards))
    assert metrics['edl/pseudo_gt_active_rate'] == 1.0
    assert metrics['edl/aux_gt_active_rate'] == 1.0
    assert metrics['edl/missing_gt_skipped_rate'] == 1.0
    assert set(aux_gt_by_group) == {0}
    assert torch.allclose(aux_gt_by_group[0]['advantage'], expected_pseudo_reward)


def test_compute_global_edl_rewards_missing_gt_creates_aux_info(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    mean_logps = torch.tensor([-0.2, -1.0], dtype=torch.float32, device=trainer.accelerator.device)
    completion_prompt = 'Reasoning : My answer is 1'

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one', 'Solve one'],
        completions_text=['My answer is 2', 'My answer is 3'],
        final_answers=['1', '1'],
        completion_prompts=[completion_prompt, completion_prompt],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )
    expected_rewards, expected_pseudo_reward, expected_pseudo_prob, expected_mean_logp, expected_log_weight = (
        _expected_missing_gt_group_reward(
            trainer,
            prompt='Solve one',
            completions_text=['My answer is 2', 'My answer is 3'],
            final_answer='1',
            completion_prompt=completion_prompt,
            mean_logps=mean_logps,
        ))

    assert torch.allclose(rewards, expected_rewards)
    assert metrics['edl/gt_miss_rate'] == 1.0
    assert metrics['edl/pseudo_gt_active_rate'] == 1.0
    assert metrics['edl/aux_gt_active_rate'] == 1.0
    assert torch.allclose(torch.tensor(metrics['edl/pseudo_gt_reward']), expected_pseudo_reward.cpu())
    assert torch.allclose(torch.tensor(metrics['edl/pseudo_gt_logp']), expected_mean_logp.cpu())
    assert torch.allclose(torch.tensor(metrics['edl/pseudo_gt_log_weight']), expected_log_weight.cpu())
    assert torch.allclose(torch.tensor(metrics['edl/pseudo_gt_prob']), expected_pseudo_prob.cpu())
    assert 0 in aux_gt_by_group
    assert torch.allclose(aux_gt_by_group[0]['advantage'], expected_pseudo_reward)
    assert torch.allclose(aux_gt_by_group[0]['mean_logp'], expected_mean_logp)
    assert torch.allclose(aux_gt_by_group[0]['log_weight'], expected_log_weight)


def test_compute_global_edl_rewards_handles_mixed_gt_present_and_missing_groups(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    mean_logps = torch.tensor([-0.2, -1.0, -0.3, -0.8], dtype=torch.float32, device=trainer.accelerator.device)
    completion_prompt = 'Reasoning : My answer is 1'

    rewards, metrics, aux_gt_by_group = trainer._compute_global_edl_rewards(
        prompts=['Solve one'] * 4,
        completions_text=['My answer is 1', 'My answer is 2', 'My answer is 2', 'My answer is 3'],
        final_answers=['1'] * 4,
        completion_prompts=[None, None, completion_prompt, completion_prompt],
        mean_logps=mean_logps,
        num_generations=2,
        mode='train',
    )

    expected_present = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps[:2],
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )
    expected_missing_rewards, expected_pseudo_reward, _, expected_mean_logp, expected_log_weight = (
        _expected_missing_gt_group_reward(
            trainer,
            prompt='Solve one',
            completions_text=['My answer is 2', 'My answer is 3'],
            final_answer='1',
            completion_prompt=completion_prompt,
            mean_logps=mean_logps[2:],
        ))

    assert torch.allclose(rewards[:2], expected_present.rewards)
    assert torch.allclose(rewards[2:], expected_missing_rewards)
    assert rewards[0] > 0
    assert torch.all(rewards[2:] < 0)
    assert metrics['edl/gt_miss_rate'] == 0.5
    assert metrics['edl/pseudo_gt_active_rate'] == 0.5
    assert metrics['edl/aux_gt_active_rate'] == 0.5
    assert set(aux_gt_by_group) == {1}
    assert torch.allclose(aux_gt_by_group[1]['advantage'], expected_pseudo_reward)
    assert torch.allclose(aux_gt_by_group[1]['mean_logp'], expected_mean_logp)
    assert torch.allclose(aux_gt_by_group[1]['log_weight'], expected_log_weight)


def test_prepare_local_aux_gt_inputs_are_batch_aligned(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    inputs = [
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
    ]

    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=['Solve one', 'Solve one'],
        aux_gt_by_group={0: {
            'advantage': torch.tensor(0.25, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=0,
    )

    assert aux_inputs is not None
    assert aux_inputs['aux_gt_mask'].tolist() == [True, False]
    assert aux_inputs['aux_gt_prompt_ids'].shape[0] == len(inputs)
    assert aux_inputs['aux_gt_completion_ids'].shape[0] == len(inputs)
    assert aux_inputs['aux_gt_old_per_token_logps'].shape == aux_inputs['aux_gt_completion_ids'].shape
    assert aux_inputs['aux_gt_advantages'].tolist() == [0.25, 0.0]


def test_prepare_local_aux_gt_inputs_respects_disable_and_global_offset(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    inputs = [
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
    ]
    trainer.edl_aux_gt_enabled = False
    assert trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=['Solve one', 'Solve one'],
        aux_gt_by_group={1: {
            'advantage': torch.tensor(0.5, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=2,
    ) is None

    trainer.edl_aux_gt_enabled = True
    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=['Solve one', 'Solve one'],
        aux_gt_by_group={1: {
            'advantage': torch.tensor(0.5, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=2,
    )

    assert aux_inputs is not None
    assert aux_inputs['aux_gt_mask'].tolist() == [True, False]
    assert aux_inputs['aux_gt_advantages'].tolist() == [0.5, 0.0]


def test_prepare_local_aux_gt_inputs_aligns_mixed_batch_missing_group(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    inputs = [
        {
            'completion_prompt': None
        },
        {
            'completion_prompt': None
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
    ]

    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=['Solve one'] * 4,
        aux_gt_by_group={1: {
            'advantage': torch.tensor(0.75, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=0,
    )

    assert aux_inputs is not None
    assert aux_inputs['aux_gt_mask'].tolist() == [False, False, True, False]
    assert aux_inputs['aux_gt_advantages'].tolist() == [0.0, 0.0, 0.75, 0.0]


def test_prepare_local_aux_gt_inputs_packs_three_group_values_correctly(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    prompts = ['Solve one', 'Solve one', 'Solve one', 'Solve one', 'Solve', 'Solve']
    inputs = [
        {
            'completion_prompt': None
        },
        {
            'completion_prompt': None
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': '1'
        },
        {
            'completion_prompt': '1'
        },
    ]

    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=prompts,
        aux_gt_by_group={
            1: {
                'advantage': torch.tensor(0.75, device=trainer.accelerator.device)
            },
            2: {
                'advantage': torch.tensor(-0.25, device=trainer.accelerator.device)
            },
        },
        num_generations=2,
        local_process_offset=0,
    )
    expected_prompt_ids, expected_prompt_mask, expected_completion_ids, expected_completion_mask, expected_logps = (
        trainer._compute_text_completion_logps(
            prompts=['Solve one', 'Solve'],
            completions=['Reasoning : My answer is 1', '1'],
            no_grad=True,
        ))
    device = trainer.accelerator.device
    active_rows = torch.tensor([2, 4], dtype=torch.long, device=trainer.accelerator.device)
    inactive_rows = torch.tensor([0, 1, 3, 5], dtype=torch.long, device=trainer.accelerator.device)
    expected_inactive_prompt_ids = torch.full((4, expected_prompt_ids.size(1)), trainer.pad_token_id, device=device)
    expected_inactive_prompt_mask = torch.zeros((4, expected_prompt_mask.size(1)), dtype=torch.long, device=device)
    expected_inactive_completion_ids = torch.full(
        (4, expected_completion_ids.size(1)), trainer.pad_token_id, device=device)
    expected_inactive_completion_mask = torch.zeros(
        (4, expected_completion_mask.size(1)), dtype=torch.long, device=device)
    expected_inactive_logps = torch.zeros((4, expected_logps.size(1)), device=device)

    assert aux_inputs is not None
    assert aux_inputs['aux_gt_mask'].tolist() == [False, False, True, False, True, False]
    assert torch.equal(aux_inputs['aux_gt_prompt_ids'][active_rows], expected_prompt_ids)
    assert torch.equal(aux_inputs['aux_gt_prompt_mask'][active_rows], expected_prompt_mask)
    assert torch.equal(aux_inputs['aux_gt_completion_ids'][active_rows], expected_completion_ids)
    assert torch.equal(aux_inputs['aux_gt_completion_mask'][active_rows], expected_completion_mask)
    assert torch.allclose(aux_inputs['aux_gt_old_per_token_logps'][active_rows], expected_logps)
    assert torch.allclose(
        aux_inputs['aux_gt_advantages'],
        torch.tensor([0.0, 0.0, 0.75, 0.0, -0.25, 0.0], device=trainer.accelerator.device),
    )
    assert torch.equal(
        aux_inputs['aux_gt_prompt_ids'][inactive_rows],
        expected_inactive_prompt_ids,
    )
    assert torch.equal(aux_inputs['aux_gt_prompt_mask'][inactive_rows], expected_inactive_prompt_mask)
    assert torch.equal(
        aux_inputs['aux_gt_completion_ids'][inactive_rows],
        expected_inactive_completion_ids,
    )
    assert torch.equal(
        aux_inputs['aux_gt_completion_mask'][inactive_rows],
        expected_inactive_completion_mask,
    )
    assert torch.allclose(aux_inputs['aux_gt_old_per_token_logps'][inactive_rows], expected_inactive_logps)


def test_prepare_local_aux_gt_inputs_includes_ref_logps_when_beta_enabled(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.beta = 0.1
    trainer.ref_model = copy.deepcopy(trainer.model)
    with torch.no_grad():
        target_token_id = trainer.processing_class.convert_tokens_to_ids('1')
        trainer.ref_model.lm_head.weight[target_token_id, 0].add_(5.0)
    inputs = [
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'completion_prompt': 'Reasoning : My answer is 1'
        },
    ]

    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=inputs,
        prompts=['Solve one', 'Solve one'],
        aux_gt_by_group={0: {
            'advantage': torch.tensor(0.25, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=0,
    )

    assert aux_inputs is not None
    assert 'aux_gt_ref_per_token_logps' in aux_inputs
    assert aux_inputs['aux_gt_ref_per_token_logps'].shape == aux_inputs['aux_gt_old_per_token_logps'].shape
    _, _, _, _, expected_ref_logps = trainer._compute_text_completion_logps(
        prompts=['Solve one'],
        completions=['Reasoning : My answer is 1'],
        model=trainer.ref_model,
        no_grad=True,
    )
    assert torch.allclose(aux_inputs['aux_gt_ref_per_token_logps'][0:1], expected_ref_logps)
    assert torch.allclose(aux_inputs['aux_gt_ref_per_token_logps'][1], torch.zeros_like(expected_ref_logps[0]))
    assert not torch.allclose(
        aux_inputs['aux_gt_ref_per_token_logps'][0:1], aux_inputs['aux_gt_old_per_token_logps'][0:1])


def test_generate_and_score_completions_returns_expected_edl_values(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    prompt_ids_list = [[3, 4], [3, 4], [3, 4], [3, 4]]
    completion_ids_list = [
        [7, 8, 9, 10],
        [7, 8, 9, 11],
        [7, 8, 9, 11],
        [7, 8, 9, 11],
    ]

    def fake_generate(prompts):
        assert prompts == ['Solve one'] * 4
        return prompt_ids_list, completion_ids_list, None, None, None, None, {'rollout_id': [0, 1, 2, 3]}

    trainer._generate = fake_generate
    inputs = [
        {
            'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
        },
        {
            'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
        },
        {
            'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': 'Reasoning : My answer is 1'
        },
        {
            'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': 'Reasoning : My answer is 1'
        },
    ]

    output = trainer._generate_and_score_completions(inputs)
    manual_logps = _manual_completion_logps(
        trainer,
        output['prompt_ids'],
        output['prompt_mask'],
        output['completion_ids'],
        output['completion_mask'],
    )
    mean_logps = trainer._mean_logps(output['old_per_token_logps'], output['completion_mask'])
    expected_present = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * mean_logps[:2],
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    )
    expected_missing_rewards, expected_pseudo_reward, _, _, _ = _expected_missing_gt_group_reward(
        trainer,
        prompt='Solve one',
        completions_text=['My answer is 2', 'My answer is 2'],
        final_answer='1',
        completion_prompt='Reasoning : My answer is 1',
        mean_logps=mean_logps[2:],
    )
    expected_advantages = torch.cat([expected_present.rewards, expected_missing_rewards])

    assert torch.equal(output['prompt_ids'], torch.tensor(prompt_ids_list, device=trainer.accelerator.device))
    assert torch.equal(output['completion_ids'], torch.tensor(completion_ids_list, device=trainer.accelerator.device))
    assert torch.allclose(output['old_per_token_logps'], manual_logps)
    assert torch.allclose(output['advantages'], expected_advantages)
    assert output['aux_gt_mask'].tolist() == [False, False, True, False]
    assert torch.allclose(output['aux_gt_advantages'][2], expected_pseudo_reward)
    assert torch.allclose(output['aux_gt_advantages'][[0, 1, 3]], torch.zeros(3, device=trainer.accelerator.device))
    assert [example['rollout_id'] for example in inputs] == [0, 1, 2, 3]
    assert trainer._metrics['eval']['edl/gt_miss_rate'][-1] == 0.5
    assert trainer._metrics['eval']['edl/aux_gt_active_rate'][-1] == 0.5


def test_generate_and_score_completions_uses_sampled_tokens_for_truncated_edl_rewards(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.mask_truncated_completions = True
    prompt_ids_list = [[3, 4], [3, 4]]
    completion_ids_list = [
        [7, 8, 9, 10],
        [7, 8, 9, 11],
    ]

    def fake_generate(prompts):
        assert prompts == ['Solve one'] * 2
        return prompt_ids_list, completion_ids_list, None, None, None, None, None

    trainer._generate = fake_generate
    output = trainer._generate_and_score_completions(
        [
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
        ])
    sampled_completion_mask = torch.ones_like(output['completion_mask'])
    expected_old_logps = _manual_completion_logps(
        trainer,
        output['prompt_ids'],
        output['prompt_mask'],
        output['completion_ids'],
        sampled_completion_mask,
    )
    expected_mean_logps = trainer._mean_logps(expected_old_logps, sampled_completion_mask)
    expected_rewards = compute_edl_group_rewards(
        sample_log_weights=(trainer.edl_beta - 1.0) * expected_mean_logps,
        cluster_indices=[[0], [1]],
        correct_idx=0,
        log_u_min=trainer.edl_log_u_min,
        reward_scale=trainer.edl_reward_scale,
    ).rewards

    assert torch.equal(output['completion_mask'], torch.zeros_like(output['completion_mask']))
    assert torch.allclose(output['old_per_token_logps'], expected_old_logps)
    assert torch.allclose(output['advantages'], expected_rewards)
    assert not torch.allclose(output['advantages'], torch.zeros_like(output['advantages']))


def test_generate_and_score_completions_computes_vllm_sequence_truncate_ratio(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_correction = True
    trainer.vllm_importance_sampling_mode = 'sequence_truncate'
    trainer.vllm_importance_sampling_cap = 2.0

    def fake_generate(prompts):
        assert prompts == ['Solve one'] * 2
        prompt_ids_list = [[3, 4], [3, 4]]
        completion_ids_list = [[10, 1], [11, 12, 1]]
        sampling_logps = [[0.0, 0.0], [0.0, 0.0, 0.0]]
        return prompt_ids_list, completion_ids_list, None, None, None, sampling_logps, None

    def fake_logps(*args, **kwargs):
        logps = torch.tensor([[0.3, 0.1, 0.0], [1.0, -0.2, 0.0]], device=trainer.accelerator.device)
        return logps, None

    trainer._generate = fake_generate
    trainer._get_per_token_logps_and_entropies = fake_logps

    output = trainer._generate_and_score_completions(
        [
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
        ])

    expected = torch.exp(torch.tensor([[0.4], [0.8]], device=trainer.accelerator.device))
    expected = torch.clamp(expected, max=trainer.vllm_importance_sampling_cap)
    assert torch.allclose(output['importance_sampling_ratio'], expected)
    assert output['importance_sampling_ratio'].shape == torch.Size([2, 1])
    expected_old_logps = torch.tensor([[0.3, 0.1, 0.0], [1.0, -0.2, 0.0]], device=trainer.accelerator.device)
    assert torch.allclose(output['old_per_token_logps'], expected_old_logps)


def test_generate_and_score_completions_computes_vllm_token_mask_ratio(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_correction = True
    trainer.vllm_importance_sampling_mode = 'token_mask'
    trainer.vllm_importance_sampling_cap = 1.2

    def fake_generate(prompts):
        assert prompts == ['Solve one'] * 2
        prompt_ids_list = [[3, 4], [3, 4]]
        completion_ids_list = [[10, 12, 1], [11, 1]]
        sampling_logps = [[0.0, 0.0, 0.0], [0.0, 0.0]]
        return prompt_ids_list, completion_ids_list, None, None, None, sampling_logps, None

    def fake_logps(*args, **kwargs):
        logps = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.2, 0.0]], device=trainer.accelerator.device)
        return logps, None

    trainer._generate = fake_generate
    trainer._get_per_token_logps_and_entropies = fake_logps

    output = trainer._generate_and_score_completions(
        [
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
            {
                'prompt': 'Solve one', 'final_answer': '1', 'completion_prompt': None
            },
        ])

    expected = torch.exp(torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.2, 0.0]], device=trainer.accelerator.device))
    expected = expected.masked_fill(expected > trainer.vllm_importance_sampling_cap, value=0.0)
    assert torch.allclose(output['importance_sampling_ratio'], expected)
    assert output['importance_sampling_ratio'].shape == torch.Size([2, 3])
    assert torch.allclose(output['sampling_per_token_logps'], torch.zeros_like(output['sampling_per_token_logps']))


@pytest.mark.gpu
def test_new_edl_trainer_vllm_server_generates_and_scores(tmp_path) -> None:
    """Manual integration test for TRL vLLM server mode.

    Run on a vLLM-capable server after starting a matching server, for example:
    ``trl vllm-serve --model $S_EDL_VLLM_MODEL --host 0.0.0.0 --port 8000``.
    Then run:
    ``S_EDL_RUN_VLLM_TESTS=1 S_EDL_VLLM_MODEL=<model> pytest -q tests/test_new_edl_trainer.py -k vllm_server``.
    """
    if os.environ.get('S_EDL_RUN_VLLM_TESTS') != '1':
        pytest.skip('Set S_EDL_RUN_VLLM_TESTS=1 to run real vLLM server integration tests.')
    model_id = os.environ.get('S_EDL_VLLM_MODEL')
    if not model_id:
        pytest.skip('Set S_EDL_VLLM_MODEL to the model served by `trl vllm-serve`.')

    server_url = os.environ.get('S_EDL_VLLM_SERVER_URL', 'http://localhost:8000')
    try:
        with urlopen(f'{server_url.rstrip("/")}/v1/models', timeout=5) as response:
            assert response.status == 200
    except URLError as exc:
        pytest.fail(f'vLLM server is not reachable at {server_url}: {exc}')

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=os.environ.get('S_EDL_VLLM_TRUST_REMOTE_CODE', '0') == '1',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(
        [
            {
                'prompt': 'What is 1 + 1? Answer with only the number.',
                'completion_prompt': ' 2',
                'final_answer': '2',
            },
            {
                'prompt': 'What is 1 + 1? Answer with only the number.',
                'completion_prompt': ' 2',
                'final_answer': '2',
            },
        ])
    model_init_kwargs = {'trust_remote_code': os.environ.get('S_EDL_VLLM_TRUST_REMOTE_CODE', '0') == '1'}
    args = NewEDLConfig(
        output_dir=str(tmp_path / 'new_edl_vllm'),
        model_init_kwargs=model_init_kwargs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        generation_batch_size=2,
        num_generations=2,
        max_completion_length=int(os.environ.get('S_EDL_VLLM_MAX_COMPLETION_LENGTH', '8')),
        use_vllm=True,
        vllm_mode='server',
        vllm_server_base_url=server_url,
        vllm_server_timeout=float(os.environ.get('S_EDL_VLLM_SERVER_TIMEOUT', '120')),
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode=os.environ.get('S_EDL_VLLM_IMPORTANCE_SAMPLING_MODE', 'sequence_truncate'),
        report_to=[],
        save_strategy='no',
        logging_strategy='no',
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        disable_tqdm=True,
    )
    trainer = NewEDLTrainer(
        model=model_id,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.model.eval()

    output = trainer._generate_and_score_completions(list(dataset))

    assert output['prompt_ids'].shape[0] == 2
    assert output['completion_ids'].shape[0] == 2
    assert output['sampling_per_token_logps'].shape == output['old_per_token_logps'].shape
    assert 'importance_sampling_ratio' in output
    assert torch.isfinite(output['old_per_token_logps']).all()
    assert torch.isfinite(output['advantages']).all()


def test_compute_aux_gt_loss_matches_unit_ratio_ppo_objective(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    aux_inputs = trainer._prepare_local_aux_gt_inputs(
        inputs=[
            {
                'completion_prompt': 'Reasoning : My answer is 1'
            },
            {
                'completion_prompt': 'Reasoning : My answer is 1'
            },
        ],
        prompts=['Solve one', 'Solve one'],
        aux_gt_by_group={0: {
            'advantage': torch.tensor(0.25, device=trainer.accelerator.device)
        }},
        num_generations=2,
        local_process_offset=0,
    )

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)

    assert torch.allclose(aux_loss, torch.tensor(-0.25, device=aux_loss.device), atol=1e-5)


def test_aux_gt_row_matched_normalization_matches_rollout_row_fraction(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    active_mean_loss = torch.tensor(-0.4, device=trainer.accelerator.device)
    aux_mask = torch.tensor([False, False, True, False, False, False, False, False], device=trainer.accelerator.device)

    trainer.edl_aux_gt_normalization = 'active_mean'
    legacy_loss, row_fraction = trainer._normalize_aux_gt_loss(active_mean_loss, aux_mask)
    trainer.edl_aux_gt_normalization = 'row_matched'
    row_matched_loss, row_fraction_again = trainer._normalize_aux_gt_loss(active_mean_loss, aux_mask)

    assert torch.allclose(row_fraction, torch.tensor(1 / 8, device=row_fraction.device))
    assert torch.allclose(row_fraction_again, row_fraction)
    assert torch.allclose(legacy_loss, active_mean_loss)
    assert torch.allclose(row_matched_loss, active_mean_loss / 8)


@pytest.mark.parametrize(
    ('loss_type', 'expected'),
    [
        ('grpo', -0.125),
        ('bnpo', -0.05),
        ('dapo', -0.05),
        ('dr_grpo', -0.03125),
        ('luspo', -0.125),
        ('cispo', 0.0),
        ('vespo', 0.0),
    ],
)
def test_compute_aux_gt_loss_uses_configured_grpo_reduction(tmp_path, loss_type, expected) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = loss_type
    trainer.max_completion_length = 4
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['num_items_in_batch'] = torch.tensor(5.0, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)

    assert torch.allclose(aux_loss, torch.tensor(expected, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_uses_sapo_temperature_objective(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'sapo'
    trainer.args.sapo_temperature_pos = 2.0
    trainer.args.sapo_temperature_neg = 4.0
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    expected_row0 = -0.5
    expected_row1 = 0.125
    expected = (expected_row0 + expected_row1) / 2.0

    assert torch.allclose(aux_loss, torch.tensor(expected, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_supports_sequence_importance_sampling(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'bnpo'
    trainer.importance_sampling_level = 'sequence'
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['aux_gt_old_per_token_logps'][:] = 0.0

    def fake_logps(*args, **kwargs):
        logps = torch.tensor([[0.4, 0.4, 0.0], [-0.4, -0.4, -0.4]], device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    positive_coef = 1.0 + trainer.epsilon_high
    negative_coef = 1.0 - trainer.epsilon_low
    expected_sum = (-positive_coef * 0.5 * 2) + (-negative_coef * -0.25 * 3)
    expected = expected_sum / 5.0

    assert torch.allclose(aux_loss, torch.tensor(expected, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_includes_reference_kl_when_beta_enabled(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'bnpo'
    trainer.beta = 0.5
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['aux_gt_ref_per_token_logps'] = torch.full((3, 3), -0.2, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    kl = torch.exp(torch.tensor(-0.2, device=aux_loss.device)) + 0.2 - 1.0
    expected = torch.tensor(-0.05, device=aux_loss.device) + trainer.beta * kl

    assert torch.allclose(aux_loss, expected, atol=1e-6)


def test_compute_aux_gt_loss_bias_corrects_reference_kl_with_current_ratio(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'bnpo'
    trainer.beta = 0.5
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.args.use_bias_correction_kl = True
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['aux_gt_ref_per_token_logps'] = torch.full((3, 3), -0.2, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.full((2, 3), 0.4, device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    ratio = torch.exp(torch.tensor(0.4, device=aux_loss.device))
    positive_coef = torch.tensor(1.0 + trainer.epsilon_high, device=aux_loss.device)
    negative_coef = ratio
    ppo_sum = (-positive_coef * 0.5 * 2) + (-negative_coef * -0.25 * 3)
    kl = torch.exp(torch.tensor(-0.6, device=aux_loss.device)) + 0.6 - 1.0
    expected = (ppo_sum + 5 * trainer.beta * kl * ratio) / 5.0

    assert torch.allclose(aux_loss, expected, atol=1e-6)


def test_compute_aux_gt_loss_applies_off_policy_mask_to_token_losses(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'bnpo'
    trainer.off_policy_mask_threshold = 0.1
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    def fake_off_policy_mask(**kwargs):
        assert kwargs['sampling_per_token_logps'].shape == torch.Size([2, 3])
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=trainer.accelerator.device)

    trainer._get_per_token_logps_and_entropies = fake_logps
    trainer.get_off_policy_mask = fake_off_policy_mask

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)

    assert torch.allclose(aux_loss, torch.tensor(0.0, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_applies_entropy_mask_to_token_losses(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'bnpo'
    trainer.top_entropy_quantile = 0.5
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        assert kwargs.get('compute_entropy') is True
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        return logps, torch.ones_like(logps)

    def fake_entropy_mask(entropies, mask, quantile):
        assert entropies.shape == torch.Size([2, 3])
        assert mask.shape == torch.Size([2, 3])
        assert quantile == pytest.approx(0.5)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=trainer.accelerator.device)

    trainer._get_per_token_logps_and_entropies = fake_logps
    trainer.get_high_entropy_mask = fake_entropy_mask

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)

    assert torch.allclose(aux_loss, torch.tensor(-0.1, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_uses_num_items_in_batch_for_dapo_normalizer(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'dapo'
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['num_items_in_batch'] = torch.tensor(10.0, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)

    assert torch.allclose(aux_loss, torch.tensor(-0.025, device=aux_loss.device), atol=1e-6)


def test_compute_aux_gt_loss_cispo_uses_logp_weighted_objective(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'cispo'
    trainer.epsilon_high = 0.7
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['num_items_in_batch'] = torch.tensor(5.0, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.full((2, 3), -0.3, device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    clamped_ratio = torch.tensor(trainer.epsilon_high, device=aux_loss.device)
    expected_sum = (-clamped_ratio * 0.5 * -0.3 * 2) + (-clamped_ratio * -0.25 * -0.3 * 3)
    expected = expected_sum / 5.0

    assert torch.allclose(aux_loss, expected, atol=1e-6)


def test_compute_aux_gt_loss_vespo_uses_gamma_weighted_objective(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.loss_type = 'vespo'
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)
    aux_inputs['num_items_in_batch'] = torch.tensor(5.0, device=trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.full((2, 3), -0.4, device=trainer.accelerator.device)
        entropies = torch.zeros_like(logps) if kwargs.get('compute_entropy') else None
        return logps, entropies

    def fake_gamma_weights(**kwargs):
        assert kwargs['importance_sampling_ratio'] is None
        return torch.tensor([[2.0], [0.5]], device=trainer.accelerator.device)

    trainer._get_per_token_logps_and_entropies = fake_logps
    trainer.get_gamma_weights = fake_gamma_weights

    aux_loss = trainer._compute_aux_gt_loss(trainer.model, aux_inputs)
    expected_sum = (-2.0 * 0.5 * -0.4 * 2) + (-0.5 * -0.25 * -0.4 * 3)
    expected = torch.tensor(expected_sum / 5.0, device=aux_loss.device)

    assert torch.allclose(aux_loss, expected, atol=1e-6)


def test_compute_aux_gt_loss_requires_reference_logps_when_beta_enabled(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.model.eval()
    trainer.beta = 0.1
    aux_inputs = _make_aux_loss_inputs(trainer.accelerator.device)

    def fake_logps(*args, **kwargs):
        logps = torch.zeros((2, 3), device=trainer.accelerator.device)
        return logps, None

    trainer._get_per_token_logps_and_entropies = fake_logps

    with pytest.raises(ValueError, match='aux_gt_ref_per_token_logps'):
        trainer._compute_aux_gt_loss(trainer.model, aux_inputs)


def test_new_edl_trainer_tiny_smoke_use_vllm_false(tmp_path) -> None:
    trainer = _make_tiny_trainer(tmp_path)
    trainer.args.max_steps = 1

    trainer.train()

    assert trainer.state.global_step == 1
    assert any('edl/reward_mean' in logs for logs in trainer.state.log_history)
    assert any('edl/aux_gt_active_rate' in logs for logs in trainer.state.log_history)
