"""GRPO training script for LoRA fine-tuning with reward-based RL."""
import argparse
import os
import os.path as osp
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime

from mmengine import Config, DictAction, mkdir_or_exist
from peft import LoraConfig, PeftModel
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.evaluation import are_semantically_equivalent, extract_prediction
from s_edl.models import load_model_and_tokenizer
from s_edl.utils import log_training_examples, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LoRA GRPO training')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/', help='Working directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--project', type=str, default='s_edl', help='Wandb project name')
    parser.add_argument('--run-name', type=str, help='Wandb run name')
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='Path to SFT LoRA checkpoint to continue training from')
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (e.g., training.learning_rate=1e-6 training.num_generations=8)',
    )
    return parser.parse_args()


def math_reward_fn(completions: list[str], final_answer: list[str], **kwargs) -> list[float]:
    """Reward function: 1.0 if model answer matches ground truth, 0.0 otherwise.

    Args:
        completions: Model-generated completion texts.
        final_answer: Ground truth answers from the dataset.

    Returns:
        List of reward values (1.0 or 0.0).
    """
    rewards = []
    for completion, gt in zip(completions, final_answer):
        pred = extract_prediction(completion)
        if are_semantically_equivalent(pred, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def build_reward_fn(cfg: Config, logger) -> Callable:
    """Build a reward function based on config.

    Args:
        cfg: Full training config. Uses ``cfg.reward_type`` to select the backend:
            - ``'math_verify'`` (default): SymPy-based symbolic equivalence.
            - ``'llm_judge'``: LLM-based pairwise equivalence via subprocess judge.
        logger: Logger instance.

    Returns:
        A reward function compatible with TRL's GRPOTrainer.
    """
    reward_type = cfg.get('reward_type', 'math_verify')

    if reward_type == 'math_verify':
        logger.info('Using math_verify reward function')
        return math_reward_fn

    if reward_type == 'llm_judge':
        from s_edl.evaluation.judge import SubprocessPairwiseJudge

        judge_cfg = cfg.judge_engine
        if judge_cfg.get('tensor_parallel_size') is not None:
            logger.warning(
                'judge_engine.tensor_parallel_size is ignored; '
                'tensor parallelism is derived from judge_engine.cuda_devices')
        sampling_kwargs = dict(cfg.get('judge_sampling', {}))
        judge = SubprocessPairwiseJudge(
            model_name=judge_cfg.model_name,
            cuda_devices=judge_cfg.cuda_devices,
            gpu_memory_utilization=judge_cfg.get('gpu_memory_utilization', 0.4),
            max_model_len=judge_cfg.get('max_model_len', 4096),
            max_num_seqs=judge_cfg.get('max_num_seqs', 64),
            sampling_kwargs=sampling_kwargs,
            reasoning_effort=judge_cfg.get('reasoning_effort', 'low'),
        )

        def llm_judge_reward_fn(completions: list[str], final_answer: list[str], **kwargs) -> list[float]:
            preds = [extract_prediction(c) for c in completions]
            matches = judge.judge_batch(preds, final_answer)
            return [1.0 if m else 0.0 for m in matches]

        logger.info(f'Using LLM judge reward function (model={judge_cfg.model_name}, cuda={judge_cfg.cuda_devices})')
        return llm_judge_reward_fn

    raise ValueError(f"Unknown reward_type: '{reward_type}'. Expected 'math_verify' or 'llm_judge'.")


def log_trainable_parameters(model, logger) -> None:
    """Log total/trainable parameter counts for quick training sanity checks."""
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    logger.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({pct:.4f}%)")
    if trainable_params == 0:
        logger.warning('No trainable parameters found. Training will produce grad_norm=0.0.')


def train(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir
    seed = args.seed
    set_seed(seed)

    mkdir_or_exist(work_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger('s_edl', filepath=osp.join(work_dir, f'{timestamp}.log'), reset=True)
    logger.info('Config:\n' + '=' * 60 + '\n' + cfg.pretty_text + '\n' + '=' * 60)

    # Set wandb config via env vars — GRPOTrainer handles init on main process only
    os.environ.setdefault('WANDB_PROJECT', args.project)
    if args.run_name:
        os.environ.setdefault('WANDB_NAME', args.run_name)
    os.environ.setdefault('WANDB_DIR', work_dir)

    # Load base model + tokenizer (no LoRA, no device_map — GRPOTrainer handles both)
    model, tokenizer = load_model_and_tokenizer(
        model=cfg.model,
        device=None,
    )

    # If SFT checkpoint provided: load as PeftModel, GRPO continues training the same adapter.
    # Uses vLLM server mode (not colocate) — server mode supports PeftModel (TRL #3671).
    # If no checkpoint: GRPOTrainer creates a fresh LoRA via peft_config.
    peft_config = None
    if args.checkpoint:
        logger.info(f"Loading SFT LoRA checkpoint: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        model.train()
        logger.info('SFT LoRA loaded in trainable mode — GRPO will continue training this adapter')
    else:
        lora_cfg = deepcopy(dict(cfg.lora))
        lora_cfg.setdefault('bias', 'none')
        lora_cfg.setdefault('task_type', 'CAUSAL_LM')
        peft_config = LoraConfig(**lora_cfg)
        logger.info(f"No checkpoint — will create fresh LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    logger.info(f"Loading dataset: {cfg.data.train}")
    raw_dataset = load_dataset_split(cfg.data.train, split='train')
    prompt_builder = PROMPT_BUILDERS.build({**cfg.data.prompt_builder, 'seed': seed})

    datasets = prompt_builder.prepare_datasets(
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
    )
    train_dataset = datasets['train']

    # Log examples before renaming columns (log_training_examples expects 'prefill_prompt')
    log_training_examples(train_dataset, work_dir, logger)

    # GRPOTrainer expects a "prompt" column (plain text, pre-formatted with chat template)
    train_dataset = train_dataset.rename_column('prefill_prompt', 'prompt')

    grpo_config = GRPOConfig(
        output_dir=work_dir,
        **cfg.training,
        seed=seed,
        run_name=args.run_name,
        report_to='wandb',
    )

    reward_fn = build_reward_fn(cfg, logger)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    log_trainable_parameters(trainer.model, logger)

    trainer.train()

    # Save LoRA checkpoint (relative to base model, usable directly with evaluate.py)
    lora_path = osp.join(work_dir, 'lora_checkpoint')
    trainer.save_model(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f'LoRA checkpoint saved to: {lora_path}')

    logger.info('Training complete.')


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
