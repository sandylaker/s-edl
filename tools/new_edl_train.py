"""Pure RL EDL training script built on TRL GRPOTrainer."""

import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime

from mmengine import Config, DictAction, mkdir_or_exist
from peft import LoraConfig, PeftModel
from transformers import set_seed

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.evaluation import SubprocessClusterJudge
from s_edl.models import load_model_and_tokenizer
from s_edl.training import NewEDLConfig, NewEDLTrainer
from s_edl.utils import log_training_examples, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LoRA NewEDL GRPO-style training')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/', help='Working directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--project', type=str, default='s_edl', help='Wandb project name')
    parser.add_argument('--run-name', type=str, help='Wandb run name')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to SFT LoRA checkpoint to continue training from',
    )
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (e.g., training.learning_rate=1e-6 training.num_generations=8)',
    )
    return parser.parse_args()


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


def build_cluster_classifier(cfg: Config, logger):
    """Build an optional cluster classifier for non-math semantic clustering."""
    if cfg.get('judge_engine') is None:
        logger.info('Using math-verify clustering')
        return None

    judge_cfg = cfg.judge_engine
    if judge_cfg.get('tensor_parallel_size') is not None:
        logger.warning(
            'judge_engine.tensor_parallel_size is ignored; '
            'tensor parallelism is derived from judge_engine.cuda_devices')
    sampling_kwargs = dict(cfg.get('judge_sampling', {}))
    classifier = SubprocessClusterJudge(
        model_name=judge_cfg.model_name,
        cuda_devices=judge_cfg.cuda_devices,
        gpu_memory_utilization=judge_cfg.get('gpu_memory_utilization', 0.4),
        max_model_len=judge_cfg.get('max_model_len', 4096),
        max_num_seqs=judge_cfg.get('max_num_seqs', 64),
        sampling_kwargs=sampling_kwargs,
        reasoning_effort=judge_cfg.get('reasoning_effort', 'low'),
    )
    logger.info(f'Using LLM cluster judge (model={judge_cfg.model_name}, cuda={judge_cfg.cuda_devices})')
    return classifier


def validate_dataset_columns(train_dataset) -> None:
    """Validate columns required by NewEDL before handing the dataset to TRL."""
    required_columns = {'prefill_prompt', 'completion_prompt', 'final_answer'}
    missing = required_columns - set(train_dataset.column_names)
    if missing:
        raise ValueError(f'NewEDL training dataset is missing required columns: {sorted(missing)}')
    if 'prompt' in train_dataset.column_names:
        raise ValueError("NewEDL expects 'prefill_prompt' to be renamed to 'prompt', but 'prompt' already exists.")


def build_training_args(cfg: Config, work_dir: str, seed: int, run_name: str | None, logger) -> NewEDLConfig:
    """Build NewEDLConfig while letting script-level settings override config duplicates."""
    training_kwargs = dict(cfg.training)
    if training_kwargs.get('remove_unused_columns', False):
        raise ValueError(
            'training.remove_unused_columns must be false for NewEDL because final_answer and completion_prompt are '
            'read inside _generate_and_score_completions.')

    overrides = {
        'output_dir': work_dir,
        'seed': seed,
        'run_name': run_name,
    }
    for key, value in overrides.items():
        if key in training_kwargs and training_kwargs[key] != value:
            logger.warning(f'Overriding training.{key}={training_kwargs[key]!r} with script value {value!r}.')
        training_kwargs[key] = value
    training_kwargs.setdefault('report_to', 'wandb')
    return NewEDLConfig(**training_kwargs)


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

    os.environ.setdefault('WANDB_PROJECT', args.project)
    if args.run_name:
        os.environ.setdefault('WANDB_NAME', args.run_name)
    os.environ.setdefault('WANDB_DIR', work_dir)

    model, tokenizer = load_model_and_tokenizer(
        model=cfg.model,
        device=None,
    )

    peft_config = None
    if args.checkpoint:
        logger.info(f"Loading SFT LoRA checkpoint: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        model.train()
        logger.info('SFT LoRA loaded in trainable mode - NewEDL will continue training this adapter')
    else:
        lora_cfg = deepcopy(dict(cfg.lora))
        lora_cfg.setdefault('bias', 'none')
        lora_cfg.setdefault('task_type', 'CAUSAL_LM')
        peft_config = LoraConfig(**lora_cfg)
        logger.info(f"No checkpoint - will create fresh LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    logger.info(f"Loading dataset: {cfg.data.train}")
    raw_dataset = load_dataset_split(cfg.data.train, split='train')
    prompt_builder = PROMPT_BUILDERS.build({**cfg.data.prompt_builder, 'seed': seed})

    datasets = prompt_builder.prepare_datasets(
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
    )
    train_dataset = datasets['train']
    log_training_examples(train_dataset, work_dir, logger)

    validate_dataset_columns(train_dataset)
    train_dataset = train_dataset.rename_column('prefill_prompt', 'prompt')
    cluster_classifier = build_cluster_classifier(cfg, logger)

    try:
        training_args = build_training_args(cfg, work_dir, seed, args.run_name, logger)
        trainer = NewEDLTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            cluster_classifier=cluster_classifier,
        )

        log_trainable_parameters(trainer.model, logger)
        trainer.train()
    finally:
        if cluster_classifier is not None:
            cluster_classifier.shutdown()

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
