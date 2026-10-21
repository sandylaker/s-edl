"""RCFT training script for LoRA SFT + calibration-aware fine-tuning."""
import argparse
import os
from datetime import datetime

import wandb
from mmengine import Config, DictAction, mkdir_or_exist
from transformers import set_seed

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.models import load_model_and_tokenizer
from s_edl.training import RCFTConfig, RCFTTrainer
from s_edl.utils import log_training_examples, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LoRA RCFT training')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/', help='Working directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--project', type=str, default='s_edl', help='Wandb project name')
    parser.add_argument('--run-name', type=str, help='Wandb run name')
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (e.g., model.max_seq_length=4096 training.learning_rate=1e-4)',
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir
    seed = args.seed
    set_seed(seed)

    mkdir_or_exist(work_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger('s_edl', filepath=os.path.join(work_dir, f'{timestamp}.log'), reset=True)
    logger.info('Config:\n' + '=' * 60 + '\n' + cfg.pretty_text + '\n' + '=' * 60)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=cfg.to_dict(),
        dir=work_dir,
    )

    model, tokenizer = load_model_and_tokenizer(
        model=cfg.model,
        lora=cfg.lora,
    )

    logger.info(f"Loading dataset: {cfg.data.train}")
    raw_dataset = load_dataset_split(cfg.data.train, split='train')
    prompt_builder = PROMPT_BUILDERS.build({**cfg.data.prompt_builder, 'seed': seed})

    logger.info(f"Building prompts: {cfg.data.prompt_builder}")
    datasets = prompt_builder.prepare_datasets(
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
    )
    train_dataset = datasets['train']
    log_training_examples(train_dataset, work_dir, logger)

    training_args = RCFTConfig(
        output_dir=work_dir,
        **cfg.training,
        seed=seed,
        max_length=cfg.model.max_seq_length,
        report_to='wandb',
    )

    trainer = RCFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()

    lora_path = os.path.join(work_dir, 'lora_checkpoint')
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    wandb.finish()

    logger.info(f'Training complete. LoRA checkpoint saved to: {lora_path}')


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
