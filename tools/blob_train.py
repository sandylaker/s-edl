"""BLoB training script for LoRA SFT with Bayesian LoRA regularization."""
import argparse
import json
import os
from datetime import datetime

import wandb
from mmengine import Config, DictAction, mkdir_or_exist
from transformers import set_seed

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.models import load_blob_model_and_tokenizer
from s_edl.models.load_blob_model import build_mean_lora_state_dict, get_blob_state_dict
from s_edl.training import BLoBConfig, BLoBTrainer
from s_edl.utils import log_training_examples, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LoRA BLoB training')
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


def _save_blob_metadata(work_dir: str, training_args: BLoBConfig) -> None:
    metadata = {
        'method':
        'BLoB',
        'posterior_eval_note': (
            'Use posterior sampling over full BLoB weights for Bayesian evaluation. '
            'Mean-only LoRA checkpoint is provided for temporary compatibility with current evaluate.py.'),
        'bayes_train_n_samples':
        int(training_args.bayes_train_n_samples),
        'bayes_eps':
        float(training_args.bayes_eps),
        'bayes_beta':
        float(training_args.bayes_beta),
        'bayes_kllr':
        float(training_args.bayes_kllr),
        'bayes_kl_reweighting':
        bool(training_args.bayes_kl_reweighting),
    }
    with open(os.path.join(work_dir, 'blob_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


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

    model, tokenizer = load_blob_model_and_tokenizer(
        model=cfg.model,
        lora=cfg.lora,
        blob=cfg.training,
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

    training_args = BLoBConfig(
        output_dir=work_dir,
        **cfg.training,
        seed=seed,
        max_length=cfg.model.max_seq_length,
        report_to='wandb',
    )

    trainer = BLoBTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()

    model_to_save = trainer.accelerator.unwrap_model(trainer.model)

    blob_path = os.path.join(work_dir, 'lora_checkpoint_blob')
    model_to_save.save_pretrained(blob_path)
    tokenizer.save_pretrained(blob_path)

    lora_path = os.path.join(work_dir, 'lora_checkpoint')
    blob_state = get_blob_state_dict(model_to_save)
    mean_state = build_mean_lora_state_dict(blob_state)
    model_to_save.save_pretrained(lora_path, state_dict=mean_state)
    tokenizer.save_pretrained(lora_path)

    _save_blob_metadata(work_dir=work_dir, training_args=training_args)
    wandb.finish()

    logger.info(f'Training complete. BLoB checkpoint saved to: {blob_path}')
    logger.info(f'Mean-only LoRA checkpoint saved to: {lora_path}')


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
