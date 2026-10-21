"""EDL training script for LoRA SFT + self-evidence loss."""
import argparse
import os
from datetime import datetime

import wandb
from mmengine import Config, DictAction, mkdir_or_exist
from peft import PeftModel
from transformers import set_seed

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.evaluation import SubprocessClusterJudge
from s_edl.models import load_model_and_tokenizer
from s_edl.training import EDLConfig, EDLTrainer
from s_edl.utils import log_training_examples, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LoRA EDL training')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/', help='Working directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--project', type=str, default='s_edl', help='Wandb project name')
    parser.add_argument('--run-name', type=str, help='Wandb run name')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to LoRA checkpoint to initialize from (e.g., baseline SFT checkpoint)')
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (e.g., model.max_seq_length=4096 training.learning_rate=1e-4)',
    )
    return parser.parse_args()


def _validate_edl_sampling(cfg: Config, logger) -> None:
    edl_sampling = cfg.training.edl_sampling
    if edl_sampling is None:
        raise ValueError('training.edl_sampling must be set for EDL training.')
    required_keys = {'n', 'temperature', 'top_p', 'max_tokens', 'logprobs'}
    missing = required_keys - set(edl_sampling.keys())
    if missing:
        raise ValueError(f"training.edl_sampling missing required keys: {sorted(missing)}")

    vllm_sampling = cfg.get('vllm_sampling')
    if vllm_sampling is None:
        logger.warning('vllm_sampling is not set; ensure eval sampling matches edl_sampling if needed.')
        return
    compare_keys = {'n', 'temperature', 'top_p', 'max_tokens'}
    for key in compare_keys:
        if key not in vllm_sampling or vllm_sampling[key] != edl_sampling[key]:
            logger.warning(
                f"edl_sampling.{key} ({edl_sampling.get(key)}) differs from vllm_sampling."
                f"{key} ({vllm_sampling.get(key)}).")
            break


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
    _validate_edl_sampling(cfg, logger)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=cfg.to_dict(),
        dir=work_dir,
    )

    if args.checkpoint:
        # Load base model without LoRA, then load checkpoint adapter
        logger.info(f"Loading SFT LoRA checkpoint: {args.checkpoint}")
        model, tokenizer = load_model_and_tokenizer(model=cfg.model)
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        if cfg.model.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            logger.info('Gradient checkpointing enabled (use_reentrant=False)')
        model.train()
        model.print_trainable_parameters()
        logger.info('SFT LoRA loaded in trainable mode — EDL will continue training this adapter')
    else:
        model, tokenizer = load_model_and_tokenizer(model=cfg.model, lora=cfg.lora)

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

    # Initialize judge LLM in a subprocess for LLM-based clustering (if configured)
    cluster_classifier = None
    if cfg.get('judge_engine') is not None:
        judge_cfg = cfg.judge_engine
        if judge_cfg.get('tensor_parallel_size') is not None:
            logger.warning(
                'judge_engine.tensor_parallel_size is ignored; '
                'tensor parallelism is derived from judge_engine.cuda_devices')
        cluster_classifier = SubprocessClusterJudge(
            model_name=judge_cfg.model_name,
            cuda_devices=judge_cfg.cuda_devices,
            gpu_memory_utilization=judge_cfg.get('gpu_memory_utilization', 0.4),
            max_model_len=judge_cfg.get('max_model_len', 4096),
            max_num_seqs=judge_cfg.get('max_num_seqs', 64),
            sampling_kwargs=dict(cfg.get('judge_sampling', {})),
            reasoning_effort=judge_cfg.get('reasoning_effort', 'low'),
        )

    training_args = EDLConfig(
        output_dir=work_dir,
        **cfg.training,
        seed=seed,
        max_length=cfg.model.max_seq_length,
        report_to='wandb',
    )

    trainer = EDLTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        cluster_classifier=cluster_classifier,
    )

    trainer.train()

    lora_path = os.path.join(work_dir, 'lora_checkpoint')
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    wandb.finish()

    if cluster_classifier is not None:
        cluster_classifier.shutdown()

    logger.info(f'Training complete. LoRA checkpoint saved to: {lora_path}')


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
