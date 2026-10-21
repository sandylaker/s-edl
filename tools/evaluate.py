"""Evaluation script with semantic clustering for LoRA checkpoints."""
import argparse
import copy
import gc
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from mmengine import Config, DictAction, mkdir_or_exist
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.lora.request import LoRARequest

from s_edl.datasets import PROMPT_BUILDERS, load_dataset_split
from s_edl.evaluation import (
    SampleEvalResult,
    build_cluster_judge,
    check_judge_cuda_devices,
    cluster_predictions_with_ground_truth,
    compute_aggregate_metrics,
    extract_prediction,
    normalize_cuda_devices,
    save_eval_samples,
)
from s_edl.utils import setup_logger
from s_edl.utils.chat_templates import CHAT_TEMPLATES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate LoRA checkpoint with semantic clustering')
    parser.add_argument('config', type=str, help='Path to evaluation config file')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to LoRA checkpoint (optional, evaluates base model if not provided)')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/eval/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options',
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('judge_engine') is not None:
        # Fail now rather than after generation when the judge GPUs do not match the launch GPUs.
        check_judge_cuda_devices(cfg.judge_engine.get('cuda_devices', '0'))

    work_dir = args.work_dir
    mkdir_or_exist(work_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger('s_edl', filepath=osp.join(work_dir, f'{timestamp}.log'), reset=True)

    logger.info('Config:\n' + '=' * 60 + '\n' + cfg.pretty_text + '\n' + '=' * 60)

    # Load datasets
    logger.info(f"Loading datasets: train={cfg.data.train}, eval={cfg.data.eval}")
    train_dataset = load_dataset_split(cfg.data.train, split='train')
    test_dataset = load_dataset_split(cfg.data.eval, split='test')

    # Build prompts
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if cfg.model.model_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[cfg.model.model_name]
        logger.info(f"Set chat template for base model: {cfg.model.model_name}")
    else:
        logger.warning(f"No custom chat template found for: {cfg.model.model_name}")
    prompt_builder = PROMPT_BUILDERS.build({**cfg.data.prompt_builder, 'seed': args.seed})

    logger.info(f"Building prompts: {cfg.data.prompt_builder}")
    datasets = prompt_builder.prepare_datasets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    test_data = datasets['test']

    prompts = test_data['prefill_prompt']
    ground_truths = test_data['final_answer']
    questions = test_data['question']

    # Log first 4 samples for sanity check
    logger.info('First 4 prompts and answers:')
    for i in range(min(4, len(prompts))):
        logger.info(f'[{i}] prompt:\n{prompts[i]}\ngt_answer: {ground_truths[i]}')

    # Initialize vLLM
    logger.info(f"Loading model: {cfg.model.model_name}")
    # Sanity check: vllm_engine should not override settings from cfg.model
    conflict_keys = {'model', 'dtype', 'max_model_len'}
    vllm_engine_keys = set(cfg.vllm_engine.keys())
    if conflict_keys & vllm_engine_keys:
        raise ValueError(
            f'vllm_engine should not contain {conflict_keys & vllm_engine_keys}, '
            'these are set from cfg.model')
    llm = LLM(
        model=cfg.model.model_name,
        enable_lora=args.checkpoint is not None,
        dtype=cfg.model.get('dtype', 'bfloat16'),
        max_model_len=cfg.model.max_seq_length,
        **cfg.vllm_engine,
    )

    lora_request = None
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        lora_request = LoRARequest(
            lora_name='eval_adapter',
            lora_int_id=1,
            lora_path=args.checkpoint,
        )
    else:
        logger.info('Evaluating base model (no LoRA checkpoint)')

    sampling_params = SamplingParams(**cfg.vllm_sampling)

    # Generate
    logger.info(f"Generating {cfg.vllm_sampling.n} samples for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Initialize judge LLM for LLM-based clustering (if configured)
    cluster_classifier = None
    if cfg.get('judge_engine') is not None:
        # Free generation LLM before loading judge to allow GPU memory reuse
        destroy_model_parallel()
        destroy_distributed_environment()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        judge_engine_cfg = copy.deepcopy(cfg.judge_engine)
        judge_model_name = judge_engine_cfg.pop('model_name')
        reasoning_effort = judge_engine_cfg.pop('reasoning_effort', 'low')
        # cuda_devices are physical GPU IDs: they replace CUDA_VISIBLE_DEVICES for the judge workers.
        cuda_devices = normalize_cuda_devices(judge_engine_cfg.pop('cuda_devices', '0'))
        if judge_engine_cfg.pop('tensor_parallel_size', None) is not None:
            logger.warning(
                'judge_engine.tensor_parallel_size is ignored; '
                'tensor parallelism is derived from judge_engine.cuda_devices')
        tensor_parallel_size = len(cuda_devices.split(','))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        logger.info(f'Loading judge model: {judge_model_name} on CUDA:{cuda_devices}')
        judge_llm = LLM(
            model=judge_model_name,
            tensor_parallel_size=tensor_parallel_size,
            **judge_engine_cfg,
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        judge_sampling_params = SamplingParams(**cfg.judge_sampling)
        cluster_classifier = build_cluster_judge(
            llm=judge_llm,
            sampling_params=judge_sampling_params,
            tokenizer=judge_tokenizer,
            chat_template_kwargs={'reasoning_effort': reasoning_effort},
        )
        logger.info('Judge LLM initialized for clustering.')

    # Process results
    logger.info('Processing predictions and clustering...')
    eval_samples = []

    for i, output in enumerate(tqdm(outputs, desc='Clustering')):
        generations = [o.text for o in output.outputs]
        final_preds = [extract_prediction(gen) for gen in generations]

        # Cluster predictions
        clusters, correct_cluster_idx = cluster_predictions_with_ground_truth(
            final_preds,
            ground_truths[i],
            classifier=cluster_classifier,
        )

        # Find argmax cluster
        cluster_sizes = np.array([len(c) for c in clusters])
        n_preds = len(final_preds)
        assert cluster_sizes.sum() == n_preds, f'Cluster sizes {cluster_sizes.sum()} != n_preds {n_preds}'
        argmax_cluster_idx = int(np.argmax(cluster_sizes))
        argmax_prediction = clusters[argmax_cluster_idx].representative

        # Compute confidence (n_preds is always > 0 since we generate cfg.generation.n samples)
        confidence = cluster_sizes[argmax_cluster_idx] / n_preds

        eval_samples.append(
            SampleEvalResult(
                question=questions[i],
                prefill_prompt=prompts[i],
                final_answer=ground_truths[i],
                generations=generations,
                final_preds=final_preds,
                semantic_clusters=[c.predictions for c in clusters],
                correct_cluster_idx=correct_cluster_idx,
                argmax_cluster_idx=argmax_cluster_idx,
                argmax_prediction=argmax_prediction,
                is_correct=(argmax_cluster_idx == correct_cluster_idx),
                confidence=confidence,
            ))

    # Compute aggregate metrics and plot calibration curve
    logger.info('Computing aggregate metrics...')
    calibration_plot_path = osp.join(work_dir, 'calibration_curve.png')
    cluster_dist_plot_path = osp.join(work_dir, 'cluster_distribution.png')
    aggregate = compute_aggregate_metrics(
        eval_samples,
        ece_bins=cfg.metrics.ece_bins,
        k_values=cfg.metrics.k_values,
        calibration_plot_path=calibration_plot_path,
        cluster_dist_plot_path=cluster_dist_plot_path,
    )

    logger.info(f'Semantic accuracy: {aggregate.semantic_accuracy:.4f}')
    logger.info(f'ECE: {aggregate.ece}')
    logger.info(f'smECE: {aggregate.smece:.4f}')
    logger.info(f'Pass@k: {aggregate.pass_at_k}')
    logger.info(f'Cluster stats: {aggregate.cluster_stats}')
    logger.info(f'Calibration curve saved: {calibration_plot_path}')
    logger.info(f'Cluster distribution saved: {cluster_dist_plot_path}')

    # Save results
    samples_path = osp.join(work_dir, 'eval_samples.jsonl')
    save_eval_samples(eval_samples, samples_path)

    summary_path = osp.join(work_dir, 'metrics_summary.json')
    summary = {
        'semantic_accuracy': aggregate.semantic_accuracy,
        'ece': aggregate.ece,
        'smece': aggregate.smece,
        'pass_at_k': aggregate.pass_at_k,
        'num_samples': aggregate.num_samples,
        'cluster_stats': aggregate.cluster_stats,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f'Results saved: samples={samples_path}, summary={summary_path}')


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
