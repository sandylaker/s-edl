"""Posterior-sampled BLoB evaluation using vLLM LLM + LoRARequest."""
import argparse
import copy
import gc
import json
import os
import os.path as osp
import shutil
from datetime import datetime

import numpy as np
import torch
from mmengine import Config, DictAction, mkdir_or_exist
from safetensors.torch import save_file
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
from s_edl.models.load_blob_model import load_lora_state_dict_from_checkpoint, sample_blob_lora_state_dict
from s_edl.utils import setup_logger
from s_edl.utils.chat_templates import CHAT_TEMPLATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate BLoB posterior with sampled LoRA adapters')
    parser.add_argument('config', type=str, help='Path to evaluation config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to BLoB LoRA checkpoint directory')
    parser.add_argument('-w', '--work-dir', type=str, default='workdirs/debug/eval_blob/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for posterior sampling')
    parser.add_argument(
        '-o',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options',
    )
    return parser.parse_args()


def _load_bayes_eps(cfg: Config, checkpoint_dir: str, logger) -> float:
    if 'training' in cfg and 'bayes_eps' in cfg.training:
        return float(cfg.training.bayes_eps)

    metadata_path = osp.join(checkpoint_dir, 'blob_metadata.json')
    if osp.isfile(metadata_path):
        with open(metadata_path, encoding='utf-8') as f:
            metadata = json.load(f)
        if 'bayes_eps' in metadata:
            return float(metadata['bayes_eps'])

    logger.warning('bayes_eps not found in config/checkpoint metadata. Falling back to 0.05.')
    return 0.05


def _materialize_sampled_adapters(
    posterior_state: dict[str, torch.Tensor],
    checkpoint_dir: str,
    temp_root: str,
    n_samples: int,
    bayes_eps: float,
    seed: int,
) -> list[str]:
    adapter_config_src = osp.join(checkpoint_dir, 'adapter_config.json')
    if not osp.isfile(adapter_config_src):
        raise FileNotFoundError(f'Missing adapter_config.json in checkpoint: {checkpoint_dir}')

    run_root = osp.join(temp_root, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}')
    os.makedirs(run_root, exist_ok=True)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    adapter_dirs: list[str] = []

    for sample_idx in range(n_samples):
        sampled_state = sample_blob_lora_state_dict(
            posterior_state_dict=posterior_state,
            bayes_eps=bayes_eps,
            generator=generator,
        )
        sample_dir = osp.join(run_root, f'sample_{sample_idx:03d}')
        os.makedirs(sample_dir, exist_ok=True)

        save_file(sampled_state, osp.join(sample_dir, 'adapter_model.safetensors'))
        shutil.copy2(adapter_config_src, osp.join(sample_dir, 'adapter_config.json'))
        adapter_dirs.append(sample_dir)

    return adapter_dirs


def _generate_posterior_rollouts(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    adapter_dirs: list[str],
    per_weight_generations: int,
    logger,
) -> list[list[str]]:
    n_prompts = len(prompts)
    n_weight_samples = len(adapter_dirs)
    total_steps = n_prompts * n_weight_samples * per_weight_generations
    all_generations: list[list[str]] = [[] for _ in range(n_prompts)]

    with tqdm(total=total_steps, desc='Generated rollouts') as pbar:
        for sample_idx, adapter_dir in enumerate(adapter_dirs):
            logger.info(f'Posterior weight {sample_idx + 1}/{n_weight_samples}: start generation')
            pbar.set_postfix_str(f'weight={sample_idx + 1}/{n_weight_samples}')
            lora_request = LoRARequest(
                lora_name=f'blob_sample_{sample_idx}',
                lora_int_id=sample_idx + 1,
                lora_path=adapter_dir,
            )

            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            if len(outputs) != n_prompts:
                raise RuntimeError(
                    f'vLLM returned {len(outputs)} outputs for {n_prompts} prompts '
                    f'(sample_idx={sample_idx}).')

            generated_this_weight = 0
            for prompt_idx, output in enumerate(outputs):
                batch_generations = [o.text for o in output.outputs]
                all_generations[prompt_idx].extend(batch_generations)
                generated_this_weight += len(batch_generations)

            pbar.update(generated_this_weight)

    return all_generations


def evaluate(args: argparse.Namespace) -> None:
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

    if not osp.isdir(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint directory not found: {args.checkpoint}')

    blob_eval_cfg = cfg.get('blob_eval', {})
    n_weight_samples = int(blob_eval_cfg.get('n_weight_samples', 10))
    temp_adapter_root = str(blob_eval_cfg.get('temp_adapter_root', '/tmp/s_edl_blob_eval_adapters'))
    if n_weight_samples <= 0:
        raise ValueError(f'blob_eval.n_weight_samples must be > 0, got {n_weight_samples}')

    # Load datasets
    logger.info(f'Loading datasets: train={cfg.data.train}, eval={cfg.data.eval}')
    train_dataset = load_dataset_split(cfg.data.train, split='train')
    test_dataset = load_dataset_split(cfg.data.eval, split='test')

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if cfg.model.model_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[cfg.model.model_name]
        logger.info(f'Set chat template for base model: {cfg.model.model_name}')
    else:
        logger.warning(f'No custom chat template found for: {cfg.model.model_name}')

    prompt_builder = PROMPT_BUILDERS.build({**cfg.data.prompt_builder, 'seed': args.seed})
    datasets = prompt_builder.prepare_datasets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    test_data = datasets['test']
    prompts = test_data['prefill_prompt']
    ground_truths = test_data['final_answer']
    questions = test_data['question']
    logger.info(f'Prepared {len(prompts)} prompts.')

    posterior_state = load_lora_state_dict_from_checkpoint(args.checkpoint)
    bayes_eps = _load_bayes_eps(cfg, args.checkpoint, logger)
    logger.info(f'Using bayes_eps={bayes_eps} for posterior sampling.')

    adapter_dirs: list[str] = []
    temp_run_root = ''
    try:
        adapter_dirs = _materialize_sampled_adapters(
            posterior_state=posterior_state,
            checkpoint_dir=args.checkpoint,
            temp_root=temp_adapter_root,
            n_samples=n_weight_samples,
            bayes_eps=bayes_eps,
            seed=args.seed,
        )
        temp_run_root = osp.dirname(adapter_dirs[0]) if adapter_dirs else ''
        logger.info(f'Materialized {len(adapter_dirs)} sampled adapters under: {temp_run_root}')

        conflict_keys = {'model', 'dtype', 'max_model_len'}
        vllm_engine_keys = set(cfg.vllm_engine.keys())
        if conflict_keys & vllm_engine_keys:
            raise ValueError(
                f'vllm_engine should not contain {conflict_keys & vllm_engine_keys}, '
                'these are set from cfg.model')

        llm = LLM(
            model=cfg.model.model_name,
            enable_lora=True,
            dtype=cfg.model.get('dtype', 'bfloat16'),
            max_model_len=cfg.model.max_seq_length,
            **cfg.vllm_engine,
        )
        sampling_params = SamplingParams(**cfg.vllm_sampling)

        per_weight_generations = int(cfg.vllm_sampling.n)
        total_generations_per_prompt = n_weight_samples * per_weight_generations
        logger.info(
            f'BLoB posterior evaluation setup: N={n_weight_samples} weight samples, '
            f'M={per_weight_generations} generations/weight, '
            f'total={total_generations_per_prompt} generations/prompt.')

        all_generations = _generate_posterior_rollouts(
            llm=llm,
            sampling_params=sampling_params,
            prompts=prompts,
            adapter_dirs=adapter_dirs,
            per_weight_generations=per_weight_generations,
            logger=logger,
        )

        # Free generation LLM before loading judge to allow GPU memory reuse
        destroy_model_parallel()
        destroy_distributed_environment()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Initialize judge LLM for LLM-based clustering (if configured)
        cluster_classifier = None
        if cfg.get('judge_engine') is not None:
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

        for idx, generations in enumerate(all_generations):
            if len(generations) != total_generations_per_prompt:
                raise AssertionError(
                    f'Prompt index {idx} has {len(generations)} generations; '
                    f'expected {total_generations_per_prompt}.')

        logger.info('Processing predictions and semantic clustering...')
        eval_samples: list[SampleEvalResult] = []
        for i, generations in enumerate(tqdm(all_generations, desc='Clustering')):
            final_preds = [extract_prediction(gen) for gen in generations]
            clusters, correct_cluster_idx = cluster_predictions_with_ground_truth(
                final_preds,
                ground_truths[i],
                classifier=cluster_classifier,
            )

            cluster_sizes = np.array([len(c) for c in clusters])
            n_preds = len(final_preds)
            assert cluster_sizes.sum() == n_preds, f'Cluster sizes {cluster_sizes.sum()} != n_preds {n_preds}'
            argmax_cluster_idx = int(np.argmax(cluster_sizes))
            argmax_prediction = clusters[argmax_cluster_idx].representative
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
            'blob_eval': {
                'n_weight_samples': n_weight_samples,
                'n_generations_per_weight': per_weight_generations,
                'total_generations_per_prompt': total_generations_per_prompt,
            },
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f'Results saved: samples={samples_path}, summary={summary_path}')
    finally:
        if temp_run_root and osp.isdir(temp_run_root):
            shutil.rmtree(temp_run_root, ignore_errors=True)
            logger.info(f'Cleaned temporary adapters: {temp_run_root}')


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
