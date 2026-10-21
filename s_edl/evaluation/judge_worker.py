"""Subprocess workers for LLM judge models.

This module is intentionally kept free of vLLM/CUDA imports at the top level.
Imports happen inside worker functions AFTER ``CUDA_VISIBLE_DEVICES`` is set,
ensuring the subprocess uses the correct GPU.
"""
import multiprocessing as mp
import os


def _init_judge_llm(
    cuda_devices: str,
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    sampling_kwargs: dict,
    reasoning_effort: str,
) -> tuple:
    """Initialize vLLM judge model on the specified GPU(s).

    Sets ``CUDA_VISIBLE_DEVICES`` and performs late imports of vLLM/CUDA.

    Args:
        cuda_devices: Comma-separated GPU IDs, e.g. ``"1"`` or ``"6,7"``.

    Returns:
        Tuple of (llm, tokenizer, sampling_params, chat_template_kwargs).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tensor_parallel_size = len(cuda_devices.split(','))
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling_params = SamplingParams(**sampling_kwargs)
    chat_template_kwargs = {'reasoning_effort': reasoning_effort}
    return llm, tokenizer, sampling_params, chat_template_kwargs


def cluster_judge_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    cuda_devices: str,
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    sampling_kwargs: dict,
    reasoning_effort: str,
) -> None:
    """Worker process for cluster judge (listwise classification)."""
    try:
        llm, tokenizer, sampling_params, template_kwargs = _init_judge_llm(
            cuda_devices, model_name, gpu_memory_utilization, max_model_len, max_num_seqs,
            sampling_kwargs, reasoning_effort,
        )
        # Import after _init_judge_llm sets CUDA_VISIBLE_DEVICES
        from s_edl.evaluation.judge import build_cluster_judge

        classifier = build_cluster_judge(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            chat_template_kwargs=template_kwargs,
        )
    except Exception as e:
        response_queue.put(e)
        return

    response_queue.put('ready')

    while True:
        request = request_queue.get()
        if request is None:
            break
        prediction, representatives = request
        try:
            result = classifier(prediction, representatives)
        except Exception as e:
            response_queue.put(e)
            return
        response_queue.put(result)


def pairwise_judge_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    cuda_devices: str,
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    sampling_kwargs: dict,
    reasoning_effort: str,
) -> None:
    """Worker process for pairwise judge (batched equivalence checking)."""
    try:
        llm, tokenizer, sampling_params, template_kwargs = _init_judge_llm(
            cuda_devices, model_name, gpu_memory_utilization, max_model_len, max_num_seqs,
            sampling_kwargs, reasoning_effort,
        )
        # Import after _init_judge_llm sets CUDA_VISIBLE_DEVICES
        from s_edl.evaluation.judge import build_pairwise_judge

        judge_batch = build_pairwise_judge(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            chat_template_kwargs=template_kwargs,
        )
    except Exception as e:
        response_queue.put(e)
        return

    response_queue.put('ready')

    while True:
        request = request_queue.get()
        if request is None:
            break
        candidates, references = request
        try:
            result = judge_batch(candidates, references)
        except Exception as e:
            response_queue.put(e)
            return
        response_queue.put(result)
