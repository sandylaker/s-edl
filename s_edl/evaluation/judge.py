"""LLM judge for answer equivalence checking.

Provides two judge modes:

- Cluster judge (listwise): classifies a prediction against a list of cluster
  representatives. Used during evaluation for semantic clustering.
- Pairwise judge (batched): checks if candidate answers are equivalent to
  reference answers. Used during GRPO training as a reward function.

Both modes support subprocess execution on a dedicated GPU via
``SubprocessClusterJudge`` and ``SubprocessPairwiseJudge``.

.. note::
   Worker entry points live in :mod:`s_edl.evaluation.judge_worker` which is
   intentionally free of top-level vLLM/CUDA imports so that
   ``CUDA_VISIBLE_DEVICES`` can be set before CUDA initialization in spawned
   subprocesses.
"""
import atexit
import multiprocessing as mp
import os
import re
from collections.abc import Callable

from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from ..utils import setup_logger

logger = setup_logger('s_edl')


def normalize_cuda_devices(cuda_devices: str | list[int]) -> str:
    """Normalize cuda_devices to a CUDA_VISIBLE_DEVICES-compatible string.

    Handles both string (``"0,1"``) and list (``[0, 1]``) formats, the latter
    being produced by mmengine's ``DictAction`` when overriding config values
    from the command line (e.g. ``-o judge_engine.cuda_devices=0,1``).
    """
    if isinstance(cuda_devices, (list, tuple)):
        return ','.join(str(d) for d in cuda_devices)
    return str(cuda_devices)


def check_judge_cuda_devices(cuda_devices: str | list[int]) -> None:
    """Check that the judge GPUs are within the GPUs visible to this process.

    ``cuda_devices`` are physical GPU IDs that replace ``CUDA_VISIBLE_DEVICES``
    when the judge is loaded, so a judge GPU outside the launch-time
    ``CUDA_VISIBLE_DEVICES`` almost always means a misconfigured launch script.
    The check is skipped when ``CUDA_VISIBLE_DEVICES`` is unset (all GPUs visible).
    """
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        return
    judge_devices = normalize_cuda_devices(cuda_devices)
    outside = {d.strip() for d in judge_devices.split(',')} - {d.strip() for d in visible_devices.split(',')}
    if outside:
        raise ValueError(
            f'judge_engine.cuda_devices={judge_devices} includes GPUs {sorted(outside)} '
            f'outside CUDA_VISIBLE_DEVICES={visible_devices}')


ClusterClassifier = Callable[[str, list[str]], int | None]


def _parse_cluster_response(text: str) -> int | None:
    """Parse 'Matching index: <int>' from judge LLM output.

    Uses the last occurrence since the prompt asks for the final line.

    Returns:
        The matched index (>= 0), or None if not found or negative (e.g. -1 means no match).
    """
    matches = re.findall(r'Matching index:\s*(-?\d+)', text)
    if not matches:
        return None
    idx = int(matches[-1])
    return idx if idx >= 0 else None


def build_cluster_judge(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: dict | None = None,
) -> ClusterClassifier:
    """Build a ClusterClassifier backed by a vLLM judge model.

    Args:
        llm: A vllm.LLM instance for the judge model.
        sampling_params: SamplingParams for judge generation.
        tokenizer: Tokenizer for the judge model (used for chat template).
        chat_template_kwargs: Extra kwargs passed to ``tokenizer.apply_chat_template``
            (e.g. ``{"reasoning_effort": "low"}`` for GPT-OSS models).

    Returns:
        A ClusterClassifier callable.
    """
    from s_edl.models.system_prompts import CLUSTER_JUDGE_SYSTEM_PROMPT, CLUSTER_JUDGE_USER_PROMPT

    template_kwargs = chat_template_kwargs or {}

    def _classify(prediction: str, representatives: list[str]) -> int | None:
        if not representatives:
            return None

        representatives_rendered = '\n'.join(f'{i}: {rep}' for i, rep in enumerate(representatives))
        user_prompt = CLUSTER_JUDGE_USER_PROMPT.format(
            candidate=prediction,
            representatives_rendered=representatives_rendered,
        )
        messages = [
            {
                'role': 'system', 'content': CLUSTER_JUDGE_SYSTEM_PROMPT
            },
            {
                'role': 'user', 'content': user_prompt
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text

        idx = _parse_cluster_response(response_text)
        if idx is not None and idx >= len(representatives):
            return None
        return idx

    return _classify


class SubprocessClusterJudge:
    """A ClusterClassifier that runs the judge LLM in a subprocess.

    Uses ``mp.get_context('spawn')`` so ``CUDA_VISIBLE_DEVICES`` is set before
    CUDA initialization in the child process, allowing GPU placement independent
    of the parent process.

    Satisfies the ``ClusterClassifier`` interface via ``__call__``.
    """

    def __init__(
        self,
        model_name: str,
        cuda_devices: str | list[int],
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 4096,
        max_num_seqs: int = 64,
        sampling_kwargs: dict | None = None,
        reasoning_effort: str = 'low',
    ) -> None:
        cuda_devices = normalize_cuda_devices(cuda_devices)
        ctx = mp.get_context('spawn')
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        from s_edl.evaluation.judge_worker import cluster_judge_worker

        self._process = ctx.Process(
            target=cluster_judge_worker,
            args=(
                self._request_queue,
                self._response_queue,
                cuda_devices,
                model_name,
                gpu_memory_utilization,
                max_model_len,
                max_num_seqs,
                sampling_kwargs or {},
                reasoning_effort,
            ),
            daemon=False,
        )
        logger.info(f'Starting judge subprocess on CUDA:{cuda_devices} (model={model_name})...')
        self._process.start()

        # Wait for the subprocess to load the model and signal ready
        ready = self._response_queue.get(timeout=600)
        if isinstance(ready, Exception):
            raise RuntimeError(f'Judge subprocess failed to initialize: {ready}') from ready
        assert ready == 'ready', f'Unexpected ready signal: {ready}'
        logger.info('Judge subprocess ready.')

        self._shutdown = False
        atexit.register(self.shutdown)

    def __call__(self, prediction: str, representatives: list[str]) -> int | None:
        self._request_queue.put((prediction, representatives))
        result = self._response_queue.get()
        if isinstance(result, Exception):
            raise RuntimeError(f'Judge subprocess failed: {result}') from result
        return result

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._request_queue.put(None)
        self._process.join(timeout=30)
        if self._process.is_alive():
            logger.warning('Judge subprocess did not exit gracefully, terminating.')
            self._process.terminate()


def _parse_pairwise_response(text: str) -> bool | None:
    """Parse 'Equivalent: yes' or 'Equivalent: no' from judge LLM output.

    Uses the last occurrence since the prompt asks for the final line.

    Returns:
        True if equivalent, False if not, None on parse failure.
    """
    matches = re.findall(r'Equivalent:\s*(yes|no)', text, re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].lower() == 'yes'


def build_pairwise_judge(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: dict | None = None,
) -> Callable[[list[str], list[str]], list[bool]]:
    """Build a batched pairwise equivalence judge backed by a vLLM model.

    Args:
        llm: A vllm.LLM instance for the judge model.
        sampling_params: SamplingParams for judge generation.
        tokenizer: Tokenizer for the judge model (used for chat template).
        chat_template_kwargs: Extra kwargs passed to ``tokenizer.apply_chat_template``
            (e.g. ``{"reasoning_effort": "low"}`` for GPT-OSS models).

    Returns:
        A callable ``judge_batch(candidates, references) -> list[bool]``.
    """
    from s_edl.models.system_prompts import PAIRWISE_JUDGE_SYSTEM_PROMPT, PAIRWISE_JUDGE_USER_PROMPT

    template_kwargs = chat_template_kwargs or {}

    def judge_batch(candidates: list[str], references: list[str]) -> list[bool]:
        """Judge equivalence for a batch of (candidate, reference) pairs.

        Args:
            candidates: List of candidate answer strings.
            references: List of reference answer strings.

        Returns:
            List of booleans indicating equivalence. Parse failures default to False.
        """
        assert len(candidates) == len(references), (
            f'candidates and references must have the same length, got {len(candidates)} and {len(references)}')
        prompts = []
        for candidate, reference in zip(candidates, references):
            user_prompt = PAIRWISE_JUDGE_USER_PROMPT.format(
                candidate=candidate,
                reference=reference,
            )
            messages = [
                {
                    'role': 'system', 'content': PAIRWISE_JUDGE_SYSTEM_PROMPT
                },
                {
                    'role': 'user', 'content': user_prompt
                },
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            response_text = output.outputs[0].text
            parsed = _parse_pairwise_response(response_text)
            results.append(parsed if parsed is not None else False)
        return results

    return judge_batch


class SubprocessPairwiseJudge:
    """A batched pairwise equivalence judge that runs in a subprocess.

    Uses ``mp.get_context('spawn')`` so ``CUDA_VISIBLE_DEVICES`` is set before
    CUDA initialization in the child process, allowing GPU placement independent
    of the parent process.

    Designed for use as a GRPO reward function backend: receives a batch of
    (candidate, reference) pairs and returns a list of equivalence booleans.
    """

    def __init__(
        self,
        model_name: str,
        cuda_devices: str | list[int],
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 4096,
        max_num_seqs: int = 64,
        sampling_kwargs: dict | None = None,
        reasoning_effort: str = 'low',
    ) -> None:
        cuda_devices = normalize_cuda_devices(cuda_devices)
        ctx = mp.get_context('spawn')
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        from s_edl.evaluation.judge_worker import pairwise_judge_worker

        self._process = ctx.Process(
            target=pairwise_judge_worker,
            args=(
                self._request_queue,
                self._response_queue,
                cuda_devices,
                model_name,
                gpu_memory_utilization,
                max_model_len,
                max_num_seqs,
                sampling_kwargs or {},
                reasoning_effort,
            ),
            daemon=False,
        )
        logger.info(f'Starting pairwise judge subprocess on CUDA:{cuda_devices} (model={model_name})...')
        self._process.start()

        # Wait for the subprocess to load the model and signal ready
        ready = self._response_queue.get(timeout=600)
        if isinstance(ready, Exception):
            raise RuntimeError(f'Pairwise judge subprocess failed to initialize: {ready}') from ready
        assert ready == 'ready', f'Unexpected ready signal: {ready}'
        logger.info('Pairwise judge subprocess ready.')

        self._shutdown = False
        atexit.register(self.shutdown)

    def judge_batch(self, candidates: list[str], references: list[str]) -> list[bool]:
        """Judge equivalence for a batch of (candidate, reference) pairs.

        Args:
            candidates: List of candidate answer strings.
            references: List of reference answer strings.

        Returns:
            List of booleans indicating equivalence.
        """
        self._request_queue.put((candidates, references))
        result = self._response_queue.get()
        if isinstance(result, Exception):
            raise RuntimeError(f'Pairwise judge subprocess failed: {result}') from result
        return result

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._request_queue.put(None)
        self._process.join(timeout=30)
        if self._process.is_alive():
            logger.warning('Pairwise judge subprocess did not exit gracefully, terminating.')
            self._process.terminate()
