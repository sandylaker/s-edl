"""Model loading utilities with transformers + peft."""
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..utils import CHAT_TEMPLATES, setup_logger

logger = setup_logger('s_edl')

DTYPE_MAP = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}


def load_model_and_tokenizer(
    model: dict[str, Any],
    lora: dict[str, Any] | None = None,
    device: str | int | None = 0,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with transformers + peft, optionally applying LoRA.

    Args:
        model: Model config dict.
            Required: model_name, max_seq_length
            Common optional: dtype, load_in_4bit, gradient_checkpointing
        lora: LoRA config dict. Passed directly to LoraConfig.
            When None, the base model is returned without LoRA.
            Required: r, lora_alpha, lora_dropout, target_modules
            Common optional: bias
        device: Device index or name to load model on (e.g., 0, 'cuda:0').
            When None, no device_map is set (useful when accelerate manages placement).

    Returns:
        Tuple of (model, tokenizer).
    """
    model_cfg = deepcopy(model)
    dtype = model_cfg.get('dtype', 'bfloat16')
    if isinstance(dtype, str):
        dtype = DTYPE_MAP[dtype]

    model_name = model_cfg['model_name']
    load_in_4bit = model_cfg.get('load_in_4bit', False)
    gradient_checkpointing = model_cfg.get('gradient_checkpointing', False)

    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.eos_token is None:
        logger.warning(f"tokenizer.eos_token is None for {model_name}, setting to '</s>'")
        tokenizer.eos_token = '</s>'

    # Override chat template for base models (see chat_templates.py for details)
    if model_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[model_name]
        logger.info(f"Set chat template for {model_name}")
    else:
        logger.warning(f"No custom chat template found for: {model_name}")

    # Load model
    model_kwargs = {
        'dtype': dtype,
        'trust_remote_code': True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        model_kwargs['device_map'] = 'auto'  # Required for 4bit
    elif device is not None:
        # For non-quantized models, load to specific device to avoid device_map issues
        model_kwargs['device_map'] = {'': device}
    # When device is None: no device_map, let accelerate handle placement

    loaded_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare model for training (required for quantized models, good practice for all)
    if load_in_4bit:
        loaded_model = prepare_model_for_kbit_training(loaded_model)

    if lora is not None:
        # Apply LoRA
        lora_cfg = deepcopy(lora)
        lora_cfg.setdefault('bias', 'none')
        lora_cfg.setdefault('task_type', 'CAUSAL_LM')
        logger.info(f"Applying LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
        lora_config = LoraConfig(**lora_cfg)
        loaded_model = get_peft_model(loaded_model, lora_config)

        # Enable gradient checkpointing if requested
        # Must use use_reentrant=False for peft models to maintain gradient flow
        if gradient_checkpointing:
            loaded_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            logger.info('Gradient checkpointing enabled (use_reentrant=False)')

        # Ensure model is in training mode
        loaded_model.train()

        loaded_model.print_trainable_parameters()

    return loaded_model, tokenizer
