"""Model loading and Bayesian LoRA patching utilities for BLoB training."""
from __future__ import annotations
import math
import os
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from peft import get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils import setup_logger
from .load_model import load_model_and_tokenizer

logger = setup_logger('s_edl')


def _blob_sigma_from_rho(rho: torch.Tensor, bayes_eps: float) -> torch.Tensor:
    if bayes_eps < 0:
        return torch.log1p(torch.exp(rho))
    return rho.square()


def _blob_sample(self, status: bool = True) -> None:
    self.blobsample = status


def _blob_div_posterior_prior(self) -> torch.Tensor:
    eps = 1e-6
    sigma_p = float(self.bayes_beta)
    if sigma_p <= 0:
        raise ValueError(f'bayes_beta must be positive, got {sigma_p}.')

    total = None
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        if active_adapter not in self.lora_A_rho:
            continue

        mu_q = self.lora_A[active_adapter].weight
        sigma_q = _blob_sigma_from_rho(self.lora_A_rho[active_adapter], self.bayes_eps)
        kl = (
            math.log(sigma_p + eps) - torch.log(sigma_q.to(torch.float64) + eps) +
            (sigma_q.to(torch.float64).square() + mu_q.to(torch.float64).square()) / (2.0 * sigma_p * sigma_p + eps) -
            0.5)
        kl = kl.sum().to(mu_q.dtype)
        total = kl if total is None else total + kl

    if total is None:
        first_param = next(self.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        return first_param.new_zeros(())
    return total


def _blob_layer_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    result = self._blob_original_forward(x, *args, **kwargs)
    if not getattr(self, 'blobsample', False):
        return result
    if getattr(self, 'disable_adapters', False) or getattr(self, 'merged', False):
        return result
    if kwargs.get('adapter_names') is not None:
        # Mixed-batch adapter routing is kept deterministic for simplicity.
        return result

    result_dtype = result.dtype
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        if active_adapter not in self.lora_A_rho:
            continue
        if active_adapter in getattr(self, 'lora_variant', {}):
            # Keep PEFT LoRA variants unchanged.
            continue

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_cast = self._cast_input_dtype(x, lora_A.weight.dtype) if hasattr(self, '_cast_input_dtype') else x.to(
            lora_A.weight.dtype)
        x_drop = dropout(x_cast)

        sigma = _blob_sigma_from_rho(self.lora_A_rho[active_adapter], self.bayes_eps)
        lora_noise_a = sigma * torch.randn_like(lora_A.weight)

        prefix_shape = x_drop.shape[:-1]
        r_a = torch.empty(
            *prefix_shape,
            self.in_features,
            device=x_drop.device,
            dtype=x_drop.dtype,
        ).uniform_(-1, 1).sign()
        s_a = torch.empty(
            *prefix_shape,
            self.r[active_adapter],
            device=x_drop.device,
            dtype=x_drop.dtype,
        ).uniform_(-1, 1).sign()

        noise_hidden = torch.matmul(x_drop * r_a, lora_noise_a.transpose(0, 1))
        noise_hidden = noise_hidden * s_a
        noise = torch.matmul(noise_hidden, lora_B.weight.transpose(0, 1))
        result = result + noise.to(result_dtype) * scaling

    return result.to(result_dtype)


def _supports_blob_layer(module: nn.Module) -> bool:
    if not isinstance(module, LoraLayer):
        return False
    if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
        return False
    if not hasattr(module, 'in_features'):
        return False
    if not hasattr(module, 'r'):
        return False
    return True


def _patch_blob_layer(module: nn.Module, bayes_eps: float, bayes_beta: float) -> None:
    module.bayes_eps = bayes_eps
    module.bayes_beta = bayes_beta
    module.blobsample = True
    module._blob_layer = True

    if not hasattr(module, 'lora_A_rho'):
        module.lora_A_rho = nn.ParameterDict({})
    for adapter_name, lora_a in module.lora_A.items():
        if adapter_name in module.lora_A_rho:
            continue
        rho = nn.Parameter(lora_a.weight.new_zeros(module.r[adapter_name], module.in_features))
        if bayes_eps < 0:
            nn.init.uniform_(rho, bayes_eps - 1, bayes_eps)
        else:
            nn.init.uniform_(rho, bayes_eps / math.sqrt(2), bayes_eps)
        module.lora_A_rho[adapter_name] = rho

    if not hasattr(module, '_blob_original_forward'):
        module._blob_original_forward = module.forward
    module.forward = MethodType(_blob_layer_forward, module)
    module.sample = MethodType(_blob_sample, module)
    module.div_posterior_prior = MethodType(_blob_div_posterior_prior, module)
    module._blob_patched = True


def patch_model_for_blob(
    model: PreTrainedModel,
    bayes_eps: float,
    bayes_beta: float,
) -> int:
    if bayes_beta <= 0:
        raise ValueError(f'bayes_beta must be > 0, got {bayes_beta}.')

    patched = 0
    for module in model.modules():
        if _supports_blob_layer(module):
            _patch_blob_layer(module, bayes_eps=bayes_eps, bayes_beta=bayes_beta)
            patched += 1

    if patched == 0:
        raise ValueError('No LoRA layers were found for BLoB patching. Ensure LoRA is enabled before BLoB setup.')
    logger.info(f'Patched {patched} LoRA layers for BLoB (bayes_eps={bayes_eps}, bayes_beta={bayes_beta}).')
    return patched


def set_blob_sampling(model: nn.Module, status: bool) -> None:
    for module in model.modules():
        if getattr(module, '_blob_layer', False) and hasattr(module, 'sample'):
            module.sample(status)


def compute_blob_kl_divergence(model: nn.Module) -> torch.Tensor:
    total = None
    for module in model.modules():
        if getattr(module, '_blob_layer', False) and hasattr(module, 'div_posterior_prior'):
            kl = module.div_posterior_prior()
            total = kl if total is None else total + kl

    if total is None:
        first_param = next(model.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        return first_param.new_zeros(())
    return total


def build_mean_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if 'lora_A_rho' not in k}


def get_blob_state_dict(model: nn.Module, adapter_name: str = 'default') -> dict[str, torch.Tensor]:
    return get_peft_model_state_dict(model, adapter_name=adapter_name)


def load_lora_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter state dict from a PEFT checkpoint directory."""
    safetensor_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    bin_path = os.path.join(checkpoint_path, 'adapter_model.bin')

    if os.path.isfile(safetensor_path):
        return load_file(safetensor_path)
    if os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            return state['state_dict']
        if isinstance(state, dict):
            return state
    raise FileNotFoundError(
        f'No adapter weights found in {checkpoint_path}. Expected adapter_model.safetensors or adapter_model.bin.')


def sample_blob_lora_state_dict(
    posterior_state_dict: dict[str, torch.Tensor],
    bayes_eps: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample a mean-only LoRA state dict from BLoB posterior parameters.

    This returns sampled ``lora_A.weight`` tensors using ``(mu, rho)`` while
    dropping all ``lora_A_rho`` entries for vLLM/PEFT compatibility.
    """
    sampled_state: dict[str, torch.Tensor] = {}
    saw_rho = False

    for key, value in posterior_state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(f'Expected tensor value in state_dict for key {key}, got {type(value).__name__}')

        if 'lora_A_rho' in key:
            saw_rho = True
            continue

        if key.endswith('lora_A.weight'):
            rho_key = key.replace('lora_A.weight', 'lora_A_rho')
            rho = posterior_state_dict.get(rho_key)
            if rho is not None:
                if not torch.is_tensor(rho):
                    raise TypeError(f'Expected tensor rho in state_dict for key {rho_key}, got {type(rho).__name__}')
                saw_rho = True
                mu = value.detach().to(torch.float32)
                sigma = _blob_sigma_from_rho(rho.detach().to(torch.float32), bayes_eps)
                eps = torch.randn(mu.shape, generator=generator, dtype=torch.float32, device=mu.device)
                sampled = mu + sigma * eps
                sampled_state[key] = sampled.to(value.dtype).cpu().contiguous()
                continue

        sampled_state[key] = value.detach().cpu().clone().contiguous()

    if not saw_rho:
        raise ValueError('No lora_A_rho entries found. Use BLoB posterior checkpoint (lora_checkpoint_blob).')
    return sampled_state


def load_blob_model_and_tokenizer(
    model: dict[str, Any],
    lora: dict[str, Any] | None,
    blob: dict[str, Any],
    device: str | int | None = 0,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if lora is None:
        raise ValueError('BLoB training requires LoRA config.')
    if 'bayes_eps' not in blob or 'bayes_beta' not in blob:
        raise ValueError('BLoB training config must include bayes_eps and bayes_beta.')

    loaded_model, tokenizer = load_model_and_tokenizer(
        model=model,
        lora=lora,
        device=device,
    )
    patch_model_for_blob(
        model=loaded_model,
        bayes_eps=float(blob['bayes_eps']),
        bayes_beta=float(blob['bayes_beta']),
    )
    loaded_model.train()
    return loaded_model, tokenizer
