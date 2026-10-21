"""ChatML chat templates for base models.

Key Design Decision: Skip <|im_end|> for the last assistant message
====================================================================
When training LoRA on base models (e.g. Qwen3-Base), the model outputs garbage
tokens (Thai, Chinese, Arabic characters) after \\boxed{} at inference time when
temperature > 0. This happens because:

1. Base models haven't seen <|im_end|> during pretraining (they only know <|endoftext|>)
2. During LoRA SFT, embedding and lm_head are frozen, so the model cannot learn the
   "stop-generation" signal for <|im_end|>
3. Training with <|im_end|><|endoftext|> asks LoRA to learn: } -> <|im_end|> -> <|endoftext|>
4. But LoRA has limited capacity to override the base model's prior on this transition
5. Result: Mixed distribution where garbage tokens get sampled before eventual EOS

Solution: Skip <|im_end|> for the LAST assistant message, producing: } -> <|endoftext|>
This aligns with the base model's natural behavior and eliminates garbage tokens.
Non-last assistant messages (few-shot examples) still use <|im_end|>\\n for proper formatting.
"""

# =============================================================================
# Previous version (with <|im_end|> for last assistant message):
# Produces: ...\boxed{X}<|im_end|><|endoftext|>
# Issue: LoRA model outputs garbage tokens before <|endoftext|> when temp > 0
# =============================================================================
# _CUSTOM_QWEN3_CHAT_TEMPLATE_V1 = (
#     "{%- if messages[0].role == 'system' %}"
#     "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
#     '{%- endif %}'
#     '{%- for message in messages %}'
#     "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
#     "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
#     "{%- elif message.role == 'assistant' %}"
#     "{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>' }}"
#     "{%- if not loop.last %}{{- '\\n' }}{%- endif %}"
#     '{%- endif %}'
#     '{%- endfor %}'
#     '{%- if add_generation_prompt %}'
#     "{{- '<|im_start|>assistant\\n' }}"
#     '{%- endif %}'
# )

# =============================================================================
# Current version (skip <|im_end|> for last assistant message):
# Produces: ...\boxed{X}<|endoftext|>
# Aligns with base model's natural behavior of outputting EOS directly after content.
# Non-last assistant messages (few-shot) still have <|im_end|>\n.
# =============================================================================
_CUSTOM_QWEN3_CHAT_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    '{%- endif %}'
    '{%- for message in messages %}'
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' + message.content }}"
    "{%- if not loop.last %}{{- '<|im_end|>\\n' }}{%- endif %}"
    '{%- endif %}'
    '{%- endfor %}'
    '{%- if add_generation_prompt %}'
    "{{- '<|im_start|>assistant\\n' }}"
    '{%- endif %}')

# =============================================================================
# Mistral-style template for Ministral base models
# =============================================================================
# Mistral uses [INST]/[/INST] delimiters (token IDs 3, 4) instead of ChatML.
# System message is inlined into the first [INST] block, separated by \n\n.
# No BOS in template -- SentencePiece tokenizer prepends BOS automatically.
# Skip end token for last assistant message (code appends eos_token manually).
# Non-last assistant messages end with </s> for proper multi-turn formatting.
_CUSTOM_MISTRAL_CHAT_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    '{%- set system_message = messages[0].content %}'
    '{%- set loop_messages = messages[1:] %}'
    '{%- else %}'
    '{%- set loop_messages = messages %}'
    '{%- endif %}'
    '{%- for message in loop_messages %}'
    "{%- if message.role == 'user' %}"
    '{%- if loop.first and system_message is defined %}'
    "{{- '[INST] ' + system_message + '\\n\\n' + message.content + ' [/INST]' }}"
    '{%- else %}'
    "{{- ' [INST] ' + message.content + ' [/INST]' }}"
    '{%- endif %}'
    "{%- elif message.role == 'assistant' %}"
    "{{- ' ' + message.content }}"
    '{%- if not loop.last %}{{- eos_token }}{%- endif %}'
    '{%- endif %}'
    '{%- endfor %}'
    '{%- if add_generation_prompt %}'
    "{{- ' ' }}"
    '{%- endif %}')

# =============================================================================
# Nemotron-H template for NVIDIA Nemotron Nano 9B v2 Base
# =============================================================================
# Nemotron-H is a Mamba-2/Transformer hybrid (not a standard transformer).
# The instruct variant uses <SPECIAL_10> (system), <SPECIAL_11> (user/assistant),
# <SPECIAL_12> (end-of-turn) as delimiters instead of ChatML or Mistral tokens.
# These tokens are already in the base model's vocabulary (IDs 10, 11, 12).
# Same skip-end-token design: last assistant message omits <SPECIAL_12>,
# code appends eos_token (</s>) directly after content.
_CUSTOM_NEMOTRON_CHAT_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<SPECIAL_10>System\\n' + messages[0].content + '\\n' }}"
    '{%- endif %}'
    '{%- for message in messages %}'
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
    "{{- '<SPECIAL_11>' + message.role|capitalize + '\\n' + message.content + '\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<SPECIAL_11>Assistant\\n' + message.content }}"
    "{%- if not loop.last %}{{- '<SPECIAL_12>\\n' }}{%- endif %}"
    '{%- endif %}'
    '{%- endfor %}'
    '{%- if add_generation_prompt %}'
    "{{- '<SPECIAL_11>Assistant\\n' }}"
    '{%- endif %}')

CHAT_TEMPLATES = {
    # Qwen3 base models - use Qwen3-Instruct format
    'Qwen/Qwen3-8B-Base': _CUSTOM_QWEN3_CHAT_TEMPLATE,
    'Aratako/Ministral-3-8B-Base-2512-TextOnly': _CUSTOM_MISTRAL_CHAT_TEMPLATE,
    'Aratako/Ministral-3-8B-Instruct-2512-BF16-TextOnly': _CUSTOM_MISTRAL_CHAT_TEMPLATE,
    'nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base': _CUSTOM_NEMOTRON_CHAT_TEMPLATE,
    'XiaomiMiMo/MiMo-7B-Base': _CUSTOM_QWEN3_CHAT_TEMPLATE,
}
