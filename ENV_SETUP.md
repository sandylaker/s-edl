# Environment Setup

## Dependencies

```bash
pip uninstall flash-attn  # uninstall flash-attn if already installed
pip install vllm==0.17.1 transformers==5.4.0 trl==1.0.0rc1
```

## Ministral-3-8B-Base-2512

### Use TextOnly variant

The original `mistralai/Ministral-3-8B-Base-2512` is a multimodal model
(`Mistral3ForConditionalGeneration`) and cannot be loaded with
`AutoModelForCausalLM`. Use `Aratako/Ministral-3-8B-Base-2512-TextOnly`
(text-only extraction, `Ministral3ForCausalLM`) instead.

Pre-download:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Aratako/Ministral-3-8B-Base-2512-TextOnly'); AutoModelForCausalLM.from_pretrained('Aratako/Ministral-3-8B-Base-2512-TextOnly')"
```

### Using `trl vllm-serve` for GRPO training

`trl vllm-serve` uses a stricter code path that rejects
`Ministral3ForCausalLM`. Pass `--vllm_model_impl transformers` to force
the transformers modeling backend (which is what `vllm.LLM()` falls back
to automatically during evaluation):

```bash
trl vllm-serve \
    --model Aratako/Ministral-3-8B-Base-2512-TextOnly \
    --vllm_model_impl transformers \
    ...
```
