## [S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs (NeurIPS 2026)](https://openreview.net/forum?id=ub8LNj7RrR#discussion)

## Installation

```bash
conda create -n s-edl python=3.12 -y && conda activate s-edl
pip install -e ".[dev]"
pre-commit install  # optional
```

See [ENV_SETUP.md](ENV_SETUP.md) for model-specific setup notes (Ministral3-8B, MiMo-7B).

## Running EDL Experiments

[`scripts/old_edl.sh`](scripts/old_edl.sh) runs EDL training and evaluation with one command. Pass the model and the
dataset; the script picks the matching config in `configs/edl/`, starts the vLLM rollout server, trains, stops the
server, and evaluates the trained LoRA checkpoint:

```bash
scripts/old_edl.sh --model qwen3-8b --dataset gsm8k
```

| Option | Description | Default |
| --- | --- | --- |
| `--model` | `qwen3-8b`, `ministral3-8b`, or `mimo-7b` | `qwen3-8b` |
| `--dataset` | `math`, `gsm8k`, `chemistryqa`, or `formulareasoning` | `math` |
| `--edl-weight` | EDL loss weight | `5.0` |
| `--edl-beta` | EDL beta | `1.2` |
| `--sft-checkpoint` | SFT checkpoint to continue from (see [SFT Warm-up](#sft-warm-up-for-ministral3-8b-and-mimo-7b)) | see below |
| `--train-gpu` | GPU for the training loop | `0` |
| `--vllm-gpu` | GPU for the vLLM rollout server and the LLM judge | `1` |
| `--vllm-port` | vLLM server port | `8000` |

All three models are pretrained base models, not their instruction-tuned versions: `qwen3-8b` is
`Qwen/Qwen3-8B-Base`, `ministral3-8b` is `Aratako/Ministral-3-8B-Base-2512-TextOnly`, and `mimo-7b` is
`XiaomiMiMo/MiMo-7B-Base`.

Run `scripts/old_edl.sh --help` for details, or add `--dry-run` to print the resolved config, paths, and options
without starting anything.

### Outputs

Each run writes to `workdirs/<model>/<dataset>/edl/<run_key>/`. Re-running the same command skips training if the checkpoint exists.

### GPU Usage

The script uses two GPUs:

- **Training:** the training loop runs on `--train-gpu`. The vLLM server on `--vllm-gpu` generates rollouts with the
  current LoRA weights. For ChemistryQA and FormulaReasoning, the LLM judge (`openai/gpt-oss-20b`) that groups answers
  by meaning also runs on `--vllm-gpu`, next to the vLLM server.
- **Evaluation:** after training, the vLLM server is stopped and both GPUs are used. Generation runs with tensor
  parallelism across both GPUs; for ChemistryQA and FormulaReasoning, the judge is then loaded on the same two GPUs.

### SFT Warm-up for Ministral3-8B and MiMo-7B

Ministral3-8B and MiMo-7B need a short SFT warm-up before EDL (or Dr.GRPO) training: instead of starting from the
base model, training continues from a checkpoint fine-tuned for 1 epoch. For EDL this applies to every dataset except GSM8K. First train the warm-up checkpoint with [`scripts/baseline_sft.sh`](scripts/baseline_sft.sh), then run
the EDL script:

```bash
# 1. Fine-tune the base model for one epoch (the script also evaluates the SFT model).
scripts/baseline_sft.sh --model mimo-7b --dataset math --epochs 1

# 2. Continue with EDL training from the SFT checkpoint.
scripts/old_edl.sh --model mimo-7b --dataset math
```

Step 1 saves the checkpoint to `workdirs/<model>/<dataset>/baseline/e-1/train/lora_checkpoint`, which is where the EDL script looks by default. To continue from a different SFT checkpoint, pass its path with `--sft-checkpoint`. The EDL script stops with an error if the SFT checkpoint does not exist.

## Citation

```bibtex
@inproceedings{
li2026sedl,
title={S-{EDL}: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of {LLM}s},
author={Yawei Li and Jiazheng Li and David R{\"u}gamer and Bernd Bischl and Yulan He and Mina Rezaei},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=ub8LNj7RrR}
}
```
