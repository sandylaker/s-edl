# New EDL Results with GRPO-like Updates

As discussed in Appendix A.5 of the paper, S-EDL can also be viewed as a GRPO-like update. This lets us implement S-EDL on top of `trl.GRPOTrainer`, which improves data efficiency and GPU utilization. We call this implementation New EDL and the original, naive implementation Old EDL. In several model-dataset settings, New EDL performs better
than Old EDL.

In the following tables, New EDL uses the same training hyperparameters as Dr.GRPO wherever the two methods share a hyperparameter. In addition, Old EDL and New EDL both use n=8 rollouts per prompt. For each dataset, the table reports the better of Old EDL and New EDL. The "Config and overrides" column links to the config file and lists any overrides as `-o` config options.
As for Old EDL (see [SFT Warm-up](../README.md#sft-warm-up-for-ministral3-8b-and-mimo-7b)), `Ministral3-8B-Base` and `MiMo-7B-Base` are first fine-tuned with SFT for 1 epoch, and EDL training then continues from this warm-up checkpoint.

## Qwen3-8B-Base

| Dataset | Config and overrides | Pass@1 | smECE |
| --- | --- | --- | --- |
| GSM8K | Old EDL: [edl_gsm8k_qwen3-8b.yaml](../configs/edl/edl_gsm8k_qwen3-8b.yaml)<br>`training.edl_sampling.n=8` | 83.15 | 9.93 |
| MATH500 | New EDL: [new_edl_math_qwen3-8b.yaml](../configs/new_edl/new_edl_math_qwen3-8b.yaml) | 74.68 | 7.82 |
| ChemistryQA | Old EDL: [edl_chemistryqa_qwen3-8b.yaml](../configs/edl/edl_chemistryqa_qwen3-8b.yaml) | 52.62 | 13.48 |
| FormulaReasoning | New EDL: [new_edl_formulareasoning_qwen3-8b.yaml](../configs/new_edl/new_edl_formulareasoning_qwen3-8b.yaml) | 80.39 | 5.04 |
| Average | – | 72.71 | 9.07 |

## Ministral3-8B-Base

| Dataset | Config and overrides | Pass@1 | smECE |
| --- | --- | --- | --- |
| GSM8K | New EDL: [new_edl_gsm8k_ministral3-8b.yaml](../configs/new_edl/new_edl_gsm8k_ministral3-8b.yaml) | 89.48 | 2.92 |
| MATH500 | New EDL: [new_edl_math_ministral3-8b.yaml](../configs/new_edl/new_edl_math_ministral3-8b.yaml) | 58.24 | 8.18 |
| ChemistryQA | New EDL: [new_edl_chemistryqa_ministral3-8b.yaml](../configs/new_edl/new_edl_chemistryqa_ministral3-8b.yaml) | 50.16 | 11.67 |
| FormulaReasoning | New EDL: [new_edl_formulareasoning_ministral3-8b.yaml](../configs/new_edl/new_edl_formulareasoning_ministral3-8b.yaml) | 80.34 | 8.67 |
| Average | – | 69.56 | 7.86 |

## MiMo-7B-Base

| Dataset | Config and overrides | Pass@1 | smECE |
| --- | --- | --- | --- |
| GSM8K | New EDL: [new_edl_gsm8k_mimo-7b.yaml](../configs/new_edl/new_edl_gsm8k_mimo-7b.yaml) | 83.45 | 5.63 |
| MATH500 | New EDL: [new_edl_math_mimo-7b.yaml](../configs/new_edl/new_edl_math_mimo-7b.yaml) | 63.82 | 10.68 |
| ChemistryQA | New EDL: [new_edl_chemistryqa_mimo-7b.yaml](../configs/new_edl/new_edl_chemistryqa_mimo-7b.yaml) | 45.10 | 7.84 |
| FormulaReasoning | New EDL: [new_edl_formulareasoning_mimo-7b.yaml](../configs/new_edl/new_edl_formulareasoning_mimo-7b.yaml) | 71.68 | 10.08 |
| Average | – | 66.01 | 8.56 |
