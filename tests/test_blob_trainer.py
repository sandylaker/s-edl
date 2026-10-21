"""Unit tests for BLoB trainer and BLoB model utilities."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from s_edl.models.load_blob_model import (
    build_mean_lora_state_dict,
    compute_blob_kl_divergence,
    patch_model_for_blob,
    set_blob_sampling,
)
from s_edl.training import BLoBConfig, BLoBDataCollator, BLoBTrainer


class DummyTokenizer:
    """Whitespace tokenizer with configurable padding/truncation behavior."""

    def __init__(self) -> None:
        self.vocab = {'<pad>': 0, '<eos>': 1}
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = 'right'
        self.truncation_side = 'right'

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids

    def __call__(
        self,
        text: str | list[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        pad_to_multiple_of: int | None = None,
    ):
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        encoded = []
        for sample in texts:
            ids = self._encode(sample)
            if add_special_tokens:
                ids = ids + [self.eos_token_id]
            if truncation and max_length is not None and len(ids) > max_length:
                if self.truncation_side == 'left':
                    ids = ids[-max_length:]
                else:
                    ids = ids[:max_length]
            encoded.append(ids)

        if padding:
            target_len = max((len(ids) for ids in encoded), default=0)
            if pad_to_multiple_of is not None and target_len > 0:
                target_len = ((target_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            padded_ids = []
            padded_mask = []
            for ids in encoded:
                pad_len = target_len - len(ids)
                if self.padding_side == 'left':
                    padded_ids.append([self.pad_token_id] * pad_len + ids)
                    padded_mask.append([0] * pad_len + [1] * len(ids))
                else:
                    padded_ids.append(ids + [self.pad_token_id] * pad_len)
                    padded_mask.append([1] * len(ids) + [0] * pad_len)
        else:
            padded_ids = encoded
            padded_mask = [[1] * len(ids) for ids in encoded]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(padded_mask, dtype=torch.long),
            }

        if is_single:
            return {'input_ids': padded_ids[0], 'attention_mask': padded_mask[0]}
        return {'input_ids': padded_ids, 'attention_mask': padded_mask}


class DummyBlobLayer(torch.nn.Module):
    """Minimal module exposing BLoB interfaces used by the trainer."""

    def __init__(self) -> None:
        super().__init__()
        self._blob_layer = True
        self.blobsample = True
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def sample(self, status: bool = True) -> None:
        self.blobsample = status

    def div_posterior_prior(self) -> torch.Tensor:
        return self.weight.square().sum()


class DummyCausalLM(torch.nn.Module):
    """Minimal causal LM with an attached BLoB-compatible regularizer."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.blob_regularizer = DummyBlobLayer()

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        hidden = self.embed(input_ids)
        if self.blob_regularizer.blobsample:
            hidden = hidden + torch.randn_like(hidden) * 0.01
        logits = self.lm_head(hidden)
        if labels is None:
            loss = logits.new_zeros(())
        else:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


class SequenceLossModel(torch.nn.Module):
    """Model that emits a deterministic sequence of scalar losses per forward call."""

    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.losses = losses
        self.calls = 0
        self.blob_regularizer = DummyBlobLayer()

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        value = self.losses[self.calls % len(self.losses)]
        self.calls += 1
        loss = self.anchor * 0 + torch.tensor(value, device=input_ids.device)
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8, device=input_ids.device)
        return SimpleNamespace(logits=logits, loss=loss)


def _make_features() -> list[dict[str, str]]:
    return [
        {
            'full_prompt': 'p0 p1 p2'
        },
        {
            'full_prompt': 'x0 x1'
        },
    ]


def test_blob_data_collator_masks_padding() -> None:
    tokenizer = DummyTokenizer()
    collator = BLoBDataCollator(tokenizer=tokenizer)
    batch = collator(_make_features())

    assert batch['input_ids'].shape == batch['labels'].shape
    assert (batch['labels'][batch['attention_mask'] == 0] == -100).all()


def test_build_mean_lora_state_dict_filters_rho() -> None:
    state_dict = {
        'model.q_proj.lora_A.default.weight': torch.ones(2, 2),
        'model.q_proj.lora_A_rho.default': torch.zeros(2, 2),
        'model.q_proj.lora_B.default.weight': torch.ones(2, 2),
    }
    mean_state = build_mean_lora_state_dict(state_dict)
    assert 'model.q_proj.lora_A_rho.default' not in mean_state
    assert 'model.q_proj.lora_A.default.weight' in mean_state
    assert 'model.q_proj.lora_B.default.weight' in mean_state


def test_blob_patch_stochastic_and_kl_are_active() -> None:
    peft = __import__('peft')
    transformers = __import__('transformers')
    LoraConfig = getattr(peft, 'LoraConfig')
    get_peft_model = getattr(peft, 'get_peft_model')
    LlamaConfig = getattr(transformers, 'LlamaConfig')
    LlamaForCausalLM = getattr(transformers, 'LlamaForCausalLM')

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type='CAUSAL_LM',
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=['q_proj', 'v_proj'],
        ),
    )
    patch_model_for_blob(model, bayes_eps=0.05, bayes_beta=0.2)
    for module in model.modules():
        if getattr(module, '_blob_layer', False):
            for active_adapter in module.active_adapters:
                if active_adapter in module.lora_B:
                    module.lora_B[active_adapter].weight.data.fill_(0.01)
                if active_adapter in module.lora_A_rho:
                    module.lora_A_rho[active_adapter].data.fill_(0.5)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    set_blob_sampling(model, False)
    logits_det = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    set_blob_sampling(model, True)
    logits_sto = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    assert not torch.allclose(logits_det, logits_sto)

    kl = compute_blob_kl_divergence(model)
    assert torch.isfinite(kl)
    assert float(kl.detach().item()) > 0.0


def test_blob_compute_loss_averages_k_samples(tmp_path) -> None:
    tokenizer = DummyTokenizer()
    collator = BLoBDataCollator(tokenizer=tokenizer)
    batch = collator([{'full_prompt': 'a b c d'}])
    model = SequenceLossModel(losses=[1.0, 2.0, 3.0])

    args = BLoBConfig(
        output_dir=str(tmp_path / 'blob_loss_avg'),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        no_cuda=True,
        report_to=[],
        save_strategy='no',
        remove_unused_columns=False,
        bayes_train_n_samples=3,
        bayes_kllr=0.1,
    )
    trainer = BLoBTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=[{
            'full_prompt': 'a b c d'
        }],
    )

    device_batch = {k: (v.to(trainer.args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.train()
    loss = trainer.compute_loss(model=trainer.model, inputs=device_batch)
    assert torch.allclose(loss.detach(), torch.tensor(2.0, device=loss.device), atol=1e-6)


def test_blob_dual_optimizer_kl_step_updates_params(tmp_path) -> None:
    tokenizer = DummyTokenizer()
    collator = BLoBDataCollator(tokenizer=tokenizer)
    batch = collator([{'full_prompt': 'a b c d'}])
    model = DummyCausalLM(vocab_size=max(tokenizer.vocab.values()) + 32)

    args = BLoBConfig(
        output_dir=str(tmp_path / 'blob_kl_step'),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        no_cuda=True,
        report_to=[],
        save_strategy='no',
        remove_unused_columns=False,
        bayes_train_n_samples=1,
        bayes_kllr=0.1,
    )
    trainer = BLoBTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=[{
            'full_prompt': 'a b c d'
        }],
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=4)

    trainer.model.train()
    before = trainer.model.blob_regularizer.weight.detach().clone()
    trainer._blob_callback.on_train_begin(
        args=trainer.args,
        state=trainer.state,
        control=trainer.control,
        train_dataloader=[batch],
    )
    trainer._blob_callback.on_optimizer_step(args=trainer.args, state=trainer.state, control=trainer.control)
    after = trainer.model.blob_regularizer.weight.detach().clone()

    assert not torch.allclose(before, after)
    assert trainer._last_blob_kl_loss > 0.0


def test_blob_kl_pi_large_m_no_overflow(tmp_path) -> None:
    tokenizer = DummyTokenizer()
    model = DummyCausalLM(vocab_size=max(tokenizer.vocab.values()) + 32)

    args = BLoBConfig(
        output_dir=str(tmp_path / 'blob_large_m'),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        no_cuda=True,
        report_to=[],
        save_strategy='no',
        remove_unused_columns=False,
        bayes_train_n_samples=1,
        bayes_kllr=0.01,
        bayes_kl_reweighting=True,
    )
    trainer = BLoBTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=[{
            'full_prompt': 'a b c d'
        }],
    )

    trainer._blob_M = 1869
    trainer._blob_kl_cycle_step = 1
    pi_early = trainer._compute_kl_pi()
    assert pi_early >= 0.0
    assert pi_early <= 1.0
    assert torch.isfinite(torch.tensor(pi_early, dtype=torch.float64))

    trainer._blob_kl_cycle_step = trainer._blob_M
    pi_late = trainer._compute_kl_pi()
    assert pi_late > 0.0
    assert abs(pi_late - 0.5) < 1e-6
