"""Unit tests for RCFT trainer and data collator."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import TrainerState

from s_edl.training import RCFTConfig, RCFTDataCollator, RCFTTrainer


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
    ) -> dict[str, torch.Tensor | list[list[int]] | list[int]]:
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        encoded: list[list[int]] = []
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


class DummyCausalLM(torch.nn.Module):
    """Minimal causal LM module for trainer unit tests."""

    def __init__(self, vocab_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        if labels is None:
            loss = logits.new_zeros(())
        else:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def _make_features() -> list[dict[str, str]]:
    return [
        {
            'full_prompt': 'p0 p1 c0 c1',
            'prefill_prompt': 'p0 p1',
            'completion_prompt': 'c0 c1',
        },
        {
            'full_prompt': 'x0 y0',
            'prefill_prompt': 'x0',
            'completion_prompt': 'y0',
        },
    ]


def _make_trainer(tmp_path, token_chunk=4, vocab_chunk=8) -> tuple[RCFTTrainer, dict[str, torch.Tensor]]:
    tokenizer = DummyTokenizer()
    collator = RCFTDataCollator(tokenizer=tokenizer)
    batch = collator(_make_features())
    vocab_size = max(tokenizer.vocab.values()) + 8
    model = DummyCausalLM(vocab_size=vocab_size)

    args = RCFTConfig(
        output_dir=str(tmp_path / 'rcft'),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        bf16=False,
        fp16=False,
        report_to=[],
        save_strategy='no',
        remove_unused_columns=False,
        rcft_num_bins=10,
        rcft_token_chunk_size=token_chunk,
        rcft_vocab_chunk_size=vocab_chunk,
    )
    trainer = RCFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=_make_features(),
    )
    return trainer, batch


def test_rcft_data_collator_completion_mask_alignment() -> None:
    tokenizer = DummyTokenizer()
    collator = RCFTDataCollator(tokenizer=tokenizer)
    batch = collator(_make_features())

    completion_counts = batch['rcft_completion_mask'].sum(dim=1).tolist()
    assert completion_counts == [2, 1]

    first_indices = torch.nonzero(batch['rcft_completion_mask'][0], as_tuple=False).squeeze(-1).tolist()
    second_indices = torch.nonzero(batch['rcft_completion_mask'][1], as_tuple=False).squeeze(-1).tolist()
    assert first_indices == [2, 3]
    assert second_indices == [1]


def test_rcft_bin_edge_behavior_with_right_closed_bins(tmp_path) -> None:
    trainer, _ = _make_trainer(tmp_path)
    probs = torch.tensor([0.0, 0.1, 0.10001, 0.2, 1.0], dtype=torch.float32, device=trainer.args.device)
    bins = torch.bucketize(probs, trainer._bin_upper_boundaries, right=False)
    assert bins.tolist() == [0, 0, 1, 1, 9]


def test_rcft_per_epoch_bin_refresh_recomputes(tmp_path, monkeypatch) -> None:
    trainer, batch = _make_trainer(tmp_path)
    refresh_calls = {'count': 0}

    def fake_accumulate(self, logits, labels, completion_mask, bin_correct, bin_total):
        refresh_calls['count'] += 1
        bin_correct[0] += 1.0
        bin_total[0] += 1.0

    monkeypatch.setattr(RCFTTrainer, '_accumulate_bin_stats', fake_accumulate)
    state = TrainerState()

    state.epoch = 0.0
    trainer._refresh_bin_stats_for_epoch(state=state, train_dataloader=[batch])
    first_total = float(trainer._bin_total_count.sum().item())

    # Same epoch should skip refresh.
    state.epoch = 0.0
    trainer._refresh_bin_stats_for_epoch(state=state, train_dataloader=[batch])
    second_total = float(trainer._bin_total_count.sum().item())

    # Next epoch should refresh again.
    state.epoch = 1.0
    trainer._refresh_bin_stats_for_epoch(state=state, train_dataloader=[batch])
    third_total = float(trainer._bin_total_count.sum().item())

    assert refresh_calls['count'] == 2
    assert first_total == second_total
    assert third_total == first_total


def test_rcft_chunk_size_invariance(tmp_path) -> None:
    torch.manual_seed(7)
    trainer, _ = _make_trainer(tmp_path, token_chunk=16, vocab_chunk=64)
    trainer._bin_accuracy = torch.linspace(0.1, 0.9, trainer.rcft_num_bins, device=trainer.args.device)

    logits = torch.randn(2, 6, 23, device=trainer.args.device)
    labels = torch.randint(low=0, high=23, size=(2, 6), device=trainer.args.device)
    completion_mask = torch.zeros_like(labels, dtype=torch.bool)
    completion_mask[:, 2:] = True

    loss_a, active_a = trainer._compute_rcft_loss(logits=logits, labels=labels, completion_mask=completion_mask)

    trainer.rcft_token_chunk_size = 1
    trainer.rcft_vocab_chunk_size = 5
    loss_b, active_b = trainer._compute_rcft_loss(logits=logits, labels=labels, completion_mask=completion_mask)

    assert active_a == active_b
    assert torch.allclose(loss_a, loss_b, atol=1e-5, rtol=1e-5)


def test_rcft_numerical_stability_for_extreme_confidence(tmp_path) -> None:
    trainer, _ = _make_trainer(tmp_path, token_chunk=2, vocab_chunk=4)
    trainer._bin_accuracy = torch.full((trainer.rcft_num_bins, ), 0.999999, device=trainer.args.device)

    logits_chunk = torch.tensor(
        [[50.0, -60.0, -70.0, -80.0], [40.0, -50.0, -70.0, -90.0]],
        dtype=torch.float32,
        device=trainer.args.device,
    )
    loss = trainer._compute_rcft_chunk_loss_sum(logits_chunk)
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_rcft_total_loss_composition(tmp_path, monkeypatch) -> None:
    trainer, batch = _make_trainer(tmp_path)
    trainer.model.train()
    trainer._bin_accuracy = torch.linspace(0.1, 0.9, trainer.rcft_num_bins, device=trainer.args.device)
    device_batch = {k: (v.to(trainer.args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def fake_rcft_loss(self, logits, labels, completion_mask):
        return logits.new_tensor(2.0), 4

    monkeypatch.setattr(RCFTTrainer, '_compute_rcft_loss', fake_rcft_loss)
    outputs = trainer.model(
        input_ids=device_batch['input_ids'],
        attention_mask=device_batch['attention_mask'],
        labels=device_batch['labels'],
        use_cache=False,
    )
    sft_loss = outputs.loss.detach()

    total_loss = trainer.compute_loss(model=trainer.model, inputs=device_batch)
    expected = (1.0 - trainer.rcft_alpha) * sft_loss
    expected = expected + trainer.rcft_alpha * trainer.rcft_calibration_scale * 2.0

    assert torch.allclose(total_loss.detach(), expected, atol=1e-6, rtol=1e-6)
