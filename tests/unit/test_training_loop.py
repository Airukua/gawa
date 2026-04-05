import torch

from model.char_vocab import CharVocab
from training.loop import evaluate, run_epoch, sample_reconstructions


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = torch.nn.Linear(4, vocab_size)

    def forward(self, char_ids, lengths, target_ids=None, teacher_forcing=True):
        batch = char_ids.size(0)
        seq_len = target_ids.size(1) if target_ids is not None else char_ids.size(1)
        dummy = torch.zeros(batch * seq_len, 4, device=char_ids.device)
        logits = self.proj(dummy).view(batch, seq_len, self.vocab_size)
        eword = torch.zeros(batch, 8, device=char_ids.device)
        return logits, eword

    def reconstruct(self, char_ids, lengths, vocab):
        batch = char_ids.size(0)
        return ["x"] * batch


def _make_batch(batch_size: int, seq_len: int, vocab_size: int):
    return {
        "char_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "lengths": torch.full((batch_size,), seq_len, dtype=torch.long),
        "target": torch.randint(1, vocab_size, (batch_size, seq_len)),
    }


def test_run_epoch_and_eval_smoke():
    model = DummyModel(vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch(batch_size=2, seq_len=5, vocab_size=10)
    loader = [batch, batch]

    loss, acc, step = run_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        scaler=None,
        grad_clip_norm=1.0,
        log_every=0,
        pad_idx=0,
        teacher_forcing=True,
        use_tqdm=False,
        wandb_run=None,
        epoch=1,
        global_step=0,
    )

    assert loss >= 0
    assert 0.0 <= acc <= 1.0
    assert step == 2

    val_loss = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        pad_idx=0,
    )
    assert val_loss >= 0


def test_sample_reconstructions_smoke():
    model = DummyModel(vocab_size=10)
    vocab = CharVocab()
    sample_reconstructions(
        model=model,
        vocab=vocab,
        words=["hi", "ok"],
        max_len=6,
        device=torch.device("cpu"),
    )
