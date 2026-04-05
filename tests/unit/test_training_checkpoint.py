import tempfile

import torch

from training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_roundtrip():
    model = torch.nn.Linear(4, 3)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/ckpt.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optim,
            scheduler=None,
            epoch=2,
            step=10,
            best_metric=0.5,
            config={"seed": 1},
        )

        state = load_checkpoint(
            path=path,
            model=model,
            optimizer=optim,
            scheduler=None,
        )

    assert state.epoch == 2
    assert state.step == 10
    assert state.best_metric == 0.5
