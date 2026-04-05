import sys
from pathlib import Path

import yaml

from training import trainer


def test_training_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "toy.txt"
    data_path.write_text(
        "\n".join(
            [
                "makan",
                "minum",
                "tidur",
                "pergi",
                "pulang",
                "sekolah",
                "rumah",
                "buku",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = {
        "seed": 1,
        "device": "cpu",
        "data": {
            "train_path": str(data_path),
            "val_path": None,
            "test_path": None,
            "val_split": 0.2,
            "test_split": 0.2,
            "max_word_len": 16,
        },
        "model": {
            "char_emb_dim": 16,
            "pos_enc_dim": 16,
            "hidden_dim": 32,
            "eword_dim": 64,
            "max_word_len": 16,
            "encoder_lambda_adjust": 0.1,
            "decoder_num_layers": 1,
            "decoder_num_heads": 2,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "teacher_forcing": True,
            "use_tqdm": False,
            "log_every": 0,
            "eval_every": 1,
            "sample_every": 0,
            "sample_count": 2,
            "num_workers": 0,
            "pin_memory": False,
            "amp": False,
            "checkpoint": {
                "dir": str(tmp_path / "ckpt"),
                "save_every": 1,
                "save_best": True,
                "resume_path": None,
            },
        },
        "scheduler": {
            "name": "constant",
            "warmup_steps": 0,
            "min_lr": 0.0,
        },
        "wandb": {
            "enabled": False,
            "project": "gawa",
            "entity": None,
            "run_name": None,
        },
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    argv = sys.argv
    try:
        sys.argv = [argv[0], "--config", str(cfg_path)]
        trainer.main()
    finally:
        sys.argv = argv

    # Verify checkpoints were created.
    assert (tmp_path / "ckpt" / "epoch_001.pt").exists()
