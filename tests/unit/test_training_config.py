import tempfile

from training.config import GAWAConfig, load_config


def test_load_config_allows_val_split_zero_with_val_path():
    yaml_content = """
seed: 123
data:
  train_path: "data/raw/train.txt"
  val_path: "data/raw/val.txt"
  val_split: 0.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as fh:
        fh.write(yaml_content)
        path = fh.name

    cfg = load_config(path)
    typed = GAWAConfig.from_dict(cfg)

    assert typed.seed == 123
    assert typed.data.val_path == "data/raw/val.txt"
    assert typed.data.val_split == 0.0
