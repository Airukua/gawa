import torch

from model.char_vocab import CharVocab
from training.data import WordDataset


def test_word_dataset_padding_and_targets():
    vocab = CharVocab()
    ds = WordDataset(["hi"], vocab, max_len=6)
    item = ds[0]

    assert item["char_ids"].shape == torch.Size([6])
    assert item["lengths"].item() == 2
    assert item["target"].shape == torch.Size([6])

    target = item["target"].tolist()
    assert target[0] == vocab.BOS
    assert vocab.EOS in target


def test_word_dataset_filters_long_words():
    vocab = CharVocab()
    ds = WordDataset(["abcd", "abcdefgh"], vocab, max_len=6)
    # max_len=6 -> allowed lengths 1..4 (max_len-2)
    assert len(ds) == 1
