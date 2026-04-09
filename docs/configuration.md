# Configuration Reference

GAWA uses a YAML config with three main sections: `data`, `model`, and `training`.

## Data

Common keys:

- `train_path` path to the word list
- `val_path` optional validation list
- `test_path` optional test list
- `val_split` fraction of train data used for validation
- `test_split` fraction of train data used for test
- `max_word_len` maximum word length (must match `model.max_word_len` to avoid a length mismatch error)

## Model

Common keys:

- `char_emb_dim`
- `pos_enc_dim`
- `hidden_dim`
- `eword_dim`
- `max_word_len`
- `encoder_lambda_adjust`
- `decoder_num_layers`
- `decoder_num_heads`

## Training

Common keys:

- `batch_size`
- `epochs`
- `lr`
- `weight_decay`
- `log_every`
- `eval_every`
- `grad_clip_norm`
- `teacher_forcing`
- `checkpoint.dir`
- `checkpoint.save_every`
- `checkpoint.max_keep`
- `checkpoint.resume_path`

## Notes

- `data.max_word_len` must match `model.max_word_len` or training will fail with `data.max_word_len must match model.max_word_len`.
- If `val_path` and `test_path` are not provided, the train list is split by `val_split` and `test_split`.
