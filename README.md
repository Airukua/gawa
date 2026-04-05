# GAWA — Gaussian-Weighted Abstraction for Word Architecture

![GAWA Banner](gawa.jpg)

GAWA adalah encoder–decoder karakter tingkat kata yang membentuk vektor **`eword`** untuk setiap kata menggunakan **Gaussian positional encoding** dan **Gaussian-weighted pooling**. Proyek ini dirancang sebagai modul front-end yang menggantikan tokenizer subword, sehingga panjang urutan mengikuti jumlah kata, bukan jumlah subword.

---

## Ringkasan

**GAWA** bekerja pada level karakter dan merangkum satu kata menjadi satu embedding tetap. Keunggulan utamanya:

- **Language-agnostic**: Tidak membutuhkan vocabulary subword.
- **Morphology-aware**: Bobot posisi memberi penekanan pada prefiks/sufiks.
- **Sequence lebih pendek**: 1 kata = 1 vektor.
- **Siap integrasi**: Cocok sebagai front-end untuk model transformer global.

---

## Instalasi

### 1. Install via GitHub (pip)

```bash
pip install git+https://github.com/AiRukua/gawa.git
```

### 2. Install untuk Development Lokal

```bash
git clone https://github.com/AiRukua/gawa.git
cd gawa
pip install -e .
```

### 3. Dependensi Tambahan (opsional)

```bash
pip install -e ".[dev]"
```

---

## Quick Start (CLI)

### 1. Menyiapkan Data

GAWA membutuhkan file **word list** (satu kata per baris). Kamu bisa menyiapkan dari teks mentah:

```bash
gawa-prepare --input data/raw.txt --output data/processed/train.txt --lower
```

### 2. Training

Gunakan konfigurasi YAML yang tersedia di `configs/`:

```bash
gawa-train --config configs/gawa_small.yaml
```

Checkpoint akan tersimpan di folder yang ditentukan pada config (default: `checkpoints/`).

### 3. Encoding Word Embeddings

```bash
gawa-encode \
  --checkpoint checkpoints/gawa_small/best.pt \
  --words "makan,memakan,makanan"
```

Output default adalah JSONL (bisa diarahkan ke file dengan `--output`).

### 4. Evaluasi / Rekonstruksi

```bash
gawa-evaluate --config configs/gawa_small.yaml --checkpoint checkpoints/gawa_small/best.pt
```

---

## Quick Start (Python)

```python
from gawa import GAWAModel, CharVocab, encode_words, train_from_config, load_config

# Load config dan training
cfg = load_config("configs/gawa_small.yaml")
train_from_config(cfg)

# Encoding kata dari checkpoint
kept_words, embeddings = encode_words(
    checkpoint_path="checkpoints/gawa_small/best.pt",
    words=["makan", "memakan", "makanan"],
)
print(embeddings.shape)
```

---

## Struktur Proyek

- `model/`: Implementasi encoder, decoder, dan model inti GAWA.
- `training/`: Pipeline training, scheduler, checkpointing.
- `data/`: Utilitas pembersihan dan pembuatan dataset word list.
- `eval/`: Fungsi evaluasi dan encoding.
- `scripts/`: CLI untuk training, encoding, evaluasi, dan data prep.
- `configs/`: Contoh konfigurasi YAML.

---

## Konfigurasi

GAWA memakai konfigurasi YAML terstruktur. Contoh tersedia di `configs/`:

- `configs/gawa_small.yaml`
- `configs/gawa_medium.yaml`
- `configs/gawa_large.yaml`

Parameter penting:

- `data.max_word_len`: Panjang kata maksimal (harus sama dengan `model.max_word_len`).
- `model.eword_dim`: Dimensi embedding output.
- `training.batch_size`, `training.epochs`, `training.lr`: Hyperparameter training.

---

## Lisensi

MIT License. Lihat `LICENSE` untuk detail.

---

## Kontak

**Abdul Wahid Rukua**  
🤗 HuggingFace: AiRukua  
LinkedIn: Abdul Wahid Rukua
