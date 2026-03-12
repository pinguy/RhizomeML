# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

RhizomeML fine-tunes a language model on the user's own AI conversations and a library of PDFs. The user drops in their conversation exports and some books, runs two commands, and gets a fine-tuned model that reflects their thinking and reading.

## Installation

```bash
pip install -r requirements.txt --upgrade
pip install bitsandbytes keybert pyipf
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## The full pipeline — two commands

```bash
python build_dataset.py
python train_script.py
```

That's it. Everything else is handled automatically.

---

## Step 1: Inputs

| What | Where |
|------|-------|
| Conversation exports (ChatGPT / Claude JSON) | `conversations.json` and/or `conversations2.json` in repo root |
| Books | `PDFs/` folder (any `.pdf`) |

- ChatGPT: Settings → Data Controls → Export Data → `conversations.json`
- Claude: Settings → Account → Export Data → `conversations.json`

## Step 2: `build_dataset.py`

Runs four stages in order:

1. **`batch_embedder.py`** — Embeds all conversations with SentenceTransformer (`all-MiniLM-L12-v2`, 384-dim) into a FAISS index. Writes `memory_texts.npy`, `memory_metadata.pkl`, `memory.index` to repo root.

2. **`data_formatter.py`** — Reads the FAISS arrays, deduplicates, scores, and generates Q&A pairs from conversations. Writes `data_finetune/conv_{train,validation,test}.jsonl.gz`.

3. **`PDFs/adaptive_semantic.py`** (once per PDF) — Extracts text via pdfminer.six, chunks it (~500 words, 100 overlap), labels themes, generates Q&A pairs. Writes `data_finetune/pdf_{bookname}_qa_{split}.jsonl.gz`. Already-processed books are skipped automatically.

4. **Merge** — Combines all `conv_*` and `pdf_*_qa_*` files into `data_finetune/dataset_{train,validation,test}.jsonl.gz`.

### Useful flags

```bash
python build_dataset.py --force          # Re-run everything from scratch
python build_dataset.py --skip-embed     # Skip batch_embedder.py (dangerous — see gotcha below)
python build_dataset.py --skip-conv      # Skip data_formatter.py
python build_dataset.py --skip-pdfs      # Skip all PDF processing
python build_dataset.py --skip-merge     # Skip final merge
python build_dataset.py --pdf "My Book.pdf"  # Process only one PDF
```

### CRITICAL GOTCHA — memory file conflict

`adaptive_semantic.py` overwrites `memory_texts.npy` and `memory.index` in the repo root every time it processes a PDF. If you run PDFs and then run `data_formatter.py`, it will read the last PDF's data instead of your conversations, producing ~150 conversation records instead of ~76,000.

**`build_dataset.py` handles this correctly** by always re-running `batch_embedder.py` before `data_formatter.py`. Never use `--skip-embed` unless you are certain no PDFs have been processed since the last embed.

If you end up with suspiciously few conversation records (hundreds instead of tens of thousands), re-run:
```bash
python build_dataset.py --skip-pdfs --force
```

## Step 3: `train_script.py`

QLoRA 4-bit fine-tuning via PEFT. Key settings to edit in the file:

```python
model_name = "LiquidAI/LFM2.5-1.2B-Base"  # any HF CausalLM
# LoRA
r = 64                  # rank — must match any checkpoint you resume from
lora_alpha = 128
# Memory
default_batch_size = 2
default_grad_accum = 8  # effective batch = batch × accum, target ~16
```

If training fails with a size mismatch error like:
```
copying a param with shape torch.Size([16, 2048]) from checkpoint,
shape in current model is torch.Size([64, 2048])
```
It means the checkpoint's LoRA rank doesn't match the current config. Rename the old checkpoint dir (e.g. append `-backup`) and delete `data_finetune/tokenized_cache/`, then re-run.

### Tokenization cache

Training caches the tokenized dataset to `data_finetune/tokenized_cache/`. If you rebuild the dataset or change the model, delete this cache or training will use stale data:

```bash
rm -rf data_finetune/tokenized_cache
```

### Early stopping

Training stops at whichever comes first: epoch limit (default 3) or 82% semantic theme coverage. Best checkpoint is usually the last or second-to-last. Configurable in `train_script.py`:

```python
metrics['coverage'] >= 0.82
```

## Chat interface

```bash
python chat.py                          # Uses RhizomeML-finetuned/ by default
python gradio_chat_tts.py --tts-cpu --no-stt --quant-bits 4
python gradio_chat_tts.py --model ./RhizomeML-finetuned/checkpoint-6000/
```

## Export to GGUF

```bash
python -m venv venv_gguf
source venv_gguf/bin/activate
pip install --use-deprecated=legacy-resolver peft
python convert_to_gguf.py   # auto 4-bit
deactivate
```

Then run with Koboldcpp or llama-server. See `How_To_Run.md` for full CUDA build instructions.

---

## Architecture

```
conversations.json / conversations2.json
PDFs/*.pdf
        │
        ▼
build_dataset.py  ─────────────────────────────────────────────────────────
        │
        ├─ [1] batch_embedder.py
        │        Reads conversations.json + conversations2.json
        │        Embeds with SentenceTransformer (all-MiniLM-L12-v2, 384-dim)
        │        → memory_texts.npy, memory_metadata.pkl, memory.index
        │
        ├─ [2] data_formatter.py --output-prefix conv --output-dir data_finetune
        │        Reads memory_texts.npy + memory_metadata.pkl
        │        Deduplicates via cosine similarity (FAISS)
        │        Generates Q&A pairs from conversation chunks
        │        Theme extraction: TF-IDF + optional KeyBERT
        │        IPF calibration of theme co-occurrence weights
        │        Quality scoring + 80/10/10 stratified split
        │        → data_finetune/conv_{train,validation,test}.jsonl.gz
        │
        ├─ [3] PDFs/adaptive_semantic.py  (once per PDF, skips already done)
        │        Extracts text via pdfminer.six
        │        Chunks ~500 words / 100 overlap
        │        Semantic labeling + Q&A generation
        │        → data_finetune/pdf_{name}_qa_{split}.jsonl.gz
        │        (also writes pdf_{name}_knowledge_{split}.jsonl.gz)
        │
        └─ [4] Merge
                 Combines conv_* + pdf_*_qa_* per split
                 → data_finetune/dataset_{train,validation,test}.jsonl.gz

train_script.py
        Reads data_finetune/dataset_train.jsonl.gz
        QLoRA 4-bit (bitsandbytes NF4) via PEFT LoraConfig (r=64, alpha=128)
        WeightedRandomSampler for rare-theme oversampling
        Sequence packing (~25% throughput gain)
        Tokenized cache → data_finetune/tokenized_cache/
        Checkpoints every 150 steps, auto-resumes
        → RhizomeML-finetuned/checkpoint-*/
```

## Key files

| File | Role |
|------|------|
| `build_dataset.py` | Master orchestrator — runs all 4 dataset-building stages |
| `batch_embedder.py` | Embeds conversations → FAISS arrays |
| `data_formatter.py` | Conversation Q&A generation, dedup, theme labeling, splitting |
| `PDFs/adaptive_semantic.py` | PDF extraction, chunking, Q&A generation |
| `train_script.py` | QLoRA fine-tuning |
| `chat.py` | Terminal chat interface (streaming, /eval, /info) |
| `gradio_chat_tts.py` | Gradio web UI with Vosk STT and Kokoro TTS |
| `convert_to_gguf.py` | Merge LoRA + convert to GGUF via llama.cpp |

## Generated files — not committed

```
memory_texts.npy, memory_metadata.pkl, memory.index  — conversation embeddings
semantic_memory.pkl                                   — PDF semantic state (grows each book)
data_finetune/conv_*.jsonl.gz                         — conversation training data
data_finetune/pdf_*_qa_*.jsonl.gz                     — per-book Q&A data
data_finetune/dataset_{train,validation,test}.jsonl.gz — final merged dataset
data_finetune/tokenized_cache/                        — cached tokenized training data
RhizomeML-finetuned/                                  — LoRA checkpoints
gguf_models/                                          — exported GGUF
```

## Typical dataset size

~3,000 conversations + 25 PDFs produces:
- **~115,000 train** / 14,500 validation / 14,500 test
- Conversations ~65%, PDFs ~35%

## PDFs that fail

Some scanned PDFs have corrupt font encoding (`(cid:XX)` characters) and produce 0 chunks. Check with:

```python
from pdfminer.high_level import extract_text
print(extract_text("PDFs/mybook.pdf")[:500])
```

If the output is mostly `(cid:XX)`, the PDF cannot be processed without OCR.

## Platform

- Linux (Ubuntu 22.04 reference, works on other distros with Python 3.10+)
- NVIDIA GPU ≥ Compute 6.0 recommended; CPU fallback fully supported
- Non-Ubuntu: use `Dockerfile.rhizome` + Distrobox with `--nvidia`; see `How_To_Run.md`
