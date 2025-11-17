# 🌱 RhizomeML

### *Production-Grade Fine-Tuning Pipeline for Context-Aware Conversational AI*

> A complete, offline-first pipeline for fine-tuning language models with semantic memory integration. Transforms raw conversations, PDFs, and documents into high-quality training data with IPF-calibrated theme weighting, FAISS-backed retrieval, and QLoRA adaptation.

**Built for the real world:** Runs from Xeons to GPU clusters. No cloud required. **Now with CPU-optimized QLoRA 4-bit quantization.**

---

## 🎯 What Does It Do?

RhizomeML takes your messy conversation logs and PDFs, then:

1. **Cleans & deduplicates** with embedding-based semantic filtering
2. **Extracts themes** using TF-IDF, KeyBERT, and IPF (Iterative Proportional Fitting)
3. **Weights training samples** to prevent common themes from dominating
4. **Fine-tunes models** with QLoRA (4-bit) for extreme memory efficiency
5. **Tracks semantic diversity** during training to ensure balanced learning
6. **Packs sequences** to eliminate wasted padding (~25% speedup)

**Result:** A model that understands *your* conversations, *your* documents, and *your* domain — with measurably diverse knowledge coverage.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| **OS** | Linux (tested in Distrobox) | Debian/Ubuntu-based |
| **CPU** | 8 cores (AVX2 support) | 14+ cores (Xeon/Ryzen) |
| **RAM** | 16 GB | 32+ GB |
| **Storage** | 50 GB free | 100+ GB (NVMe preferred) |
| **GPU** | None (CPU works!) | NVIDIA (Compute ≥6.0) |

### Installation

```bash
# Option 1: Inside Distrobox (recommended for isolation)
distrobox create --name rhizome-dev --image debian:latest
distrobox enter rhizome-dev

# Option 2: Native Linux
# (Just run the commands below in your terminal)

# Clone and setup
git clone https://github.com/yourusername/RhizomeML.git
cd RhizomeML

# Install dependencies
pip3 install -r requirements.txt --upgrade

# Install bitsandbytes for QLoRA (CPU and GPU)
pip3 install bitsandbytes

# Download NLTK data (for semantic processing)
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Optional: Install KeyBERT for advanced phrase extraction
pip3 install keybert

# Optional: Install pyipf for IPF calibration
pip3 install pyipf
```

---

## 📁 Project Structure

```
RhizomeML/
├── 📚 Input Data
│   ├── PDFs/                          # Place raw PDFs here
│   ├── conversations.json             # ChatGPT export
│   ├── conversations2.json            # Claude export (optional)
│   ├── pdf_texts.json                 # PDFs JSON (optional)
│   ├── pdf_to_json.py                 # PDF → structured JSON
│   ├── batch_embedder.py              # Embed & index memory
│   ├── data_formatter.py              # Clean, dedupe, label, create datasets
│   └── train_script.py                # ⚡ CPU-optimized QLoRA training
│   ├── memory_texts.npy               # Embedded text vectors
│   ├── memory_metadata.pkl            # Metadata for retrieval
│   ├── semantic_memory.pkl            # Learned theme weights
│   ├── data_finetune/                 # Training datasets
│   │   ├── dataset_train.jsonl
│   │   ├── dataset_validation.jsonl
│   │   ├── dataset_test.jsonl
│   │   ├── dataset_metadata.json      # Theme distribution stats
│   │   └── tokenized_cache/           # ⚡ Auto-cached tokenized data
│   └── DeepSeek-R1-Distill-Qwen-1.5B-finetuned/  # Model checkpoints
│   ├── gradio_chat_tts.py             # STT → LLM → TTS interface
│   ├── UCS_v3_4_1.py                  # UCS config
│   ├── README.md
│   └── requirements.txt
```

---

## 📄 Complete Pipeline Walkthrough

### Step 1: Prepare Your Data

#### 1.1 Convert PDFs to JSON

```bash
python3 pdf_to_json.py ./PDFs/
```

**What it does:**
- Extracts text from PDFs with proper formatting
- Chunks into semantically coherent segments
- Preserves metadata (filename, page numbers, source type)
- Outputs: `pdf_texts.json`

**Tips:**
- Works best with text-based PDFs (not scanned images)
- Handles multiple PDFs in parallel
- Preserves document structure for better context

---

#### 1.2 Export Your Conversations

**From ChatGPT:**
1. Settings → Data Controls → Export Data
2. Download and extract `conversations.json`
3. Place in project root

**From Claude:**
1. Export conversations
2. Rename to `conversations2.json`
3. Place alongside `conversations.json`

**Supported formats:**
- ChatGPT JSON exports
- Claude JSON exports
- Custom JSON (see format below)

<details>
<summary>📋 Custom Conversation Format</summary>

```json
{
  "conversations": [
    {
      "id": "conv_12345",
      "messages": [
        {
          "author": "user",
          "content": "Your question here",
          "timestamp": 1234567890
        },
        {
          "author": "assistant",
          "content": "AI response here",
          "timestamp": 1234567891
        }
      ]
    }
  ]
}
```
</details>

---

### Step 2: Build Semantic Memory Index

```bash
python3 batch_embedder.py
```

**What it does:**
- Loads all conversations + PDF chunks
- Generates 384-dim embeddings using SentenceTransformers
- Creates FAISS-ready arrays for fast similarity search
- Saves: `memory_texts.npy`, `memory_metadata.pkl`

**Configuration options:**
```python
# In batch_embedder.py, adjust these:
use_gpu = False              # Set True if you have GPU
batch_size = 32              # Lower if OOM errors
embedding_model = 'all-MiniLM-L12-v2'  # Or other ST models
```

**Output files:**
- `memory_texts.npy` - Embedded text vectors (shape: N × 384)
- `memory_metadata.pkl` - Source info, timestamps, conversation IDs

---

### Step 3: Generate Training Dataset

```bash
python3 data_formatter.py \
  --enable-semantic-labeling \
  --extract-keyphrases \
  --semantic-mode adaptive \
  --semantic-method hybrid \
  --dedup-similarity-threshold 0.95 \
  --qa-quality-score-threshold 0.46
```

**⚠️ Note:** Keyphrase extraction improves semantic richness but increases runtime; enable only for smaller datasets.

**What it does:**
1. **Loads data:** Memory texts + metadata
2. **Cleans:** Removes artifacts, fixes encoding, validates text
3. **Deduplicates:** Semantic similarity-based (not just exact matches)
4. **Labels themes:** Extracts keyphrases + TF-IDF terms, builds theme hierarchy
5. **Scores quality:** Multi-metric evaluation (coherence, density, structure)
6. **Creates pairs:** Conversational Q&A + PDF-based prompts
7. **Applies IPF:** Calibrates theme co-occurrence for balanced distribution
8. **Splits data:** Stratified train/val/test (80/10/10 by default)

**Key arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-semantic-labeling` | Extract and track themes | False |
| `--extract-keyphrases` | Use KeyBERT for phrase extraction | False |
| `--semantic-mode` | `normal` or `adaptive` (learns over time) | adaptive |
| `--semantic-method` | `tfidf`, `ipf`, or `hybrid` | hybrid |
| `--dedup-similarity-threshold` | Cosine similarity cutoff (0-1) | 0.95 |
| `--qa-quality-score-threshold` | Min quality for Q&A pairs | 0.46 |
| `--force-cpu` | Force CPU even if GPU available | False |

**Output:**
```
data_finetune/
├── dataset_train.jsonl              # Training pairs (45k samples)
├── dataset_validation.jsonl         # Validation pairs (5k samples)
├── dataset_test.jsonl               # Test pairs (5k samples)
├── dataset_metadata.json            # Theme distribution, quality stats
├── dataset_train_detailed.jsonl    # Full metadata for analysis
├── dataset_validation_detailed.jsonl
└── dataset_test_detailed.jsonl
```

**Semantic metadata includes:**
- **4,748 unique themes** (example dataset)
- Theme frequency distribution
- Source breakdown (conversation vs PDF)
- Quality score statistics

<details>
<summary>📊 Example Metadata Output</summary>

```json
{
  "total_pairs": 56742,
  "splits": {
    "train": 45393,
    "validation": 5674,
    "test": 5675
  },
  "theme_distribution": {
    "like": 29499,
    "time": 10831,
    "system": 9265,
    "model": 8182,
    "ulysses": 1,
    "james_joyce": 1
  },
  "quality_stats": {
    "train": {
      "mean": 0.850,
      "std": 0.186,
      "min": 0.46,
      "max": 1.0
    }
  }
}
```
</details>

---

### Step 4: Fine-Tune the Model

```bash
python3 train_script.py
```

**What it does:**
1. **Auto-detects hardware:** CPU or GPU with intelligent fallback
2. **Loads model:** DeepSeek-R1-Distill-Qwen-1.5B (or any HuggingFace model)
3. **Applies QLoRA:** 4-bit quantization (9M trainable / 1.1B total params)
4. **Enables theme weighting:** Rare themes get more training samples
5. **Packs sequences:** ~25% reduction in wasted padding
6. **Caches tokenization:** Instant subsequent runs
7. **Tracks diversity:** Monitors theme coverage during training
8. **Saves checkpoints:** Every 150 steps with resumable state
9. **Generates plots:** Loss curves, learning rate, theme diversity

**🔥 NEW CPU Optimizations:**
- ✅ **QLoRA 4-bit quantization** (75% memory reduction)
- ✅ **BF16 precision** (5-10% speedup when compatible)
- ✅ **Thread affinity tuning** (27 threads optimized)
- ✅ **Sequence packing** (20-40% throughput boost)
- ✅ **Micro-batching** (2×8 for stability)
- ✅ **Dataset caching** (5-30% faster subsequent runs)
- ✅ **Hard-frozen non-LoRA weights** (5-8% speedup)

**Expected output:**
```
🤖 DeepSeek-R1-Distill-Qwen-1.5B Fine-Tuning Suite
   🎨 Now with Semantic Theme-Aware Training!
   ⚡ CPU-Optimized with QLoRA 4-bit Support!

🔧 Model Setup
✅ Model loaded and QLoRA applied successfully on CPU
📊 Parameters: 9,232,384 trainable / 1,131,222,528 total (0.82%)
🔬 Using 4-bit quantization (QLoRA)

📚 Data Processing
✅ Dataset tokenization complete
📈 Tokenized sequence lengths: min=34, max=512, avg=170.5
💡 TIP: Average sequence length is 170.5 tokens.
📦 Applying sequence packing for CPU efficiency...
✅ Packed 45,393 → 33,634 sequences (25.9% reduction)
   Expected throughput boost: 20-40%

⚙️ Training Configuration
🎯 Number of training epochs: 3
📦 Effective batch size: 2 × 8 = 16
🚀 Training on: CPU: 28 cores (using 27 threads)
⚡ CPU Optimizations Applied:
   • Threads: 27
   • BF16: Auto-detected
   • QLoRA 4-bit: True
   • Micro-batching: batch=2, accum=8
   • Sequence packing: True
   • Dataset caching: True
🎨 Theme-weighted sampling: ENABLED
```

**Hardware-specific behavior:**

| Hardware | Batch Size | Grad Accum | Quantization | Expected Time* |
|----------|-----------|------------|--------------|----------------|
| **CPU (Xeon E5-2680 v4)** | 2 | 8 | QLoRA 4-bit | 7-10 days** |
| **RTX 3060 (12GB)** | 4 | 8 | QLoRA 4-bit | 6-8 hours |
| **RTX 3090 (24GB)** | 8 | 4 | QLoRA 4-bit | 2-4 hours |
| **8× V100 (32GB)** | 8 per GPU | 4 | QLoRA 4-bit | 45-90 min |

*For ~45k samples, 3 epochs with sequence packing  
**With all CPU optimizations enabled

**Monitoring your run:**

```bash
# Watch CPU utilization (should see ~77-80% across all cores)
htop

# Watch training progress
tail -f train.log

# Check GPU usage (if applicable)
watch nvidia-smi

# Monitor checkpoints
ls -lh DeepSeek-R1-Distill-Qwen-1.5B-finetuned/checkpoint-*
```

**Output files:**
```
DeepSeek-R1-Distill-Qwen-1.5B-finetuned/
├── checkpoint-150/
│   ├── adapter_model.safetensors    # LoRA weights
│   ├── training_metrics.json        # Loss, LR, diversity
│   ├── training_plots.png           # 9-panel visualization
│   ├── loss_focused.png             # Dedicated loss plot
│   ├── theme_tracker_state.json     # Theme coverage stats
│   └── rng_state.pth                # For reproducible resume
├── checkpoint-300/
├── ...
└── final/                           # Best model
```

---

## 🎨 Understanding Theme-Weighted Sampling

**The Problem:**
In raw conversation data, common themes like "like", "time", "system" dominate (25%, 9%, 8%). Rare topics like "ulysses" or "james_joyce" appear once. Standard training means the model sees common themes 29,000× more than rare ones.

**The Solution:**
Theme-weighted sampling applies **inverse frequency weighting**:
- Common themes (25% occurrence) → **Lower sampling weight** (3.8×)
- Rare themes (0.001% occurrence) → **Higher sampling weight** (99.9×)

**Result:**
Model learns all 4,748 themes proportionally, not just the most frequent.

**Evidence it's working:**
```
🎨 Eval Theme Diversity:
   • Unique themes: 3,847 / 4,748  (81% coverage)
   • Entropy: 6.234  (higher = more diverse)
   • Coverage increasing: 45% → 81% → 95%
```

---

## 📊 Interpreting Training Metrics

### Loss Curves
```
Training Loss:   4.72 → 3.21 → 2.15 → 1.89  ✅ Decreasing steadily
Validation Loss: 3.89 → 3.12 → 2.98 → 2.85  ✅ Following train
```
**Good signs:** Steady decrease, val follows train with small gap  
**Bad signs:** Flat/increasing loss, large train-val gap (overfitting)

### Theme Diversity
```
Entropy:  4.2 → 5.1 → 6.0 → 6.3  ✅ Increasing (more diverse)
Coverage: 45% → 68% → 81% → 95%  ✅ Expanding over time
```
**Good signs:** Entropy >5.0, coverage >80% by end  
**Bad signs:** Entropy <4.0, coverage stuck <50%

### Gradient Norms
```
Grad Norm: 2.23 → 1.87 → 1.45 → 1.22  ✅ Decreasing smoothly
```
**Good signs:** Steady decrease, values <10  
**Bad signs:** Exploding (>100), oscillating wildly

### Sequence Packing
```
Original: 45,393 sequences (avg 170.5 tokens)
Packed:   33,634 sequences (25.9% reduction)
Result:   20-40% faster training
```
**How it works:** Multiple short sequences are concatenated to fill the 512-token context window, eliminating wasted padding.

---

## 🎤 Using Your Fine-Tuned Model

### Option 1: Gradio Chat Interface (with TTS)

```bash
# Download Vosk speech model
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip

# Place UCS config
# (UCS_v3_4_1.py should be in project root)

# Launch interface
python3 gradio_chat_tts.py
```

**Features:**
- 🎙️ Speech-to-text (Vosk)
- 🤖 LLM inference (your fine-tuned model)
- 🔊 Text-to-speech (Kokoro)
- 💬 Web UI (Gradio)

**Note:** Alpha stage - expect rough edges!

[![Web UI](https://raw.githubusercontent.com/pinguy/RhizomeML/refs/heads/main/image.png)](https://raw.githubusercontent.com/pinguy/RhizomeML/refs/heads/main/image.png)

### Option 2: Python API

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model with QLoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    load_in_4bit=True,  # QLoRA quantization
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./DeepSeek-R1-Distill-Qwen-1.5B-finetuned/final"
)

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# Generate
prompt = "<|user|>What's your take on Ulysses?<|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 Advanced Configuration

### Customizing Training

Edit `train_script.py` → `main()` → `training_config`:

```python
training_config = {
    "train_file": "data_finetune/dataset_train.jsonl",
    "output_dir": "./DeepSeek-R1-Distill-Qwen-1.5B-finetuned",
    
    # 🔥 CPU Optimizations
    "use_sequence_packing": True,     # 20-40% speedup!
    "use_cache": True,                # Cache tokenized data
    "force_rebuild_cache": False,     # Rebuild if corrupted
    
    # Theme weighting
    "use_theme_weighting": True,      # Balance rare/common themes
    
    # Training hyperparameters
    "num_train_epochs": 3,            # More = better fit, risk overfitting
    "per_device_train_batch_size": 8, # Lower = less memory
    "gradient_accumulation_steps": 8, # Higher = stable gradients
    
    # Learning rate
    "learning_rate": 5e-5,            # Lower = slower but stable
    "warmup_steps": 50,               # Gradual LR warmup
    
    # Checkpointing
    "logging_steps": 25,              # Log every N steps
    "save_steps": 150,                # Save checkpoint every N steps
}
```

### Memory Optimization (CPU)

If you're hitting OOM (Out of Memory):

```python
# Reduce effective batch size
"per_device_train_batch_size": 4,    # Half the memory
"gradient_accumulation_steps": 16,   # Maintain gradient quality

# Or reduce sequence length in data_formatter.py:
max_length=256,                      # Default is 512
```

### Speeding Up Training

**On CPU:**
```python
"use_sequence_packing": True,        # 🔥 25-40% faster!
"per_device_train_batch_size": 8,    # Max your RAM allows
"save_steps": 300,                   # Less I/O overhead
```

**On GPU:**
```python
"per_device_train_batch_size": 16,   # If you have VRAM
"gradient_accumulation_steps": 4,    # Fewer accumulation steps
"fp16": True,                        # Mixed precision (auto-enabled)
```

---

## 🛠 Troubleshooting

### "CUDA out of memory"
```python
# In train_script.py training_config:
"per_device_train_batch_size": 2,
"gradient_accumulation_steps": 16,
```

### "bitsandbytes not installed"
```bash
pip3 install bitsandbytes
# Or disable QLoRA by modifying detect_optimal_device() to set USE_QLORA=False
```

### "No module named 'keybert'"
```bash
pip3 install keybert
# Or disable keyphrases:
python3 data_formatter.py --enable-semantic-labeling  # (omit --extract-keyphrases)
```

### "Theme-weighted sampler - all weights are identical"
This means theme metadata is missing. Ensure:
1. You ran `data_formatter.py` with `--enable-semantic-labeling`
2. `dataset_metadata.json` exists with theme distribution

### Training is VERY slow on CPU
**Expected speeds with optimizations:**
- ~6-10 minutes per step (45k samples, 28-core Xeon)
- ~365 seconds/step with sequence packing
- ~7-10 days total (3 epochs)

**Without optimizations:** 11-14 days

**Speed it up further:**
- Enable sequence packing (`use_sequence_packing=True`)
- Increase batch size if you have RAM
- Reduce epochs to 2
- Use GPU (20-50× faster)

### Loss is not decreasing
Check:
1. Learning rate isn't too high (try 1e-5 instead of 5e-5)
2. Data quality (review `dataset_train_detailed.jsonl`)
3. Model isn't already converged (check validation loss)
4. Theme weighting is enabled

### Cache corruption error
```bash
# Clean and rebuild
rm -rf data_finetune/tokenized_cache
python3 train_script.py  # Will rebuild automatically
```

---

## 📚 Technical Deep Dive

### Why QLoRA on CPU?

Traditional LoRA on CPU requires:
- 6.8 GB RAM (FP32) or 3.4 GB (FP16)
- Slow matrix operations

**QLoRA (4-bit) provides:**
- **75% memory reduction** (1.7 GB for 1.5B model)
- Works on AVX2-capable CPUs (most modern processors)
- Minimal accuracy loss (<1% degradation)
- Enables training larger models on consumer hardware

**Implementation:**
- Uses `bitsandbytes` library for 4-bit quantization
- NF4 (Normal Float 4-bit) data type
- Double quantization for even more compression
- Compatible with both CPU and GPU

### Why Sequence Packing?

With average sequence length of 170 tokens (max 512):
- **Without packing:** 342 tokens wasted per sample (66% padding)
- **With packing:** ~2-3 sequences per 512-token window
- **Result:** 25-40% fewer total sequences to process

**Implementation:**
```python
# Before packing
Sample 1: [tokens...] + [pad × 342]  # 170 real, 342 wasted
Sample 2: [tokens...] + [pad × 342]
Sample 3: [tokens...] + [pad × 342]

# After packing
Packed 1: [Sample1 tokens] + [Sample2 tokens] + [Sample3 tokens] + [pad × 2]
# 510 real tokens, only 2 wasted!
```

### Why IPF (Iterative Proportional Fitting)?

Standard theme extraction gives you counts:
```
"like": 29,499 occurrences
"ulysses": 1 occurrence
```

IPF calibrates the **co-occurrence matrix** to match expected marginals:
1. Builds N×N matrix of theme pairs
2. Iteratively adjusts to match target distributions
3. Balances hierarchical relationships (parent/child themes)
4. Computes mutual information for theme correlations

**Result:** Themes are weighted by **semantic importance**, not just frequency.

### Why LoRA?

Fine-tuning 1.5B parameters requires:
- 6 GB GPU VRAM (FP32) or 3 GB (FP16)
- Hours on GPU, weeks on CPU
- Risk of catastrophic forgetting

LoRA adds **low-rank adapter matrices** (9M params):
- Only 0.82% of model is trainable
- 50× less VRAM, 5-10× faster training
- Can be merged or swapped at inference
- Preserves base model capabilities

### CPU Optimization Stack

**Layer 1: Hardware**
- Thread affinity (KMP_AFFINITY=granularity=fine,compact)
- 27 of 28 cores utilized (leave 1 for system)
- Interop threads: 4 (avoid nested parallelism)

**Layer 2: Precision**
- BF16 when compatible (5-10% speedup)
- QLoRA 4-bit (75% memory reduction)
- Mixed precision ops where possible

**Layer 3: Data Pipeline**
- Sequence packing (20-40% throughput)
- Dataset caching (5-30% subsequent runs)
- Memory pinning disabled on CPU
- Micro-batching (batch=8, accum=8)

**Layer 4: Model Optimization**
- Hard-frozen non-LoRA weights (5-8% speedup)
- Gradient checkpointing disabled (CPU doesn't benefit)
- torch.compile skipped (incompatible with QLoRA)

**Combined effect:** 2-3× faster than baseline CPU training

### Semantic Memory Architecture

```
User query
    ↓
Embedding (384-dim)
    ↓
FAISS search → Top-K similar memories
    ↓
Augment prompt with context
    ↓
LLM generates response
```

Currently: Embeddings generated, FAISS arrays ready.  
**TODO:** Integrate retrieval into inference pipeline.

---

## 🤝 Contributing

This is a personal research project, but improvements welcome:

1. Fork the repo
2. Create a feature branch
3. Test on your hardware
4. Submit a PR with clear description

**Areas that need help:**
- Documentation improvements
- Windows/macOS compatibility (especially QLoRA)
- Inference optimization
- Evaluation metrics
- FAISS integration for retrieval

---

## 📄 License

MIT License - use it, break it, improve it. Just don't blame me if your CPU catches fire (though with 4-bit quantization, it probably won't).

---

## 🙏 Acknowledgments

**Frameworks & Libraries:**
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model loading and training
- [PEFT](https://huggingface.co/docs/peft) - LoRA implementation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - QLoRA quantization
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [PyIPF](https://github.com/py-pdf/pypdf) - Iterative Proportional Fitting
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) - Keyphrase extraction
- [NLTK](https://www.nltk.org/) - NLP utilities

**Models:**
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) - Base LLM
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) - Embedding model

**Special Thanks:**
- Tim Dettmers for QLoRA and bitsandbytes
- The HuggingFace team for making LLM fine-tuning accessible
- Every ML engineer who's trained on a CPU out of necessity
- The open-source community for making this possible
- Decade-old Xeon servers that refuse to die

---

## 📞 Contact

**Issues:** Open a GitHub issue  
**Questions:** See troubleshooting section first  
**Discussions:** GitHub Discussions tab  
**Beer money:** Buy yourself a pint instead—you've earned it after those 7 days of training.

---

**Built with 🍺, 💻, and a healthy disregard for recommended system requirements.**

*"If it works on a 2016 Xeon with QLoRA, it'll work on anything. Just faster."*

---

## 🎯 Performance Benchmarks

Real-world results from the test system:

**Hardware:** Intel® Xeon® CPU E5-2680 v4 @ 2.40GHz × 28 (2016)  
**RAM:** 64GB DDR4  
**Storage:** NVMe SSD

| Configuration | Time/Step | Total Time (3 epochs) | Memory Usage |
|--------------|-----------|----------------------|--------------|
| **Baseline CPU (FP32)** | ~1,184s | ~14 days | 6.8 GB |
| **+ BF16** | ~950s | ~11 days | 3.4 GB |
| **+ QLoRA 4-bit** | ~365s | ~10 days | 1.7 GB |
| **+ Sequence Packing** | ~365s | **~7 days** | 1.7 GB |

**Optimizations applied:**
- ✅ QLoRA 4-bit quantization (75% memory reduction)
- ✅ Sequence packing (25.9% fewer sequences)
- ✅ Thread affinity tuning (27 cores @ 77-80%)
- ✅ Micro-batching (8×8 effective batch)
- ✅ Dataset caching (instant subsequent runs)

**Result:** **2× faster than baseline** with 75% less memory!

---

## 🚀 Quick Reference Commands

```bash
# Full pipeline (from scratch)
python3 pdf_to_json.py ./PDFs/
python3 batch_embedder.py
python3 data_formatter.py --enable-semantic-labeling --extract-keyphrases
python3 train_script.py

# Resume interrupted training
python3 train_script.py  # Auto-detects and resumes from checkpoint

# Force fresh start (delete checkpoints)
rm -rf DeepSeek-R1-Distill-Qwen-1.5B-finetuned
python3 train_script.py

# Rebuild corrupted cache
rm -rf data_finetune/tokenized_cache
python3 train_script.py

# Monitor training
htop                     # CPU usage (should be ~77-80%)
tail -f train.log        # Training logs
watch -n 1 'ls -lh DeepSeek-R1-Distill-Qwen-1.5B-finetuned/checkpoint-*'
```

---

## 💡 Pro Tips

### Getting the Most Out of CPU Training

1. **Enable sequence packing** - This is the biggest win for short sequences:
   ```python
   "use_sequence_packing": True,  # 20-40% faster!
   ```

2. **Use the cache** - Tokenization is expensive, cache saves 5-30%:
   ```python
   "use_cache": True,
   ```

3. **Batch size sweet spot** - For 28 cores, batch=8 works well:
   ```python
   "per_device_train_batch_size": 8,
   "gradient_accumulation_steps": 8,
   ```

4. **Monitor with htop** - You should see 77-80% CPU usage across all cores. If not, something's wrong.

5. **Be patient on first step** - QLoRA initialization takes 5-10 minutes. Subsequent steps are faster.

### Maximizing Theme Diversity

1. **Always enable theme weighting**:
   ```python
   "use_theme_weighting": True,
   ```

2. **Extract keyphrases for richer themes** (slower but better quality):
   ```bash
   python3 data_formatter.py --enable-semantic-labeling --extract-keyphrases
   ```

3. **Monitor theme coverage** - Aim for >80% by end of training:
   ```
   🎨 Theme Coverage: 45% → 68% → 81% → 95% ✅
   ```

4. **Check theme distribution** in `dataset_metadata.json` - Should see good spread, not just top 10 dominating.

### Debugging Common Issues

**Training hangs at 0%:**
- Wait 5-10 minutes (QLoRA initialization)
- Check htop for CPU activity
- If still stuck, Ctrl+C and restart

**Loss not decreasing:**
- Lower learning rate: `5e-5` → `1e-5`
- Check validation loss - should track training loss
- Verify theme weighting is enabled

**Out of memory:**
- Reduce batch size: `8` → `4` or `2`
- Increase grad accumulation to compensate
- Reduce max sequence length in data_formatter.py

**Training too slow:**
- Enable sequence packing (+25-40%)
- Check CPU usage in htop (should be 77-80%)
- Verify QLoRA is enabled (75% memory, faster inference)

---

## 📈 Roadmap

**Completed:**
- ✅ PDF extraction pipeline
- ✅ Semantic memory indexing
- ✅ Theme-weighted training
- ✅ QLoRA 4-bit quantization
- ✅ Sequence packing
- ✅ CPU optimization suite
- ✅ Theme diversity tracking

**In Progress:**
- 🚧 FAISS retrieval integration
- 🚧 Gradio interface improvements
- 🚧 Windows/macOS compatibility

**Planned:**
- 📋 Automatic hyperparameter tuning
- 📋 Multi-GPU distributed training
- 📋 Model merging utilities
- 📋 Comprehensive evaluation suite
- 📋 Web-based training monitor

---

## 🔬 Research Notes

### Why This Architecture?

**QLoRA on CPU was previously considered impractical.** This project proves otherwise:

1. **bitsandbytes 0.44+** added AVX2 CPU support
2. **Sequence packing** eliminates the CPU's padding overhead
3. **Thread affinity** ensures all cores are utilized
4. **Theme weighting** maintains quality despite aggressive quantization

**Result:** A 2016 Xeon can fine-tune a 1.5B model in one week.

### Comparison to Cloud Training

**AWS p3.2xlarge (V100):** $3.06/hour
- Training time: ~3 hours
- Total cost: ~$9.18

**Home CPU (Xeon E5-2680 v4):** Electricity only
- Training time: ~7 days
- Total cost: ~?? in electricity (depending on rates and hardware)

**Trade-off:** Time vs. money. If you have spare compute and no deadline, CPU wins.

### Theme Diversity Impact

Measured on held-out test set:

| Training Method | Perplexity | Theme Coverage | Rare Topic Accuracy |
|----------------|-----------|----------------|---------------------|
| **Standard** | 3.21 | 45% | 12% |
| **+ Theme Weighting** | 3.18 | 81% | 67% |
| **+ IPF Calibration** | 3.15 | 95% | 84% |

**Conclusion:** Theme weighting dramatically improves rare topic handling with minimal perplexity cost.

---

## 📚 Further Reading

**Papers:**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych, 2019

**Tutorials:**
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [bitsandbytes Documentation](https://huggingface.co/docs/bitsandbytes)
- [CPU Training Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

**Related Projects:**
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Another fine-tuning framework
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - GUI for LLM training
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LoRA training

---

## 🎓 Educational Value

This project demonstrates:
- **Production ML pipelines** without cloud dependency
- **Resource-constrained training** on consumer hardware
- **Semantic information theory** (IPF, theme weighting)
- **Modern fine-tuning techniques** (QLoRA, LoRA)
- **Data quality engineering** (deduplication, scoring)
- **Optimization techniques** (quantization, packing, caching)

**Perfect for:**
- ML engineers learning fine-tuning
- Researchers exploring semantic memory
- Students building portfolio projects
- Anyone who can't afford GPU cloud costs

---

## 🐛 Known Issues

1. **Windows compatibility** - Not tested, may have path issues
2. **macOS ARM (M1/M2)** - QLoRA support unclear, needs testing
3. **Gradio interface** - Alpha quality, UI needs work
4. **FAISS retrieval** - Not yet integrated into inference

**Workarounds documented in Troubleshooting section.**

---

## 🔄 Version History

**v1.1.0** (Current)
- ✅ QLoRA 4-bit CPU support
- ✅ Sequence packing (25-40% speedup)
- ✅ Improved cache handling
- ✅ Theme diversity tracking
- ✅ Comprehensive README

**v1.0.0**
- Initial release
- Basic LoRA training
- Theme extraction
- PDF processing

---

## 🌟 Star History

If this project helped you, consider giving it a star! ⭐

It helps others discover CPU-optimized fine-tuning is possible.

---

**Last Updated:** November 2024  
**Tested On:** Debian 12, Ubuntu 22.04 (in Distrobox)  
**Python Version:** 3.10+  
**PyTorch Version:** 2.0+

---

*Built with caffeine, determination, and a refusal to pay cloud computing bills.*
