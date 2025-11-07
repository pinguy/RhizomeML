# 🌱 RhizomeML

### *Production-Grade Fine-Tuning Pipeline for Context-Aware Conversational AI*

> A complete, offline-first pipeline for fine-tuning language models with semantic memory integration. Transforms raw conversations, PDFs, and documents into high-quality training data with IPF-calibrated theme weighting, FAISS-backed retrieval, and LoRA adaptation.

**Built for the real world:** Runs from Xeons to GPU clusters. No cloud required.

---

## 🎯 What Does It Do?

RhizomeML takes your messy conversation logs and PDFs, then:

1. **Cleans & deduplicates** with embedding-based semantic filtering
2. **Extracts themes** using TF-IDF, KeyBERT, and IPF (Iterative Proportional Fitting)
3. **Weights training samples** to prevent common themes from dominating
4. **Fine-tunes models** with LoRA for memory-efficient updates
5. **Tracks semantic diversity** during training to ensure balanced learning

**Result:** A model that understands *your* conversations, *your* documents, and *your* domain — with measurably diverse knowledge coverage.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| **OS** | Linux (tested in Distrobox) | Debian/Ubuntu-based |
| **CPU** | 8 cores | 14+ cores (Xeon/Ryzen) |
| **RAM** | 24 GB | 32+ GB |
| **Storage** | 50 GB free | 100+ GB (NVMe preferred) |
| **GPU** | None (CPU works!) | NVIDIA (Compute ≥6.0) |

### Installation

```bash
# Option 1: Inside Distrobox (recommended for isolation)
distrobox create --name rhizome-dev --image rhizome-devbox
distrobox enter rhizome-dev

# Option 2: Native Linux
# (Just run the commands below in your terminal)

# Clone and setup
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

# Install dependencies
pip3 install -r requirements.txt

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
│   ├── data_formatter.py            # Clean, dedupe, label, create datasets
│   └── train_script.py             # Fine-tune with theme tracking
│   ├── memory_texts.npy               # Embedded text vectors
│   ├── memory_metadata.pkl            # Metadata for retrieval
│   ├── semantic_memory.pkl            # Learned theme weights
│   ├── data_finetune/                 # Training datasets
│   │   ├── dataset_train.jsonl
│   │   ├── dataset_validation.jsonl
│   │   ├── dataset_test.jsonl
│   │   └── dataset_metadata.json      # Theme distribution stats
│   └── DeepSeek-R1-Distill-Qwen-1.5B-finetuned_*/  # Model checkpoints
│   ├── gradio_chat_tts.py             # STT → LLM → TTS interface
│   ├── UCS_v3_4_1.py                  # UCS config
    ├── README.md
    └── requirements.txt
```

---

## 🔄 Complete Pipeline Walkthrough

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

* **Keyphrase extraction** improves semantic richness but increases runtime; enable only for smaller datasets.

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
- **4,744 unique themes** (Example dataset)
- Theme frequency distribution
- Source breakdown (conversation vs PDF)
- Quality score statistics

<details>
<summary>📊 Example Metadata Output</summary>

```json
{
  "total_pairs": 56664,
  "splits": {
    "train": 45331,
    "validation": 5667,
    "test": 5666
  },
  "theme_distribution": {
    "like": 28390,
    "time": 12037,
    "system": 7056,
    "model": 6295,
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
1. **Auto-detects hardware:** CPU or GPU (no changes needed)
2. **Loads model:** DeepSeek-R1-Distill-Qwen-1.5B by default (But most HF models can be used as a drop in replacment).
3. **Applies LoRA:** Efficient fine-tuning (9M trainable / 1.7B total params)
4. **Enables theme weighting:** Rare themes get more training samples
5. **Tracks diversity:** Monitors theme coverage during training
6. **Saves checkpoints:** Every 100 steps with resumable state
7. **Generates plots:** Loss curves, learning rate, theme diversity

**Expected output:**
```
🤖 DeepSeek-R1-Distill-Qwen-1.5B Fine-Tuning Suite
   🎨 Now with Semantic Theme-Aware Training!

🔧 Model Setup
✅ Model loaded and LoRA applied successfully on CPU
📊 Parameters: 9,232,384 trainable / 1,786,320,384 total (0.52%)

📚 Data Processing
✅ Dataset tokenization complete
🎨 ThemeTracker initialized with 4744 unique themes
🔝 Top 5 most common themes:
   • like: 28390 (24.3%)
   • time: 12037 (10.3%)
   • system: 7056 (6.0%)

⚙️ Training Configuration
🎯 Number of training epochs: 3
📦 Effective batch size: 2 × 32 = 64
🚀 Training on: CPU: 28 cores (using 27 threads)
🎨 Theme-weighted sampling: ENABLED
```

**Hardware-specific behavior:** CPU Tested on Intel® Xeon(R) CPU E5-2680 v4 @ 2.40GHz × 28

| Hardware | Batch Size | Grad Accum | FP16 | Expected Time* |
|----------|-----------|------------|------|----------------|
| **CPU Only** | 2 | 32 | No | 11-14 days |
| **RTX 3060** | 4 | 8 | Yes | 8-12 hours |
| **RTX 3090** | 8 | 4 | Yes | 3-5 hours |
| **8× V100** | 8 per GPU | 4 | Yes | 1-2 hours |

*For ~45k samples, 3 epochs

**Monitoring your run:**

```bash
# Watch training progress
tail -f train.log

# Check GPU usage (if applicable)
watch nvidia-smi

# Monitor checkpoints
ls -lh DeepSeek-R1-Distill-Qwen-1.5B-finetuned_*/checkpoint-*
```

**Output files:**
```
DeepSeek-R1-Distill-Qwen-1.5B-finetuned_20251106_061947/
├── checkpoint-100/
│   ├── adapter_model.safetensors    # LoRA weights
│   ├── training_metrics.json        # Loss, LR, diversity
│   ├── training_plots.png           # 6-panel visualization
│   ├── loss_focused.png             # Dedicated loss plot
│   ├── theme_tracker_state.json     # Theme coverage stats
│   └── rng_state.pth                # For reproducible resume
├── checkpoint-200/
├── ...
└── final/                           # Best model
```

---

## 🎨 Understanding Theme-Weighted Sampling

**The Problem:**
In raw conversation data, common themes like "like", "time", "system" dominate (24%, 10%, 6% in the data). Rare topics like "ulysses" or "james_joyce" appear once. Standard training means the model sees common themes 28,000× more than rare ones.

**The Solution:**
Theme-weighted sampling applies **inverse frequency weighting**:
- Common themes (24% occurrence) → **Lower sampling weight**
- Rare themes (0.001% occurrence) → **Higher sampling weight**

**Result:**
Model learns all 4,744 themes proportionally, not just the most frequent.

**Evidence it's working:**
```
🎨 Eval Theme Diversity:
   • Unique themes: 3,847 / 4,744  (81% coverage)
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

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./DeepSeek-R1-Distill-Qwen-1.5B-finetuned_20251106_061947/final"
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Generate
prompt = "<|user|>What's your take on Ulysses?<|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

---

## 🔧 Advanced Configuration

### Customizing Training

Edit `train_script.py` → `main()`:

```python
result = trainer.train(
    train_file="data_finetune/dataset_train.jsonl",
    val_file="data_finetune/dataset_validation.jsonl",
    
    # Enable theme weighting
    use_theme_weighting=True,
    
    # Training hyperparameters
    num_train_epochs=3,              # More epochs = better fit, risk overfitting
    per_device_train_batch_size=2,   # Lower = less memory, slower
    gradient_accumulation_steps=32,  # Higher = stable gradients, slower
    
    # Learning rate
    learning_rate=5e-5,              # Lower = slower but more stable
    warmup_steps=25,                 # Gradual LR warmup
    
    # Checkpointing
    logging_steps=25,                # Log every N steps
    save_steps=100,                  # Save checkpoint every N steps
    eval_steps=100,                  # Evaluate every N steps
)
```

### Memory Optimization (CPU)

If you're hitting OOM (Out of Memory):

```python
# Reduce effective batch size
per_device_train_batch_size=1,      # Half the memory
gradient_accumulation_steps=64,     # Maintain gradient quality

# Or reduce sequence length
# In data_formatter.py:
max_length=256,                     # Default is 512
```

### Speeding Up Training (GPU)

```python
# Larger batches
per_device_train_batch_size=8,      # If you have VRAM
gradient_accumulation_steps=4,      # Fewer accumulation steps

# Mixed precision (automatic on GPU)
fp16=True,                          # Already enabled by default
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size in train_script.py
per_device_train_batch_size=1
gradient_accumulation_steps=64
```

### "No module named 'keybert'"
```bash
pip3 install keybert
# Or disable keyphrases:
python3 data_formatter.py --enable-semantic-labeling  # (omit --extract-keyphrases)
```

### "Theme-weighted sampler created but all weights are identical"
This means theme metadata is missing from your training data. Ensure:
1. You ran `data_formatter.py` with `--enable-semantic-labeling`
2. `dataset_metadata.json` exists with theme distribution

### Training is VERY slow
**CPU training is slow by design.** Expected speeds:
- ~5-10 minutes per step (45k samples, CPU)
- ~368 seconds/step = ~11-14 days total (3 epochs)

**Speed it up:**
- Use GPU (20-50× faster)
- Reduce epochs to 2
- Increase `logging_steps` and `save_steps` to reduce I/O overhead

### Loss is not decreasing
Check:
1. Learning rate isn't too high (try 1e-5 instead of 5e-5)
2. Data quality (review `dataset_train_detailed.jsonl`)
3. Model isn't already converged (check validation loss)

---

## 📚 Technical Deep Dive

### Why IPF (Iterative Proportional Fitting)?

Standard theme extraction gives you counts:
```
"like": 28,390 occurrences
"ulysses": 1 occurrence
```

IPF calibrates the **co-occurrence matrix** to match expected marginals:
1. Builds N×N matrix of theme pairs
2. Iteratively adjusts to match target distributions
3. Balances hierarchical relationships (parent/child themes)
4. Computes mutual information for theme correlations

**Result:** Themes are weighted by **semantic importance**, not just frequency.

### Why LoRA?

Fine-tuning 1.7B parameters requires:
- 6.8 GB GPU VRAM (FP32) or 3.4 GB (FP16)
- Hours on GPU, weeks on CPU
- Risk of catastrophic forgetting

LoRA adds **low-rank adapter matrices** (9M params):
- Only 0.52% of model is trainable
- 50× less VRAM, 5-10× faster training
- Can be merged or swapped at inference

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

## 🤝 Contributing

This is a personal research project, but improvements welcome:

1. Fork the repo
2. Create a feature branch
3. Test on your hardware
4. Submit a PR with clear description

**Areas that need help:**
- Documentation improvements
- Windows/macOS compatibility
- Inference optimization
- Evaluation metrics

---

## 📄 License

MIT License - use it, break it, improve it. Just don't blame me if your CPU catches fire.

---

## 🙏 Acknowledgments

**Frameworks & Libraries:**
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model loading and training
- [PEFT](https://huggingface.co/docs/peft) - LoRA implementation
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [PyIPF](https://github.com/py-pdf/pypdf) - Iterative Proportional Fitting
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) - Keyphrase extraction
- [NLTK](https://www.nltk.org/) - NLP utilities

**Models:**
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) - Base LLM
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) - Embedding model

**Inspiration:**
- Every ML engineer who's trained on a CPU out of necessity
- The open-source community for making this possible
- Decade-old Xeon servers that refuse to die

---

## 📞 Contact

**Issues:** Open a GitHub issue  
**Questions:** See troubleshooting section first  
**Beer money:** Buy yourself a pint instead—you've earned it after waiting 11 days for training to finish.

---

**Built with 🍺, 💻, and a healthy disregard for recommended system requirements.**

*"If it works on a 2016 Xeon, it'll work on a V100 cluster. Just faster."*
