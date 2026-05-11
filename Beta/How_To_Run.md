# **RhizomeML – Setup & Workflow (Ubuntu 22.04)**

### **NVIDIA Driver Setup**

```bash
# Update to the latest driver
sudo apt update
sudo ubuntu-drivers autoinstall
# OR specifically:
sudo apt install --fix-missing nvidia-driver-580

# Reboot required
sudo reboot
```

**Note:** Verified to work with the `5.11.16_lowlatency` kernel on older systems. Use newer kernels when available for better performance and stability.

---

### **Running on Non-Ubuntu Systems with Distrobox**

On non-Ubuntu hosts, Distrobox can launch an isolated Ubuntu 22.04 container with full GPU passthrough. Running Nativity directly on other distributions is fine as long as you’re using Python 3.12 — Ubuntu is simply the known-good baseline.

```bash
# Clone and install Distrobox
git clone https://github.com/89luca89/distrobox.git
cd distrobox
sudo ./install --prefix /usr/local
distrobox version

# Install Podman using your distro’s package manager
sudo apt install podman

# Add NVIDIA container repository and key (DEB-based systems only)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
# Or use your distro’s package manager (e.g., Arch: sudo pacman -S nvidia-container-toolkit)
sudo apt install nvidia-container-toolkit

# Generate CDI configuration for Podman
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Reset Podman (if a previous setup failed)
podman stop --all
podman rm --all --force
podman rmi --all --force
rm -rf ~/.local/share/containers ~/.config/containers

# Prepare a temp directory for large image builds
mkdir -p ~/.podman-tmp

# Build the image (ensure Dockerfile.rhizome is in the current directory)
TMPDIR=$HOME/.podman-tmp podman build -t rhizome-img -f Dockerfile.rhizome

# If 'unexpected EOF' appears, rerun until it completes successfully
podman pull ubuntu:22.04

# Create a Distrobox container with NVIDIA passthrough
distrobox create --name rhizome-dev --image rhizome-img --nvidia

# Enter the container
distrobox enter rhizome-dev
# Note: If it hangs on first entry, open another terminal and rerun the same command.
# It may take a few retries to initialize properly. Once set up, it’s stable.

# Stop the container when finished
distrobox stop rhizome-dev
```

---

### **Clone the Repository**

```bash
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

pip install -r requirements.txt --upgrade
```

---

---

## **Building the Dataset (New Automated Pipeline)**

This is the recommended way to build the training dataset from scratch.
Drop your files in, run two scripts, train.

### **What goes where**

| What | Where |
|------|-------|
| Conversation exports (ChatGPT / Claude) | `conversations.json` and/or `conversations2.json` in the repo root |
| Books / PDFs | `PDFs/` folder |

Conversation exports can be downloaded from:
- **ChatGPT** → Settings → Data Controls → Export Data → `conversations.json`
- **Claude** → Settings → Account → Export Data → `conversations.json`

### **Step 1 — Build the dataset**

```bash
python build_dataset.py
```

This runs four stages automatically:

1. **`batch_embedder.py`** — embeds all conversations into a FAISS vector index (`memory_texts.npy`, `memory.index`, `memory_metadata.pkl`).
2. **`data_formatter.py`** — generates Q&A pairs from the conversation embeddings → `data_finetune/conv_{train,validation,test}.jsonl.gz`
3. **`PDFs/adaptive_semantic.py`** (once per PDF) — extracts text, chunks it, generates Q&A pairs per book → `data_finetune/pdf_{bookname}_qa_{split}.jsonl.gz`. Already-processed books are skipped automatically.
4. **Merge** — combines all `conv_*` and `pdf_*_qa_*` files into the final `data_finetune/dataset_{train,validation,test}.jsonl.gz`.

At the end a summary prints how many records came from each source and split.

### **Step 2 — Train**

```bash
python train_script.py
```

---

### **Re-running / incremental updates**

The script skips work that's already been done. Add new PDFs and re-run — only the new books get processed, then the merge runs again.

```bash
# Add a new PDF to PDFs/ then:
python build_dataset.py
```

Force everything to re-run from scratch:

```bash
python build_dataset.py --force
```

Process only one specific PDF (useful for testing):

```bash
python build_dataset.py --skip-embed --skip-conv --pdf "My Book.pdf"
```

Just re-merge existing outputs (e.g. after manually deleting a bad PDF's output):

```bash
python build_dataset.py --skip-embed --skip-conv --skip-pdfs
```

### **All flags**

| Flag | Effect |
|------|--------|
| `--force` | Re-run all steps, ignore cached outputs |
| `--skip-embed` | Skip `batch_embedder.py` |
| `--skip-conv` | Skip `data_formatter.py` |
| `--skip-pdfs` | Skip all PDF processing |
| `--skip-merge` | Skip the final merge step |
| `--pdf NAME` | Process only this one PDF file (basename, e.g. `"My Book.pdf"`) |
| `--pdf-workers N` | Thread count passed to `adaptive_semantic.py` |
| `--no-gzip` | Write plain `.jsonl` instead of `.jsonl.gz` |

### **Important: memory file conflict**

`adaptive_semantic.py` overwrites `memory_texts.npy` and `memory.index` in the repo root when it processes each PDF. Because of this, `build_dataset.py` always re-runs `batch_embedder.py` before `data_formatter.py`, even if the memory files exist, so the conversation data is never replaced by PDF data when formatting.

**Do not use `--skip-embed` if you have run PDFs since the last embed** — `data_formatter.py` will read the last PDF's data instead of your conversations and produce a tiny dataset.

### **PDFs with unreadable text**

Some scanned PDFs use corrupt font encoding (`(cid:XX)` characters). These produce 0 valid chunks and are skipped with an error. There is no fix without OCR. You can check a PDF manually:

```python
from pdfminer.high_level import extract_text
t = extract_text("PDFs/mybook.pdf")
print(t[:500])
```

If the output is mostly `(cid:XX)`, the PDF cannot be processed.

### **Typical dataset sizes**

A run with ~3,000 conversations and 25 books produces roughly:

| Split | Records |
|-------|---------|
| train | ~115,000 |
| validation | ~14,500 |
| test | ~14,500 |
| **Total** | **~144,000** |

Conversations make up ~65%, PDFs ~35%.

---

## **Data Preparation (Legacy — replaced by build_dataset.py above)**

```bash
python pdf_to_json.py      # Old PDF → JSON step, no longer needed
python batch_embedder.py   # Now called automatically by build_dataset.py
python data_formatter.py   # Now called automatically by build_dataset.py
```

---

## **Training**

```bash
python train_script.py
```

If tokenization errors occur, clear the cached tokenized dataset directory.
This usually happens when reusing cached data with a different base model:

```bash
rm -rf data_finetune/tokenized_cache
```

---

## **Model Selection**

Edit `train_script.py`:

```python
model_name = "google/gemma-3-1b-it-qat-int4-unquantized"  # Any Hugging Face CAUSAL_LM model
```

Notes:

* Training requires int4 / NF4 quantization. `q4_0` models are inference-only.
* Some models require a Hugging Face access token.
* Set `YOUR_HF_TOKEN_HERE` in the script to your token if required.

---

## **Gradio Chat + TTS (RAM Heavy)**

```bash
python gradio_chat_tts.py --tts-cpu    # Force CPU - Recommended
python gradio_chat_tts.py --tts-gpu    # Force CUDA 
python gradio_chat_tts.py --tts-auto   # Auto-detect best device
python gradio_chat_tts.py --tts-mps    # Apple Silicon
python gradio_chat_tts.py --model Qwen/Qwen3-4B-Instruct-2507
python gradio_chat_tts.py --model ./RhizomeML-finetuned/checkpoint-6000/
```

---

## **Gradio Chat (Low RAM/VRAM Mode)**

```bash
python gradio_chat_tts.py --quantize
python gradio_chat_tts.py --quant-bits 4          # 4 or 8 (default: 4)
python gradio_chat_tts.py --quant-type nf4        # nf4 or fp4 (default: nf4)
python gradio_chat_tts.py --no-quantize
python gradio_chat_tts.py --enable-stt
python gradio_chat_tts.py --no-stt
python gradio_chat_tts.py --tts-cpu --no-stt --quant-bits 4 --model ./RhizomeML-finetuned/checkpoint-3000/
python gradio_chat_tts.py --tts-cpu --no-stt --quant-bits 4 --model DavidAU/LFM2.5-1.2B-Instruct-Thinking-Claude-High-Reasoning
```

---

### **STT Setup (Vosk)**

Download a Vosk speech model (large model shown below; smaller ones also work):

```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip
```

---

## **Export to GGUF (for llama.cpp)**

### GPU Support for llama.cpp (skip if CPU-only)

Remove the old CUDA toolkit:

```bash
sudo apt remove nvidia-cuda-toolkit
```

Add NVIDIA’s repo for the latest CUDA:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
```

Symlink CUDA (only needed if cmake can’t find `libcuda` during build):

```bash
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```

Verify installation:

```bash
nvcc --version
nvidia-smi
```

---

### Convert to GGUF

```bash
python3.12 -m venv venv_gguf
source venv_gguf/bin/activate
pip install --upgrade pip setuptools wheel
pip install --use-deprecated=legacy-resolver peft

python convert_to_gguf.py              # Auto 4-bit medium quantization
python convert_to_gguf.py --quant f16  # No quantization (options: q2_k–q8_k)
python convert_to_gguf.py --gpu        # CUDA build
python convert_to_gguf.py --cpu        # CPU-only build

deactivate
```

The venv isolates llama.cpp build dependencies. Once compiled, you can safely delete it.

---

### Running the Model

```bash
# GPU (CUDA)
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 -ngl 99 --port 8081

# CPU only
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 --threads 14 --port 8081

# Offload to GPU
./llama.cpp/build/bin/llama-server \
  -m ./gguf_models/*.gguf \
  --n-gpu-layers 40 \
  --ctx-size 4096 \
  --batch-size 512 \
  --flash-attn on \
  --threads 20 \
  --threads-batch 20 \
  --port 8081
```

### Convert to GGUF GUI

```bash
python gguf_gui.py
```

### Koboldcpp

Koboldcpp is recommended once you have the GGUF model:

```bash
curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64
chmod +x koboldcpp
./koboldcpp
```

---

# **OOM Adjustments**

Edit these values in `train_script.py`:

```python
default_batch_size = 2    # Higher = faster training, more memory
default_grad_accum = 8    # Effective batch = batch_size × grad_accum
                          # Higher = slower but same memory
                          # Target effective batch: 16 (e.g., 4×4, 2×8, 1×16)
```

---

## **GPU Memory vs Speed (GTX / Older GPUs)**

On GTX-class or older GPUs, disabling some features can reduce memory use.
If your GPU supports FP16, leave it **enabled** — it provides a major speedup.

```python
# GPU defaults
default_batch_size = 2
default_grad_accum = 8
default_fp16 = False
```

---

# **Theme-Based Early Stopping**

Training stops when:

* The epoch limit is reached (default: 3), **or**
* 82% of semantic themes have been observed.

Best results typically come from the last or second-to-last checkpoint.

To adjust, modify:

```python
metrics['coverage'] >= 0.82
```

---
