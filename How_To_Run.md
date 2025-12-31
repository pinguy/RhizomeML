# **RhizomeML – Setup & Workflow - Ubuntu 22.04**

### NVIDIA Driver Setup
```bash
# Update to latest driver
sudo apt update
sudo ubuntu-drivers autoinstall
# OR specifically:
sudo apt install --fix-missing nvidia-driver-580

# Reboot required
sudo reboot
```

**Note:** Tested and known to work on the `5.11.16_lowlatency` kernel for older distributions. Newer kernels are recommended where available.

### Run it Using Distrobox

```bash
# Clone Distrobox
git clone https://github.com/89luca89/distrobox.git
cd distrobox

# Install
sudo ./install --prefix /usr/local
distrobox version

# Install Podman
sudo apt install podman

# On HOST set-up nvidia-container
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit

# Configure for podman
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# If you had a pass failed install - To start fresh
podman stop --all
podman rm --all --force
podman rmi --all --force
rm -rf ~/.local/share/containers
rm -rf ~/.config/containers

# Build the image. Download the Dockerfile.rhizome file from the repo.
mkdir -p ~/.podman-tmp

TMPDIR=$HOME/.podman-tmp podman build \
  --format docker \
  -t rhizome-img \
  -f Dockerfile.rhizome

# Only needed if you hit unexpected EOF pulling ubuntu:22.04
docker pull ubuntu:22.04
docker save ubuntu:22.04 -o ubuntu-22.04.tar
podman load -i ubuntu-22.04.tar

# Create container with nvidia passthrough
distrobox create --name rhizome-dev --image rhizome-img --nvidia
distrobox enter rhizome-dev # May hang or fail a few times. When it happens open a new Terminal while keeping the hanged one open and run it again. At some point it will go through then will be fine.

distrobox stop rhizome-dev # To try again.

```

---

### **Clone the Repo**

```bash
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

pip3 install -r requirements.txt --upgrade
```

---

## **Data Preparation**

Place your PDFs inside:

```
./PDFs/
```

```bash
python3 pdf_to_json.py
```

---

## **Embedding Stage**

Ensure `conversations.json` or `conversations2.json`, exported from ChatGPT/Claude, is in the working directory.
Only `conversations.json` is required — if it's missing, the conversation-embedding step is skipped and only the PDF-derived `pdf_texts.json` (if generated) will be used.

```bash
python3 batch_embedder.py
```

---

## **Semantic Processing**

```bash
python3 data_formatter.py \
    --enable-semantic-labeling \
    --semantic-mode normal \
    --semantic-method hybrid \
    --batch-size 256 # Larger uses more compute but faster
```
Add `--force-cpu` to use the CPU.

---

## **Training**

```bash
python3 train_script.py
```

If you encounter tokenization errors, clear the cached tokenized dataset.
This occurs when reusing cached data with a different base model than the one it was created for:

```bash
rm -rf data_finetune/tokenized_cache
```

## **Model Selection**

Set the base model in `train_script.py`:

```python
model_name = "google/gemma-3-1b-it-qat-int4-unquantized"  # Any Hugging Face CAUSAL_LM model
```

Note:
- Training requires int4 / NF4 quantization. q4_0 models are inference-only.
- Some models require a Hugging Face access token to download.
- Set `HF_TOKEN` in the script to your token if needed.

---

## **Gradio Chat + TTS (RAM Heavy)**

```bash
python3 gradio_chat_tts.py --tts-cpu    # Force CPU - Recommended
python3 gradio_chat_tts.py --tts-gpu    # Force CUDA 
python3 gradio_chat_tts.py --tts-auto   # Auto-detect best device
python3 gradio_chat_tts.py --tts-mps    # Apple Silicon
python3 gradio_chat_tts.py  --model Qwen/Qwen3-4B-Instruct-2507    # Load and use a model from Hugging Face
```

### **STT Setup (Vosk)**

Download a Vosk speech model (large shown here; smaller ones work too):

```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip
```

---

## **Export to GGUF (for llama.cpp)**

### GPU Support for llama.cpp (skip if CPU-only)

This section is only needed if you want to run inference with GPU acceleration via llama.cpp. Training uses PyTorch's own CUDA and doesn't require this.

Remove old cuda toolkit:
```bash
sudo apt remove nvidia-cuda-toolkit
```

Add NVIDIA's repo for newer CUDA:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
```

Symlink CUDA so it can be found (only needed if cmake can't find libcuda during the llama.cpp build):

```bash
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```

Verify installation:
```bash
nvcc --version   # Should show 12.4
nvidia-smi       # Should show your GPU
```

---

### Convert to GGUF

```bash
python3 -m venv venv_gguf
source venv_gguf/bin/activate
pip3 install --use-deprecated=legacy-resolver peft

python3 convert_to_gguf.py              # Auto quantization, 4-bit medium
python3 convert_to_gguf.py --quant f16  # No quantization (can go as small as q2_k - 2-bit)
python3 convert_to_gguf.py --gpu        # Build with CUDA (default)
python3 convert_to_gguf.py --cpu        # Build without CUDA (CPU-only)

deactivate

# The venv isolates llama.cpp's build dependencies from the pipeline. Once compiled, you don't need it anymore.
```

### Running the Model

```bash
# GPU (if CUDA enabled)
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 -ngl 99 --port 8081

# CPU only
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 --threads 14 --port 8081

# Offloding
./llama.cpp/build/bin/llama-server \
  -m ./gguf_models/*.gguf \
  --n-gpu-layers 999 \
  --ctx-size 10240 \
  --batch-size 248 \
  --port 8081
```

---

# **OOM Adjustments**

Edit these values in `train_script.py`:

```python
default_batch_size = 2   # Higher = faster training, but more memory. Use 1 for lowest footprint.
default_grad_accum = 8   # Effective batch = batch_size × grad_accum.
                         # Higher = slower training, no extra memory.
                         # Target effective batch: 16 (e.g., 4×4, 2×8, 1×16).
```

## **GPU Memory vs Speed Trade-offs (GTX / Older GPUs)**

On GTX-class or older GPUs, disabling some features can slightly reduce memory usage. However, if your GPU supports FP16, **leave it enabled**—it typically provides a significant speedup.

```python
# GPU defaults
default_batch_size = 2
default_grad_accum = 8
default_fp16 = False                  
default_gradient_checkpointing = False
```
---

# **Theme-Based Early Stopping**

Training stops when:

* The epoch limit is reached (default: 3), **or**
* All semantic themes have been observed.

To change this behavior, modify:

```python
metrics['coverage'] >= 1.0
```
