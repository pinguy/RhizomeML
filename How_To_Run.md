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

## **Data Preparation**

Place your source PDFs in:

```
./PDFs/
```

```bash
python pdf_to_json.py
```

---

## **Embedding Stage**

Ensure `conversations.json` or `conversations2.json` (exported from ChatGPT or Claude) is in the working directory.
If only `pdf_texts.json` exists, conversation embeddings will be skipped automatically — not recommended.
You can also run **without PDFs first** to test the conversation embedding pipeline before adding large PDF data.

```bash
python batch_embedder.py
```

---

## **Semantic Processing**

```bash
python data_formatter.py \
    --enable-semantic-labeling \
    --semantic-mode normal \
    --semantic-method hybrid \
    --batch-size 256  # Larger values increase speed but require more compute
```

Add `--force-cpu` to override GPU usage.

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
python -m venv venv_gguf
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
  --n-gpu-layers 999 \
  --ctx-size 10240 \
  --batch-size 248 \
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
