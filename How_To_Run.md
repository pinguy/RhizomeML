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

Note: Works with 5.11.16_lowlatency Kernel for older distros.

### Running on Older Distros Using Distrobox

```bash
# Clone Distrobox
git clone https://github.com/89luca89/distrobox.git
cd distrobox

# Install
sudo ./install --prefix /usr/local
distrobox version

# Install Podman
sudo apt install podman

# On HOST set-up nvidia-containe
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit

# Configure for podman
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Build the image. Download the Dockerfile.rhizome file from the repo.
mkdir -p ~/tmp-podman
podman build -t rhizome-img -f Dockerfile.rhizome

# Create container with nvidia passthrough
distrobox create --name rhizome-dev --image rhizome-img --nvidia
distrobox enter rhizome-dev # May hang or fail a few time. When it happens open a new Terminal while keeping the hanged one open and run it again. At some point it will go though then will be fine.
```

---

### **Clone the Repo**

```bash
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

pip3 install -r requirements.txt --upgrade
```

### DeepSpeed (ZeRO-Offload) - Optional, Not Recommended
DeepSpeed provides minimal benefit for single-GPU training and frequently causes version conflicts with PyTorch and Transformers. Skip unless you specifically need ZeRO-Offload for CPU memory offloading.
```bash
DS_SKIP_CUDA_CHECK=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip3 install deepspeed

python3 -m deepspeed.env_report
# Adam should be enabled. ZeRO-Offload is mostly configured but disabled by default.
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
    --semantic-method hybrid
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
python3 gradio_chat_tts.py --tts-cpu    # Force CPU
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

### GPU Support (skip if CPU-only)

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

Add the CUDA paths to your `.bashrc`:
```bash
nano ~/.bashrc
```

Add these lines at the end:
```bash
# CUDA 12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

Save with `Ctrl+O`, Enter, then `Ctrl+X` to exit. Then reload:
```bash
source ~/.bashrc
```
Symlink CUDA so it can be found:

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

deactivate
```

### Running the Model

```bash
# GPU (if CUDA enabled)
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 -ngl 99 --port 8081

# CPU only
./llama.cpp/build/bin/llama-server -m gguf_models/*.gguf -c 8192 --threads 14 --port 8081
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

---

# **Theme-Based Early Stopping**

Training stops when:

* The epoch limit is reached (default: 3), **or**
* All semantic themes have been observed.

To change this behavior, modify:

```python
metrics['coverage'] >= 1.0
```
