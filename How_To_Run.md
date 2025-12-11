# **RhizomeML – Setup & Workflow**

```bash
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

pip3 install -r requirements.txt --upgrade

# Install DeepSpeed (CPU-optimized build)
DS_SKIP_CUDA_CHECK=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip3 install deepspeed

# Check the DeepSpeed environment:
python3 -m deepspeed.env_report
# Adam should be enabled. ZeRO-Offload is mostly configured but disabled by default.
```

---

## **Data Preparation**

```bash
python3 pdf_to_json.py
```

### **Embedding Stage**

```bash
python3 batch_embedder.py  # Defaults to CPU. Set use_gpu=True if you have CUDA.
```

### **Semantic Processing**

```bash
python3 data_formatter.py \
    --force-cpu \
    --enable-semantic-labeling \
    --semantic-mode normal \
    --semantic-method hybrid
# Remove --force-cpu when using a compatible GPU.
```

If you retrain on the same base model and hit tokenized dataset errors, clear the cached tokenization:

```bash
rm -rf data_finetune/tokenized_cache
```

---

## **Training**

```bash
python3 train_script.py
```

---

## **Gradio Chat + TTS (RAM heavy)**

```bash
python3 gradio_chat_tts.py
```

### **STT Setup (Vosk)**

Gradio STT requires a speech model. Download the large pack below (smaller ones also work):

```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip
```

---

## **Convert the Final Model to GGUF (for llama.cpp)**

```bash
python3 -m venv venv_gguf
source venv_gguf/bin/activate
pip3 install peft
python3 convert_to_gguf.py
deactivate
```

---

# **OOM Adjustments**

Modify these lines in `train_script.py`:

```python
default_batch_size = 2   # Higher value faster but higher activation memory. Use 1 for lowest memory footprint.
default_grad_accum = 8   # Effective batch = batch_size × grad_accum. 
                         # Affects speed, not memory. Default target: 16 (e.g., 4x4, 2×8, 1×16).
```

---

# **Theme-Based Early Stopping**

Training ends when either:

* The epoch limit is reached (default: 3), **or**
* All semantic themes have been observed.

To adjust the early-stopping rule, look for:

```python
metrics['coverage'] >= 1.0
```
