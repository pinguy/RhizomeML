# **RhizomeML – Setup & Workflow**

```bash
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML

pip3 install -r requirements.txt --upgrade
```

### **Install DeepSpeed (ZeRO-Offload)**

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
Only `conversations.json` is required — if it’s missing, the conversation-embedding step is skipped and only the PDF-derived `pdf_texts.json` (if generated) will be used.

```bash
python3 batch_embedder.py
# Defaults to CPU. Set use_gpu=True if you have CUDA.
```

---

## **Semantic Processing**

```bash
python3 data_formatter.py \
    --force-cpu \
    --enable-semantic-labeling \
    --semantic-mode normal \
    --semantic-method hybrid
# Remove --force-cpu when using a compatible GPU.
```

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

---

## **Gradio Chat + TTS (RAM Heavy)**

```bash
python3 gradio_chat_tts.py
```

### **STT Setup (Vosk)**

Download a Vosk speech model (large shown here; smaller ones work too):

```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip
```

---

## **Export to GGUF (for llama.cpp)**

```bash
python3 -m venv venv_gguf
source venv_gguf/bin/activate

pip3 install peft
python3 convert_to_gguf.py

deactivate
```

---

# **OOM Adjustments**

Edit these values in `train_script.py`:

```python
default_batch_size = 2   # Higher value = faster training, but higher activation memory. Use 1 for the lowest memory footprint.
default_grad_accum = 8   # Effective batch = batch_size × grad_accum.
                         # Affects speed, not memory. Target effective batch: 16 (e.g., 4×4, 2×8, 1×16).
```

---

# **Theme-Based Early Stopping**

Training stops when:

* the epoch limit is reached (default: 3), **or**
* all semantic themes have been observed.

To change this behavior, modify:

```python
metrics['coverage'] >= 1.0
```

---
