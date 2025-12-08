```
pip3 install -r requirements.txt --upgrade
pip3 install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl
python3 pdf_to_json.py
python3 batch_embedder.py # use_gpu=False,  # Set to True if you have a compatible GPU (CUDA)
python3 data_formatter.py --force-cpu --enable-semantic-labeling --semantic-mode normal --semantic-method hybrid # Using a GPU remove --force-cpu
rm -rf data_finetune/tokenized_cache
python3 train_script.py
python3 gradio_chat_tts.py # Ram heavy

# Once happy can be turned into a GGUF for llama
python3 -m venv venv_gguf
source venv_gguf/bin/activate
pip3 install peft
python3 convert_to_gguf.py
deactivate
```
---
```
OOM Issues adjust these:

default_batch_size = 2 # Doubles activation memory but Faster.
default_grad_accum = 8 # Increases the effective batch → more time, not more VRAM/RAM.
```
