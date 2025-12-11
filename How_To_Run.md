```
git clone https://github.com/pinguy/RhizomeML.git
cd RhizomeML
pip3 install -r requirements.txt --upgrade
DS_SKIP_CUDA_CHECK=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip3 install deepspeed
python3 -m deepspeed.env_report
python3 pdf_to_json.py
python3 batch_embedder.py # use_gpu=False,  # Set to True if you have a compatible GPU (CUDA)
python3 data_formatter.py --force-cpu --enable-semantic-labeling --semantic-mode normal --semantic-method hybrid # Using a compatible GPU remove --force-cpu
rm -rf data_finetune/tokenized_cache # Removes the semantic themes arrow. If retraining the say type of base mode don't have to remove but get and error probably because of that.
python3 train_script.py
python3 gradio_chat_tts.py # Ram heavy

# For STT to work with graio download and unzip this pack. This will download the largest one but smaller more memory friendly ones are available

wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip
unzip vosk-model-en-us-0.42-gigaspeech.zip

# Once happy can be turned into a GGUF for llama
python3 -m venv venv_gguf
source venv_gguf/bin/activate
pip3 install peft
python3 convert_to_gguf.py
deactivate
```
---
```
OOM Issues adjust these in train_script.py:

default_batch_size = 2 # Doubles activation memory but Faster.
default_grad_accum = 8 # Increases the effective batch → more time, not more VRAM/RAM.
```
---
```
The fine tune stops when it hits the set Epoch (default 3) or when all themes have been seen. 

To adjust Theme stopping look for:

metrics['coverage'] >= 1.0:
```
