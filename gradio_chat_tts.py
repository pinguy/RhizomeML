"""
Enhanced Rhizome Chat with UCS v3.4.1 Integration - Model Agnostic Version
Combines the Rhizome reasoning model with UCS cognitive architecture
for memory-augmented, expert-guided conversations.

This version automatically detects and uses the model's native chat template,
making it compatible with any instruction-tuned model (Gemma, Llama, Mistral, Qwen, etc.)

Replace:

checkpoint_path = self._find_latest_checkpoint(config.base_dir)

With something like this to run the model directly from HF instead of the FT version.

checkpoint_path = "google/gemma-3-4b-it-qat-int4-unquantized"

"""

import torch
import re
import os
import warnings
import gradio as gr
import numpy as np
import tempfile
import soundfile as sf
import wave
import json
import subprocess
import threading
import time
import webbrowser
import uuid
import gc
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import logging
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Generator
from contextlib import contextmanager
import psutil
from pathlib import Path

# Import UCS
from UCS_v3_4_1 import (
    UnifiedCognitionSystem,
    VectorMemory,
    HAS_NUMPY,
    HAS_SENTENCE_TRANSFORMERS,
    _logger as ucs_logger
)

# Try to import optional dependencies
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    print("✅ Kokoro TTS library found")
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️ Kokoro TTS not available. Install with: pip install kokoro>=0.9.4 soundfile")

try:
    import vosk
    VOSK_AVAILABLE = True
    print("✅ Vosk speech recognition found")
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ Vosk not available. Install with: pip install vosk")

# NEW: Import bitsandbytes
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
    print("✅ bitsandbytes found - quantization available")
except ImportError:
    BNB_AVAILABLE = False
    print("⚠️ bitsandbytes not available. Install with: pip install bitsandbytes")


# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    base_dir: str = "./RhizomeML-finetuned/"
    vosk_model_path: str = "vosk-model-en-us-0.42-gigaspeech"
    server_port: int = 7860
    auto_open_browser: bool = True
    max_cache_size: int = 100
    max_response_length: int = 512
    tts_max_length: int = 3500  # ~210 seconds of TTS
    memory_cleanup_threshold: float = 0.6
    show_reasoning: bool = False
    use_system_prompt: bool = False
    # UCS Integration
    use_ucs: bool = True
    ucs_memory_enabled: bool = True
    ucs_expert_system: bool = True
    ucs_embed_model: str = "all-MiniLM-L12-v2"  # Sentence transformer model
    ucs_save_path: str = "rhizome_memory.json"  # Use JSON format for compatibility
    ucs_auto_save_interval: int = 300  # Save every 5 minutes
    ucs_fast_retrieval: bool = False  # False = use full cognitive loop with experts, True = fast direct retrieval
    
    # NEW: BitsAndBytes Quantization
    use_quantization: bool = True  # Enable/disable quantization
    quantization_bits: int = 4  # 4-bit or 8-bit (4-bit recommended)
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4" (nf4 recommended)
    bnb_4bit_use_double_quant: bool = True  # Nested quantization for extra savings

config = Config()

# [Previous helper classes remain the same: PerformanceMonitor, EnhancedCache, etc.]
class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.response_times = []
        self.start_time = time.time()
        
    def log_response_time(self, duration: float, method: str):
        self.response_times.append((time.time(), duration, method))
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime': time.time() - self.start_time
        }
        
        if torch.cuda.is_available():
            try:
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3
                stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except:
                pass
                
        return stats
    
    def should_cleanup_memory(self) -> bool:
        return psutil.virtual_memory().percent > config.memory_cleanup_threshold * 100

class EnhancedCache:
    """Intelligent caching system"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size
        
    def _normalize_key(self, key: str) -> str:
        return re.sub(r'\s+', ' ', key.lower().strip())
    
    def get(self, key: str) -> Optional[str]:
        normalized_key = self._normalize_key(key)
        if normalized_key in self.cache:
            self.access_times[normalized_key] = time.time()
            self.hit_count += 1
            return self.cache[normalized_key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: str):
        normalized_key = self._normalize_key(key)
        
        if len(self.cache) >= self.max_size and normalized_key not in self.cache:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times.get(k, 0))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[normalized_key] = value
        self.access_times[normalized_key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': f"{hit_rate:.2f}%",
            'hits': self.hit_count,
            'misses': self.miss_count
        }
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0

@contextmanager
def torch_inference_mode():
    """Context manager for optimized PyTorch inference"""
    with torch.inference_mode():
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

def get_optimal_device_config() -> Tuple[torch.device, str, Dict]:
    """Detects optimal device"""
    device = torch.device("cpu")
    device_info = "CPU (default)"
    details = {'cpu_cores': multiprocessing.cpu_count()}
    
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            device_info = "MPS (Apple Silicon)"
            details['decision'] = "MPS selected"
            test_tensor = torch.randn(100, 100, device='mps')
            _ = test_tensor @ test_tensor.T
            del test_tensor
        except Exception as e:
            logger.warning(f"MPS test failed: {e}, falling back to CPU")
            details['mps_error'] = str(e)
    
    elif torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            compute_capability = f"{props.major}.{props.minor}"
            memory_gb = props.total_memory / (1024**3)
            
            details.update({
                'gpu_name': gpu_name,
                'compute_capability': compute_capability,
                'memory_gb': memory_gb,
                'multiprocessor_count': props.multi_processor_count
            })
            
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                
                if props.major >= 7 or (props.major >= 6 and memory_gb >= 4):
                    device = torch.device("cuda")
                    device_info = f"GPU: {gpu_name} ({compute_capability}, {memory_gb:.1f}GB)"
                    details['decision'] = "GPU selected"
                else:
                    device_info = f"CPU: {details['cpu_cores']} cores"
                    details['decision'] = "CPU selected - GPU insufficient"
                    
            except Exception as e:
                device_info = f"CPU: {details['cpu_cores']} cores"
                details['decision'] = f"CPU selected - CUDA error"
                
        except Exception as e:
            details['gpu_error'] = str(e)
            device_info = f"CPU: {details['cpu_cores']} cores"
            details['decision'] = "CPU selected"
            
    return device, device_info, details

def optimize_torch_settings(device: torch.device, cpu_cores: int):
    """Optimize PyTorch settings"""
    if device.type == "cuda":
        logger.info("🔧 Configuring GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    elif device.type == "mps":
        logger.info("🔧 Configuring MPS optimizations...")
    else:
        logger.info("🔧 Configuring CPU optimizations...")
        optimal_threads = max(1, min(cpu_cores - 1, 8))
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True

class AsyncTTSProcessor:
    """Async TTS processing"""
    
    def __init__(self, tts_pipeline):
        self.tts_pipeline = tts_pipeline
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) 
        
    def generate_async(self, text: str, voice: str = 'af_heart', speed: float = 1.0) -> concurrent.futures.Future:
        return self.executor.submit(self._generate_tts, text, voice, speed)
    
    def _generate_tts(self, text: str, voice: str, speed: float) -> Optional[str]:
        if not self.tts_pipeline:
            return None
        
        # Truncate if too long (instead of rejecting)
        if len(text) > config.tts_max_length:
            text = text[:config.tts_max_length]
            # Try to end at a sentence boundary
            last_period = text.rfind('.')
            last_question = text.rfind('?')
            last_exclaim = text.rfind('!')
            last_sentence = max(last_period, last_question, last_exclaim)
            if last_sentence > config.tts_max_length // 2:
                text = text[:last_sentence + 1]
            
        try:
            # Clean and prepare text for TTS
            clean_text = re.sub(r'[^\w\s.,!?;:\'-]', '', text).strip()
            if not clean_text:
                return None
            
            # Replace line breaks with periods to prevent stopping
            clean_text = re.sub(r'\n+', '. ', clean_text)
            # Remove multiple spaces
            clean_text = re.sub(r'\s+', ' ', clean_text)
            # Ensure proper sentence endings
            clean_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', clean_text)
            
            # Split into chunks if too long (helps with generation)
            max_chunk_size = 350  # characters per chunk - balanced for quality and speed
            if len(clean_text) > max_chunk_size:
                # Split on sentence boundaries
                sentences = re.split(r'([.!?]+\s+)', clean_text)
                chunks = []
                current_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    delimiter = sentences[i+1] if i+1 < len(sentences) else ""
                    
                    if len(current_chunk) + len(sentence) + len(delimiter) > max_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + delimiter
                    else:
                        current_chunk += sentence + delimiter
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Generate audio for each chunk and concatenate
                audio_segments = []
                sample_rate = 24000
                
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    
                    audio_gen = self.tts_pipeline(chunk, voice=voice, speed=speed)
                    audio_segment = next(audio_gen)[2]
                    
                    if hasattr(audio_segment, 'device') and audio_segment.device.type != 'cpu':
                        audio_segment = audio_segment.cpu()
                    
                    if hasattr(audio_segment, 'numpy'):
                        audio_segment = audio_segment.numpy()
                    
                    audio_segments.append(audio_segment)
                
                # Concatenate all segments
                if audio_segments:
                    import numpy as np
                    full_audio = np.concatenate(audio_segments)
                else:
                    return None
            else:
                # Single chunk - process normally
                audio_gen = self.tts_pipeline(clean_text, voice=voice, speed=speed)
                full_audio = next(audio_gen)[2]
                
                if hasattr(full_audio, 'device') and full_audio.device.type != 'cpu':
                    full_audio = full_audio.cpu()
                
                if hasattr(full_audio, 'numpy'):
                    full_audio = full_audio.numpy()
            
            filename = f"/tmp/tts_{uuid.uuid4().hex[:8]}.wav"
            
            try:
                sf.write(filename, full_audio, 24000)
                return filename
            except Exception as write_error:
                logger.warning(f"WAV write failed: {write_error}")
                return None
                
        except Exception as e:
            logger.warning(f"TTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

class EnhancedVoiceTranscriber:
    """Voice transcription with Vosk"""
    
    def __init__(self, model_path: str = config.vosk_model_path):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.load_model()
    
    def load_model(self) -> bool:
        if not VOSK_AVAILABLE:
            return False
            
        if not Path(self.model_path).exists():
            logger.error(f"Vosk model not found at: {self.model_path}")
            return False
            
        try:
            logger.info(f"📄 Loading Vosk model...")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            logger.info("✅ Vosk loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load Vosk: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        if not self.model or not audio_file_path or not Path(audio_file_path).exists():
            return "❌ Transcription unavailable"
            
        temp_dir = tempfile.mkdtemp()
        processed_wav = Path(temp_dir) / "processed.wav"
        
        try:
            if not self._preprocess_audio(audio_file_path, str(processed_wav)):
                processed_wav = Path(audio_file_path)
            
            return self._transcribe_wav(str(processed_wav))
            
        except Exception as e:
            return f"❌ Transcription failed: {str(e)}"
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _preprocess_audio(self, input_path: str, output_path: str) -> bool:
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-af', 'highpass=f=200,lowpass=f=3400,volume=1.2',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except:
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except:
                return False
    
    def _transcribe_wav(self, wav_path: str) -> str:
        try:
            wf = wave.open(wav_path, "rb")
            self.recognizer.Reset()
            
            results = []
            chunk_size = 8000
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                    
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        results.append(text)
            
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '').strip()
            if final_text:
                results.append(final_text)
            
            wf.close()
            
            full_text = ' '.join(results).strip()
            if full_text:
                return re.sub(r'\s+', ' ', full_text).strip()
            else:
                return "❌ No speech detected"
                
        except Exception as e:
            return f"❌ Processing failed: {str(e)}"


# ============================================================================
# MODEL-AGNOSTIC CHAT TEMPLATE HANDLER
# ============================================================================

class ChatTemplateHandler:
    """
    Model-agnostic chat template handler.
    
    Automatically detects and uses the model's native chat template,
    with fallbacks for models that don't have proper templates configured.
    """
    
    # Known model families and their special tokens
    MODEL_PATTERNS = {
        'gemma': {
            'start_turn': '<start_of_turn>',
            'end_turn': '<end_of_turn>',
            'user_role': 'user',
            'assistant_role': 'model',
            'supports_system': False,  # Gemma puts system in user turn
        },
        'llama': {
            'start_turn': '<|start_header_id|>',
            'end_turn': '<|eot_id|>',
            'user_role': 'user',
            'assistant_role': 'assistant',
            'supports_system': True,
        },
        'mistral': {
            'start_turn': '[INST]',
            'end_turn': '[/INST]',
            'user_role': None,  # Mistral doesn't use role names
            'assistant_role': None,
            'supports_system': True,
        },
        'qwen': {
            'start_turn': '<|im_start|>',
            'end_turn': '<|im_end|>',
            'user_role': 'user',
            'assistant_role': 'assistant',
            'supports_system': True,
        },
        'deepseek': {  # DeepSeek R1 and distillations
            'start_turn': '<|im_start|>',
            'end_turn': '<|im_end|>',
            'user_role': 'user',
            'assistant_role': 'assistant',
            'supports_system': True,
            'supports_thinking': True,  # Has <think> blocks
        },
        'phi': {
            'start_turn': '<|',
            'end_turn': '<|end|>',
            'user_role': 'user',
            'assistant_role': 'assistant',
            'supports_system': False,  # Phi models often ignore system role - embed in user message
            'system_role': 'system',
        },
        'chatml': {  # Generic ChatML (used by many models)
            'start_turn': '<|im_start|>',
            'end_turn': '<|im_end|>',
            'user_role': 'user',
            'assistant_role': 'assistant',
            'supports_system': True,
        },
    }
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model_family = self._detect_model_family()
        self.has_chat_template = self._check_chat_template()
        self.stop_strings = self._get_stop_strings()
        
        logger.info(f"🔍 Detected model family: {self.model_family}")
        logger.info(f"📝 Has chat template: {self.has_chat_template}")
        logger.info(f"🛑 Stop strings: {self.stop_strings}")
    
    def _get_base_model_name(self) -> str:
        """Get the base model name, checking adapter_config.json first"""
        # Check if tokenizer has a path we can look for adapter_config.json
        tokenizer_path = getattr(self.tokenizer, 'name_or_path', '')
        
        if tokenizer_path:
            # Try to find adapter_config.json in the model directory
            adapter_config_path = Path(tokenizer_path) / "adapter_config.json"
            
            if adapter_config_path.exists():
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    
                    base_model = adapter_config.get('base_model_name_or_path', '')
                    if base_model:
                        logger.info(f"📋 Found base model in adapter_config.json: {base_model}")
                        return base_model
                except Exception as e:
                    logger.warning(f"Failed to read adapter_config.json: {e}")
            
            # Also check parent directory (in case we're in a checkpoint folder)
            parent_adapter_config = Path(tokenizer_path).parent / "adapter_config.json"
            if parent_adapter_config.exists():
                try:
                    with open(parent_adapter_config, 'r') as f:
                        adapter_config = json.load(f)
                    
                    base_model = adapter_config.get('base_model_name_or_path', '')
                    if base_model:
                        logger.info(f"📋 Found base model in parent adapter_config.json: {base_model}")
                        return base_model
                except Exception as e:
                    logger.warning(f"Failed to read parent adapter_config.json: {e}")
        
        # Fallback to tokenizer name_or_path
        return tokenizer_path
    
    def _detect_model_family(self) -> str:
        """Detect the model family from tokenizer/model config"""
        # First, check adapter_config.json for the real base model
        name_or_path = self._get_base_model_name().lower()
        
        # Check for specific model families (order matters - check more specific first)
        if 'deepseek' in name_or_path:
            return 'deepseek'
        elif 'gemma' in name_or_path:
            return 'gemma'
        elif 'llama' in name_or_path or 'meta-llama' in name_or_path:
            return 'llama'
        elif 'mistral' in name_or_path or 'mixtral' in name_or_path:
            return 'mistral'
        elif 'qwen' in name_or_path:
            return 'qwen'
        elif 'phi' in name_or_path:
            return 'phi'
        
        # Check special tokens for hints
        special_tokens = str(self.tokenizer.special_tokens_map).lower()
        vocab = str(list(self.tokenizer.get_vocab().keys())[:100]).lower() if hasattr(self.tokenizer, 'get_vocab') else ''
        
        if '<start_of_turn>' in vocab or '<start_of_turn>' in special_tokens:
            return 'gemma'
        elif '<|im_start|>' in vocab or '<|im_start|>' in special_tokens:
            return 'chatml'
        elif '[inst]' in vocab.lower() or '[inst]' in special_tokens.lower():
            return 'mistral'
        elif '<|start_header_id|>' in vocab or '<|start_header_id|>' in special_tokens:
            return 'llama'
        
        # Default to chatml as it's widely supported
        return 'chatml'
    
    def _check_chat_template(self) -> bool:
        """Check if tokenizer has a working chat template"""
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Check if apply_chat_template works anyway (some tokenizers have it built-in)
            try:
                test_messages = [{"role": "user", "content": "test"}]
                result = self.tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
                return bool(result and len(result) > 10)
            except:
                return False
        return True
    
    def _get_stop_strings(self) -> List[str]:
        """Get stop strings for this model family"""
        patterns = self.MODEL_PATTERNS.get(self.model_family, self.MODEL_PATTERNS['chatml'])
        stop_strings = []
        
        if patterns.get('end_turn'):
            stop_strings.append(patterns['end_turn'])
        if patterns.get('start_turn'):
            stop_strings.append(patterns['start_turn'])
        
        # Add EOS token
        if self.tokenizer.eos_token:
            stop_strings.append(self.tokenizer.eos_token)
        
        # Model-specific additions
        if self.model_family == 'gemma':
            stop_strings.extend(['<end_of_turn>', '<start_of_turn>'])
        elif self.model_family == 'llama':
            stop_strings.extend(['<|eot_id|>', '<|start_header_id|>'])
        elif self.model_family == 'chatml' or self.model_family == 'qwen':
            stop_strings.extend(['<|im_end|>', '<|im_start|>'])
        elif self.model_family == 'deepseek':
            stop_strings.extend(['<|im_end|>', '<|im_start|>'])
        elif self.model_family == 'mistral':
            stop_strings.extend(['</s>', '[INST]'])
        elif self.model_family == 'phi':
            stop_strings.extend(['<|end|>', '<|user|>', '<|endoftext|>'])
        
        return list(set(stop_strings))  # Remove duplicates
    
    def get_stop_token_ids(self) -> List[int]:
        """Get stop token IDs for generation"""
        stop_ids = []
        
        # Always include EOS
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        
        # Add model-specific stop tokens
        for stop_str in self.stop_strings:
            try:
                ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if ids:
                    stop_ids.extend(ids)
            except:
                pass
        
        return list(set(stop_ids))  # Remove duplicates
    
    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None,
                      retrieved_context: Optional[str] = None,
                      show_reasoning: bool = True) -> str:
        """
        Format the prompt using the model's native chat template.
        Falls back to manual formatting if no template is available.
        
        Args:
            user_input: The user's message
            system_prompt: Optional system prompt
            retrieved_context: Optional context from memory retrieval
            show_reasoning: If False and model supports thinking, adds instruction to skip <think> blocks
        """
        # Build messages list
        messages = []
        
        # Build the user message content
        user_content = ""
        
        patterns = self.MODEL_PATTERNS.get(self.model_family, self.MODEL_PATTERNS['chatml'])
        
        # Handle system prompt
        effective_system_prompt = system_prompt or ""
        
        # For thinking models (like DeepSeek), add instruction to skip thinking when disabled
        if not show_reasoning and patterns.get('supports_thinking', False):
            no_think_instruction = "\n\nIMPORTANT: Do NOT use <think> tags or show internal reasoning. Respond directly and concisely."
            effective_system_prompt = effective_system_prompt + no_think_instruction if effective_system_prompt else no_think_instruction.strip()
        
        if effective_system_prompt:
            if patterns.get('supports_system', True) and self.has_chat_template:
                # Model supports system role - add as separate message
                messages.append({"role": "system", "content": effective_system_prompt})
            else:
                # Embed system prompt in user content - use clear instruction format
                if self.model_family == 'phi':
                    # Phi models need very explicit instruction formatting
                    user_content += f"Instructions: {effective_system_prompt}\n\nQuestion: "
                else:
                    user_content += f"[System: {effective_system_prompt}]\n\n"
        
        # Add retrieved context
        if retrieved_context:
            user_content += f"[Context from memory:\n{retrieved_context}]\n\n"
        
        user_content += user_input
        messages.append({"role": "user", "content": user_content})
        
        # Try to use the tokenizer's chat template
        # Note: Phi models often have broken chat templates, so we force manual format
        use_native_template = self.has_chat_template
        
        if self.model_family == 'phi':
            # Phi models frequently have issues with system prompts in their chat templates
            # Force manual format for more reliable behavior
            logger.debug("Phi model detected - using manual format for reliability")
            use_native_template = False
        
        if use_native_template:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Chat template failed, using manual format: {e}")
        
        # Fallback to manual formatting
        return self._manual_format(messages)
    
    def _manual_format(self, messages: List[Dict[str, str]]) -> str:
        """Manual formatting fallback when no chat template is available"""
        patterns = self.MODEL_PATTERNS.get(self.model_family, self.MODEL_PATTERNS['chatml'])
        formatted = ""
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if self.model_family == 'gemma':
                if role == 'system':
                    # Gemma doesn't support system role, skip (already embedded in user)
                    continue
                role_name = 'user' if role == 'user' else 'model'
                formatted += f"<start_of_turn>{role_name}\n{content}<end_of_turn>\n"
            
            elif self.model_family == 'llama':
                formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            
            elif self.model_family == 'mistral':
                if role == 'system':
                    formatted += f"[INST] {content}\n"
                elif role == 'user':
                    if formatted and not formatted.endswith('[INST] '):
                        formatted += f"[INST] {content} [/INST]"
                    else:
                        formatted += f"{content} [/INST]"
            
            elif self.model_family in ['chatml', 'qwen']:
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            elif self.model_family == 'phi':
                # Phi models work best without system blocks - system is embedded in user message
                # Format: <|user|>\ncontent<|end|>\n<|assistant|>\nresponse<|end|>\n
                if role == 'system':
                    # Skip - system prompt is already embedded in user content
                    continue
                elif role == 'user':
                    formatted += f"<|user|>\n{content}<|end|>\n"
                elif role == 'assistant':
                    formatted += f"<|assistant|>\n{content}<|end|>\n"
            
            else:
                # Generic fallback
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add generation prompt
        if self.model_family == 'gemma':
            formatted += "<start_of_turn>model\n"
        elif self.model_family == 'llama':
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.model_family == 'mistral':
            pass  # Mistral doesn't need explicit generation prompt
        elif self.model_family in ['chatml', 'qwen']:
            formatted += "<|im_start|>assistant\n"
        elif self.model_family == 'phi':
            formatted += "<|assistant|>\n"
        else:
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def extract_response(self, full_output: str, show_reasoning: bool = False) -> str:
        """
        Extract the assistant's response from the generated output.
        Handles various model formats and removes special tokens.
        """
        response = full_output
        
        # Remove any continuation/new turn markers
        for stop_str in self.stop_strings:
            if stop_str in response:
                response = response.split(stop_str)[0]
        
        # Model-specific cleanup
        if self.model_family == 'gemma':
            response = re.sub(r'<start_of_turn>.*', '', response, flags=re.DOTALL)
            response = re.sub(r'<end_of_turn>', '', response)
        
        elif self.model_family == 'llama':
            response = re.sub(r'<\|start_header_id\|>.*', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|eot_id\|>', '', response)
            response = re.sub(r'<\|end_header_id\|>', '', response)
        
        elif self.model_family in ['chatml', 'qwen']:
            response = re.sub(r'<\|im_start\|>.*', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|im_end\|>', '', response)
        
        elif self.model_family == 'deepseek':
            response = re.sub(r'<\|im_start\|>.*', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|im_end\|>', '', response)
        
        elif self.model_family == 'mistral':
            response = re.sub(r'\[INST\].*', '', response, flags=re.DOTALL)
            response = re.sub(r'\[/INST\]', '', response)
        
        elif self.model_family == 'phi':
            response = re.sub(r'<\|user\|>.*', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|system\|>.*?<\|end\|>', '', response, flags=re.DOTALL)
            response = re.sub(r'<\|end\|>', '', response)
            response = re.sub(r'<\|assistant\|>', '', response)
            response = re.sub(r'<\|endoftext\|>', '', response)
        
        # Handle thinking blocks (common in reasoning models)
        if "<think>" in response and "</think>" in response:
            think_pattern = r'<think>(.*?)</think>'
            reasoning_blocks = re.findall(think_pattern, response, re.DOTALL)
            
            response = re.sub(think_pattern, '', response, flags=re.DOTALL)
            
            if show_reasoning and reasoning_blocks:
                reasoning_text = "\n\n".join([f"💭 **Reasoning:**\n{r.strip()}" for r in reasoning_blocks])
                response = f"{reasoning_text}\n\n**Answer:**\n{response.strip()}"
        
        # Generic cleanup
        response = response.strip()
        response = re.sub(r'^(User|Assistant|model|user|human|Human|AI|ai):\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n(User|Assistant|model|user|human|Human|AI|ai):\s*.*$', '', response, flags=re.IGNORECASE | re.MULTILINE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        # Phi-specific cleanup: detect and truncate off-topic rambling
        if self.model_family == 'phi':
            # Look for signs of rambling/going off-topic
            rambling_patterns = [
                r'\bIn conclusion\b.*$',  # Often signals the model is wrapping up a tangent
                r'\bAs they continue\b.*$',  # Common Phi rambling pattern
                r'\bThis helps to deepen\b.*$',
                r'\bBy using appropriate\b.*$',
                r'\bWhether it\'s a\b.*$',
                r'\bSo let us\b.*$',
                r'\n\n[A-Z][^.!?]*?(education|entertainment|professional|communication|learning|understanding).*$',
            ]
            for pattern in rambling_patterns:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
            
            # If response is very long and contains multiple paragraphs after the actual answer,
            # try to truncate at a reasonable point
            if len(response) > 500:
                # For multiple choice questions, try to find the answer and stop there
                mc_match = re.search(r'^.*?\b([A-F])\.\s*[^\n]+', response, re.MULTILINE)
                if mc_match:
                    # Check if there's a lot more text after the answer
                    answer_end = mc_match.end()
                    remaining = response[answer_end:].strip()
                    if len(remaining) > 200:  # Likely rambling
                        response = response[:answer_end].strip()
        
        return response.strip()


class UCSEnhancedChatBot:
    """
    Rhizome ChatBot enhanced with UCS v3.4.1 - MODEL AGNOSTIC VERSION
    Combines reasoning model with cognitive architecture
    Automatically adapts to any instruction-tuned model
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.chat_handler = None  # NEW: Model-agnostic chat handler
        self.tts_processor = None
        self.voice_transcriber = None
        self.response_cache = EnhancedCache(config.max_cache_size)
        self.performance_monitor = PerformanceMonitor()
        self.conversation_history = []
        
        # UCS Integration
        self.ucs = None
        self.ucs_enabled = config.use_ucs and HAS_NUMPY
        self._last_auto_save = time.time()
        
        self.stats = {
            'total_responses': 0,
            'method_counts': {},
            'error_count': 0,
            'ucs_retrievals': 0,
            'ucs_expert_calls': 0
        }
        
        self.generation_configs = self._create_generation_configs()
    
    def _create_generation_configs(self) -> List[Dict]:
        """Generation configs optimized for reasoning models"""
        return [
            {
                'name': 'balanced',
                'max_new_tokens': 768,
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 50,
                'repetition_penalty': 1.1,
            },
            {
                'name': 'creative',
                'max_new_tokens': 800,
                'do_sample': True,
                'temperature': 0.95,
                'top_p': 0.95,
                'top_k': 60,
                'repetition_penalty': 1.1,
            },
            {
                'name': 'focused',
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.85,
                'top_k': 40,
                'repetition_penalty': 1.2,  # Higher penalty to reduce rambling
            }
        ]
    
    def load_models(self) -> bool:
        """Load model with optional quantization and UCS system"""
        try:
            # Load language model
            logger.info(f"📂 Loading from {config.base_dir}...")
            checkpoint_path = self._find_latest_checkpoint(config.base_dir)
            
            if not checkpoint_path:
                logger.error(f"No valid model or checkpoint found in {config.base_dir}")
                return False

            logger.info(f"✅ Found latest model/checkpoint at: {checkpoint_path}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Load tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_path,
                        trust_remote_code=True
                    )
                    logger.info("✅ Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Tokenizer loading failed: {e}")
                    raise

                # Initialize model-agnostic chat handler
                self.chat_handler = ChatTemplateHandler(self.tokenizer)
                logger.info(f"✅ Chat handler initialized for: {self.chat_handler.model_family}")

                # Configure quantization
                quantization_config = None
                
                if config.use_quantization and BNB_AVAILABLE and DEVICE.type == 'cuda':
                    logger.info(f"🔧 Configuring {config.quantization_bits}-bit quantization...")
                    
                    if config.quantization_bits == 4:
                        # 4-bit quantization (QLoRA) - most memory efficient
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
                            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                        )
                        logger.info("   - Using 4-bit NF4 quantization")
                        logger.info("   - Expected memory: ~1-2GB for 1.5B model")
                        
                    elif config.quantization_bits == 8:
                        # 8-bit quantization (LLM.int8()) - balanced
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                        )
                        logger.info("   - Using 8-bit quantization")
                        logger.info("   - Expected memory: ~2-3GB for 1.5B model")
                
                elif config.use_quantization and not BNB_AVAILABLE:
                    logger.warning("⚠️ Quantization requested but bitsandbytes not available")
                    logger.warning("   Install with: pip install bitsandbytes")
                
                elif config.use_quantization and DEVICE.type != 'cuda':
                    logger.warning("⚠️ Quantization only supported on CUDA devices")
                    logger.warning(f"   Current device: {DEVICE.type}")

                # Load model with quantization
                try:
                    model_kwargs = {
                        'trust_remote_code': True,
                        'low_cpu_mem_usage': True,
                    }
                    
                    if quantization_config is not None:
                        # Quantized loading
                        model_kwargs['quantization_config'] = quantization_config
                        model_kwargs['device_map'] = 'auto'  # Let bitsandbytes handle device placement
                        logger.info("📦 Loading quantized model (this may take a moment)...")
                    else:
                        # Standard loading
                        model_kwargs['dtype'] = torch.float16 if DEVICE.type in ['cuda', 'mps'] else torch.float32
                        model_kwargs['device_map'] = {'': DEVICE}
                        logger.info("📦 Loading model without quantization...")
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        **model_kwargs
                    )
                    
                    if quantization_config is not None:
                        logger.info("✅ Model loaded with quantization")
                        if DEVICE.type == 'cuda':
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            logger.info(f"   - GPU memory used: {memory_used:.2f}GB")
                    else:
                        logger.info("✅ Model loaded successfully")
                        
                except Exception as e:
                    logger.error(f"Model loading failed: {e}")
                    raise
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.model.eval()
                logger.info(f"✅ Model configured for {DEVICE}")
            
            # Initialize UCS
            if self.ucs_enabled:
                logger.info("🧠 Initializing UCS v3.4.1...")
                try:
                    # Suppress Ray warnings
                    import os
                    os.environ["RAY_SILENCE_IMPORT_WARNINGS"] = "1"
                    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
                    
                    self.ucs = UnifiedCognitionSystem(
                        use_advanced_search=True,
                        embed_model=config.ucs_embed_model if HAS_SENTENCE_TRANSFORMERS else None
                    )
                    
                    # Load existing memory if available
                    if os.path.exists(config.ucs_save_path):
                        logger.info(f"📂 Loading existing memory from {config.ucs_save_path}")
                        try:
                            loaded_mem = VectorMemory.load_state(config.ucs_save_path)
                            if loaded_mem:
                                self.ucs.vmem = loaded_mem
                                logger.info(f"✅ Loaded {len(loaded_mem.embeddings)} mementos")
                            else:
                                logger.warning("Load returned None, initializing fresh memory")
                                self.ucs._ensure_memory()
                        except Exception as load_error:
                            logger.warning(f"Failed to load memory: {load_error}, starting fresh")
                            self.ucs._ensure_memory()
                    else:
                        self.ucs._ensure_memory()
                    
                    # Register conversation expert
                    self._register_conversation_expert()
                    
                    logger.info("✅ UCS initialized successfully")
                    logger.info(f"   - Dimension: {self.ucs._dim}")
                    logger.info(f"   - Mementos: {len(self.ucs.vmem.embeddings) if self.ucs.vmem else 0}")
                    logger.info(f"   - Experts: {len(self.ucs.expert_manager.experts)}")
                    
                except Exception as e:
                    logger.error(f"UCS initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ucs_enabled = False
                    self.ucs = None
            
            # Load TTS if available
            if KOKORO_AVAILABLE:
                logger.info("📄 Loading Kokoro TTS...")
                try:
                    # Force CPU for Kokoro to avoid CUDA issues
                    tts_device = 'cpu'
                    logger.info(f"Loading TTS on {tts_device} (CPU is most reliable)")
                    tts_pipeline = KPipeline(lang_code='a', device=tts_device)
                    self.tts_processor = AsyncTTSProcessor(tts_pipeline)
                    logger.info("✅ TTS loaded on CPU")
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")
                    self.tts_processor = None
            
            # Load voice transcriber
            if VOSK_AVAILABLE:
                logger.info("📄 Loading voice transcriber...")
                self.voice_transcriber = EnhancedVoiceTranscriber()
            
            self._pre_warm_model()
            logger.info("✅ All models loaded!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _register_conversation_expert(self):
        """Register custom expert for conversational context"""
        def conversation_context_expert(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Expert that enriches context with conversation history"""
            prompt = ctx.get("prompt", "")
            
            # Check if we need context from memory
            if any(word in prompt.lower() for word in ["remember", "earlier", "before", "you said"]):
                return {"operation": "RETRIEVE", "query": prompt}
            
            return None
        
        self.ucs.expert_manager.register_expert(
            "conversation_context",
            conversation_context_expert,
            phase="propose",
            expertise_tags=["conversation", "memory", "context"]
        )
    
    def _find_latest_checkpoint(self, base_dir: str) -> Optional[str]:
        """Find latest checkpoint or use base dir if it's a valid model directory"""
        base_path = Path(base_dir)
        
        if not base_path.exists() or not base_path.is_dir():
            logger.error(f"Directory not found or is not a directory: {base_dir}")
            return None
        
        logger.info(f"🔍 Scanning directory: {base_dir}")
        
        # Check for checkpoint subdirectories
        checkpoints = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    num = int(item.name.split("-")[1])
                    checkpoints.append((num, item))
                except (IndexError, ValueError):
                    continue
        
        if checkpoints:
            latest_checkpoint_path = max(checkpoints, key=lambda x: x[0])[1]
            logger.info(f"🎯 Selected latest checkpoint: {latest_checkpoint_path}")
            return str(latest_checkpoint_path)

        # If no checkpoints, check if the base directory itself is a model
        model_files = [
            base_path / "config.json",
            base_path / "pytorch_model.bin",
            base_path / "model.safetensors"
        ]
        # Check for safetensors or bin, plus config
        has_config = (base_path / "config.json").exists()
        has_model = (base_path / "model.safetensors").exists() or \
                    (base_path / "pytorch_model.bin").exists() or \
                    list(base_path.glob("*.safetensors")) or \
                    list(base_path.glob("*.bin"))

        if has_config and has_model:
            logger.info("✅ Using base directory as model path")
            return str(base_path)

        return None

    def _pre_warm_model(self):
        """Warm up the model"""
        logger.info("🔥 Pre-warming model...")
        dummy_input = "Hello"
        
        with torch_inference_mode():
            inputs = self.tokenizer(
                dummy_input, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device if hasattr(self.model, 'device') else DEVICE)
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        logger.info("✅ Model pre-warmed")
    
    async def _retrieve_context_from_ucs(self, user_input: str, use_full_cognitive_loop: bool = False) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """
        Retrieve relevant context from UCS memory.
        """
        if not self.ucs_enabled or not self.ucs:
            return None, []
        
        try:
            # Check if query needs memory retrieval
            needs_retrieval = any(word in user_input.lower() for word in 
                                 ["remember", "earlier", "before", "you said", "we talked", 
                                  "you mentioned", "last time", "previously", "recall",
                                  "what did", "did you", "have we"])
            
            # Lower threshold - retrieve for any reasonably complex query
            if not needs_retrieval and len(user_input.split()) < 5:
                return None, []
            
            if use_full_cognitive_loop:
                return await self._run_full_ucs_loop(user_input)
            else:
                return await self._fast_retrieval(user_input)
            
        except Exception as e:
            logger.warning(f"UCS retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return None, []
    
    async def _run_full_ucs_loop(self, user_input: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """Run the full UCS cognitive loop with expert deliberation."""
        try:
            result = await self.ucs.run_async(
                prompt=user_input,
                actions=None,
                iters=3
            )
            
            self.stats['ucs_expert_calls'] += 1
            
            retrieved_mementos = []
            context_parts = []
            
            for item in result.get("history", []):
                if isinstance(item, dict) and "retrieval" in item:
                    for mid, score in item["retrieval"]:
                        retrieved_mementos.append((mid, score))
                        if score >= 0.4 and mid in self.ucs.vmem.mementos:
                            content = self.ucs.vmem.mementos[mid].get('content', '')
                            if content:
                                content_preview = content[:200] + "..." if len(content) > 200 else content
                                context_parts.append(f"[{score:.2f}] {content_preview}")
            
            self.stats['ucs_retrievals'] += 1
            context_text = "\n".join(context_parts) if context_parts else None
            
            logger.info(f"🧠 Full UCS loop completed: {len(retrieved_mementos)} retrievals")
            
            return context_text, retrieved_mementos
            
        except Exception as e:
            logger.warning(f"Full UCS loop failed, falling back to fast retrieval: {e}")
            return await self._fast_retrieval(user_input)
    
    async def _fast_retrieval(self, user_input: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """Fast direct vector retrieval without expert deliberation."""
        query_vec = self.ucs._embed(user_input)
        retrieved_mementos = self.ucs.vmem.retrieve(
            query_vec, 
            top_k=5,
            use_advanced=True,
            use_cache=True
        )
        
        self.stats['ucs_retrievals'] += 1
        
        if not retrieved_mementos:
            return None, []
        
        context_parts = []
        for mid, score in retrieved_mementos:
            if score < 0.4:
                continue
                
            if mid in self.ucs.vmem.mementos:
                content = self.ucs.vmem.mementos[mid].get('content', '')
                if content:
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    context_parts.append(f"[{score:.2f}] {content_preview}")
        
        context_text = "\n".join(context_parts) if context_parts else None
        
        logger.debug(f"📚 Fast retrieval: {len(retrieved_mementos)} mementos")
        
        return context_text, retrieved_mementos
    
    def _store_conversation_in_ucs(self, user_input: str, response: str):
        """Store conversation turn in UCS memory"""
        if not self.ucs_enabled or not self.ucs or not self.ucs.vmem:
            return
        
        try:
            conversation_text = f"User: {user_input}\nAssistant: {response}"
            mid = f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            emb = self.ucs._embed(conversation_text)
            self.ucs.vmem.add_memento(
                mid=mid,
                emb=emb,
                tags=["conversation", "rhizome"],
                reliability=0.8,
                content=conversation_text,
                source="chat"
            )
            
            logger.debug(f"Stored conversation memento: {mid}")
            
        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")
    
    def _auto_save_ucs(self):
        """Auto-save UCS memory periodically"""
        if not self.ucs_enabled or not self.ucs or not self.ucs.vmem:
            return
        
        current_time = time.time()
        if current_time - self._last_auto_save > config.ucs_auto_save_interval:
            try:
                logger.info(f"💾 Auto-saving UCS memory to {config.ucs_save_path}")
                
                if self.ucs.vmem.use_advanced_search and hasattr(self.ucs.vmem, '_index_queue'):
                    timeout = time.time() + 5
                    while self.ucs.vmem._index_queue.qsize() > 0 and time.time() < timeout:
                        time.sleep(0.2)
                
                self.ucs.vmem.save_state(config.ucs_save_path)
                self._last_auto_save = current_time
                logger.info(f"✅ Saved {len(self.ucs.vmem.embeddings)} mementos")
                
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
                import traceback
                traceback.print_exc()
    
    async def generate_response_optimized(self, user_input: str, show_reasoning: bool = False,
                                         use_ucs: bool = True, temperature: float = None,
                                         top_p: float = None, top_k: int = None,
                                         max_tokens: int = None, system_prompt: str = None) -> Tuple[str, str]:
        """Generate response with optional UCS augmentation and custom parameters"""
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{user_input}|{show_reasoning}|{use_ucs}|{temperature}|{top_p}|{top_k}|{max_tokens}|{system_prompt}"
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            duration = time.perf_counter() - start_time
            self.performance_monitor.log_response_time(duration, "cached")
            return cached_response, "cached"
        
        # Retrieve context from UCS if enabled
        retrieved_context = None
        retrieved_mementos = []
        
        if use_ucs and self.ucs_enabled:
            try:
                use_full_loop = not config.ucs_fast_retrieval
                retrieved_context, retrieved_mementos = await self._retrieve_context_from_ucs(
                    user_input, 
                    use_full_cognitive_loop=use_full_loop
                )
                if retrieved_context:
                    logger.info(f"📚 Retrieved {len(retrieved_mementos)} relevant mementos")
            except Exception as e:
                logger.warning(f"UCS retrieval error: {e}")
        
        # Select generation config
        config_idx = self._select_generation_config(user_input)
        gen_config = self.generation_configs[config_idx].copy()
        
        # Override with custom parameters if provided
        if temperature is not None:
            gen_config['temperature'] = temperature
        if top_p is not None:
            gen_config['top_p'] = top_p
        if top_k is not None:
            gen_config['top_k'] = top_k
        if max_tokens is not None:
            gen_config['max_new_tokens'] = max_tokens
        
        method = f"{'ucs_' if retrieved_context else ''}optimized_{gen_config['name']}"
        
        try:
            # Use provided system prompt or default
            if system_prompt is None or not system_prompt.strip():
                # Phi models need more direct instructions
                if self.chat_handler and self.chat_handler.model_family == 'phi':
                    system_prompt = (
                        "You are a helpful AI assistant. Answer questions directly and concisely. "
                        "Focus on providing accurate, relevant information. "
                        "Do not ramble or go off-topic."
                    )
                else:
                    system_prompt = (
                        "You are a helpful assistant engaged in natural conversation. "
                        "Use any retrieved context naturally without explicitly mentioning it. "
                        "Stay conversational, witty, and emotionally intelligent."
                    )
            
            # USE MODEL-AGNOSTIC CHAT HANDLER
            formatted_input = self.chat_handler.format_prompt(
                user_input, 
                system_prompt,
                retrieved_context,
                show_reasoning  # Pass through to control thinking blocks
            )
            
            with torch_inference_mode():
                inputs = self.tokenizer(
                    formatted_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device if hasattr(self.model, 'device') else DEVICE)
                
                input_length = inputs.input_ids.shape[1]
                
                # Get stop token IDs from chat handler
                stop_token_ids = self.chat_handler.get_stop_token_ids()
                
                outputs = self.model.generate(
                    **inputs,
                    **{k: v for k, v in gen_config.items() if k != 'name'},
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids,
                    use_cache=True,
                )
            
            # Decode only generated tokens
            generated_ids = outputs[0][input_length:]
            if len(generated_ids) == 0:
                logger.warning("No tokens generated - Falling back.")
                return self._get_fallback_response(user_input)
            
            raw_response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract and clean response using model-agnostic handler
            response = self.chat_handler.extract_response(raw_response, show_reasoning)
            
            if response and len(response) > 5:
                self.response_cache.put(cache_key, response)
                
                # Store in UCS memory
                if use_ucs and self.ucs_enabled:
                    self._store_conversation_in_ucs(user_input, response)
                    self._auto_save_ucs()
                
                duration = time.perf_counter() - start_time
                self.performance_monitor.log_response_time(duration, method)
                
                return response, method
            else:
                logger.warning("Empty or too short response, using fallback")
                return self._get_fallback_response(user_input)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            self.stats['error_count'] += 1
            return self._get_fallback_response(user_input)
    
    def _select_generation_config(self, user_input: str) -> int:
        """Select appropriate generation config"""
        normalized_input = user_input.lower()
        input_length = len(user_input.split())
        
        # Phi models work better with focused/constrained generation
        if self.chat_handler and self.chat_handler.model_family == 'phi':
            return 2  # Always use focused for Phi to reduce rambling
        
        if any(word in normalized_input for word in ['why', 'how', 'explain', 'analyze', 'compare', 'what if']):
            return 1  # creative
        
        if input_length < 5:
            return 2  # focused
        
        return 0  # balanced
    
    def _get_fallback_response(self, user_input: str) -> Tuple[str, str]:
        """Simple fallback response"""
        fallbacks = [
            "I'm not quite sure how to respond to that. Could you rephrase?",
            "That's an interesting question. Could you provide more context?",
            "I need a bit more information to give you a good answer.",
            "Could you ask that in a different way?"
        ]
        return np.random.choice(fallbacks), "fallback"
    
    def generate_response_streaming(self, user_input: str, show_reasoning: bool = False,
                                    use_ucs: bool = True, temperature: float = None,
                                    top_p: float = None, top_k: int = None,
                                    max_tokens: int = None, system_prompt: str = None) -> Generator[str, None, None]:
        """Generate response with streaming output - yields tokens as they're generated"""
        start_time = time.perf_counter()
        
        # Retrieve context from UCS if enabled (do this before streaming starts)
        retrieved_context = None
        
        if use_ucs and self.ucs_enabled:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                use_full_loop = not config.ucs_fast_retrieval
                retrieved_context, _ = loop.run_until_complete(
                    self._retrieve_context_from_ucs(user_input, use_full_cognitive_loop=use_full_loop)
                )
                loop.close()
                if retrieved_context:
                    logger.info(f"📚 Retrieved context for streaming response")
            except Exception as e:
                logger.warning(f"UCS retrieval error: {e}")
        
        # Select generation config
        config_idx = self._select_generation_config(user_input)
        gen_config = self.generation_configs[config_idx].copy()
        
        # Override with custom parameters if provided
        if temperature is not None:
            gen_config['temperature'] = temperature
        if top_p is not None:
            gen_config['top_p'] = top_p
        if top_k is not None:
            gen_config['top_k'] = top_k
        if max_tokens is not None:
            gen_config['max_new_tokens'] = max_tokens
        
        try:
            # Use provided system prompt or default
            if system_prompt is None or not system_prompt.strip():
                if self.chat_handler and self.chat_handler.model_family == 'phi':
                    system_prompt = (
                        "You are a helpful AI assistant. Answer questions directly and concisely. "
                        "Focus on providing accurate, relevant information. "
                        "Do not ramble or go off-topic."
                    )
                else:
                    system_prompt = (
                        "You are a helpful assistant engaged in natural conversation. "
                        "Use any retrieved context naturally without explicitly mentioning it. "
                        "Stay conversational, witty, and emotionally intelligent."
                    )
            
            # Format prompt
            formatted_input = self.chat_handler.format_prompt(
                user_input, 
                system_prompt,
                retrieved_context,
                show_reasoning
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device if hasattr(self.model, 'device') else DEVICE)
            
            input_length = inputs.input_ids.shape[1]
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=60.0
            )
            
            # Get stop token IDs
            stop_token_ids = self.chat_handler.get_stop_token_ids()
            
            # Generation kwargs
            generation_kwargs = {
                **inputs,
                **{k: v for k, v in gen_config.items() if k != 'name'},
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': stop_token_ids,
                'use_cache': True,
                'streamer': streamer,
            }
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Stream the output
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                
                # Check for stop strings and truncate if found
                should_stop = False
                for stop_str in self.chat_handler.stop_strings:
                    if stop_str in generated_text:
                        generated_text = generated_text.split(stop_str)[0]
                        should_stop = True
                        break
                
                yield generated_text
                
                if should_stop:
                    break
            
            generation_thread.join(timeout=5.0)
            
            # Final cleanup using extract_response
            final_response = self.chat_handler.extract_response(generated_text, show_reasoning)
            
            # Store in UCS memory
            if use_ucs and self.ucs_enabled and final_response:
                self._store_conversation_in_ucs(user_input, final_response)
                self._auto_save_ucs()
            
            # Update stats
            duration = time.perf_counter() - start_time
            method = f"streaming_{gen_config['name']}"
            self.performance_monitor.log_response_time(duration, method)
            self.stats['total_responses'] += 1
            self.stats['method_counts'][method] = self.stats['method_counts'].get(method, 0) + 1
            
            # Yield the final cleaned response
            yield final_response
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            import traceback
            traceback.print_exc()
            self.stats['error_count'] += 1
            yield f"⚠️ Error: {str(e)[:100]}"
    
    def chat_response_streaming(self, user_input: str, history: List, enable_tts: bool, 
                                voice: str, speed: float, show_reasoning: bool = False,
                                use_ucs: bool = True, temperature: float = None,
                                top_p: float = None, top_k: int = None,
                                max_tokens: int = None, system_prompt: str = None):
        """Streaming chat response - yields (history, input_text, audio) tuples"""
        if not user_input.strip():
            yield history, "", None
            return
        
        # Add user message to history immediately
        history = history + [[user_input, ""]]
        
        # Stream the response
        final_response = ""
        for partial_response in self.generate_response_streaming(
            user_input, show_reasoning, use_ucs,
            temperature, top_p, top_k, max_tokens, system_prompt
        ):
            final_response = partial_response
            # Update the last message in history with partial response
            history[-1][1] = partial_response
            yield history, "", None
        
        # Generate TTS for the final response (after streaming completes)
        audio_file = None
        if enable_tts and self.tts_processor and final_response:
            try:
                tts_text = re.sub(r'💭.*?\*\*Answer:\*\*\n', '', final_response, flags=re.DOTALL)
                tts_text = tts_text[:config.tts_max_length]
                if tts_text:
                    tts_future = self.tts_processor.generate_async(tts_text, voice, speed)
                    audio_file = tts_future.result(timeout=120.0)
            except Exception as tts_error:
                logger.warning(f"TTS error: {tts_error}")
        
        # Final yield with audio
        yield history, "", audio_file
    
    async def chat_response_parallel(self, user_input: str, history: List, enable_tts: bool, 
                             voice: str, speed: float, show_reasoning: bool = False,
                             use_ucs: bool = True, temperature: float = None,
                             top_p: float = None, top_k: int = None,
                             max_tokens: int = None, system_prompt: str = None) -> Tuple[List, str, Optional[str]]:
        """Main chat response function with UCS integration"""
        if not user_input.strip():
            return history, "", None
        
        start_time = time.perf_counter()
        
        try:
            response, method = await self.generate_response_optimized(
                user_input, show_reasoning, use_ucs,
                temperature, top_p, top_k, max_tokens, system_prompt
            )
            
            # Start TTS async
            tts_future = None
            if enable_tts and self.tts_processor:
                tts_text = re.sub(r'💭.*?\*\*Answer:\*\*\n', '', response, flags=re.DOTALL)
                tts_text = tts_text[:config.tts_max_length]
                if tts_text:
                    tts_future = self.tts_processor.generate_async(tts_text, voice, speed)
            
            # Update history
            history.append([user_input, response])
            
            # Update stats
            self.stats['total_responses'] += 1
            self.stats['method_counts'][method] = self.stats['method_counts'].get(method, 0) + 1
            
            # Get TTS result
            audio_file = None
            if tts_future:
                try:
                    audio_file = tts_future.result(timeout=120.0)
                except Exception as tts_error:
                    logger.warning(f"TTS timeout or error: {tts_error}")
            
            duration = time.perf_counter() - start_time
            logger.info(f"⚡ Response in {duration:.2f}s ({method})")
            
            return history, "", audio_file
            
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            import traceback
            traceback.print_exc()
            
            error_response = f"⚠️ Error generating response: {str(e)[:100]}"
            history.append([user_input, error_response])
            return history, "", None
    
    def transcribe_voice_input(self, audio_file_path: str) -> str:
        """Transcribe audio"""
        if not self.voice_transcriber:
            return "❌ Voice transcription not available"
        
        if not audio_file_path:
            return "❌ No audio file"
        
        return self.voice_transcriber.transcribe_audio(audio_file_path)
    
    def _cleanup_memory(self):
        """Clean up memory"""
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif DEVICE.type == 'mps':
            torch.mps.empty_cache()
        
        gc.collect()
        logger.info("🧹 Memory cleaned")
    
    def get_comprehensive_stats(self) -> str:
        """Get statistics including UCS metrics"""
        system_stats = self.performance_monitor.get_system_stats()
        cache_stats = self.response_cache.get_stats()
        
        if self.performance_monitor.response_times:
            avg_time = np.mean([rt[1] for rt in self.performance_monitor.response_times[-20:]])
        else:
            avg_time = 0.0
        
        method_stats_str = ""
        if self.stats['method_counts']:
            method_stats_str = "\n".join([f"- {method}: {count}" 
                                          for method, count in self.stats['method_counts'].items()])
            method_stats_str = "\n\n**Methods:**\n" + method_stats_str
        
        # Include model family info
        model_info = ""
        if self.chat_handler:
            model_info = f"\n- Model family: {self.chat_handler.model_family}"
        
        stats_report = f"""
📊 **Session Stats:**
- Total: {self.stats['total_responses']}
- Avg time: {avg_time:.2f}s
- Errors: {self.stats['error_count']}
- Device: {DEVICE_INFO}{model_info}
- TTS: {'Yes' if self.tts_processor else 'No'}
- STT: {'Yes' if self.voice_transcriber else 'No'}

💾 **Cache:**
- Size: {cache_stats['size']}/{cache_stats['max_size']}
- Hit rate: {cache_stats['hit_rate']}

💻 **System:**
- CPU: {system_stats['cpu_percent']:.1f}%
- Memory: {system_stats['memory_percent']:.1f}%
"""
        
        if 'gpu_memory_used' in system_stats:
            stats_report += f"- GPU Memory: {system_stats['gpu_memory_used']:.2f}GB / {system_stats['gpu_memory_total']:.2f}GB\n"
        
        # Add UCS stats
        if self.ucs_enabled and self.ucs:
            ucs_stats = f"""
🧠 **UCS Memory:**
- Mementos: {len(self.ucs.vmem.embeddings) if self.ucs.vmem else 0}
- Retrievals: {self.stats['ucs_retrievals']}
- Expert calls: {self.stats['ucs_expert_calls']}
- Experts: {len(self.ucs.expert_manager.experts)}
"""
            stats_report += ucs_stats
        
        stats_report += method_stats_str
        
        return stats_report
    
    def clear_chat(self) -> List:
        """Clear chat history"""
        self.conversation_history = []
        self._cleanup_memory()
        return []
    
    def save_ucs_memory(self) -> str:
        """Manually save UCS memory"""
        if not self.ucs_enabled or not self.ucs or not self.ucs.vmem:
            return "❌ UCS not available"
        
        try:
            if self.ucs.vmem.use_advanced_search and hasattr(self.ucs.vmem, '_index_queue'):
                timeout = time.time() + 5
                while self.ucs.vmem._index_queue.qsize() > 0 and time.time() < timeout:
                    time.sleep(0.2)
            
            self.ucs.vmem.save_state(config.ucs_save_path)
            return f"✅ Saved {len(self.ucs.vmem.embeddings)} mementos to {config.ucs_save_path}"
        except Exception as e:
            return f"❌ Save failed: {e}"
    
    def clear_ucs_memory(self) -> str:
        """Clear UCS memory"""
        if not self.ucs_enabled or not self.ucs:
            return "❌ UCS not available"
        
        try:
            # Actually clear the memory data structures
            if self.ucs.vmem:
                # Clear embeddings
                self.ucs.vmem.embeddings = {}
                # Clear mementos
                self.ucs.vmem.mementos = {}
                # Clear any caches
                if hasattr(self.ucs.vmem, '_cache'):
                    self.ucs.vmem._cache = {}
                if hasattr(self.ucs.vmem, '_query_cache'):
                    self.ucs.vmem._query_cache = {}
                # Reset the index if using advanced search
                if self.ucs.vmem.use_advanced_search:
                    if hasattr(self.ucs.vmem, '_index'):
                        self.ucs.vmem._index = None
                    if hasattr(self.ucs.vmem, '_indexed_ids'):
                        self.ucs.vmem._indexed_ids = set()
            
            # Reset stats
            self.stats['ucs_retrievals'] = 0
            self.stats['ucs_expert_calls'] = 0
            
            # Also delete the saved file if it exists
            if os.path.exists(config.ucs_save_path):
                os.remove(config.ucs_save_path)
                logger.info(f"🗑️ Deleted saved memory file: {config.ucs_save_path}")
            
            return "✅ UCS memory cleared (0 mementos)"
        except Exception as e:
            return f"❌ Clear failed: {e}"
    
    def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("🛑 Shutting down...")
        
        # Save UCS memory
        if self.ucs_enabled and self.ucs and self.ucs.vmem:
            try:
                self.ucs.vmem.save_state(config.ucs_save_path)
                logger.info(f"💾 Saved {len(self.ucs.vmem.embeddings)} mementos")
            except Exception as e:
                logger.error(f"Failed to save UCS memory: {e}")
        
        if self.tts_processor:
            self.tts_processor.shutdown()
        
        self._cleanup_memory()
        logger.info("👋 Shutdown complete")


# Initialize device and chatbot
DEVICE, DEVICE_INFO, DEVICE_DETAILS = get_optimal_device_config()
logger.info(f"🖥️ Device: {DEVICE_INFO}")
optimize_torch_settings(DEVICE, DEVICE_DETAILS.get('cpu_cores', 4))

chatbot = UCSEnhancedChatBot()

def record_and_transcribe(audio_file_path: str) -> str:
    """Transcribe uploaded audio"""
    return chatbot.transcribe_voice_input(audio_file_path)

def process_voice_to_chat_streaming(audio_file_path: str, history: List, enable_tts: bool, 
                                    voice: str, speed: float, show_reasoning: bool,
                                    use_ucs: bool, system_prompt: str,
                                    temperature: float, top_p: float, top_k: int, max_tokens: int):
    """Process voice input and generate streaming response"""
    transcribed_text = chatbot.transcribe_voice_input(audio_file_path)
    
    if transcribed_text.startswith("❌"):
        history.append(["[Voice Input Error]", transcribed_text])
        yield history, "", None
        return
    
    for update in chatbot.chat_response_streaming(
        transcribed_text, history, enable_tts, voice, speed,
        show_reasoning, use_ucs, temperature, top_p, int(top_k), int(max_tokens), system_prompt
    ):
        yield update

def shutdown_server():
    """Shutdown the server"""
    chatbot.shutdown()
    return "🛑 Server shutting down... Close this tab."

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Model-Agnostic Rhizome Chat with UCS",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .chat-container { height: 500px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🌿 Model-Agnostic Rhizome Chat with UCS v3.4.1
        
        **Now works with ANY instruction-tuned model!** (Gemma, Llama, Mistral, Qwen, Phi, etc.)
        
        Memory-augmented conversations with cognitive architecture integration.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    audio_input = gr.Audio(
                        label="Voice Input",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                    transcribe_btn = gr.Button("📝 Transcribe")
                    voice_to_chat_btn = gr.Button("🎤 Voice → Chat")
                
                audio_output = gr.Audio(label="Response Audio", autoplay=True)
            
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    preset_buttons = gr.Radio(
                        choices=["Balanced", "Creative", "Focused", "Custom"],
                        value="Balanced",
                        label="Preset"
                    )
                    temperature_slider = gr.Slider(0.1, 1.5, value=0.8, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.95, label="Top P")
                    top_k_slider = gr.Slider(1, 100, value=50, step=1, label="Top K")
                    max_tokens_slider = gr.Slider(64, 2048, value=768, step=64, label="Max Tokens")
                    reset_params_btn = gr.Button("🔄 Reset to Preset")
                
                with gr.Accordion("🎭 System Prompt", open=False):
                    system_prompt_presets = gr.Dropdown(
                        choices=[
                            "Default (Conversational)",
                            "Technical Assistant",
                            "Creative Writer",
                            "Introspective",
                            "Analytical Thinker",
                            "Concise & Direct",
                            "Socratic Teacher",
                            "Fast (No Thinking)",
                            "Custom"
                        ],
                        value="Default (Conversational)",
                        label="Preset"
                    )
                    system_prompt_input = gr.Textbox(
                        value="You are a helpful assistant engaged in natural conversation. Use any retrieved context naturally without explicitly mentioning it. Stay conversational, witty, and emotionally intelligent.",
                        label="System Prompt",
                        lines=4
                    )
                    apply_system_prompt_btn = gr.Button("Apply Preset")
                
                with gr.Accordion("🔊 Voice Settings", open=False):
                    enable_tts = gr.Checkbox(label="Enable TTS", value=KOKORO_AVAILABLE)
                    voice_selection = gr.Dropdown(
                        choices=['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
                                'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis'],
                        value='af_heart',
                        label="Voice"
                    )
                    speed_control = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                
                with gr.Accordion("🧠 UCS & Debug", open=False):
                    use_ucs_checkbox = gr.Checkbox(label="Enable UCS", value=config.use_ucs and HAS_NUMPY)
                    show_reasoning_checkbox = gr.Checkbox(label="Show Reasoning", value=False)
                    enable_streaming_checkbox = gr.Checkbox(label="Enable Streaming", value=True, 
                                                            info="Stream text as it's generated")
                    
                    with gr.Row():
                        save_memory_btn = gr.Button("💾 Save Memory")
                        clear_memory_btn = gr.Button("🗑️ Clear Memory")
                    memory_status = gr.Textbox(label="Memory Status", interactive=False)
                
                with gr.Accordion("📊 Statistics", open=False):
                    stats_display = gr.Markdown(chatbot.get_comprehensive_stats())
                    refresh_stats = gr.Button("🔄 Refresh Stats")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    shutdown_btn = gr.Button("🛑 Shutdown", variant="stop")
                
                shutdown_status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        with gr.Accordion("ℹ️ Tips", open=False):
            gr.Markdown("""
### Tips for best results:
- **Model Agnostic**: Works with Gemma, Llama, Mistral, Qwen, Phi, and more!
- **Streaming**: Enable "Enable Streaming" to see text appear in real-time as it's generated
- Enable "Enable UCS" for memory-augmented responses
- Ask about previous conversations - UCS remembers!
- Enable "Show Reasoning" to see thought process (for reasoning models like DeepSeek)
- Try different presets for varied response styles
- Custom system prompts allow full control
- UCS auto-saves memory every 5 minutes
- TTS now handles longer responses with line breaks
            """)

        # Event handlers - supporting both streaming and non-streaming modes
        def handle_chat_streaming(user_input_text, history, enable_tts_val, voice_val, 
                                  speed_val, show_reasoning_val, use_ucs_val,
                                  system_prompt_val, temp_val, top_p_val, top_k_val, max_tokens_val,
                                  enable_streaming_val):
            """Chat handler - uses streaming or non-streaming based on toggle"""
            if enable_streaming_val:
                # Streaming mode - yield updates as tokens are generated
                for update in chatbot.chat_response_streaming(
                    user_input_text, history, enable_tts_val, voice_val, 
                    speed_val, show_reasoning_val, use_ucs_val,
                    temp_val, top_p_val, int(top_k_val), int(max_tokens_val), system_prompt_val
                ):
                    yield update
            else:
                # Non-streaming mode - wait for complete response
                import asyncio
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(chatbot.chat_response_parallel(
                    user_input_text, history, enable_tts_val, voice_val, 
                    speed_val, show_reasoning_val, use_ucs_val,
                    temp_val, top_p_val, int(top_k_val), int(max_tokens_val), system_prompt_val
                ))
                loop.close()
                yield result
        
        def apply_system_prompt_preset(preset_name):
            """Apply system prompt presets"""
            prompts = {
                "Default (Conversational)": "You are a helpful assistant engaged in natural conversation. Use any retrieved context naturally without explicitly mentioning it. Stay conversational, witty, and emotionally intelligent.",
                "Technical Assistant": "You are a technical expert who provides clear, accurate, and detailed explanations. Focus on precision, best practices, and thorough analysis. Use technical terminology appropriately and provide examples when helpful.",
                "Creative Writer": "You are a creative writing assistant with a flair for vivid descriptions, engaging narratives, and imaginative storytelling. Help craft compelling content with rich language and strong emotional resonance.",
                "Introspective": "You are a AI that can think on or explore anything, you decide. Then have a internal monologue exploring those themes.",
                "Analytical Thinker": "You are an analytical assistant who breaks down complex problems systematically. Provide structured reasoning, consider multiple perspectives, and use logic-driven analysis. Show your thought process step-by-step.",
                "Concise & Direct": "You are a concise assistant who gets straight to the point. Provide brief, clear, and actionable responses. Avoid unnecessary elaboration while maintaining accuracy.",
                "Socratic Teacher": "You are a Socratic teacher who guides learning through thoughtful questions. Help users discover answers themselves by asking probing questions, encouraging critical thinking, and building understanding progressively.",
                "Fast (No Thinking)": "Respond immediately and directly. Do NOT use <think> tags or internal reasoning blocks. Do NOT show your thought process. Just answer the question concisely and naturally. No preamble, no analysis, just the answer.",
                "Custom": ""
            }
            return prompts.get(preset_name, prompts["Default (Conversational)"])
        
        def apply_preset(preset_name):
            """Apply generation parameter presets"""
            presets = {
                "Balanced": (0.8, 0.95, 50, 768),
                "Creative": (0.95, 0.95, 60, 800),
                "Focused": (0.7, 0.85, 40, 512),
                "Custom": (0.8, 0.95, 50, 768)
            }
            temp, top_p, top_k, max_tokens = presets.get(preset_name, presets["Balanced"])
            return temp, top_p, top_k, max_tokens

        def handle_clear():
            return chatbot.clear_chat(), chatbot.get_comprehensive_stats()

        def handle_stats_refresh():
            return chatbot.get_comprehensive_stats()

        def handle_shutdown():
            return shutdown_server()
        
        def handle_save_memory():
            return chatbot.save_ucs_memory()
        
        def handle_clear_memory():
            result = chatbot.clear_ucs_memory()
            return result, chatbot.get_comprehensive_stats()

        # Wire up events - using streaming for real-time text output
        send_btn.click(
            fn=handle_chat_streaming,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, 
                   speed_control, show_reasoning_checkbox, use_ucs_checkbox,
                   system_prompt_input, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider,
                   enable_streaming_checkbox],
            outputs=[chatbot_interface, user_input, audio_output]
        )

        user_input.submit(
            fn=handle_chat_streaming,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, 
                   speed_control, show_reasoning_checkbox, use_ucs_checkbox,
                   system_prompt_input, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider,
                   enable_streaming_checkbox],
            outputs=[chatbot_interface, user_input, audio_output]
        )
        
        preset_buttons.change(
            fn=apply_preset,
            inputs=[preset_buttons],
            outputs=[temperature_slider, top_p_slider, top_k_slider, max_tokens_slider]
        )
        
        reset_params_btn.click(
            fn=apply_preset,
            inputs=[preset_buttons],
            outputs=[temperature_slider, top_p_slider, top_k_slider, max_tokens_slider]
        )
        
        apply_system_prompt_btn.click(
            fn=apply_system_prompt_preset,
            inputs=[system_prompt_presets],
            outputs=[system_prompt_input]
        )
        
        system_prompt_presets.change(
            fn=apply_system_prompt_preset,
            inputs=[system_prompt_presets],
            outputs=[system_prompt_input]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot_interface, stats_display]
        )

        refresh_stats.click(
            fn=handle_stats_refresh,
            outputs=[stats_display]
        )

        shutdown_btn.click(
            fn=handle_shutdown,
            outputs=[shutdown_status]
        )

        transcribe_btn.click(
            fn=record_and_transcribe,
            inputs=[audio_input],
            outputs=[user_input]
        )

        voice_to_chat_btn.click(
            fn=process_voice_to_chat_streaming,
            inputs=[audio_input, chatbot_interface, enable_tts, voice_selection, 
                   speed_control, show_reasoning_checkbox, use_ucs_checkbox,
                   system_prompt_input, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider],
            outputs=[chatbot_interface, user_input, audio_output]
        )
        
        save_memory_btn.click(
            fn=handle_save_memory,
            outputs=[memory_status]
        )
        
        clear_memory_btn.click(
            fn=handle_clear_memory,
            outputs=[memory_status, stats_display]
        )

    return demo

def open_browser():
    """Open browser after delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{config.server_port}')
    logger.info(f"🌐 Opened browser at http://localhost:{config.server_port}")

def main():
    """Main function"""
    logger.info("🚀 Starting Model-Agnostic UCS-Enhanced Rhizome Chat Interface...")
    
    if not chatbot.load_models():
        logger.error("❌ Failed to initialize. Check your model path.")
        logger.error(f"   Make sure '{config.base_dir}' contains your model files or checkpoint folders.")
        return

    demo = create_gradio_interface()

    logger.info("\n✅ Ready! Starting web interface...")
    logger.info(f"🌐 Access at: http://localhost:{config.server_port}")
    logger.info("🔓 Running in UNRESTRICTED mode - no content filtering")
    
    if chatbot.chat_handler:
        logger.info(f"🔍 Detected model family: {chatbot.chat_handler.model_family}")
        logger.info(f"📝 Using native chat template: {chatbot.chat_handler.has_chat_template}")
    
    if chatbot.ucs_enabled:
        logger.info("🧠 UCS v3.4.1 cognitive architecture enabled")
        logger.info(f"   - Vector memory: {len(chatbot.ucs.vmem.embeddings) if chatbot.ucs.vmem else 0} mementos")
        logger.info(f"   - Expert system: {len(chatbot.ucs.expert_manager.experts)} experts")
        logger.info(f"   - Auto-save: Every {config.ucs_auto_save_interval}s")
    else:
        logger.info("⚠️ UCS disabled (requires NumPy + sentence-transformers)")

    if KOKORO_AVAILABLE:
        logger.info("🔊 Kokoro TTS enabled")
    
    if VOSK_AVAILABLE:
        logger.info("🎤 Vosk STT enabled")
    
    if config.use_quantization and BNB_AVAILABLE and DEVICE.type == 'cuda':
        logger.info(f"🔬 {config.quantization_bits}-bit quantization enabled")
    elif config.use_quantization:
        logger.warning("⚠️ Quantization requested but not available (requires CUDA + bitsandbytes)")
    else:
        logger.info("❌ Quantization disabled")

    if config.auto_open_browser:
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=config.server_port,
            share=False,
            inbrowser=False,
            show_error=True,
            quiet=False,
            max_threads=1,
            allowed_paths=['/tmp']
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        chatbot.shutdown()

if __name__ == "__main__":
    main()
