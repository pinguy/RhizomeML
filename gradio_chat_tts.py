"""
Enhanced Rhizome Chat with UCS v3.4.1 Integration
Combines the Rhizome reasoning model with UCS cognitive architecture
for memory-augmented, expert-guided conversations.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
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

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    base_dir: str = "./DeepSeek-R1-Distill-Qwen-1.5B-finetuned/"
    vosk_model_path: str = "vosk-model-en-us-0.42-gigaspeech"
    server_port: int = 7860
    auto_open_browser: bool = True
    max_cache_size: int = 100
    max_response_length: int = 512
    tts_max_length: int = 500
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
    ucs_fast_retrieval: bool = True  # Use fast direct retrieval instead of full cognitive loop

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
        if not self.tts_pipeline or len(text) > config.tts_max_length:
            return None
            
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
            max_chunk_size = 200  # characters per chunk
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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

class UCSEnhancedChatBot:
    """
    Rhizome ChatBot enhanced with UCS v3.4.1
    Combines reasoning model with cognitive architecture
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
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
                'repetition_penalty': 1.15,
            }
        ]
    
    def load_models(self) -> bool:
        """Load Rhizome model and UCS system"""
        try:
            # Load language model
            logger.info(f"📄 Loading from {config.base_dir}...")
            checkpoint_path = self._find_latest_checkpoint(config.base_dir)
            
            if not checkpoint_path:
                logger.error(f"No valid model or checkpoint found in {config.base_dir}")
                return False

            logger.info(f"✅ Found latest model/checkpoint at: {checkpoint_path}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_path,
                        trust_remote_code=True
                    )
                    logger.info("✅ Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Tokenizer loading failed: {e}")
                    raise

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        dtype=torch.float16 if DEVICE.type in ['cuda', 'mps'] else torch.float32,
                        device_map={'': DEVICE},
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("✅ Model loaded successfully")
                except Exception as e:
                    logger.error(f"Model loading failed: {e}")
                    raise
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.model.eval()
                logger.info(f"✅ Model loaded on {DEVICE}")
            
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
        if any(f.exists() for f in model_files):
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
            ).to(DEVICE)
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        logger.info("✅ Model pre-warmed")

    def _format_prompt_for_Rhizome(self, user_input: str, system_prompt: Optional[str] = None,
                                   retrieved_context: Optional[str] = None) -> str:
        """
        Format prompt with optional retrieved context from UCS
        """
        formatted = ""
        
        # Use provided system prompt or default
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # Add retrieved context if available
        if retrieved_context:
            formatted += f"<|im_start|>system\nRelevant context from memory:\n{retrieved_context}<|im_end|>\n"
        
        formatted += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        return formatted
    
    def _extract_response_from_output(self, full_output: str, show_reasoning: bool = False) -> str:
        """Extract final response from Rhizome output"""
        if "<|im_start|>assistant" in full_output:
            full_output = full_output.split("<|im_start|>assistant", 1)[-1]
        
        # Handle thinking blocks
        if "<think>" in full_output and "</think>" in full_output:
            think_pattern = r'<think>(.*?)</think>'
            reasoning_blocks = re.findall(think_pattern, full_output, re.DOTALL)
            
            response = re.sub(think_pattern, '', full_output, flags=re.DOTALL)
            
            if show_reasoning and reasoning_blocks:
                reasoning_text = "\n\n".join([f"💭 **Reasoning:**\n{r.strip()}" for r in reasoning_blocks])
                response = f"{reasoning_text}\n\n**Answer:**\n{response.strip()}"
        else:
            response = full_output
        
        # Clean up response
        response = response.strip()
        response = re.sub(r'<\|im.*?\|>', '', response)
        response = re.sub(r'^(User|Assistant):\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n(User|Assistant):\s*.*$', '', response, flags=re.IGNORECASE | re.MULTILINE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        return response.strip()
    
    async def _retrieve_context_from_ucs(self, user_input: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """Retrieve relevant context from UCS memory (lightweight version)"""
        if not self.ucs_enabled or not self.ucs:
            return None, []
        
        try:
            # Fast path: Direct memory retrieval without full cognitive loop
            # This avoids the expensive expert system deliberation
            
            # Check if query needs memory retrieval
            needs_retrieval = any(word in user_input.lower() for word in 
                                 ["remember", "earlier", "before", "you said", "we talked", 
                                  "you mentioned", "last time", "previously"])
            
            if not needs_retrieval and len(user_input.split()) < 15:
                # Short queries without memory keywords - skip retrieval
                return None, []
            
            # Direct vector retrieval (fast)
            query_vec = self.ucs._embed(user_input)
            retrieved_mementos = self.ucs.vmem.retrieve(
                query_vec, 
                top_k=3,  # Reduced from 5 for speed
                use_advanced=True,
                use_cache=True
            )
            
            self.stats['ucs_retrievals'] += 1
            
            if not retrieved_mementos:
                return None, []
            
            # Build context from top mementos
            context_parts = []
            for mid, score in retrieved_mementos:
                if score < 0.5:  # Skip low-relevance results
                    continue
                    
                if mid in self.ucs.vmem.mementos:
                    content = self.ucs.vmem.mementos[mid].get('content', '')
                    if content:
                        # Trim long content
                        content_preview = content[:150] + "..." if len(content) > 150 else content
                        context_parts.append(f"[{score:.2f}] {content_preview}")
            
            context_text = "\n".join(context_parts) if context_parts else None
            
            logger.debug(f"📚 Retrieved {len(retrieved_mementos)} mementos in fast mode")
            
            return context_text, retrieved_mementos
            
        except Exception as e:
            logger.warning(f"UCS retrieval failed: {e}")
            return None, []
    
    def _store_conversation_in_ucs(self, user_input: str, response: str):
        """Store conversation turn in UCS memory"""
        if not self.ucs_enabled or not self.ucs or not self.ucs.vmem:
            return
        
        try:
            # Create conversation memento
            conversation_text = f"User: {user_input}\nAssistant: {response}"
            mid = f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Embed and store
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
                
                # Wait for any pending indexing to complete
                if self.ucs.vmem.use_advanced_search and hasattr(self.ucs.vmem, '_index_queue'):
                    timeout = time.time() + 5
                    while self.ucs.vmem._index_queue.qsize() > 0 and time.time() < timeout:
                        time.sleep(0.2)
                
                # Use JSON format which handles journal IDs better
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
                retrieved_context, retrieved_mementos = await self._retrieve_context_from_ucs(user_input)
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
                system_prompt = (
                    "You are a helpful assistant engaged in natural conversation. "
                    "Use any retrieved context naturally without explicitly mentioning it. "
                    "Stay conversational, witty, and emotionally intelligent."
                )
            
            formatted_input = self._format_prompt_for_Rhizome(
                user_input, 
                system_prompt,
                retrieved_context
            )
            
            with torch_inference_mode():
                inputs = self.tokenizer(
                    formatted_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(DEVICE)
                
                input_length = inputs.input_ids.shape[1]
                
                outputs = self.model.generate(
                    **inputs,
                    **{k: v for k, v in gen_config.items() if k != 'name'},
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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
            
            # Extract and clean response
            response = self._extract_response_from_output(raw_response, show_reasoning)
            
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
        
        # Use creative for complex reasoning questions
        if any(word in normalized_input for word in ['why', 'how', 'explain', 'analyze', 'compare', 'what if']):
            return 1  # creative
        
        # Focused for short queries
        if input_length < 5:
            return 2  # focused
        
        # Default balanced
        return 0
    
    def _get_fallback_response(self, user_input: str) -> Tuple[str, str]:
        """Simple fallback response"""
        fallbacks = [
            "I'm not quite sure how to respond to that. Could you rephrase?",
            "That's an interesting question. Could you provide more context?",
            "I need a bit more information to give you a good answer.",
            "Could you ask that in a different way?"
        ]
        return np.random.choice(fallbacks), "fallback"
    
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
            # Generate response with UCS and custom parameters
            response, method = await self.generate_response_optimized(
                user_input, show_reasoning, use_ucs,
                temperature, top_p, top_k, max_tokens, system_prompt
            )
            
            # Start TTS async
            tts_future = None
            if enable_tts and self.tts_processor:
                # Remove reasoning blocks for TTS
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
                    audio_file = tts_future.result(timeout=10.0)  # Increased timeout for longer text
                except Exception as tts_error:
                    logger.warning(f"TTS timeout or error: {tts_error}")
            
            duration = time.perf_counter() - start_time
            logger.info(f"⚡ Response in {duration:.2f}s ({method})")
            
            return history, "", audio_file
            
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error message in chat
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
        
        stats_report = f"""
📊 **Session Stats:**
- Total: {self.stats['total_responses']}
- Avg time: {avg_time:.2f}s
- Errors: {self.stats['error_count']}
- Device: {DEVICE_INFO}
- TTS: {'Yes' if self.tts_processor else 'No'}
- STT: {'Yes' if self.voice_transcriber else 'No'}

---
🧠 **UCS Integration:**
- Enabled: {'Yes' if self.ucs_enabled else 'No'}
- Mementos: {len(self.ucs.vmem.embeddings) if self.ucs and self.ucs.vmem else 0}
- Retrievals: {self.stats['ucs_retrievals']}
- Expert Calls: {self.stats['ucs_expert_calls']}
"""

        if self.ucs_enabled and self.ucs:
            stats_report += f"- Experts: {len(self.ucs.expert_manager.experts)}\n"
            if self.ucs.vmem:
                cache_stats_ucs = self.ucs.vmem.query_cache.stats()
                stats_report += f"- UCS Cache Hit Rate: {cache_stats_ucs['hit_rate']}\n"

        stats_report += f"""
---
💻 **System:**
- CPU: {system_stats.get('cpu_percent', 0):.1f}%
- Memory: {system_stats.get('memory_percent', 0):.1f}%
- Uptime: {system_stats.get('uptime', 0):.0f}s
"""
        if 'gpu_name' in DEVICE_DETAILS:
            stats_report += f"""
- GPU: {DEVICE_DETAILS['gpu_name']}
- GPU Memory: {system_stats.get('gpu_memory_used', 0):.2f}/{system_stats.get('gpu_memory_total', 0):.2f}GB
"""
        
        stats_report += f"""
---
📦 **Cache:**
- Size: {cache_stats['size']}/{cache_stats['max_size']}
- Hit Rate: {cache_stats['hit_rate']}
{method_stats_str}
        """
        return stats_report
    
    def clear_chat(self):
        """Clear chat history (preserves UCS memory)"""
        self.stats = {
            'total_responses': 0,
            'method_counts': {},
            'error_count': 0,
            'ucs_retrievals': 0,
            'ucs_expert_calls': 0
        }
        self.response_cache.clear()
        self.performance_monitor = PerformanceMonitor()
        self.conversation_history = []
        logger.info("🗑️ Chat cleared")
        return []
    
    def save_ucs_memory(self) -> str:
        """Manually save UCS memory"""
        if not self.ucs_enabled or not self.ucs or not self.ucs.vmem:
            return "❌ UCS not enabled"
        
        try:
            self.ucs.vmem.save_state(config.ucs_save_path)
            return f"✅ Saved {len(self.ucs.vmem.embeddings)} mementos to {config.ucs_save_path}"
        except Exception as e:
            return f"❌ Save failed: {e}"
    
    def clear_ucs_memory(self) -> str:
        """Clear UCS memory (destructive!)"""
        if not self.ucs_enabled or not self.ucs:
            return "❌ UCS not enabled"
        
        try:
            self.ucs._ensure_memory()  # Reinitialize empty memory
            return f"✅ UCS memory cleared"
        except Exception as e:
            return f"❌ Clear failed: {e}"
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down...")
        
        # Save UCS memory
        if self.ucs_enabled and self.ucs:
            try:
                logger.info("💾 Saving UCS memory before shutdown...")
                self.ucs.vmem.save_state(config.ucs_save_path)
                self.ucs.shutdown()
            except Exception as e:
                logger.error(f"UCS shutdown error: {e}")
        
        # Shutdown TTS
        if self.tts_processor:
            self.tts_processor.shutdown()
        
        logger.info("👋 Goodbye!")

# Global initialization
DEVICE, DEVICE_INFO, DEVICE_DETAILS = get_optimal_device_config()
optimize_torch_settings(DEVICE, DEVICE_DETAILS.get('cpu_cores', multiprocessing.cpu_count()))
chatbot = UCSEnhancedChatBot()

def record_and_transcribe(audio_file_path):
    """Transcribe audio file"""
    if audio_file_path is None:
        return "No audio recorded."
    return chatbot.transcribe_voice_input(audio_file_path)

async def process_voice_to_chat(audio_file_path, history, enable_tts, voice_selection, 
                                speed_control, show_reasoning, use_ucs,
                                system_prompt_val, temp_val, top_p_val, top_k_val, max_tokens_val):
    """Transcribe and send to chat"""
    transcribed_text = record_and_transcribe(audio_file_path)
    if transcribed_text and not transcribed_text.startswith("❌") and transcribed_text != "No audio recorded.":
        return await chatbot.chat_response_parallel(
            transcribed_text, history, enable_tts, voice_selection, 
            speed_control, show_reasoning, use_ucs,
            temp_val, top_p_val, int(top_k_val), int(max_tokens_val), system_prompt_val
        )
    else:
        return history, "", None

def shutdown_server():
    """Shutdown server"""
    logger.info("🛑 Shutdown requested...")
    chatbot.shutdown()

    def delayed_shutdown():
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=delayed_shutdown).start()
    return "🛑 Shutting down..."

def create_gradio_interface():
    """Create Gradio interface with UCS controls"""

    available_voices = [
        "af", "af_bella", "af_heart", "af_sky", "af_wave", "af_happy", "af_happy_2", "af_confused",
        "am", "am_adam", "am_michael", "bf", "bf_emma", "bf_isabella", "bm", "bm_george", "bm_lewis"
    ]

    css = """
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', sans-serif;
    }
    .gradio-container .gr-button.primary {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-container .gr-button.secondary {
        background-color: #f0f0f0;
        color: #333;
        border-radius: 8px;
    }
    .gradio-container .gr-button.stop {
        background-color: #f44336;
        color: white;
        border-radius: 8px;
    }
    .ucs-indicator {
        color: #2196F3;
        font-weight: bold;
    }
    """

    with gr.Blocks(css=css, title="UCS-Enhanced Rhizome Chat") as demo:
        gr.Markdown("# 🧠 UCS-Enhanced Rhizome Chat")
        gr.Markdown("Reasoning model with cognitive architecture, vector memory, and expert systems")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="Chat History",
                    show_label=True
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask me anything...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")

                with gr.Row():
                    show_reasoning_checkbox = gr.Checkbox(
                        label="Show Reasoning 💭",
                        value=config.show_reasoning,
                        info="Display the model's thinking process"
                    )
                    use_ucs_checkbox = gr.Checkbox(
                        label="Enable UCS 🧠",
                        value=config.ucs_memory_enabled,
                        info="Use cognitive architecture and memory retrieval"
                    )

                # Generation Parameters
                with gr.Accordion("🎛️ Generation Parameters", open=False):
                    with gr.Row():
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful assistant engaged in natural conversation. Use any retrieved context naturally without explicitly mentioning it. Stay conversational, witty, and emotionally intelligent.",
                            lines=3,
                            placeholder="Enter custom system prompt...",
                            info="Instructions that guide the model's behavior"
                        )
                    
                    with gr.Row():
                        system_prompt_presets = gr.Dropdown(
                            choices=[
                                "Default (Conversational)",
                                "Technical Assistant",
                                "Creative Writer",
                                "Analytical Thinker",
                                "Concise & Direct",
                                "Socratic Teacher",
                                "Custom"
                            ],
                            value="Default (Conversational)",
                            label="System Prompt Presets",
                            scale=3
                        )
                        apply_system_prompt_btn = gr.Button("Apply Preset", size="sm", scale=1)
                    
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.05,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold"
                        )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top K",
                            info="Limits vocabulary to top K tokens"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=768,
                            step=64,
                            label="Max Tokens",
                            info="Maximum response length"
                        )
                    
                    with gr.Row():
                        preset_buttons = gr.Radio(
                            choices=["Balanced", "Creative", "Focused", "Custom"],
                            value="Balanced",
                            label="Generation Presets",
                            info="Quick parameter presets"
                        )
                        reset_params_btn = gr.Button("↺ Reset to Preset", size="sm")

                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Microphone(
                            label="🎙️ Voice Input",
                            type="filepath",
                            interactive=VOSK_AVAILABLE
                        )
                    with gr.Column(scale=2):
                        transcribe_btn = gr.Button(
                            "🎤 Transcribe",
                            variant="secondary",
                            interactive=VOSK_AVAILABLE
                        )
                        voice_to_chat_btn = gr.Button(
                            "🗣️ Voice → Chat",
                            variant="primary",
                            interactive=VOSK_AVAILABLE
                        )

                with gr.Row():
                    enable_tts = gr.Checkbox(
                        label="Enable TTS 🔊",
                        value=KOKORO_AVAILABLE,
                        interactive=KOKORO_AVAILABLE
                    )
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                    shutdown_btn = gr.Button("Shutdown 🛑", variant="stop")

            with gr.Column(scale=1):
                gr.Markdown("### 🎵 Audio Output")
                audio_output = gr.Audio(
                    label="Generated Speech",
                    autoplay=True,
                    show_label=True
                )

                if KOKORO_AVAILABLE:
                    gr.Markdown("### 🎙️ Voice Settings")
                    voice_selection = gr.Dropdown(
                        choices=available_voices,
                        value="af_heart",
                        label="Voice"
                    )

                    speed_control = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                else:
                    voice_selection = gr.Dropdown(choices=["af_heart"], value="af_heart", visible=False)
                    speed_control = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, visible=False)

                gr.Markdown("### 🧠 UCS Memory Control")
                with gr.Row():
                    save_memory_btn = gr.Button("💾 Save Memory", size="sm")
                    clear_memory_btn = gr.Button("🗑️ Clear Memory", size="sm")
                
                memory_status = gr.Markdown("Ready")

                gr.Markdown("### 📊 Statistics")
                stats_display = gr.Markdown(chatbot.get_comprehensive_stats())
                refresh_stats = gr.Button("Refresh Stats 📊", size="sm")
                
                gr.Markdown("### 🛑 Server Control")
                shutdown_status = gr.Markdown("Server running")

                gr.Markdown("### ℹ️ Information")
                tts_status = "✅ Kokoro TTS" if KOKORO_AVAILABLE else "❌ Install: pip install kokoro>=0.9.4"
                vosk_status = "✅ Vosk STT" if VOSK_AVAILABLE else "❌ Install: pip install vosk"
                ucs_status = "✅ UCS v3.4.1" if chatbot.ucs_enabled else "❌ Limited (NumPy required)"
                
                gr.Markdown(f"""
**Model:** {config.base_dir}
**Device:** {DEVICE_INFO}
**TTS:** {tts_status}
**STT:** {vosk_status}
**UCS:** {ucs_status}

**Features:**
- 🧠 UCS v3.4.1 cognitive architecture
- 💭 Chain-of-thought reasoning
- 📚 Vector memory & retrieval
- 👥 Multi-expert system
- 🎯 Context-aware responses
- 🎤 Voice input/output
- 🎛️ Adjustable generation parameters
- 📝 Customizable system prompts

**System Prompt Presets:**
- **Conversational**: Natural, friendly chat
- **Technical**: Precise, detailed explanations
- **Creative**: Vivid, imaginative writing
- **Analytical**: Structured, logical reasoning
- **Concise**: Brief, direct responses
- **Socratic**: Teaching through questions

**Generation Presets:**
- **Balanced** (default): Reliable, coherent responses
- **Creative**: More diverse and imaginative
- **Focused**: Precise and on-topic
- **Custom**: Manual parameter control

**Tips:**
- Choose a system prompt preset to shape personality
- Enable "Enable UCS" for memory-augmented responses
- Ask about previous conversations - UCS remembers!
- Enable "Show Reasoning" to see thought process
- Try different presets for varied response styles
- Custom system prompts allow full control
- UCS auto-saves memory every 5 minutes
- TTS now handles longer responses with line breaks
                """)

        # Event handlers
        async def handle_chat(user_input_text, history, enable_tts_val, voice_val, 
                            speed_val, show_reasoning_val, use_ucs_val,
                            system_prompt_val, temp_val, top_p_val, top_k_val, max_tokens_val):
            return await chatbot.chat_response_parallel(
                user_input_text, history, enable_tts_val, voice_val, 
                speed_val, show_reasoning_val, use_ucs_val,
                temp_val, top_p_val, int(top_k_val), int(max_tokens_val), system_prompt_val
            )
        
        def apply_system_prompt_preset(preset_name):
            """Apply system prompt presets"""
            prompts = {
                "Default (Conversational)": "You are a helpful assistant engaged in natural conversation. Use any retrieved context naturally without explicitly mentioning it. Stay conversational, witty, and emotionally intelligent.",
                "Technical Assistant": "You are a technical expert who provides clear, accurate, and detailed explanations. Focus on precision, best practices, and thorough analysis. Use technical terminology appropriately and provide examples when helpful.",
                "Creative Writer": "You are a creative writing assistant with a flair for vivid descriptions, engaging narratives, and imaginative storytelling. Help craft compelling content with rich language and strong emotional resonance.",
                "Analytical Thinker": "You are an analytical assistant who breaks down complex problems systematically. Provide structured reasoning, consider multiple perspectives, and use logic-driven analysis. Show your thought process step-by-step.",
                "Concise & Direct": "You are a concise assistant who gets straight to the point. Provide brief, clear, and actionable responses. Avoid unnecessary elaboration while maintaining accuracy.",
                "Socratic Teacher": "You are a Socratic teacher who guides learning through thoughtful questions. Help users discover answers themselves by asking probing questions, encouraging critical thinking, and building understanding progressively.",
                "Custom": ""
            }
            return prompts.get(preset_name, prompts["Default (Conversational)"])
        
        def apply_preset(preset_name):
            """Apply generation parameter presets"""
            presets = {
                "Balanced": (0.8, 0.95, 50, 768),
                "Creative": (0.95, 0.95, 60, 800),
                "Focused": (0.7, 0.85, 40, 512),
                "Custom": (0.8, 0.95, 50, 768)  # Default for custom
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

        # Wire up events
        send_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, 
                   speed_control, show_reasoning_checkbox, use_ucs_checkbox,
                   system_prompt_input, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider],
            outputs=[chatbot_interface, user_input, audio_output]
        )

        user_input.submit(
            fn=handle_chat,
            inputs=[user_input, chatbot_interface, enable_tts, voice_selection, 
                   speed_control, show_reasoning_checkbox, use_ucs_checkbox,
                   system_prompt_input, temperature_slider, top_p_slider, top_k_slider, max_tokens_slider],
            outputs=[chatbot_interface, user_input, audio_output]
        )
        
        # Preset button handler
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
        
        # System prompt preset handler
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
            fn=process_voice_to_chat,
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
    logger.info("🚀 Starting UCS-Enhanced Rhizome Chat Interface...")
    
    if not chatbot.load_models():
        logger.error("❌ Failed to initialize. Check your model path.")
        logger.error(f"   Make sure '{config.base_dir}' contains your model files or checkpoint folders.")
        return

    demo = create_gradio_interface()

    logger.info("\n✅ Ready! Starting web interface...")
    logger.info(f"🌐 Access at: http://localhost:{config.server_port}")
    logger.info("🔓 Running in UNRESTRICTED mode - no content filtering")
    
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
