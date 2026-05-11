import os
# Replace "DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL" with any CAUSAL_LM model you want to finetune from HF. GTX 1660 Ti with 6GB of VRAM is able to finefune models 3b and under.
# CRITICAL: Handle Memory Fragmentation before Torch loads
# This helps with "reserved but unallocated" memory issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress PyTorch cpp_extension CUDA warning when running on CPU
import logging
logging.getLogger("torch.utils.cpp_extension").setLevel(logging.ERROR)

try:
    config
except NameError:
    # Replace 'YOUR_HF_TOKEN_HERE' with your actual token. https://huggingface.co/settings/tokens
    config = {
        "HF_TOKEN": "YOUR_HF_TOKEN_HERE",
    }

os.environ["HF_TOKEN"] = config["HF_TOKEN"]

# --- Hugging Face download stability (disable CAS/Xet) ---
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Optional but recommended for slow/unstable links
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"

import torch
import json
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg', force=True)  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from tqdm import tqdm
import time
import numpy as np
import multiprocessing
import psutil
import random
from datetime import datetime
from collections import Counter, defaultdict
from torch.utils.data import WeightedRandomSampler

# Add safe globals for numpy reconstruct (fix deprecation warning)
import torch.serialization
try:
    # Try new numpy namespace first
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except (ImportError, AttributeError):
    # Fall back to old namespace
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

# -------------------------------------------------------------------------
# Theme / Training Constants
# -------------------------------------------------------------------------
THEME_COVERAGE_STOP_THRESHOLD = 0.82
MAX_THEME_WEIGHT = 20.0

# CRITICAL: Force CPU-only mode by setting CUDA_VISIBLE_DEVICES BEFORE any torch imports
def force_cpu_only():
    """Force CPU-only mode by hiding all CUDA devices"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_AVAILABLE_DEVICES"] = ""

# Configure clean logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and clean output"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging():
    """Configure clean, colorful logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        fmt='%(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Suppress noisy library logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("tokenizers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)
    logging.getLogger("deepspeed").setLevel(logging.ERROR)
    logging.getLogger("real_accelerator").setLevel(logging.ERROR)

    return logging.getLogger(__name__)

logger = setup_logging()

# Global variables for device configuration
DEVICE = None
DEVICE_INFO = "Not initialized"
DEVICE_DETAILS = {}
USE_CPU_ONLY = False
USE_QLORA = False  # Will be set based on device

def get_system_memory_info():
    """Get system memory information"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent_used': memory.percent,
        'free_gb': memory.free / (1024**3)
    }

def calculate_safe_num_proc():
    """Calculate safe number of processes based on available RAM"""
    memory_info = get_system_memory_info()
    cpu_count = multiprocessing.cpu_count()

    # Estimate ~1-2GB per process for tokenization (conservative)
    memory_per_process_gb = 1.5

    # Calculate max processes based on available memory
    max_processes_by_memory = max(1, int(memory_info['available_gb'] / memory_per_process_gb))

    # Use conservative approach: min of CPU count-1 and memory-limited processes
    safe_processes = min(cpu_count - 1, max_processes_by_memory, 8)  # Cap at 8 for safety
    safe_processes = max(1, safe_processes)  # Ensure at least 1

    logger.info(f"💾 System memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
    logger.info(f"🔧 Using {safe_processes} processes for tokenization (CPU cores: {cpu_count}, Memory-safe limit: {max_processes_by_memory})")

    return safe_processes

def check_avx2_support():
    """Check if CPU supports AVX2 instructions"""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        has_avx2 = 'avx2' in info.get('flags', [])
        logger.info(f"🔍 CPU AVX2 support: {'✅ Available' if has_avx2 else '❌ Not available'}")
        return has_avx2
    except:
        logger.warning("⚠️ Could not detect AVX2 support, assuming available")
        return True

def get_gpu_info() -> dict:
    """Get detailed GPU information including compute capability."""
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info['name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info['memory_total'] = props.total_memory / (1024**3)  # GB
            gpu_info['multiprocessor_count'] = props.multi_processor_count
            gpu_info['max_threads_per_multiprocessor'] = props.max_threads_per_multi_processor
            gpu_info['compute_capability'] = f"{props.major}.{props.minor}"
            gpu_info['compute_capability_major'] = props.major
            gpu_info['compute_capability_minor'] = props.minor
            gpu_info['is_rtx'] = 'RTX' in gpu_info['name'].upper()
            compute_capability_numeric = props.major + (props.minor / 10.0)
            gpu_info['is_supported'] = compute_capability_numeric >= 6.0
            gpu_info['is_modern'] = compute_capability_numeric >= 7.0
            if compute_capability_numeric < 3.5:
                gpu_info['classification'] = "Very Old (Pre-Kepler)"
                gpu_info['performance_expectation'] = "Not supported by PyTorch"
            elif compute_capability_numeric < 5.0:
                gpu_info['classification'] = "Old (Kepler)"
                gpu_info['performance_expectation'] = "Limited PyTorch support, likely slower than modern CPU"
            elif compute_capability_numeric < 6.0:
                gpu_info['classification'] = "Legacy (Maxwell)"
                gpu_info['performance_expectation'] = "Deprecated in modern PyTorch, CPU likely faster"
            else:
                gpu_info['classification'] = "Modern (Turing/Ampere/Ada)"
                gpu_info['performance_expectation'] = "Excellent performance"
        except Exception as e:
            gpu_info['error'] = str(e)
            logger.warning(f"Warning: Could not get full GPU info: {e}")
    return gpu_info

def check_pytorch_cuda_compatibility() -> tuple[bool, str]:
    """Check if CUDA is actually working with PyTorch."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        test_tensor = torch.randn(10, 10).cuda()
        result = test_tensor @ test_tensor.T
        result = result.cpu()
        del test_tensor, result
        torch.cuda.empty_cache()
        return True, "CUDA working correctly"
    except Exception as e:
        error_msg = str(e).lower()
        if "no longer supports" in error_msg or "too old" in error_msg:
            return False, f"GPU too old for PyTorch: {e}"
        elif "out of memory" in error_msg:
            return False, f"GPU out of memory: {e}"
        else:
            return False, f"CUDA error: {e}"

def apply_cpu_optimizations():
    """Apply aggressive CPU optimizations when forced to use CPU"""
    cpu_count = multiprocessing.cpu_count()
    optimal_threads = max(1, cpu_count - 1)
    logger.info("⚡ Applying CPU optimizations...")
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(4)
    os.environ.update({
        "OMP_NUM_THREADS": str(optimal_threads),
        "MKL_NUM_THREADS": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "KMP_BLOCKTIME": "1",
    })
    logger.info(f"  ✓ Thread count: {optimal_threads} (interop: 4)")
    try:
        if hasattr(torch, 'bfloat16'):
            test = torch.randn(2, 2, dtype=torch.bfloat16)
            _ = test @ test
            if not USE_QLORA:
                torch.set_default_dtype(torch.bfloat16)
                logger.info(f"  ✓ BF16 enabled on CPU (performance boost)")
                return True, optimal_threads
            else:
                logger.info(f"  ℹ️ BF16 available but QLoRA manages its own dtypes")
                return False, optimal_threads
    except:
        logger.info(f"  ℹ️ BF16 not available, using FP32")
    try:
        if hasattr(torch.backends, 'cpu'):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cpu.enable_sdp(True)
            torch.backends.cpu.enable_mem_efficient_sdp(True)
            torch.backends.cpu.enable_math_sdp(True)
            logger.info(f"  ✓ CPU FlashAttention enabled")
    except:
        logger.info(f"  ℹ️ CPU FlashAttention not available (PyTorch < 2.2)")
    return False, optimal_threads

def detect_optimal_device():
    """Intelligently detect the optimal device with proper GPU support checking."""
    global DEVICE, DEVICE_INFO, DEVICE_DETAILS, USE_CPU_ONLY, USE_QLORA
    device_selected = "cpu"
    device_info_str = "CPU (default)"
    all_info = {}
    USE_CPU_ONLY = True
    USE_QLORA = False
    cpu_count = multiprocessing.cpu_count()
    all_info['cpu_cores'] = cpu_count
    all_info['has_avx2'] = check_avx2_support()
    gpu_info = get_gpu_info()
    all_info.update(gpu_info)
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU detected: {gpu_info.get('name', 'Unknown')}")
        logger.info(f"📊 GPU compute capability: {gpu_info.get('compute_capability', 'Unknown')}")
        logger.info(f"🏷️ GPU classification: {gpu_info.get('classification', 'Unknown')}")
        logger.info(f"💾 GPU memory: {gpu_info.get('memory_total', 0):.1f}GB")
        is_supported = gpu_info.get('is_supported', False)
        if not is_supported:
            reason = f"GPU compute capability {gpu_info.get('compute_capability', 'unknown')} is below PyTorch minimum requirement (6.0)"
            logger.warning(f"⚠️ Forcing CPU: {reason}")
            device_info_str = f"CPU: {cpu_count} cores (GPU {gpu_info.get('name', 'Unknown')} unsupported - {reason})"
            all_info['decision_reason'] = reason
            USE_CPU_ONLY = True
        else:
            cuda_works, cuda_message = check_pytorch_cuda_compatibility()
            if not cuda_works:
                logger.warning(f"⚠️ Forcing CPU: {cuda_message}")
                device_info_str = f"CPU: {cpu_count} cores (GPU CUDA failed - {cuda_message})"
                all_info['decision_reason'] = cuda_message
                USE_CPU_ONLY = True
            else:
                device_selected = "cuda"
                USE_CPU_ONLY = False
                USE_QLORA = True
                device_info_str = f"GPU: {gpu_info.get('name', 'Unknown')} ({gpu_info.get('memory_total', 0):.1f}GB)"
                all_info['decision_reason'] = "GPU available and working"
                logger.info(f"🚀 GPU is ready! Will use CUDA with QLoRA (4-bit) for training.")
    else:
        device_info_str = f"CPU: {cpu_count} cores (CUDA not available)"
        all_info['decision_reason'] = "CUDA not available"
        USE_CPU_ONLY = True
    if USE_CPU_ONLY:
        force_cpu_only()
        device_selected = "cpu"
        if all_info.get('has_avx2', False):
            USE_QLORA = True
            logger.info("✅ AVX2 detected - QLoRA 4-bit quantization available on CPU!")
        else:
            USE_QLORA = False
            logger.warning("⚠️ AVX2 not available - QLoRA disabled on CPU")
    DEVICE = torch.device(device_selected)
    DEVICE_INFO = device_info_str
    DEVICE_DETAILS = all_info
    if DEVICE.type == "cuda" and not USE_CPU_ONLY:
        logger.info("⚡ Configuring GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info(f"✅ GPU acceleration enabled: {DEVICE_INFO}")
    else:
        uses_bf16, optimal_threads = apply_cpu_optimizations()
        DEVICE_DETAILS['cpu_threads_used'] = optimal_threads
        DEVICE_DETAILS['uses_bf16'] = uses_bf16
        DEVICE_INFO += f" ({optimal_threads} threads"
        if uses_bf16:
            DEVICE_INFO += ", BF16"
        DEVICE_INFO += ")"
    logger.info(f"🎯 Final device: {DEVICE_INFO}")
    logger.info(f"📝 Decision reason: {DEVICE_DETAILS.get('decision_reason', 'No specific reason')}")
    logger.info(f"🖥️ CPU-only mode: {USE_CPU_ONLY}")
    logger.info(f"🔬 QLoRA enabled: {USE_QLORA}")

detect_optimal_device()

if not USE_CPU_ONLY:
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "true",
        "PYTHONIOENCODING": "utf-8",
        "TRANSFORMERS_VERBOSITY": "error",
        "DATASETS_VERBOSITY": "error"
    })

torch.multiprocessing.set_start_method('spawn', force=True)

def get_model_lora_targets(model):
    model_type = getattr(model.config, 'model_type', '').lower()
    logger.info(f"🔍 Auto-discovering LoRA targets for model type: {model_type}")
    forbidden_keywords = {
        'lm_head', 'output_layer', 'embed_tokens', 'wte', 'wpe', 'shared',
        'score', 'classifier', 'predictions', 'output'
    }
    target_modules = set()
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if 'Linear' in module_type or 'Conv1D' in module_type:
            parts = name.split('.')
            leaf_name = parts[-1]
            if leaf_name in forbidden_keywords:
                continue
            target_modules.add(leaf_name)
    final_targets = list(target_modules)
    if not final_targets:
        logger.warning("⚠️ No linear layers found automatically. Falling back to standard defaults.")
        final_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.info(f"✅ Agnostic LoRA Targets Found: {final_targets}")
    return final_targets

def determine_fan_in_fan_out(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    if any(x in model_name_lower for x in ["deepseek", "qwen", "qwen2"]):
        logger.info(f"🔧 Setting fan_in_fan_out=False for {model_name} (DeepSeek/Qwen architecture)")
        return False
    if "falcon" in model_name_lower:
        logger.info(f"🔧 Setting fan_in_fan_out=True for {model_name} (Falcon architecture)")
        return True
    logger.info(f"🔧 Setting fan_in_fan_out=False for {model_name} (default for modern architectures)")
    return False

# ============================================================================
# Semantic Theme Utilities
# ============================================================================

def extract_themes_from_metadata(metadata):
    if not metadata:
        return ["general"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            return ["general"]
    if not isinstance(metadata, dict):
        return ["general"]
    candidates = (
        metadata.get("themes")
        or metadata.get("semantic_themes")
        or metadata.get("primary_theme")
    )
    if not candidates:
        user_msg = metadata.get("user_msg", {})
        assistant_msg = metadata.get("assistant_msg", {})
        candidates = (
            user_msg.get("semantic_themes")
            or assistant_msg.get("semantic_themes")
        )
    if not candidates:
        return ["general"]
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return ["general"]
    STOP_THEMES = {"like", "want", "let", "get", "general", "thing", "stuff"}
    cleaned = [str(x).strip() for x in candidates if str(x).strip()]
    cleaned = [x for x in cleaned if x not in STOP_THEMES]
    return cleaned if cleaned else ["general"]

class ThemeTracker:
    def __init__(self, theme_distribution: Dict[str, int]):
        self.global_theme_dist = theme_distribution
        self.total_themes = sum(theme_distribution.values())
        self.theme_weights = {}
        for theme, count in theme_distribution.items():
            weight = np.sqrt(self.total_themes / max(count, 1))
            weight = min(weight, MAX_THEME_WEIGHT)
            self.theme_weights[theme] = float(weight)
        self.eval_theme_counts = Counter()
        self.training_theme_counts = Counter()
        logger.info(f"🎨 ThemeTracker initialized with {len(theme_distribution)} unique themes")
        logger.info(f"📊 Total theme occurrences: {self.total_themes:,}")
        sorted_themes = sorted(theme_distribution.items(), key=lambda x: x[1], reverse=True)
        logger.info("🔝 Top 5 most common themes:")
        for theme, count in sorted_themes[:5]:
            logger.info(f"   • {theme}: {count} ({100*count/self.total_themes:.1f}%)")
        logger.info("🔻 Bottom 5 rarest themes:")
        for theme, count in sorted_themes[-5:]:
            logger.info(f"   • {theme}: {count} ({100*count/self.total_themes:.1f}%)")

    def get_sample_weight(self, themes: List[str]) -> float:
        if not themes:
            return 1.0
        weights = [self.theme_weights.get(theme, 1.0) for theme in themes]
        return sum(weights) / len(weights)

    def record_batch_themes(self, batch_themes: List[List[str]], is_training: bool = True):
        counter = self.training_theme_counts if is_training else self.eval_theme_counts
        for themes in batch_themes:
            for theme in themes:
                counter[theme] += 1

    def get_diversity_metrics(self, is_training: bool = True) -> Dict[str, float]:
        counter = self.training_theme_counts if is_training else self.eval_theme_counts
        if not counter:
            return {'unique_themes': 0, 'entropy': 0.0, 'coverage': 0.0, 'total_occurrences': 0}
        total = sum(counter.values())
        unique = len(counter)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        coverage = unique / len(self.global_theme_dist) if len(self.global_theme_dist) > 0 else 0.0
        return {
            'unique_themes': unique,
            'entropy': float(entropy),
            'coverage': float(coverage),
            'total_occurrences': total
        }

def create_theme_weighted_sampler(dataset, theme_tracker: ThemeTracker) -> Optional[WeightedRandomSampler]:
    try:
        weights = []
        missing_metadata = 0
        for example in dataset:
            themes = extract_themes_from_metadata(example.get("source_metadata"))
            weight = theme_tracker.get_sample_weight(themes)
            weights.append(weight)
            if themes == ["general"]:
                missing_metadata += 1
        if missing_metadata > 0:
            logger.warning(f"⚠️ {missing_metadata}/{len(dataset)} examples missing theme metadata")
        logger.info(f"📊 Sample weights - min: {min(weights):.3f}, max: {max(weights):.3f}, mean: {np.mean(weights):.3f}")
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    except Exception as e:
        logger.warning(f"⚠️ Could not create weighted sampler: {e}")
        return None

# ============================================================================
# Theme-Aware Trainer
# ============================================================================

class ThemeAwareTrainer(Trainer):
    def __init__(self, *args, theme_tracker: Optional[ThemeTracker] = None,
                 original_dataset: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.theme_tracker = theme_tracker
        self.original_dataset = original_dataset
        self.original_dataset_size = len(original_dataset) if original_dataset else 0
        self._last_logged_step = -1
        self._last_coverage_check_step = -1
        if not theme_tracker:
            logger.warning("ThemeAwareTrainer initialized without a ThemeTracker!")
        if not original_dataset:
            logger.warning("ThemeAwareTrainer initialized without an original_dataset!")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if self.theme_tracker and self.original_dataset and self.original_dataset_size > 0:
            try:
                batch_size = inputs["input_ids"].shape[0]
                random_indices = np.random.randint(0, self.original_dataset_size, size=batch_size)
                sampled_examples = self.original_dataset.select(random_indices)
                batch_themes = [extract_themes_from_metadata(ex.get("source_metadata")) for ex in sampled_examples]
                self.theme_tracker.record_batch_themes(batch_themes, is_training=True)
                metrics = self.theme_tracker.get_diversity_metrics(is_training=True)
                current_step = self.state.global_step
                should_log = (current_step > 0 and current_step % 100 == 0 and current_step != self._last_logged_step)
                if should_log:
                    self._last_logged_step = current_step
                    self.log({
                        "train_theme_entropy": metrics['entropy'],
                        "train_theme_coverage": metrics['coverage'],
                        "train_unique_themes": metrics['unique_themes']
                    })
                if current_step != self._last_coverage_check_step:
                    coverage = metrics['coverage']
                    should_check = False
                    if coverage >= THEME_COVERAGE_STOP_THRESHOLD:
                        should_check = True
                    elif coverage >= 0.70:
                        should_check = (current_step % 10 == 0)
                    elif coverage >= 0.65:
                        should_check = (current_step % 25 == 0)
                    else:
                        should_check = (current_step % 100 == 0)
                    if should_check:
                        self._last_coverage_check_step = current_step
                        if coverage >= THEME_COVERAGE_STOP_THRESHOLD:
                            logger.info("\n" + "=" * 70)
                            logger.info(f"🎯 THEME COVERAGE REACHED {THEME_COVERAGE_STOP_THRESHOLD:.0%} (EMPIRICAL OVERFITTING THRESHOLD)")
                            logger.info("🛑 STOPPING TRAINING EARLY")
                            logger.info("=" * 70)
                            logger.info(f"   • Step: {current_step}")
                            logger.info(f"   • Epoch: {self.state.epoch:.2f}")
                            logger.info(f"   • Coverage: {coverage:.1%}")
                            logger.info(f"   • Unique themes: {metrics['unique_themes']}")
                            logger.info(f"   • Total known themes: {len(self.theme_tracker.global_theme_dist)}")
                            logger.info(f"   • Shannon entropy: {metrics['entropy']:.3f}")
                            logger.info("=" * 70 + "\n")
                            self.control.should_training_stop = True
            except Exception as e:
                logger.warning(f"⚠️ Error during theme tracking in training_step: {e}", exc_info=False)
        return loss

# TrainingLogger class is unchanged, but it's kept for completeness
class TrainingLogger(TrainerCallback):
    def __init__(self, output_dir, theme_tracker: Optional[ThemeTracker] = None, checkpoint_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.theme_tracker = theme_tracker
        self.metrics = {
            'step': [], 'epoch': [], 'train_loss': [], 'eval_loss': [], 'learning_rate': [],
            'grad_norm': [], 'train_runtime': [], 'train_samples_per_second': [], 'train_steps_per_second': [],
            'train_theme_diversity': [], 'train_theme_coverage': [], 'eval_theme_diversity': [], 'eval_theme_coverage': [],
        }
        self.start_time = time.time()
        self.resumed_from_checkpoint = False
        self.checkpoint_start_step = 0
        self.checkpoint_start_time = None
        self._step_to_idx = {}
        self._last_speed_calc_time = None
        self._last_speed_calc_step = 0
        metrics_file = None
        main_metrics = self.output_dir / "training_metrics.json"
        if main_metrics.exists():
            metrics_file = main_metrics
        else:
            checkpoint_dirs = sorted(
                [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[-1]), reverse=True
            ) if self.output_dir.exists() else []
            for ckpt_dir in checkpoint_dirs:
                ckpt_metrics = ckpt_dir / "training_metrics.json"
                if ckpt_metrics.exists():
                    metrics_file = ckpt_metrics
                    break
        if metrics_file and metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                for key in self.metrics:
                    if key in saved_metrics and isinstance(saved_metrics[key], list):
                        self.metrics[key] = saved_metrics[key]
                self.resumed_from_checkpoint = True
                if self.metrics['step']:
                    self.checkpoint_start_step = max(self.metrics['step'])
                    self._step_to_idx = {step: idx for idx, step in enumerate(self.metrics['step'])}
                    base_length = len(self.metrics['step'])
                    for key in self.metrics:
                        if key not in ['step', 'epoch']:
                            while len(self.metrics[key]) < base_length:
                                self.metrics[key].append(None)
                logger.info(f"📊 Loaded previous metrics: {len(self.metrics['step'])} steps up to step {self.checkpoint_start_step}")
            except Exception as e:
                logger.warning(f"Could not load previous metrics: {e}")
        if checkpoint_path and not self.resumed_from_checkpoint:
            try:
                ckpt_name = Path(checkpoint_path).name
                if ckpt_name.startswith("checkpoint-"):
                    step = int(ckpt_name.split("-")[-1])
                    self.resumed_from_checkpoint = True
                    self.checkpoint_start_step = step
                    logger.info(f"📊 Detected resume from checkpoint at step {step} (no metrics file found)")
            except (ValueError, AttributeError):
                pass

    def set_checkpoint_info(self, checkpoint_path: str):
        if checkpoint_path and not self.resumed_from_checkpoint:
            try:
                ckpt_name = Path(checkpoint_path).name
                if ckpt_name.startswith("checkpoint-"):
                    step = int(ckpt_name.split("-")[-1])
                    self.resumed_from_checkpoint = True
                    self.checkpoint_start_step = step
                    logger.info(f"📊 Updated: will resume from step {step}")
            except (ValueError, AttributeError):
                pass

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resumed_from_checkpoint and self.checkpoint_start_step > 0:
            self.checkpoint_start_time = time.time()
            self._last_speed_calc_time = self.checkpoint_start_time
            self._last_speed_calc_step = self.checkpoint_start_step
            logger.info(f"🔄 Resuming training from step {self.checkpoint_start_step}")
        else:
            self.start_time = time.time()
            self.checkpoint_start_time = self.start_time
            self._last_speed_calc_time = self.start_time
            self._last_speed_calc_step = 0
            logger.info("🆕 Starting fresh training run")

    def _get_or_create_step_index(self, step, epoch):
        if step in self._step_to_idx:
            return self._step_to_idx[step]
        idx = len(self.metrics['step'])
        self.metrics['step'].append(step)
        self.metrics['epoch'].append(epoch)
        self._step_to_idx[step] = idx
        for key in self.metrics:
            if key not in ['step', 'epoch']:
                while len(self.metrics[key]) < len(self.metrics['step']):
                    self.metrics[key].append(None)
        return idx

    def _set_metric(self, key, idx, value):
        while len(self.metrics[key]) <= idx:
            self.metrics[key].append(None)
        self.metrics[key][idx] = value

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
        current_step = state.global_step
        current_epoch = state.epoch
        idx = self._get_or_create_step_index(current_step, current_epoch)
        if 'loss' in logs:
            self._set_metric('train_loss', idx, logs['loss'])
        if 'eval_loss' in logs:
            self._set_metric('eval_loss', idx, logs['eval_loss'])
        if 'learning_rate' in logs:
            self._set_metric('learning_rate', idx, logs['learning_rate'])
        if 'grad_norm' in logs:
            self._set_metric('grad_norm', idx, logs['grad_norm'])
        if 'train_theme_entropy' in logs:
            self._set_metric('train_theme_diversity', idx, logs['train_theme_entropy'])
        if 'train_theme_coverage' in logs:
            self._set_metric('train_theme_coverage', idx, logs['train_theme_coverage'])
        current_time = time.time()
        if self._last_speed_calc_time and current_step > self._last_speed_calc_step:
            elapsed = current_time - self._last_speed_calc_time
            steps_done = current_step - self._last_speed_calc_step
            if elapsed > 0:
                steps_per_second = steps_done / elapsed
                effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
                samples_per_second = steps_per_second * effective_batch
                self._set_metric('train_steps_per_second', idx, steps_per_second)
                self._set_metric('train_samples_per_second', idx, samples_per_second)
        self._last_speed_calc_time = current_time
        self._last_speed_calc_step = current_step
        if 'train_runtime' in logs:
            self._set_metric('train_runtime', idx, logs['train_runtime'])

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self.theme_tracker and metrics:
            diversity_metrics = self.theme_tracker.get_diversity_metrics(is_training=False)
            logger.info(f"🎨 Eval Theme Diversity:")
            logger.info(f"   • Unique themes: {diversity_metrics['unique_themes']}")
            logger.info(f"   • Entropy: {diversity_metrics['entropy']:.3f}")
            logger.info(f"   • Coverage: {diversity_metrics['coverage']:.1%}")
            idx = self._get_or_create_step_index(state.global_step, state.epoch if state.epoch else 0)
            self._set_metric('eval_theme_diversity', idx, diversity_metrics['entropy'])
            self._set_metric('eval_theme_coverage', idx, diversity_metrics['coverage'])

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        self.save_metrics_and_plots(self.output_dir)
        self.save_metrics_and_plots(checkpoint_dir)
        if self.theme_tracker:
            theme_state_path = checkpoint_dir / 'theme_tracker_state.json'
            theme_state = {
                'training_themes': dict(self.theme_tracker.training_theme_counts),
                'eval_themes': dict(self.theme_tracker.eval_theme_counts),
                'diversity_metrics': self.theme_tracker.get_diversity_metrics(is_training=True)
            }
            with open(theme_state_path, 'w') as f:
                json.dump(theme_state, f, indent=2)
            logger.info(f"🎨 Saved theme tracker state to {theme_state_path}")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.save_metrics_and_plots(self.output_dir, final=True)
        if self.theme_tracker:
            logger.info("\n" + "="*70)
            logger.info("🎨 FINAL THEME DIVERSITY REPORT")
            logger.info("="*70)
            train_metrics = self.theme_tracker.get_diversity_metrics(is_training=True)
            logger.info(f"Training Data:")
            logger.info(f"  • Unique themes seen: {train_metrics['unique_themes']}")
            logger.info(f"  • Shannon entropy: {train_metrics['entropy']:.3f}")
            logger.info(f"  • Theme coverage: {train_metrics['coverage']:.1%}")
            logger.info(f"  • Total occurrences: {train_metrics['total_occurrences']:,}")
            eval_metrics = self.theme_tracker.get_diversity_metrics(is_training=False)
            if eval_metrics['unique_themes'] > 0:
                logger.info(f"\nValidation Data:")
                logger.info(f"  • Unique themes seen: {eval_metrics['unique_themes']}")
                logger.info(f"  • Shannon entropy: {eval_metrics['entropy']:.3f}")
                logger.info(f"  • Theme coverage: {eval_metrics['coverage']:.1%}")
                logger.info(f"  • Total occurrences: {eval_metrics['total_occurrences']:,}")
            logger.info("="*70 + "\n")

    def save_metrics_and_plots(self, save_dir, final=False):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cleaned_metrics = {}
        base_length = len(self.metrics['step'])
        for key, values in self.metrics.items():
            if key == 'step':
                cleaned_metrics[key] = values
            else:
                if len(values) < base_length:
                    values.extend([None] * (base_length - len(values)))
                elif len(values) > base_length:
                    values = values[:base_length]
                cleaned_metrics[key] = values
        metrics_file = save_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(cleaned_metrics, f, indent=2)
        self.create_training_plots(save_dir, cleaned_metrics, final)
        if final:
            logger.info(f"📊 Final training metrics saved to: {save_dir}")

    def create_training_plots(self, save_dir, metrics, final=False):
        plt.style.use('default')
        plt.rcParams.update({'figure.figsize': (15, 12), 'font.size': 10, 'axes.linewidth': 1, 'axes.grid': True, 'grid.alpha': 0.3})
        steps = metrics['step']
        epochs = metrics['epoch']
        has_semantic = any(metrics.get('eval_theme_diversity', [None])) or any(metrics.get('train_theme_diversity', [None]))
        n_rows = 3 if has_semantic else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        fig.suptitle(f'Training Progress {"(Final)" if final else "(Checkpoint)"}', fontsize=16, fontweight='bold')
        ax1 = axes[0, 0]
        train_losses = [x for x in metrics['train_loss'] if x is not None]
        eval_losses = [x for x in metrics['eval_loss'] if x is not None]
        if train_losses:
            train_steps = [steps[i] for i, x in enumerate(metrics['train_loss']) if x is not None]
            ax1.plot(train_steps, train_losses, 'b-', label='Training Loss', linewidth=2)
        if eval_losses:
            eval_steps = [steps[i] for i, x in enumerate(metrics['eval_loss']) if x is not None]
            ax1.plot(eval_steps, eval_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Steps'); ax1.set_ylabel('Loss'); ax1.set_title('Training & Validation Loss')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        learning_rates = [x for x in metrics['learning_rate'] if x is not None]
        if learning_rates:
            lr_steps = [steps[i] for i, x in enumerate(metrics['learning_rate']) if x is not None]
            ax2.plot(lr_steps, learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Steps'); ax2.set_ylabel('Learning Rate'); ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax3 = axes[0, 2]
        grad_norms = [x for x in metrics['grad_norm'] if x is not None]
        if grad_norms:
            grad_steps = [steps[i] for i, x in enumerate(metrics['grad_norm']) if x is not None]
            ax3.plot(grad_steps, grad_norms, 'orange', linewidth=2)
        ax3.set_xlabel('Steps'); ax3.set_ylabel('Gradient Norm'); ax3.set_title('Gradient Norm')
        ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 0]
        samples_per_sec = [x for x in metrics['train_samples_per_second'] if x is not None]
        if samples_per_sec:
            speed_steps = [steps[i] for i, x in enumerate(metrics['train_samples_per_second']) if x is not None]
            ax4.plot(speed_steps, samples_per_sec, 'purple', linewidth=2)
        ax4.set_xlabel('Steps'); ax4.set_ylabel('Samples/Second'); ax4.set_title('Training Speed')
        ax4.grid(True, alpha=0.3)
        ax5 = axes[1, 1]
        steps_per_sec = [x for x in metrics['train_steps_per_second'] if x is not None]
        if steps_per_sec:
            step_speed_steps = [steps[i] for i, x in enumerate(metrics['train_steps_per_second']) if x is not None]
            ax5.plot(step_speed_steps, steps_per_sec, 'brown', linewidth=2)
        ax5.set_xlabel('Steps'); ax5.set_ylabel('Steps/Second'); ax5.set_title('Training Steps per Second')
        ax5.grid(True, alpha=0.3)
        ax6 = axes[1, 2]
        if epochs:
            ax6.plot(steps, epochs, 'teal', linewidth=2, marker='o', markersize=3)
        ax6.set_xlabel('Steps'); ax6.set_ylabel('Epoch'); ax6.set_title('Epoch Progress')
        ax6.grid(True, alpha=0.3)
        if has_semantic:
            ax7 = axes[2, 0]
            train_diversity = [x for x in metrics.get('train_theme_diversity', []) if x is not None]
            eval_diversity = [x for x in metrics.get('eval_theme_diversity', []) if x is not None]
            if train_diversity:
                div_steps = [steps[i] for i, x in enumerate(metrics.get('train_theme_diversity', [])) if x is not None]
                ax7.plot(div_steps, train_diversity, 'b-', label='Train Diversity', linewidth=2)
            if eval_diversity:
                eval_div_steps = [steps[i] for i, x in enumerate(metrics.get('eval_theme_diversity', [])) if x is not None]
                ax7.plot(eval_div_steps, eval_diversity, 'r--', label='Eval Diversity', linewidth=2)
            ax7.set_xlabel('Steps'); ax7.set_ylabel('Shannon Entropy'); ax7.set_title('Theme Diversity'); ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax8 = axes[2, 1]
            train_coverage = [x for x in metrics.get('train_theme_coverage', []) if x is not None]
            eval_coverage = [x for x in metrics.get('eval_theme_coverage', []) if x is not None]
            if train_coverage:
                cov_steps = [steps[i] for i, x in enumerate(metrics.get('train_theme_coverage', [])) if x is not None]
                ax8.plot(cov_steps, train_coverage, 'b-', label='Train Coverage', linewidth=2)
            if eval_coverage:
                eval_cov_steps = [steps[i] for i, x in enumerate(metrics.get('eval_theme_coverage', [])) if x is not None]
                ax8.plot(eval_cov_steps, eval_coverage, 'r--', label='Eval Coverage', linewidth=2)
            ax8.set_xlabel('Steps'); ax8.set_ylabel('Coverage Ratio'); ax8.set_title('Theme Coverage'); ax8.legend()
            ax8.set_ylim([0, 1.05]); ax8.grid(True, alpha=0.3)
            ax9 = axes[2, 2]
            train_diversity_data = [(steps[i], x) for i, x in enumerate(metrics.get('train_theme_diversity', [])) if x is not None]
            train_coverage_data = [(steps[i], x) for i, x in enumerate(metrics.get('train_theme_coverage', [])) if x is not None]
            if train_diversity_data and train_coverage_data:
                diversity_map = dict(train_diversity_data)
                coverage_map = dict(train_coverage_data)
                common_steps = sorted(list(set(diversity_map.keys()) & set(coverage_map.keys())))
                if common_steps:
                    combined = [diversity_map[step] * coverage_map[step] for step in common_steps]
                    ax9.plot(common_steps, combined, 'purple', linewidth=2)
            ax9.set_xlabel('Steps'); ax9.set_ylabel('Combined Score'); ax9.set_title('Semantic Quality Score')
            ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = save_dir / "training_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        if train_losses or eval_losses:
            self.create_loss_focused_plot(save_dir, metrics, final)

    def create_loss_focused_plot(self, save_dir, metrics, final=False):
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('dark_background')
        plt.figure(figsize=(12, 8))
        steps = metrics['step']
        train_losses = metrics['train_loss']
        eval_losses = metrics['eval_loss']
        train_data = [(steps[i], loss) for i, loss in enumerate(train_losses) if loss is not None]
        eval_data = [(steps[i], loss) for i, loss in enumerate(eval_losses) if loss is not None]
        if train_data:
            train_steps, train_vals = zip(*train_data)
            plt.plot(train_steps, train_vals, 'b-', label='Training Loss', linewidth=3, alpha=0.8)
        if eval_data:
            eval_steps, eval_vals = zip(*eval_data)
            plt.plot(eval_steps, eval_vals, 'r--', label='Validation Loss', linewidth=3, alpha=0.8)
        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'Loss Progression {"(Final Run)" if final else "(Checkpoint)"}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12); plt.grid(True, alpha=0.6)
        if train_data:
            min_train_loss = min(train_vals)
            final_train_loss = train_vals[-1]
            plt.text(0.02, 0.98, f'Min Train Loss: {min_train_loss:.4f}\nFinal Train Loss: {final_train_loss:.4f}',
                     transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), fontsize=10)
        if eval_data:
            min_eval_loss = min(eval_vals)
            final_eval_loss = eval_vals[-1]
            plt.text(0.98, 0.98, f'Min Val Loss: {min_eval_loss:.4f}\nFinal Val Loss: {final_eval_loss:.4f}',
                     transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8), fontsize=10)
        plt.tight_layout()
        loss_plot_file = save_dir / "loss_focused.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

@dataclass
class CustomDataCollator:
    tokenizer: Any
    max_length: int = 512
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            logger.warning("Tokenizer does not have a pad_token_id. Using eos_token_id for padding.")
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("No pad_token_id or eos_token_id found in tokenizer. Cannot pad sequences.")
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, feature in enumerate(features):
            ids = feature["input_ids"][:max_len]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def pack_sequences(examples, max_length=512, eos_token_id=None):
    packed = []
    current_ids = []
    for ex in examples:
        ids = list(ex['input_ids'])
        if eos_token_id is not None and (not ids or ids[-1] != eos_token_id):
            ids.append(eos_token_id)
        if len(current_ids) + len(ids) <= max_length:
            current_ids.extend(ids)
        else:
            if current_ids:
                packed.append({'input_ids': current_ids})
            current_ids = ids[:]
    if current_ids:
        packed.append({'input_ids': current_ids})
    return packed

def load_semantic_metadata(metadata_path: str = "data_finetune/dataset_metadata.json") -> Dict:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        logger.warning(f"⚠️ Semantic metadata not found at {metadata_path}")
        return {}
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        theme_dist = metadata.get('theme_distribution', {})
        source_dist = metadata.get('source_distribution', {})
        total_pairs = metadata.get('total_pairs', 0)
        logger.info(f"🧠 Loaded semantic metadata:")
        logger.info(f"   - Total pairs: {total_pairs:,}")
        logger.info(f"   - Unique themes: {len(theme_dist)}")
        logger.info(f"   - Data sources: {list(source_dist.keys())}")
        return metadata
    except Exception as e:
        logger.warning(f"⚠️ Failed to load semantic metadata: {e}")
        return {}

def save_tokenized_cache(dataset, cache_path):
    try:
        dataset.save_to_disk(cache_path)
        logger.info(f"💾 Saved tokenized dataset to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save dataset cache: {e}")

def load_tokenized_cache(cache_path):
    try:
        cache_path_obj = Path(cache_path)
        if cache_path_obj.exists():
            from datasets import load_from_disk
            dataset = load_from_disk(cache_path)
            logger.info(f"⚡ Loaded tokenized dataset from cache: {cache_path}")
            return dataset
    except Exception as e:
        logger.warning(f"⚠️ Failed to load dataset cache: {e}")
        logger.info(f"🔄 Cleaning corrupted cache and will rebuild...")
        try:
            import shutil
            if Path(cache_path).exists():
                shutil.rmtree(cache_path)
                logger.info(f"✅ Removed corrupted cache directory")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Could not clean cache: {cleanup_error}")
    return None

class RhizomeTrainer:
    def __init__(self, model_name="DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.start_time = time.time()
        self.semantic_metadata = {}
        self.theme_tracker = None
        self.original_train_dataset = None
        self.use_theme_weighting = False

    def print_header(self):
        print("\n" + "═" * 70)
        print("🤖 RhizomeML Fine-Tuning Suite")
        print("   🎨 Now with Semantic Theme-Aware Training!")
        print("   ⚡ CPU-Optimized with QLoRA 4-bit Support!")
        print("   Compatible with data_formatter.py output")
        print("═" * 70)

    def print_section(self, title, emoji="📋"):
        print(f"\n{emoji} {title}")
        print("─" * 50)

    def setup_model_and_tokenizer(self):
        global USE_QLORA
        self.print_section("Model Setup", "🔧")
        with tqdm(total=4, desc="Loading components", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer's pad_token was None, set to eos_token.")
            pbar.update(1)
            logger.info(f"Loading model from {self.model_name}...")
            if USE_QLORA:
                logger.info("🔬 Loading model with QLoRA 4-bit quantization...")
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE_DETAILS.get('uses_bf16', False) else torch.float32,
                        bnb_4bit_use_double_quant=True,
                    )
                    max_mem_config = None
                    if not USE_CPU_ONLY and torch.cuda.is_available():
                        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
                        usable_mem_bytes = int(gpu_mem_bytes * 0.95)
                        usable_mem_gb = usable_mem_bytes / (1024**3)
                        max_mem_config = {0: f"{usable_mem_gb:.1f}GB"}
                        logger.info(f"💾 GPU memory: {gpu_mem_bytes / (1024**3):.1f}GB total, using {usable_mem_gb:.1f}GB")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto" if not USE_CPU_ONLY else "cpu",
                        low_cpu_mem_usage=True,
                        max_memory=max_mem_config,
                    )
                    self.model = prepare_model_for_kbit_training(self.model)
                    logger.info("✅ Model loaded with 4-bit quantization")
                except ImportError:
                    logger.error("❌ bitsandbytes not installed! Install with: pip install bitsandbytes")
                    logger.info("Falling back to standard FP32 loading...")
                    USE_QLORA = False
            if not USE_QLORA:
                if USE_CPU_ONLY:
                    dtype = torch.bfloat16 if DEVICE_DETAILS.get('uses_bf16', False) else torch.float32
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                    )
                else:
                    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    ).to(DEVICE)
            pbar.update(1)
            lora_target_modules = get_model_lora_targets(self.model)
            fan_in_fan_out = determine_fan_in_fan_out(self.model_name)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="lora_only",
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=fan_in_fan_out
            )
            pbar.update(1)
            self.model = get_peft_model(self.model, lora_config)
            for name, param in self.model.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False
            self.model.train()
            pbar.update(1)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Model loaded and {'QLoRA' if USE_QLORA else 'LoRA'} applied successfully on {DEVICE.type.upper()}")
        logger.info(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
        if USE_QLORA:
            logger.info(f"🔬 Using 4-bit quantization (QLoRA)")
        if trainable_params == 0:
            raise RuntimeError("❌ Error: No trainable parameters found after applying LoRA. Check LoRA target modules.")
        if USE_CPU_ONLY and not USE_QLORA and hasattr(torch, 'compile'):
            try:
                logger.info("🔥 Attempting torch.compile for CPU optimization...")
                self.model = torch.compile(self.model, backend="inductor", mode="reduce-overhead")
                logger.info("✅ Model compiled successfully!")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed (this is OK): {e}")
        elif USE_QLORA:
            logger.info("ℹ️ Skipping torch.compile (incompatible with QLoRA quantization)")

    # ------------------------- PATCH 1: Safe cache loading -------------------------
    def load_and_tokenize_data(
        self,
        train_file,
        val_file=None,
        max_length=512,
        use_theme_weighting=True,
        use_sequence_packing=True,
        use_cache=True
    ):
        self.print_section("Data Processing", "📚")
        train_path = Path(train_file)
        if not train_path.exists():
            raise FileNotFoundError(f"❌ Training file not found: {train_file}")

        cache_dir = train_path.parent / "tokenized_cache"

        # Only use cache if the configuration matches.
        cache_config_path = cache_dir / "cache_config.json"
        cache_valid = False
        if use_cache and cache_dir.exists() and cache_config_path.exists():
            try:
                with open(cache_config_path, "r") as f:
                    cache_config = json.load(f)
                expected_config = {
                    "model_name": self.model_name,
                    "max_length": max_length,
                    "sequence_packing": use_sequence_packing,
                    "tokenizer_class": self.tokenizer.__class__.__name__,
                }
                if cache_config == expected_config:
                    cache_valid = True
                else:
                    logger.info("🔄 Cache configuration changed. Rebuilding tokenized dataset.")
            except Exception as e:
                logger.warning(f"⚠️ Could not read cache configuration: {e}")

        if use_cache and cache_valid:
            cached = load_tokenized_cache(str(cache_dir))
            if cached is not None:
                logger.info("⚡ Using cached tokenized dataset!")
                original_dataset = load_dataset("json", data_files={"train": train_file})
                self.original_train_dataset = original_dataset["train"]
                metadata_path = Path(train_file).parent / "dataset_metadata.json"
                self.semantic_metadata = load_semantic_metadata(str(metadata_path))
                if self.semantic_metadata and "theme_distribution" in self.semantic_metadata:
                    self.theme_tracker = ThemeTracker(self.semantic_metadata["theme_distribution"])
                self.use_theme_weighting = use_theme_weighting and self.theme_tracker is not None
                return cached

        # --- No valid cache, build dataset from scratch ---
        data_files = {"train": train_file}
        if val_file and Path(val_file).exists():
            data_files["validation"] = val_file
            logger.info(f"✅ Validation file found: {val_file}")
        else:
            logger.info("ℹ️ No validation file provided. Creating automatic 2% validation split...")

        dataset = load_dataset("json", data_files=data_files)
        if "validation" not in dataset:
            split = dataset["train"].train_test_split(test_size=0.02, seed=42)
            dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
            logger.info(f"📊 Split into {len(dataset['train'])} train / {len(dataset['validation'])} val")

        train_size = len(dataset['train'])
        logger.info(f"📊 Training samples: {train_size:,}")
        if "validation" in dataset:
            logger.info(f"📊 Validation samples: {len(dataset['validation']):,}")

        metadata_path = Path(train_file).parent / "dataset_metadata.json"
        self.semantic_metadata = load_semantic_metadata(str(metadata_path))
        if self.semantic_metadata and 'theme_distribution' in self.semantic_metadata:
            self.theme_tracker = ThemeTracker(self.semantic_metadata['theme_distribution'])
        else:
            logger.warning("⚠️ No theme distribution found in metadata. Theme tracking disabled.")
            use_theme_weighting = False

        def tokenize_function(examples):
            if "text" in examples:
                text_data = examples["text"]
            elif "content" in examples:
                text_data = examples["content"]
            elif "prompt" in examples:
                text_data = examples["prompt"]
            else:
                raise ValueError(f"No supported text field found. Available columns: {list(examples.keys())}")
            return self.tokenizer(
                text_data,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=True,
            )

        print("\n🔄 Tokenizing dataset...")
        original_columns = dataset["train"].column_names
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=1 if USE_CPU_ONLY else None,
            remove_columns=original_columns,
            desc="Tokenizing"
        )
        logger.info("✅ Dataset tokenization complete.")

        if use_sequence_packing and USE_CPU_ONLY:
            logger.info("📦 Applying sequence packing for CPU efficiency...")
            try:
                original_count = len(tokenized_dataset["train"])
                packed_examples = pack_sequences(
                    tokenized_dataset["train"],
                    max_length=max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                tokenized_dataset["train"] = Dataset.from_list(packed_examples)
                packed_count = len(tokenized_dataset["train"])
                efficiency_gain = (1 - packed_count / original_count) * 100
                logger.info(f"✅ Packed {original_count:,} → {packed_count:,} sequences ({efficiency_gain:.1f}% reduction)")
            except Exception as e:
                logger.warning(f"⚠️ Sequence packing failed: {e}")

        self.original_train_dataset = dataset["train"]
        self.use_theme_weighting = use_theme_weighting and self.theme_tracker is not None

        if use_cache:
            # Save tokenized dataset and configuration snapshot for future runs
            save_tokenized_cache(tokenized_dataset, str(cache_dir))
            config_snapshot = {
                "model_name": self.model_name,
                "max_length": max_length,
                "sequence_packing": use_sequence_packing,
                "tokenizer_class": self.tokenizer.__class__.__name__,
            }
            cache_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_config_path, "w") as f:
                json.dump(config_snapshot, f, indent=2)
            logger.info(f"📝 Saved cache configuration to {cache_config_path}")

        return tokenized_dataset

    def create_training_args(self, output_dir="./RhizomeML-finetuned", has_validation=False, **kwargs):
        if USE_CPU_ONLY:
            default_batch_size = 4
            default_grad_accum = 4
            default_fp16 = False
        else:
            is_rtx = DEVICE_DETAILS.get('is_rtx', False)
            if is_rtx:
                default_batch_size = 8
                default_grad_accum = 2
                default_fp16 = True
            else:
                default_batch_size = 2
                default_grad_accum = 8
                default_fp16 = False
        default_args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": 5,
            "per_device_train_batch_size": default_batch_size,
            "gradient_accumulation_steps": default_grad_accum,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 25,
            "save_steps": 150,
            "save_total_limit": 10,
            "eval_strategy": "steps" if has_validation else "no",
            "eval_steps": 150 if has_validation else None,
            "save_strategy": "steps",
            "load_best_model_at_end": has_validation,
            "metric_for_best_model": "eval_loss" if has_validation else None,
            "greater_is_better": False,
            "save_safetensors": True,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": True if not USE_CPU_ONLY else False,
            "remove_unused_columns": True,
            "seed": 42,
            "fp16": default_fp16,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "report_to": "none",
            "disable_tqdm": False,
            "log_level": "error",
            "log_level_replica": "error",
            "logging_nan_inf_filter": False,
            "log_on_each_node": False,
        }
        default_args["use_cpu"] = USE_CPU_ONLY
        default_args["ddp_find_unused_parameters"] = False
        default_args.update(kwargs)
        if "evaluation_strategy" in default_args and "eval_strategy" not in default_args:
            default_args["eval_strategy"] = default_args.pop("evaluation_strategy")
        elif "eval_strategy" in default_args and "evaluation_strategy" in default_args:
            default_args.pop("evaluation_strategy")
        return TrainingArguments(**default_args)

    def train(self, train_file, val_file=None, output_dir="./RhizomeML-finetuned",
              use_theme_weighting=True, use_sequence_packing=True, use_cache=True, **training_kwargs):
        self.print_header()
        try:
            self.setup_model_and_tokenizer()
            tokenized_dataset = self.load_and_tokenize_data(
                train_file, val_file,
                use_theme_weighting=use_theme_weighting,
                use_sequence_packing=use_sequence_packing,
                use_cache=use_cache
            )
            self.print_section("Training Configuration", "⚙️")
            has_validation = "validation" in tokenized_dataset
            if training_kwargs.get("dataloader_num_workers", 0) > 0:
                training_kwargs['dataloader_worker_init_fn'] = seed_worker
            training_args = self.create_training_args(
                output_dir=output_dir,
                has_validation=has_validation,
                **training_kwargs
            )
            effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            logger.info(f"🎯 Number of training epochs: {training_args.num_train_epochs}")
            logger.info(f"📦 Effective batch size: {effective_batch_size}")
            logger.info(f"📈 Initial learning rate: {training_args.learning_rate}")
            logger.info(f"💾 Output directory: {Path(output_dir).resolve()}")
            logger.info(f"🚀 Training on: {DEVICE_INFO}")
            logger.info(f"🔌 FP16: {training_args.fp16}")
            logger.info(f"🚫 CPU-only mode: {training_args.use_cpu}")
            logger.info(f"🔬 QLoRA: {USE_QLORA}")
            if USE_CPU_ONLY:
                logger.info(f"⚡ CPU Optimizations Applied:")
                logger.info(f"   • Threads: {DEVICE_DETAILS.get('cpu_threads_used', 'N/A')}")
                logger.info(f"   • BF16: {DEVICE_DETAILS.get('uses_bf16', False)}")
                logger.info(f"   • QLoRA: {USE_QLORA}")
                logger.info(f"   • Sequence packing: {use_sequence_packing}")
            if self.semantic_metadata:
                theme_count = len(self.semantic_metadata.get('theme_distribution', {}))
                logger.info(f"🧠 Training with {theme_count} semantic themes")
                if self.use_theme_weighting:
                    logger.info(f"🎨 Theme-weighted sampling: ENABLED")
                else:
                    logger.info(f"⚪ Theme-weighted sampling: DISABLED")
            data_collator = CustomDataCollator(self.tokenizer, max_length=512)
            training_logger = TrainingLogger(output_dir, theme_tracker=self.theme_tracker)
            train_sampler = None
            if self.use_theme_weighting and self.theme_tracker:
                logger.info("🎲 Creating theme-weighted sampler...")
                train_sampler = create_theme_weighted_sampler(
                    self.original_train_dataset, self.theme_tracker
                )
            import warnings
            warnings.filterwarnings("ignore", message=".*label_names.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*loss_type.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*checkpoint.*use_reentrant.*", category=UserWarning)
            trainer_kwargs = {
                "model": self.model,
                "args": training_args,
                "train_dataset": tokenized_dataset["train"],
                "eval_dataset": tokenized_dataset.get("validation"),
                "data_collator": data_collator,
                "callbacks": [training_logger],
            }
            if self.theme_tracker and self.original_train_dataset:
                logger.info("Using ThemeAwareTrainer to track semantic diversity")
                trainer = ThemeAwareTrainer(
                    theme_tracker=self.theme_tracker,
                    original_dataset=self.original_train_dataset,
                    **trainer_kwargs
                )
            else:
                trainer = Trainer(**trainer_kwargs)
            checkpoint_dir_path = Path(output_dir)
            last_checkpoint_path = self.find_last_checkpoint(checkpoint_dir_path)
            if last_checkpoint_path:
                training_logger.set_checkpoint_info(last_checkpoint_path)
            self.print_section("Training Progress", "🚀")
            if last_checkpoint_path:
                logger.info(f"🔄 Resuming training from checkpoint: {last_checkpoint_path}")
                theme_state_path = Path(last_checkpoint_path) / 'theme_tracker_state.json'
                if theme_state_path.exists() and self.theme_tracker:
                    logger.info(f"Loading theme tracker state from {theme_state_path}...")
                    try:
                        with open(theme_state_path, 'r') as f:
                            theme_state = json.load(f)
                        self.theme_tracker.training_theme_counts = Counter(theme_state.get('training_themes', {}))
                        self.theme_tracker.eval_theme_counts = Counter(theme_state.get('eval_themes', {}))
                        logger.info("✅ Restored theme tracker state.")
                    except Exception as e:
                        logger.warning(f"Failed to load theme tracker state: {e}")
                trainer.train(resume_from_checkpoint=True)
            else:
                logger.info("🎯 Starting fresh training run...")
                trainer.train()
            logger.info("💾 Saving final model and tokenizer...")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            elapsed = time.time() - self.start_time
            self.print_section("Training Complete", "🎉")
            logger.info(f"⏱️ Total training duration: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
            logger.info(f"📁 Final model saved to: {Path(output_dir).resolve()}")
            logger.info(f"📊 Training plots: {Path(output_dir) / 'training_plots.png'}")
            logger.info(f"📈 Loss plot: {Path(output_dir) / 'loss_focused.png'}")
            logger.info(f"📋 Metrics JSON: {Path(output_dir) / 'training_metrics.json'}")
            if self.theme_tracker:
                training_logger.on_train_end(training_args, trainer.state, None)
                logger.info(f"🎨 Theme tracker data: {Path(output_dir) / 'theme_tracker_state.json'}")
            if USE_CPU_ONLY:
                logger.info("\n" + "="*70)
                logger.info("⚡ CPU OPTIMIZATION SUMMARY")
                logger.info("="*70)
                logger.info(f"  ✓ Threads: {DEVICE_DETAILS.get('cpu_threads_used', 'N/A')}")
                logger.info(f"  ✓ BF16: {DEVICE_DETAILS.get('uses_bf16', False)}")
                logger.info(f"  ✓ QLoRA: {USE_QLORA}")
                logger.info(f"  ✓ Micro-batching: {training_args.per_device_train_batch_size}×{training_args.gradient_accumulation_steps}")
                logger.info(f"  ✓ Sequence packing: {use_sequence_packing}")
                logger.info(f"  ✓ Dataset caching: {use_cache}")
                logger.info("="*70 + "\n")
            return trainer
        except KeyboardInterrupt:
            logger.info("ℹ️ Training interrupted by user.")
            if self.theme_tracker:
                training_logger.on_train_end(None, None, None)
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during training: {e}", exc_info=True)
            raise

    @staticmethod
    def find_last_checkpoint(checkpoint_dir: Path):
        if not checkpoint_dir.exists():
            return None
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            return None
        try:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            return str(last_checkpoint)
        except:
            return None

def main():
    trainer = RhizomeTrainer(model_name="DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL")
    try:
        result = trainer.train(
            train_file="data_finetune/dataset_train.jsonl",
            # val_file="data_finetune/dataset_validation.jsonl",
            output_dir="./RhizomeML-finetuned",
            use_theme_weighting=True,
            use_sequence_packing=True,
            use_cache=True,
            num_train_epochs=3,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=25,
            save_steps=150,
            dataloader_num_workers=0,
        )
        if result:
            print("\n" + "═" * 70)
            print("🎉 Fine-tuning process successfully completed!")
            print("📁 Your fine-tuned model and training artifacts are in the output directory")
            print("🎨 Semantic diversity metrics have been tracked and saved")
            if USE_CPU_ONLY:
                print("⚡ CPU optimizations were applied for maximum performance")
            if USE_QLORA:
                print("🔬 Model was trained with QLoRA 4-bit quantization")
            print("═" * 70)
        else:
            print("\n" + "═" * 70)
            print("ℹ️ Fine-tuning process finished (possibly interrupted or encountered issues).")
            print("═" * 70)
    except Exception as e:
        logger.critical(f"\n❌ Fine-tuning terminated unexpectedly: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
