"""
RhizomeML Checkpoint to GGUF Converter
======================================
Converts the latest fine-tuned checkpoint to GGUF format for llama.cpp inference.

This script:
1. Finds the latest checkpoint in your RhizomeML output directory
2. Merges LoRA adapters into the base model (if applicable)
3. Converts the merged model to GGUF format
4. Optionally quantizes to various formats (Q4_K_M, Q5_K_M, etc.)
5. Auto-clones and builds llama.cpp if needed

Usage:
    python convert_to_gguf.py                          # Auto-detect latest checkpoint
    python convert_to_gguf.py --checkpoint checkpoint-2100
    python convert_to_gguf.py --quant q4_k_m           # Specify quantization
    python convert_to_gguf.py --output my_model.gguf   # Custom output name

Requirements:
    pip install torch transformers peft accelerate sentencepiece
"""

import os
import sys
# Disable torch compile/dynamo which pulls in triton
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1" 
os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"
import argparse
import subprocess
import shutil
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum
import re
import multiprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_FINETUNED_DIR = "./RhizomeML-finetuned"
DEFAULT_MERGED_DIR = "./merged_model"
DEFAULT_GGUF_DIR = "./gguf_models"
DEFAULT_LLAMA_CPP_DIR = "./llama.cpp"

# Quantization options
QUANT_TYPES = {
    'f16': 'F16 (no quantization, largest)',
    'q8_0': 'Q8_0 (8-bit, high quality)',
    'q6_k': 'Q6_K (6-bit, very good quality)',
    'q5_k_m': 'Q5_K_M (5-bit medium, good balance)',
    'q5_k_s': 'Q5_K_S (5-bit small, slightly smaller)',
    'q4_k_m': 'Q4_K_M (4-bit medium, recommended)',
    'q4_k_s': 'Q4_K_S (4-bit small, smaller)',
    'q3_k_m': 'Q3_K_M (3-bit medium, smaller but quality loss)',
    'q2_k': 'Q2_K (2-bit, smallest but noticeable quality loss)',
    'iq4_nl': 'IQ4_NL (4-bit importance quantization)',
    'iq3_xxs': 'IQ3_XXS (3-bit importance, very small)',
}

DEFAULT_QUANT = 'q4_k_m'


class QuantResult(Enum):
    """Result of quantization attempt."""
    SUCCESS = "success"           # Quantization completed successfully
    SKIPPED = "skipped"           # No quantize binary, F16 still available
    FAILED = "failed"             # Quantization attempted but failed


def get_llama_cpp_bin_dir(llama_cpp_dir: Path) -> Path:
    """Get the binary directory for llama.cpp (build/bin for CMake builds)."""
    return llama_cpp_dir / "build" / "bin"


def find_binary(llama_cpp_dir: Path, name: str) -> Optional[Path]:
    """Find a llama.cpp binary, checking CMake build paths first."""
    # Ensure it's an absolute Path object
    llama_cpp_dir = Path(llama_cpp_dir).resolve()
    
    # CMake build locations (preferred)
    cmake_paths = [
        llama_cpp_dir / "build" / "bin" / name,
        llama_cpp_dir / "build" / "bin" / f"{name}.exe",
        llama_cpp_dir / "build" / name,
        llama_cpp_dir / "build" / f"{name}.exe",
    ]
    
    # Legacy Makefile locations (fallback)
    legacy_paths = [
        llama_cpp_dir / name,
        llama_cpp_dir / f"{name}.exe",
    ]
    
    for p in cmake_paths + legacy_paths:
        if p.exists():
            logger.debug(f"Found binary: {p}")
            return p
    
    # Log what we checked if not found
    logger.debug(f"Binary '{name}' not found. Checked: {cmake_paths[0]}")
    
    return None


def find_latest_checkpoint(base_dir: str) -> Optional[Path]:
    """Find the latest checkpoint in the finetuned directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.error(f"Directory not found: {base_dir}")
        return None
    
    # Look for checkpoint-* directories
    checkpoints = list(base_path.glob("checkpoint-*"))
    
    if not checkpoints:
        # Maybe the base dir itself is the model
        if (base_path / "adapter_config.json").exists() or \
           (base_path / "config.json").exists():
            logger.info(f"Using base directory as model: {base_path}")
            return base_path
        logger.error(f"No checkpoints found in {base_dir}")
        return None
    
    # Sort by checkpoint number
    def get_checkpoint_num(p: Path) -> int:
        match = re.search(r'checkpoint-(\d+)', p.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_checkpoint_num)
    latest = checkpoints[-1]
    
    logger.info(f"Found {len(checkpoints)} checkpoints, using latest: {latest.name}")
    return latest


def detect_model_type(checkpoint_path: Path) -> Tuple[str, bool, Optional[str]]:
    """
    Detect if this is a LoRA adapter or full model, and find base model.
    
    Returns:
        (model_type, is_lora, base_model_name)
    """
    adapter_config = checkpoint_path / "adapter_config.json"
    config_file = checkpoint_path / "config.json"
    
    if adapter_config.exists():
        with open(adapter_config, 'r') as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get('base_model_name_or_path', None)
        logger.info(f"Detected LoRA adapter, base model: {base_model}")
        return ('lora', True, base_model)
    
    elif config_file.exists():
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        model_type = cfg.get('model_type', 'unknown')
        logger.info(f"Detected full model, type: {model_type}")
        return (model_type, False, None)
    
    else:
        logger.warning("Could not detect model type, assuming LoRA")
        return ('unknown', True, None)


def merge_lora_adapters(
    checkpoint_path: Path,
    base_model: str,
    output_dir: Path,
    device: str = "cpu"
) -> bool:
    """Merge LoRA adapters into base model."""
    logger.info("=" * 60)
    logger.info("STEP 1: Merging LoRA adapters into base model")
    logger.info("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Load base model
        logger.info(f"Loading base model from {base_model}...")
        logger.info("(This may take a while and use significant RAM)")
        
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load and merge LoRA
        logger.info(f"Loading LoRA adapters from {checkpoint_path}...")
        model = PeftModel.from_pretrained(
            base_model_obj,
            checkpoint_path,
            dtype=torch.float16
        )
        
        logger.info("Merging adapters...")
        model = model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        # Also copy any special files
        for special_file in ['generation_config.json', 'special_tokens_map.json']:
            src = checkpoint_path / special_file
            if src.exists():
                shutil.copy(src, output_dir / special_file)
        
        logger.info("✅ LoRA merge complete!")
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch transformers peft accelerate")
        return False
    except Exception as e:
        logger.error(f"Failed to merge LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_llama_cpp(llama_cpp_dir: str) -> Optional[Path]:
    """Find llama.cpp directory and conversion script."""
    llama_path = Path(llama_cpp_dir)
    
    # Check common locations
    possible_paths = [
        llama_path,
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("../llama.cpp"),
    ]
    
    for p in possible_paths:
        convert_script = p / "convert_hf_to_gguf.py"
        if convert_script.exists():
            logger.info(f"Found llama.cpp at: {p}")
            return p
    
    return None


def check_cmake_available() -> bool:
    """Check if cmake is available."""
    try:
        result = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def build_llama_cpp(llama_cpp_dir: Path) -> bool:
    """Build llama.cpp using CMake."""
    logger.info("=" * 60)
    logger.info("Building llama.cpp binaries...")
    logger.info("=" * 60)
    
    # Resolve to absolute path
    llama_cpp_dir = Path(llama_cpp_dir).resolve()
    
    if not check_cmake_available():
        logger.error("CMake not found. Please install cmake:")
        logger.error("  Ubuntu/Debian: sudo apt install cmake build-essential")
        logger.error("  Fedora: sudo dnf install cmake gcc-c++")
        logger.error("  Arch: sudo pacman -S cmake base-devel")
        return False
    
    build_dir = llama_cpp_dir / "build"
    
    try:
        # Configure with CMake
        logger.info("Configuring with CMake...")
        configure_cmd = [
            "cmake",
            "-B", str(build_dir),
            #"-DGGML_CUDA=ON",
            "-DLLAMA_CURL=OFF",  # Don't require libcurl
        ]
        
        result = subprocess.run(
            configure_cmd,
            cwd=str(llama_cpp_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"CMake configure failed: {result.stderr}")
            return False
        
        # Build with all available cores
        num_cores = multiprocessing.cpu_count()
        logger.info(f"Building with {num_cores} cores...")
        
        # Build specific targets we need
        targets = ["llama-server", "llama-quantize", "llama-cli"]
        
        for target in targets:
            logger.info(f"Building {target}...")
            build_cmd = [
                "cmake",
                "--build", str(build_dir),
                "--config", "Release",
                "--target", target,
                "-j", str(num_cores)
            ]
            
            result = subprocess.run(
                build_cmd,
                cwd=str(llama_cpp_dir),
                capture_output=False,  # Show build output
                text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to build {target}, continuing...")
        
        # Verify at least llama-quantize exists (llama-server is optional for conversion)
        expected_path = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        logger.info(f"Checking for binary at: {expected_path}")
        logger.info(f"Path exists: {expected_path.exists()}")
        
        if not find_binary(llama_cpp_dir, "llama-quantize"):
            logger.error("Failed to build llama-quantize")
            return False
        
        logger.info("✅ llama.cpp build complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during build: {e}")
        import traceback
        traceback.print_exc()
        return False


def clone_llama_cpp(target_dir: Path) -> bool:
    """Clone llama.cpp repository."""
    logger.info("llama.cpp not found, cloning repository...")
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(target_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Install requirements
        requirements = target_dir / "requirements.txt"
        if requirements.exists():
            logger.info("Installing llama.cpp Python requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
                capture_output=True,
                text=True
            )
        
        logger.info("✅ llama.cpp cloned!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone llama.cpp: {e}")
        return False
    except FileNotFoundError:
        logger.error("Git not found. Please install git or clone llama.cpp manually.")
        return False


def ensure_llama_cpp_built(llama_cpp_dir: Path) -> bool:
    """Ensure llama.cpp is cloned and built, building if necessary."""
    
    # Check if already built
    server_bin = find_binary(llama_cpp_dir, "llama-server")
    quantize_bin = find_binary(llama_cpp_dir, "llama-quantize")
    
    if server_bin and quantize_bin:
        logger.info(f"✅ llama.cpp binaries found")
        return True
    
    # Check if repo exists but not built
    if (llama_cpp_dir / "convert_hf_to_gguf.py").exists():
        logger.info("llama.cpp found but binaries not built, building now...")
        return build_llama_cpp(llama_cpp_dir)
    
    # Need to clone first
    if not clone_llama_cpp(llama_cpp_dir):
        return False
    
    # Then build
    return build_llama_cpp(llama_cpp_dir)


def convert_to_gguf(
    model_dir: Path,
    output_file: Path,
    llama_cpp_dir: Path,
    outtype: str = "f16"
) -> bool:
    """Convert HuggingFace model to GGUF format."""
    logger.info("=" * 60)
    logger.info("STEP 2: Converting to GGUF format")
    logger.info("=" * 60)
    
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        logger.error(f"Conversion script not found: {convert_script}")
        return False
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile", str(output_file),
        "--outtype", outtype
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        logger.info(f"✅ GGUF conversion complete: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        return False


def quantize_gguf(
    input_file: Path,
    output_file: Path,
    llama_cpp_dir: Path,
    quant_type: str = "q4_k_m"
) -> QuantResult:
    """
    Quantize GGUF model to smaller size.
    
    Returns:
        QuantResult.SUCCESS - Quantization completed successfully
        QuantResult.SKIPPED - No quantize binary found, F16 still available
        QuantResult.FAILED  - Quantization attempted but failed
    """
    logger.info("=" * 60)
    logger.info(f"STEP 3: Quantizing to {quant_type.upper()}")
    logger.info("=" * 60)
    
    # Find quantize binary
    quantize_bin = find_binary(llama_cpp_dir, "llama-quantize")
    
    if not quantize_bin:
        logger.warning("⚠️ llama-quantize binary not found.")
        logger.warning("Attempting to build llama.cpp...")
        
        if build_llama_cpp(llama_cpp_dir):
            quantize_bin = find_binary(llama_cpp_dir, "llama-quantize")
        
        if not quantize_bin:
            logger.error("Failed to build llama-quantize")
            logger.info(f"📁 F16 GGUF is still available at: {input_file}")
            return QuantResult.SKIPPED
    
    cmd = [str(quantize_bin), str(input_file), str(output_file), quant_type.upper()]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ Quantization complete: {output_file}")
        
        # Show size comparison
        if input_file.exists() and output_file.exists():
            input_size = input_file.stat().st_size / (1024**3)
            output_size = output_file.stat().st_size / (1024**3)
            reduction = (1 - output_size/input_size) * 100
            logger.info(f"📊 Size: {input_size:.2f}GB → {output_size:.2f}GB ({reduction:.1f}% reduction)")
        
        return QuantResult.SUCCESS
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e}")
        return QuantResult.FAILED


def generate_run_script(gguf_path: Path, model_name: str, llama_cpp_dir: Path) -> Path:
    """Generate a convenience script to run the model with llama.cpp."""
    script_path = gguf_path.parent / f"run_{model_name}.sh"
    
    # Use absolute paths for reliability
    gguf_abs = gguf_path.resolve()
    bin_dir = (llama_cpp_dir / "build" / "bin").resolve()
    server_bin = bin_dir / "llama-server"
    
    script_content = f'''#!/bin/bash
# Run script for {model_name}
# Generated by convert_to_gguf.py

MODEL="{gguf_abs}"
SERVER="{server_bin}"

# Check if server exists
if [ ! -f "$SERVER" ]; then
    echo "Error: llama-server not found at $SERVER"
    echo "Please build llama.cpp first:"
    echo "  cd {llama_cpp_dir} && cmake -B build -DLLAMA_CURL=OFF && cmake --build build --config Release -j$(nproc)"
    exit 1
fi

# Adjust these based on your hardware:
# -ngl: Number of layers to offload to GPU (use 99 for all)
# -c: Context size
# --threads: CPU threads for non-GPU layers

echo "Starting llama.cpp server with {model_name}"
echo ""
echo "Options:"
echo "  1) Full GPU:  $SERVER -m $MODEL -c 8192 -ngl 99"
echo "  2) Hybrid:    $SERVER -m $MODEL -c 8192 -ngl 99 -ot \\"*attn.*=GPU,*ffn_.*=CPU\\" --threads 14"
echo "  3) CPU only:  $SERVER -m $MODEL -c 8192 --threads 14"
echo ""
echo "Server will be available at: http://localhost:8081"
echo ""

# Default: Hybrid mode
"$SERVER" -m "$MODEL" -c 8192 -ngl 99 -ot "*attn.*=GPU,*ffn_.*=CPU" --threads 14 --port 8081
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    logger.info(f"📝 Generated run script: {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert RhizomeML checkpoints to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Convert latest checkpoint with Q4_K_M
  %(prog)s --checkpoint checkpoint-2100       # Specific checkpoint
  %(prog)s --quant q5_k_m                     # Higher quality quantization
  %(prog)s --quant f16                        # No quantization (largest)
  %(prog)s --base-model meta-llama/Llama-2-7b # Override base model detection

Quantization options (smallest to largest):
  q2_k    - 2-bit (smallest, noticeable quality loss)
  q3_k_m  - 3-bit medium
  q4_k_s  - 4-bit small
  q4_k_m  - 4-bit medium (RECOMMENDED)
  q5_k_s  - 5-bit small
  q5_k_m  - 5-bit medium (good balance)
  q6_k   - 6-bit (very good quality)
  q8_0    - 8-bit (high quality)
  f16     - No quantization (largest, best quality)
        """
    )
    
    parser.add_argument(
        "--finetuned-dir", "-d",
        default=DEFAULT_FINETUNED_DIR,
        help=f"Directory containing checkpoints (default: {DEFAULT_FINETUNED_DIR})"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Specific checkpoint to convert (default: latest)"
    )
    
    parser.add_argument(
        "--base-model", "-b",
        default=None,
        help="Override base model (auto-detected from adapter_config.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output GGUF filename (default: auto-generated)"
    )
    
    parser.add_argument(
        "--quant", "-q",
        default=DEFAULT_QUANT,
        choices=list(QUANT_TYPES.keys()),
        help=f"Quantization type (default: {DEFAULT_QUANT})"
    )
    
    parser.add_argument(
        "--merged-dir", "-m",
        default=DEFAULT_MERGED_DIR,
        help=f"Directory for merged model (default: {DEFAULT_MERGED_DIR})"
    )
    
    parser.add_argument(
        "--gguf-dir", "-g",
        default=DEFAULT_GGUF_DIR,
        help=f"Directory for GGUF output (default: {DEFAULT_GGUF_DIR})"
    )
    
    parser.add_argument(
        "--llama-cpp", "-l",
        default=DEFAULT_LLAMA_CPP_DIR,
        help=f"Path to llama.cpp (default: {DEFAULT_LLAMA_CPP_DIR})"
    )
    
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep merged model directory after conversion"
    )
    
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep F16 GGUF after quantization"
    )
    
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip LoRA merge (use if already merged)"
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building llama.cpp (use if already built)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for LoRA merge (default: cpu, use cuda if you have VRAM)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RhizomeML to GGUF Converter")
    logger.info("=" * 60)
    
    # Step 0: Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.finetuned_dir) / args.checkpoint
        if not checkpoint_path.exists():
            checkpoint_path = Path(args.checkpoint)  # Try as absolute path
    else:
        checkpoint_path = find_latest_checkpoint(args.finetuned_dir)
    
    if not checkpoint_path or not checkpoint_path.exists():
        logger.error("No valid checkpoint found!")
        sys.exit(1)
    
    logger.info(f"📁 Using checkpoint: {checkpoint_path}")
    
    # Detect model type
    model_type, is_lora, base_model = detect_model_type(checkpoint_path)
    
    if args.base_model:
        base_model = args.base_model
        logger.info(f"Using override base model: {base_model}")
    
    if is_lora and not base_model:
        logger.error("LoRA adapter detected but no base model found!")
        logger.error("Please specify with --base-model")
        sys.exit(1)
    
    # Setup paths
    merged_dir = Path(args.merged_dir)
    gguf_dir = Path(args.gguf_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp = Path(args.llama_cpp)
    
    # Generate output names
    checkpoint_name = checkpoint_path.name
    model_short = base_model.split('/')[-1] if base_model else model_type
    
    f16_gguf = gguf_dir / f"rhizome-{model_short}-{checkpoint_name}-f16.gguf"
    
    if args.output:
        quantized_gguf = gguf_dir / args.output
    else:
        quantized_gguf = gguf_dir / f"rhizome-{model_short}-{checkpoint_name}-{args.quant}.gguf"
    
    # Ensure llama.cpp is available and built
    if not args.skip_build:
        if not ensure_llama_cpp_built(llama_cpp):
            logger.error("Cannot proceed without llama.cpp")
            sys.exit(1)
    else:
        # Just check it exists
        if not find_llama_cpp(args.llama_cpp):
            logger.error(f"llama.cpp not found at {args.llama_cpp}")
            sys.exit(1)
        llama_cpp = Path(find_llama_cpp(args.llama_cpp))
    
    # Step 1: Merge LoRA (if needed)
    if is_lora and not args.skip_merge:
        if not merge_lora_adapters(checkpoint_path, base_model, merged_dir, args.device):
            logger.error("LoRA merge failed!")
            sys.exit(1)
        model_to_convert = merged_dir
    else:
        model_to_convert = checkpoint_path
        logger.info("Skipping LoRA merge (full model or --skip-merge)")
    
    # Step 2: Convert to GGUF (F16 first)
    if not convert_to_gguf(model_to_convert, f16_gguf, llama_cpp, outtype="f16"):
        logger.error("GGUF conversion failed!")
        sys.exit(1)
    
    # Step 3: Quantize (if not F16)
    final_gguf = f16_gguf  # Default to F16
    quantization_done = False
    
    if args.quant.lower() != 'f16':
        quant_result = quantize_gguf(f16_gguf, quantized_gguf, llama_cpp, args.quant)
        
        if quant_result == QuantResult.SUCCESS:
            # Quantization worked, use quantized file
            final_gguf = quantized_gguf
            quantization_done = True
            
            # Clean up F16 if not keeping
            if not args.keep_f16 and f16_gguf.exists():
                logger.info(f"Removing intermediate F16 file: {f16_gguf}")
                f16_gguf.unlink()
                
        elif quant_result == QuantResult.SKIPPED:
            # No quantize binary, keep F16
            logger.warning(f"⚠️ Quantization skipped - using F16 GGUF instead")
            final_gguf = f16_gguf
            
        elif quant_result == QuantResult.FAILED:
            # Quantization attempted but failed, keep F16
            logger.warning(f"⚠️ Quantization failed - F16 GGUF is still available")
            final_gguf = f16_gguf
    
    # Clean up merged directory
    if is_lora and not args.keep_merged and merged_dir.exists():
        logger.info(f"Cleaning up merged model directory: {merged_dir}")
        shutil.rmtree(merged_dir)
    
    # Verify final file exists
    if not final_gguf.exists():
        logger.error(f"❌ Output file not found: {final_gguf}")
        sys.exit(1)
    
    # Generate run script
    model_name = final_gguf.stem
    run_script = generate_run_script(final_gguf, model_name, llama_cpp)
    
    # Get binary paths for display
    server_bin = find_binary(llama_cpp, "llama-server")
    
    # Done!
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CONVERSION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📁 GGUF model: {final_gguf}")
    logger.info(f"📊 Size: {final_gguf.stat().st_size / (1024**3):.2f} GB")
    
    if args.quant.lower() != 'f16' and not quantization_done:
        logger.info(f"⚠️ Note: Requested {args.quant.upper()} but outputting F16 (quantization unavailable)")
    
    logger.info(f"📝 Run script: {run_script}")
    logger.info("")
    logger.info("To run the model:")
    logger.info(f"  {run_script}")
    logger.info("")
    logger.info("Or manually:")
    if server_bin:
        logger.info(f"  {server_bin} -m {final_gguf} -c 8192 -ngl 99")
        logger.info("")
        logger.info("For (hybrid offload):")
        logger.info(f"  {server_bin} -m {final_gguf} -c 8192 -ngl 99 \\")
        logger.info('    -ot "*attn.*=GPU,*ffn_.*=CPU" --threads 14')
    else:
        logger.info(f"  ./llama.cpp/build/bin/llama-server -m {final_gguf} -c 8192 -ngl 99")
    logger.info("")


if __name__ == "__main__":
    main()
