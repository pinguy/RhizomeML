"""
DeepSpeed ZeRO-Offload Integration for RhizomeML Training Pipeline

This module provides DeepSpeed configuration and integration functions
designed for systems with limited VRAM but ample CPU RAM.

Optimized for:
- GTX 1660 Ti (6GB VRAM)
- 32GB+ System RAM (with zram compression = 50GB+ effective)
- Fast NVMe storage (3.2GB/s)
- 14+ CPU cores with large L3 cache (35MB)

Usage:
    from deepspeed_integration import (
        get_deepspeed_config_for_system,
        save_deepspeed_config,
        DEEPSPEED_AVAILABLE,
    )

"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    logger.warning("accelerate not installed. Install with: pip install accelerate")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("deepspeed not installed. Install with: pip install deepspeed")

# Combined availability check
OFFLOAD_AVAILABLE = DEEPSPEED_AVAILABLE and ACCELERATE_AVAILABLE


# ============================================================================
# DEEPSPEED CONFIGURATION
# ============================================================================

def create_deepspeed_config(
    zero_stage: int = 3,
    offload_optimizer: bool = True,
    offload_param: bool = True,
    offload_device: str = "cpu",
    nvme_path: str = "/tmp/deepspeed_nvme",
    fp16: bool = True,
    bf16: bool = False,
    gradient_accumulation_steps: int = 4,
    gradient_clipping: float = 1.0,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """
    Create a DeepSpeed configuration dictionary optimized for CPU/NVMe offloading.
    
    This is tuned for systems with:
    - Limited VRAM (6-8GB)
    - Large system RAM (32GB+)
    - Fast NVMe storage
    - Multi-core CPU with large cache
    
    Args:
        zero_stage: ZeRO optimization stage (2 or 3)
            - Stage 2: Partitions optimizer states + gradients
            - Stage 3: Also partitions parameters (more memory efficient)
        offload_optimizer: Offload optimizer states to CPU/NVMe
        offload_param: Offload parameters to CPU/NVMe (ZeRO-3 only)
        offload_device: "cpu" or "nvme"
        nvme_path: Path for NVMe offload storage
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision (overrides fp16)
        gradient_accumulation_steps: Steps to accumulate gradients
        gradient_clipping: Maximum gradient norm
        pin_memory: Pin CPU memory for faster GPU transfers
    
    Returns:
        DeepSpeed configuration dictionary
    
    Example:
        >>> config = create_deepspeed_config(zero_stage=3, offload_param=True)
        >>> save_deepspeed_config(config, "ds_config.json")
    """
    
    config = {
        # Batch size settings (auto = let Trainer handle it)
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        
        # ZeRO optimization settings
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,  # Overlap communication with computation
            "contiguous_gradients": True,  # Reduce memory fragmentation
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
        },
        
        # Allow non-standard optimizers
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        
        # Logging
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    # Configure mixed precision
    if bf16:
        config["bf16"] = {"enabled": True}
        config["fp16"] = {"enabled": False}
    elif fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,  # Dynamic loss scaling
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
        config["bf16"] = {"enabled": False}
    else:
        config["fp16"] = {"enabled": False}
        config["bf16"] = {"enabled": False}
    
    # Configure optimizer offloading
    if offload_optimizer:
        if offload_device == "nvme":
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "nvme",
                "nvme_path": nvme_path,
                "pin_memory": pin_memory,
                "buffer_count": 4,
                "fast_init": True,
            }
        else:  # CPU offload
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }
    
    # Configure parameter offloading (ZeRO-3 only)
    if zero_stage == 3 and offload_param:
        if offload_device == "nvme":
            config["zero_optimization"]["offload_param"] = {
                "device": "nvme",
                "nvme_path": nvme_path,
                "pin_memory": pin_memory,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9,
            }
        else:  # CPU offload
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }
        
        # ZeRO-3 specific optimizations
        config["zero_optimization"]["sub_group_size"] = 1e9
        config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
    
    return config


def save_deepspeed_config(config: Dict[str, Any], path: str = "ds_config.json") -> str:
    """
    Save DeepSpeed configuration to a JSON file.
    
    Args:
        config: DeepSpeed configuration dictionary
        path: Output file path
    
    Returns:
        Absolute path to saved config file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"💾 DeepSpeed config saved to: {path.absolute()}")
    return str(path.absolute())


def get_deepspeed_config_for_system(
    vram_gb: float = 6.0,
    ram_gb: float = 32.0,
    has_fast_nvme: bool = True,
    model_size_b: float = 1.0,
    nvme_path: str = "/tmp/deepspeed_nvme",
) -> Dict[str, Any]:
    """
    Automatically configure DeepSpeed based on system specifications.
    
    This function analyzes your hardware and model size to determine
    the optimal DeepSpeed configuration for training.
    
    Args:
        vram_gb: GPU VRAM in gigabytes (e.g., 6.0 for GTX 1660 Ti)
        ram_gb: System RAM in gigabytes (e.g., 32.0)
        has_fast_nvme: Whether fast NVMe storage is available
        model_size_b: Model size in billions of parameters
        nvme_path: Path for NVMe offload if needed
    
    Returns:
        Optimized DeepSpeed configuration dictionary
    
    Example:
        >>> # For GTX 1660 Ti + 32GB RAM + 7B model
        >>> config = get_deepspeed_config_for_system(
        ...     vram_gb=6.0,
        ...     ram_gb=32.0,
        ...     model_size_b=7.0
        ... )
    """
    
    # Estimate memory requirements
    # Rough estimates for LoRA fine-tuning with 4-bit quantization:
    # - Base model: ~0.5-1GB per 1B parameters (quantized)
    # - Optimizer states: ~2x trainable params
    # - Gradients: ~1x trainable params
    # - Activations: Variable, depends on batch size and sequence length
    
    estimated_base_memory = model_size_b * 0.75  # 4-bit quantized base
    estimated_training_overhead = model_size_b * 0.5  # LoRA + gradients
    estimated_total = estimated_base_memory + estimated_training_overhead
    
    logger.info(f"🔧 Configuring DeepSpeed for system:")
    logger.info(f"   • VRAM: {vram_gb:.1f}GB")
    logger.info(f"   • RAM: {ram_gb:.1f}GB")
    logger.info(f"   • Model: {model_size_b:.1f}B parameters")
    logger.info(f"   • Estimated memory need: {estimated_total:.1f}GB")
    
    # Decision tree for configuration
    
    if estimated_total <= vram_gb * 0.6:
        # Model fits comfortably in VRAM
        # Use ZeRO-2 with optimizer offload for extra headroom
        logger.info("📊 Strategy: Model fits in VRAM → ZeRO-2 + optimizer offload")
        return create_deepspeed_config(
            zero_stage=2,
            offload_optimizer=True,
            offload_param=False,
            offload_device="cpu",
            fp16=True,
            gradient_accumulation_steps=4,
        )
    
    elif estimated_total <= vram_gb * 1.5:
        # Model needs some offloading but mostly fits
        # Use ZeRO-3 with optimizer offload
        logger.info("📊 Strategy: Tight VRAM fit → ZeRO-3 + optimizer offload")
        return create_deepspeed_config(
            zero_stage=3,
            offload_optimizer=True,
            offload_param=False,
            offload_device="cpu",
            fp16=True,
            gradient_accumulation_steps=8,
        )
    
    elif estimated_total <= ram_gb * 0.6:
        # Model needs full CPU offloading
        # Use ZeRO-3 with full CPU offload
        logger.info("📊 Strategy: VRAM insufficient → ZeRO-3 + full CPU offload")
        return create_deepspeed_config(
            zero_stage=3,
            offload_optimizer=True,
            offload_param=True,
            offload_device="cpu",
            fp16=True,
            gradient_accumulation_steps=8,
            pin_memory=True,
        )
    
    elif has_fast_nvme:
        # Model needs NVMe offloading
        logger.info("📊 Strategy: RAM insufficient → ZeRO-3 + NVMe offload")
        return create_deepspeed_config(
            zero_stage=3,
            offload_optimizer=True,
            offload_param=True,
            offload_device="nvme",
            nvme_path=nvme_path,
            fp16=True,
            gradient_accumulation_steps=16,
            pin_memory=True,
        )
    
    else:
        # Last resort - aggressive settings
        logger.warning("⚠️ Model may not fit! Using aggressive offloading...")
        return create_deepspeed_config(
            zero_stage=3,
            offload_optimizer=True,
            offload_param=True,
            offload_device="cpu",
            fp16=True,
            gradient_accumulation_steps=16,
            pin_memory=True,
        )


def estimate_max_model_size(
    vram_gb: float = 6.0,
    ram_gb: float = 32.0,
    use_quantization: bool = True,
    use_offload: bool = True,
) -> float:
    """
    Estimate the maximum model size trainable on given hardware.
    
    Args:
        vram_gb: Available VRAM in GB
        ram_gb: Available system RAM in GB
        use_quantization: Whether using 4-bit quantization
        use_offload: Whether using CPU/NVMe offloading
    
    Returns:
        Estimated maximum model size in billions of parameters
    """
    
    if use_offload:
        # With offloading, RAM becomes the primary constraint
        effective_memory = ram_gb * 0.7  # Leave headroom
        
        if use_quantization:
            # 4-bit: ~0.5-1GB per 1B params for inference
            # Training overhead adds ~50%
            bytes_per_billion = 1.5
        else:
            # FP16: ~2GB per 1B params for inference
            # Training overhead adds ~100%
            bytes_per_billion = 4.0
    else:
        # Without offloading, VRAM is the constraint
        effective_memory = vram_gb * 0.8
        
        if use_quantization:
            bytes_per_billion = 2.0
        else:
            bytes_per_billion = 6.0
    
    max_size = effective_memory / bytes_per_billion
    
    logger.info(f"📏 Estimated max model size: {max_size:.1f}B parameters")
    logger.info(f"   • Memory pool: {effective_memory:.1f}GB")
    logger.info(f"   • Quantization: {use_quantization}")
    logger.info(f"   • Offloading: {use_offload}")
    
    return max_size


# ============================================================================
# TRAINING ARGUMENTS INTEGRATION
# ============================================================================

def add_deepspeed_to_training_args(
    training_args_dict: Dict[str, Any],
    output_dir: str,
    model_size_b: float = 1.0,
    vram_gb: float = 6.0,
    ram_gb: float = 32.0,
) -> Dict[str, Any]:
    """
    Add DeepSpeed configuration to TrainingArguments dictionary.
    
    This is designed to integrate with the existing RhizomeTrainer.create_training_args()
    method with minimal changes.
    
    Args:
        training_args_dict: Existing training arguments dictionary
        output_dir: Output directory for saving config
        model_size_b: Model size in billions of parameters
        vram_gb: GPU VRAM in GB
        ram_gb: System RAM in GB
    
    Returns:
        Updated training arguments dictionary with DeepSpeed config
    
    Example:
        >>> args = {"output_dir": "./output", "num_train_epochs": 3, ...}
        >>> args = add_deepspeed_to_training_args(args, "./output", model_size_b=7.0)
    """
    
    if not OFFLOAD_AVAILABLE:
        logger.warning("DeepSpeed/Accelerate not available. Skipping DeepSpeed config.")
        return training_args_dict
    
    logger.info("🔥 Configuring DeepSpeed ZeRO-Offload...")
    
    # Generate optimal config
    ds_config = get_deepspeed_config_for_system(
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        has_fast_nvme=True,
        model_size_b=model_size_b,
    )
    
    # Save config
    ds_config_path = Path(output_dir) / "ds_config.json"
    save_deepspeed_config(ds_config, str(ds_config_path))
    
    # Update training args
    training_args_dict["deepspeed"] = str(ds_config_path)
    
    # DeepSpeed handles mixed precision internally
    training_args_dict["fp16"] = False
    training_args_dict["bf16"] = False
    
    # Log configuration
    zero_stage = ds_config["zero_optimization"]["stage"]
    has_opt_offload = "offload_optimizer" in ds_config["zero_optimization"]
    has_param_offload = "offload_param" in ds_config["zero_optimization"]
    
    logger.info(f"✅ DeepSpeed configured:")
    logger.info(f"   • Config file: {ds_config_path}")
    logger.info(f"   • ZeRO Stage: {zero_stage}")
    logger.info(f"   • Optimizer Offload: {has_opt_offload}")
    logger.info(f"   • Parameter Offload: {has_param_offload}")
    
    return training_args_dict


# ============================================================================
# QUICK INTEGRATION SNIPPET FOR RhizomeTrainer
# ============================================================================

INTEGRATION_SNIPPET = '''
# ============================================================================
# ADD TO RhizomeTrainer.create_training_args() METHOD
# ============================================================================

# Add this import at the top of train_script.py:
from deepspeed_integration import (
    add_deepspeed_to_training_args,
    OFFLOAD_AVAILABLE,
    estimate_max_model_size,
)

# Then modify create_training_args() - add this before the return statement:

def create_training_args(self, output_dir="./RhizomeML-finetuned", 
                        has_validation=False, 
                        use_deepspeed=True,  # NEW PARAMETER
                        model_size_b=1.0,    # NEW PARAMETER
                        **kwargs):
    """Creates and configures TrainingArguments for the Hugging Face Trainer."""
    
    # ... existing code to build default_args dict ...
    
    # Override defaults with user-provided arguments
    default_args.update(kwargs)
    
    # ========== ADD THIS BLOCK ==========
    # DeepSpeed integration
    if use_deepspeed and OFFLOAD_AVAILABLE and not USE_CPU_ONLY:
        vram_gb = DEVICE_DETAILS.get('memory_total', 6.0)
        ram_gb = DEVICE_DETAILS.get('ram_total_gb', 32.0)
        
        default_args = add_deepspeed_to_training_args(
            training_args_dict=default_args,
            output_dir=output_dir,
            model_size_b=model_size_b,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
        )
    # ========== END BLOCK ==========
    
    return TrainingArguments(**default_args)

# ============================================================================
# ADD TO RhizomeTrainer.train() METHOD  
# ============================================================================

# Modify the train() method signature:
def train(self, train_file, val_file=None, output_dir="./RhizomeML-finetuned", 
          use_theme_weighting=True, use_sequence_packing=True, use_cache=True,
          use_deepspeed=True,  # NEW PARAMETER
          **training_kwargs):

# And when calling create_training_args:
training_args = self.create_training_args(
    output_dir=output_dir,
    has_validation=has_validation,
    use_deepspeed=use_deepspeed,  # NEW
    model_size_b=1.0,  # Or detect from model config
    **training_kwargs
)
'''


# ============================================================================
# COMMAND LINE TESTING
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSpeed Configuration Generator")
    parser.add_argument("--vram", type=float, default=6.0, help="GPU VRAM in GB")
    parser.add_argument("--ram", type=float, default=32.0, help="System RAM in GB")
    parser.add_argument("--model-size", type=float, default=7.0, help="Model size in billions")
    parser.add_argument("--output", type=str, default="ds_config.json", help="Output config path")
    parser.add_argument("--show-snippet", action="store_true", help="Show integration code snippet")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔧 DeepSpeed Configuration Generator for RhizomeML")
    print("="*70)
    
    if args.show_snippet:
        print(INTEGRATION_SNIPPET)
    else:
        # Check dependencies
        print(f"\n📦 Dependencies:")
        print(f"   • accelerate: {'✅ Available' if ACCELERATE_AVAILABLE else '❌ Not installed'}")
        print(f"   • deepspeed:  {'✅ Available' if DEEPSPEED_AVAILABLE else '❌ Not installed'}")
        print(f"   • Ready for offload: {'✅ Yes' if OFFLOAD_AVAILABLE else '❌ No'}")
        
        # Estimate max model size
        print(f"\n📏 Hardware Analysis:")
        max_size = estimate_max_model_size(
            vram_gb=args.vram,
            ram_gb=args.ram,
            use_quantization=True,
            use_offload=True,
        )
        
        # Generate config
        print(f"\n🔧 Generating config for {args.model_size}B model...")
        config = get_deepspeed_config_for_system(
            vram_gb=args.vram,
            ram_gb=args.ram,
            has_fast_nvme=True,
            model_size_b=args.model_size,
        )
        
        # Save config
        save_deepspeed_config(config, args.output)
        
        # Print config
        print(f"\n📄 Generated Configuration:")
        print(json.dumps(config, indent=2))
        
        print(f"\n✅ Config saved to: {args.output}")
        print(f"\n💡 To integrate with your training script, run:")
        print(f"   python deepspeed_integration.py --show-snippet")
