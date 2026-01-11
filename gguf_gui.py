"""
GGUF Converter GUI - Easy Model Conversion Interface
====================================================
A Gradio interface for convert_to_gguf.py that makes local LLM conversion accessible.

Features:
- Search and download HuggingFace models
- Visual VRAM calculator
- Smart quantization recommendations
- One-click conversion and launch
- Model library management

Usage:
    python gguf_gui.py

Requirements:
    pip install gradio huggingface_hub torch psutil
"""

import gradio as gr
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import time
import webbrowser

# Try imports with helpful error messages
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        VRAM_GB = 0
        GPU_NAME = "No GPU detected"
except ImportError:
    CUDA_AVAILABLE = False
    VRAM_GB = 0
    GPU_NAME = "PyTorch not installed"

try:
    from huggingface_hub import HfApi, model_info
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Paths
GGUF_DIR = Path("./gguf_models")
LLAMA_CPP_DIR = Path("./llama.cpp")
HF_MODELS_DIR = Path("./hf_models")

# Quantization specs
QUANT_SPECS = {
    'f16': {'ratio': 2.0, 'quality': '100%', 'desc': 'No quantization (largest, best quality)'},
    'q8_0': {'ratio': 1.0, 'quality': '95%', 'desc': '8-bit (high quality, large)'},
    'q6_k': {'ratio': 0.65, 'quality': '90%', 'desc': '6-bit (very good quality)'},
    'q5_k_m': {'ratio': 0.58, 'quality': '85%', 'desc': '5-bit medium (good balance) ⭐'},
    'q4_k_m': {'ratio': 0.50, 'quality': '80%', 'desc': '4-bit medium (recommended) ⭐⭐'},
    'q4_k_s': {'ratio': 0.48, 'quality': '75%', 'desc': '4-bit small (smaller)'},
    'q3_k_m': {'ratio': 0.40, 'quality': '65%', 'desc': '3-bit medium (quality loss)'},
    'q2_k': {'ratio': 0.30, 'quality': '50%', 'desc': '2-bit (smallest, noticeable loss)'},
}

def estimate_model_size(params_billions: float, quant_type: str) -> float:
    """Estimate GGUF size in GB."""
    return params_billions * QUANT_SPECS[quant_type]['ratio']

def parse_model_size(model_id: str) -> Optional[float]:
    """Try to extract parameter count from model name."""
    import re
    # Look for patterns like "7b", "13b", "160m", etc.
    match = re.search(r'(\d+\.?\d*)(b|m)', model_id.lower())
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if unit == 'b':
            return num
        elif unit == 'm':
            return num / 1000
    return None

def get_model_info(model_id: str) -> Tuple[str, str, str]:
    """Fetch model info from HuggingFace."""
    if not HF_AVAILABLE:
        return "❌ huggingface_hub not installed", "", ""

    if not model_id or not model_id.strip():
        return "⚠️ Please enter a model ID", "", ""

    try:
        info = model_info(model_id)
        size_estimate = parse_model_size(model_id)

        # Try to get actual size from model files
        safetensors_size = 0
        try:
            if hasattr(info, 'siblings') and info.siblings:
                for file in info.siblings:
                    filename = file.rfilename if hasattr(file, 'rfilename') else str(file)
                    if filename.endswith(('.safetensors', '.bin', '.pth')):
                        # Safely get file size
                        if hasattr(file, 'size') and file.size is not None:
                            try:
                                safetensors_size += int(file.size)
                            except (ValueError, TypeError):
                                pass
        except (AttributeError, TypeError):
            pass

        # Calculate estimates
        if safetensors_size > 0:
            size_gb = safetensors_size / 1e9
            params_est = size_gb / 2  # Rough estimate: FP16 = 2 bytes per param
        elif size_estimate:
            params_est = size_estimate
            size_gb = params_est * 2
        else:
            # Fallback: just use name-based estimate
            params_est = size_estimate if size_estimate else 1.0
            size_gb = params_est * 2 if params_est else None

        # Build info string
        info_str = f"📦 **Model:** {model_id}\n"
        if params_est:
            info_str += f"🔢 **Parameters:** ~{params_est:.1f}B\n"
        if size_gb:
            info_str += f"💾 **Size (FP16):** ~{size_gb:.1f}GB\n"

        # Safely get tags
        try:
            if hasattr(info, 'tags') and info.tags and len(info.tags) > 0:
                tag_list = [str(t) for t in info.tags[:5]]
                info_str += f"🏷️ **Tags:** {', '.join(tag_list)}\n"
        except (AttributeError, TypeError):
            pass

        # Add warning if we're just guessing
        if not safetensors_size and size_estimate:
            info_str += "\n⚠️ *Size estimated from model name*\n"
        elif not safetensors_size and not size_estimate:
            info_str += "\n⚠️ *Could not determine size - enter manually below*\n"

        return info_str, str(params_est) if params_est else "1.0", str(size_gb) if size_gb else "2.0"

    except Exception as e:
        # Fallback to name-based parsing
        size_estimate = parse_model_size(model_id)
        if size_estimate:
            info_str = f"📦 **Model:** {model_id}\n"
            info_str += f"🔢 **Parameters:** ~{size_estimate:.1f}B *(estimated from name)*\n"
            info_str += f"💾 **Size (FP16):** ~{size_estimate * 2:.1f}GB *(estimated)*\n"
            info_str += f"\n⚠️ *API error: {str(e)}*\n"
            return info_str, str(size_estimate), str(size_estimate * 2)
        else:
            return f"❌ Error fetching model info: {str(e)}\n\n💡 **Tip:** Enter the model size manually below", "1.0", "2.0"

def calculate_quantization_table(params_billions: float) -> str:
    """Generate table showing size for each quantization level."""
    if not params_billions or params_billions <= 0:
        return "⚠️ Enter model size to see quantization options"

    table = "| Quantization | Size | Quality | Fits in VRAM? | Description |\n"
    table += "|--------------|------|---------|---------------|-------------|\n"

    for quant, specs in QUANT_SPECS.items():
        size_gb = estimate_model_size(params_billions, quant)
        quality = specs['quality']
        desc = specs['desc']

        # Check if fits (leaving 1GB headroom)
        if CUDA_AVAILABLE:
            fits = "✅ Yes" if size_gb < (VRAM_GB - 1) else "❌ No"
        else:
            fits = "❓ No GPU"

        table += f"| **{quant.upper()}** | {size_gb:.2f}GB | {quality} | {fits} | {desc} |\n"

    return table

def get_available_models() -> List[str]:
    """List already converted GGUF models."""
    if not GGUF_DIR.exists():
        return []

    models = []
    for gguf_file in GGUF_DIR.glob("*.gguf"):
        size_mb = gguf_file.stat().st_size / 1e6
        models.append(f"{gguf_file.name} ({size_mb:.1f}MB)")

    return models

def get_local_checkpoints() -> List[str]:
    """Find local checkpoints in RhizomeML-finetuned directory."""
    finetuned_dir = Path("./RhizomeML-finetuned")

    if not finetuned_dir.exists():
        return []

    checkpoints = []

    # Look for checkpoint-* directories
    for checkpoint_dir in sorted(finetuned_dir.glob("checkpoint-*"), reverse=True):
        if checkpoint_dir.is_dir():
            # Check if it has model files
            if (checkpoint_dir / "adapter_config.json").exists() or \
               (checkpoint_dir / "config.json").exists():
                checkpoints.append(checkpoint_dir.name)

    # Also check if base dir itself is a model
    if (finetuned_dir / "adapter_config.json").exists() or \
       (finetuned_dir / "config.json").exists():
        checkpoints.insert(0, "RhizomeML-finetuned (base)")

    return checkpoints if checkpoints else ["No checkpoints found"]

def run_conversion(
    model_source: str,
    model_id: str,
    checkpoint_choice: str,
    quant_type: str,
    use_cuda: bool,
    update_llama: bool,
    progress=gr.Progress()
) -> str:
    """Run the conversion script with progress tracking."""

    # Determine which source to use
    if model_source == "HuggingFace Model":
        if not model_id.strip():
            return "❌ Please enter a model ID"
        cmd = [
            sys.executable,
            "convert_to_gguf.py",
            "--hf", model_id,
            "--quant", quant_type.lower(),
        ]
    else:  # Local Checkpoint
        if checkpoint_choice == "No checkpoints found":
            return "❌ No local checkpoints found in ./RhizomeML-finetuned"

        if checkpoint_choice == "latest":
            # Let the script auto-detect latest
            cmd = [
                sys.executable,
                "convert_to_gguf.py",
                "--quant", quant_type.lower(),
            ]
        else:
            # Specific checkpoint
            checkpoint_name = checkpoint_choice.replace(" (base)", "")
            cmd = [
                sys.executable,
                "convert_to_gguf.py",
                "--checkpoint", checkpoint_name,
                "--quant", quant_type.lower(),
            ]

    if not use_cuda:
        cmd.append("--cpu")

    if update_llama:
        cmd.append("--update")

    progress(0, desc="Starting conversion...")

    try:
        # Run subprocess and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            print(line, end='')  # Also print to console

            # Update progress based on keywords
            if "Downloading" in line or "Loading" in line:
                progress(0.2, desc="Loading model...")
            elif "Merging" in line:
                progress(0.4, desc="Merging LoRA...")
            elif "Converting" in line:
                progress(0.6, desc="Converting to GGUF...")
            elif "Quantizing" in line:
                progress(0.8, desc="Quantizing...")
            elif "COMPLETE" in line:
                progress(1.0, desc="Done!")

        process.wait()

        if process.returncode == 0:
            return "✅ **Conversion complete!**\n\n" + "".join(output_lines[-20:])
        else:
            return f"❌ **Conversion failed** (exit code {process.returncode})\n\n" + "".join(output_lines[-30:])

    except FileNotFoundError:
        return "❌ convert_to_gguf.py not found in current directory!"
    except Exception as e:
        return f"❌ Error during conversion: {str(e)}"

def launch_model(model_name: str) -> str:
    """Launch llama-server with the selected model."""
    if not model_name:
        return "❌ Please select a model"

    # Extract just the filename
    model_file = model_name.split(" (")[0]
    model_path = GGUF_DIR / model_file

    if not model_path.exists():
        return f"❌ Model not found: {model_path}"

    # Find llama-server binary
    server_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        server_bin = LLAMA_CPP_DIR / "llama-server"

    if not server_bin.exists():
        return "❌ llama-server not found. Build llama.cpp first!"

    # Build command
    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-c", "8192",
        "--port", "8081"
    ]

    if CUDA_AVAILABLE:
        cmd.extend(["-ngl", "99"])  # Offload all layers to GPU

    try:
        # Start server in background
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a bit then open browser
        time.sleep(2)
        webbrowser.open("http://localhost:8081")

        return "✅ **Server starting!**\n\nOpening http://localhost:8081 in your browser...\n\nTo stop the server, close this app or kill the llama-server process."

    except Exception as e:
        return f"❌ Error launching server: {str(e)}"

# Build the Gradio interface
def build_interface():
    with gr.Blocks(title="GGUF Converter Studio", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 GGUF Converter Studio")
        gr.Markdown("Convert HuggingFace models to GGUF format with smart quantization")

        # Hardware info
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"""
                ### 🖥️ Your Hardware
                **GPU:** {GPU_NAME}
                **VRAM:** {VRAM_GB:.1f}GB {"✅" if CUDA_AVAILABLE else "❌"}
                """)

        with gr.Tabs():
            # Tab 1: Convert Model
            with gr.Tab("🔧 Convert Model"):
                # Model source selector
                with gr.Row():
                    model_source = gr.Radio(
                        choices=["HuggingFace Model", "Local Checkpoint"],
                        value="HuggingFace Model",
                        label="📂 Model Source",
                        info="Choose between downloading from HuggingFace or using your fine-tuned checkpoint"
                    )

                # HuggingFace section
                with gr.Column(visible=True) as hf_section:
                    with gr.Row():
                        with gr.Column(scale=2):
                            model_id_input = gr.Textbox(
                                label="🔍 HuggingFace Model ID",
                                placeholder="e.g., EleutherAI/pythia-160m",
                                info="Enter the model repository ID from HuggingFace"
                            )

                            fetch_btn = gr.Button("📊 Fetch Model Info", variant="secondary")

                            model_info_display = gr.Markdown("Enter a model ID and click 'Fetch Model Info'")

                            # Hidden fields for storing data
                            params_billions = gr.Textbox(visible=False)
                            size_gb = gr.Textbox(visible=False)

                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ Quick Links")
                            gr.Markdown("""
                            **Popular Models:**
                            - `EleutherAI/pythia-160m` (160M)
                            - `EleutherAI/pythia-1b` (1B)
                            - `google/gemma-2b` (2B)
                            - `Qwen/Qwen2.5-1.5B` (1.5B)
                            - `meta-llama/Llama-3.2-3B` (3B)
                            """)

                # Local checkpoint section
                with gr.Column(visible=False) as checkpoint_section:
                    gr.Markdown("### 🎯 Fine-Tuned Model Checkpoints")
                    gr.Markdown("Select a checkpoint from your `RhizomeML-finetuned` directory")

                    with gr.Row():
                        checkpoint_refresh_btn = gr.Button("🔄 Refresh Checkpoints", variant="secondary", scale=1)
                        checkpoint_dropdown = gr.Dropdown(
                            choices=["latest"] + get_local_checkpoints(),
                            value="latest",
                            label="Select Checkpoint",
                            info="'latest' will auto-detect the most recent checkpoint",
                            scale=3
                        )

                    checkpoint_info = gr.Markdown("""
                    **💡 Tips:**
                    - Select **'latest'** to automatically use your most recent checkpoint
                    - Or choose a specific checkpoint (e.g., `checkpoint-1050`)
                    - The script will auto-detect if it's a LoRA adapter and merge it
                    """)

                # Toggle visibility based on source selection
                def toggle_source(source):
                    if source == "HuggingFace Model":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                model_source.change(
                    fn=toggle_source,
                    inputs=[model_source],
                    outputs=[hf_section, checkpoint_section]
                )

                # Refresh checkpoints button
                checkpoint_refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=["latest"] + get_local_checkpoints()),
                    outputs=[checkpoint_dropdown]
                )

                gr.Markdown("---")

                # Manual size input (fallback) - only for HF models
                with gr.Row():
                    manual_params = gr.Number(
                        label="📏 Model Size (Billions of Parameters)",
                        value=0.16,
                        info="Auto-filled from model info, or enter manually"
                    )

                # Quantization table
                quant_table = gr.Markdown("Enter model size to see quantization options")

                manual_params.change(
                    fn=calculate_quantization_table,
                    inputs=[manual_params],
                    outputs=[quant_table]
                )

                gr.Markdown("---")

                with gr.Row():
                    quant_choice = gr.Dropdown(
                        choices=list(QUANT_SPECS.keys()),
                        value="q4_k_m",
                        label="🎯 Select Quantization",
                        info="Q4_K_M is recommended for best size/quality balance"
                    )

                with gr.Row():
                    use_cuda_check = gr.Checkbox(
                        label="🎮 Use CUDA for building llama.cpp",
                        value=CUDA_AVAILABLE,
                        interactive=CUDA_AVAILABLE
                    )
                    update_llama_check = gr.Checkbox(
                        label="🔄 Update llama.cpp before converting",
                        value=False,
                        info="Recommended if you're getting 'BPE pre-tokenizer' errors"
                    )

                convert_btn = gr.Button("🚀 Convert Model", variant="primary", size="lg")

                conversion_output = gr.Textbox(
                    label="📋 Conversion Log",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )

                # Wire up the fetch button (HF only)
                fetch_btn.click(
                    fn=get_model_info,
                    inputs=[model_id_input],
                    outputs=[model_info_display, params_billions, size_gb]
                ).then(
                    fn=lambda p: float(p) if p else 0.16,
                    inputs=[params_billions],
                    outputs=[manual_params]
                )

                # Wire up conversion
                convert_btn.click(
                    fn=run_conversion,
                    inputs=[model_source, model_id_input, checkpoint_dropdown, quant_choice, use_cuda_check, update_llama_check],
                    outputs=[conversion_output]
                )

            # Tab 2: Launch Model
            with gr.Tab("🎮 Launch Model"):
                gr.Markdown("### 💾 Your Converted Models")

                refresh_btn = gr.Button("🔄 Refresh Model List", variant="secondary")

                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model to Launch",
                    info="Choose a GGUF model to run with llama.cpp"
                )

                launch_btn = gr.Button("🚀 Launch llama-server", variant="primary", size="lg")

                launch_output = gr.Markdown("")

                gr.Markdown("""
                ### ℹ️ About llama-server

                The server will start on **http://localhost:8081**

                **Features:**
                - Web-based chat interface
                - OpenAI-compatible API
                - Adjustable parameters (temperature, top-p, etc.)

                **To stop the server:** Close this app or run `pkill llama-server`
                """)

                # Wire up buttons
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_available_models()),
                    outputs=[model_dropdown]
                )

                launch_btn.click(
                    fn=launch_model,
                    inputs=[model_dropdown],
                    outputs=[launch_output]
                )

            # Tab 3: Help
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## 📖 How to Use

                ### Step 1: Convert a Model
                1. Enter a HuggingFace model ID (e.g., `EleutherAI/pythia-160m`)
                2. Click "Fetch Model Info" to see details
                3. Review the quantization table to see which formats fit your VRAM
                4. Select your preferred quantization level
                5. Click "Download & Convert"

                ### Step 2: Launch the Model
                1. Go to the "Launch Model" tab
                2. Select your converted model from the dropdown
                3. Click "Launch llama-server"
                4. A browser window will open with the chat interface

                ### Quantization Guide

                **What is quantization?**
                Reduces model size by using fewer bits per parameter. Smaller = faster loading, less VRAM, but some quality loss.

                **Which quantization should I use?**
                - **Q4_K_M**: Best overall balance (recommended)
                - **Q5_K_M**: Better quality, slightly larger
                - **Q2_K**: Smallest size, for testing/demos
                - **F16**: No quantization, best quality but largest

                ### Troubleshooting

                **"BPE pre-tokenizer" error:**
                Enable "Update llama.cpp" before converting

                **Model won't fit in VRAM:**
                Try a more aggressive quantization (Q4→Q3→Q2)

                **Server won't start:**
                Make sure llama.cpp is built: `cd llama.cpp && cmake -B build && cmake --build build`

                ### Requirements

                This GUI wraps `convert_to_gguf.py` which needs:
                ```bash
                pip install torch transformers peft accelerate sentencepiece huggingface_hub gradio
                ```

                ### About

                Built with ❤️ to make local LLMs accessible.
                Powered by llama.cpp and your own hardware.
                """)

        gr.Markdown("---")
        gr.Markdown("💡 **Tip:** A RTX 3060 is perfect for running some 20b (openai/gpt-oss-20b) models with Q4 quantization!")

    return app

# Main execution
if __name__ == "__main__":
    print("🚀 Starting GGUF Converter Studio...")
    print(f"   GPU: {GPU_NAME}")
    print(f"   VRAM: {VRAM_GB:.1f}GB")
    print(f"   CUDA Available: {CUDA_AVAILABLE}")
    print()

    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # Auto-open browser
    )
