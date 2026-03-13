"""
Compare perplexity across all checkpoints.
Loads each one sequentially and reports results in a table.
"""
import os, json, math, gzip, random
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich import box
from rich.rule import Rule

ROOT       = Path(__file__).parent
TEST_FILE  = ROOT / "data_finetune" / "dataset_test.jsonl"
TEST_GZ    = Path(str(TEST_FILE) + ".gz")
CKPT_DIR   = ROOT / "RhizomeML-finetuned"
N_SAMPLES  = 60
SEED       = 42

console = Console()

# Load test records once
test_file = TEST_FILE if TEST_FILE.exists() else TEST_GZ
opener = gzip.open if test_file.suffix == ".gz" else open
with opener(test_file, "rt", encoding="utf-8") as f:
    all_records = [json.loads(l) for l in f if l.strip()]
random.seed(SEED)
samples = random.sample(all_records, min(N_SAMPLES, len(all_records)))
console.print(f"[cyan]Test samples loaded: {len(samples)}[/]")

# Find all checkpoints sorted by step
checkpoints = sorted(
    [p for p in CKPT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
    key=lambda p: int(p.name.split("-")[1])
)
console.print(f"[cyan]Checkpoints found: {len(checkpoints)}[/]\n")

results = []

for ckpt in checkpoints:
    step = ckpt.name
    console.print(Rule(f"[bold cyan]{step}[/]"))

    cfg       = json.loads((ckpt / "adapter_config.json").read_text())
    base_name = cfg["base_model_name_or_path"]
    use_quant = torch.cuda.is_available()
    quant_kw  = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16} if use_quant else {}

    with console.status(f"Loading {base_name} + {step}...", spinner="dots"):
        base = AutoModelForCausalLM.from_pretrained(
            base_name, device_map="auto",
            torch_dtype=None if use_quant else torch.bfloat16,
            trust_remote_code=True, **quant_kw,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()

    device = next(model.parameters()).device
    total_loss = total_tok = 0

    with console.status(f"Perplexity ({len(samples)} samples)...", spinner="dots"):
        for rec in samples:
            text = rec.get("text") or rec.get("content") or rec.get("prompt", "")
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            n = ids["input_ids"].shape[1]
            if n < 4:
                continue
            with torch.no_grad():
                loss = model(**ids, labels=ids["input_ids"]).loss.item()
            total_loss += loss * n
            total_tok  += n

    if total_tok:
        ppl = math.exp(total_loss / total_tok)
        ce  = total_loss / total_tok
        color = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"
        console.print(f"  CE: {ce:.4f}  PPL: [{color}]{ppl:.2f}[/]")
        results.append((step, ce, ppl))
    else:
        console.print("[red]No valid samples.[/]")
        results.append((step, None, None))

    # Free memory before next checkpoint
    del model, base, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Summary table
console.print()
console.print(Rule("[bold white]Summary[/]"))
t = Table(box=box.ROUNDED, title="Checkpoint Perplexity Comparison")
t.add_column("Checkpoint",      style="cyan")
t.add_column("Cross-Entropy",   justify="right")
t.add_column("Perplexity",      justify="right")
t.add_column("",                justify="left")

best_ppl = min(r[2] for r in results if r[2] is not None)

for step, ce, ppl in results:
    if ppl is None:
        t.add_row(step, "-", "-", "")
        continue
    color  = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"
    marker = " ◀ best" if ppl == best_ppl else ""
    t.add_row(step, f"{ce:.4f}", f"[{color}]{ppl:.2f}[/]", marker)

console.print(t)
