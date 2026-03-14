"""
eval.py — Evaluate RhizomeML checkpoints.

Usage:
  python eval.py                          # compare all checkpoints, probe the best
  python eval.py --checkpoint <path>      # eval a single checkpoint (perplexity + probes)
  python eval.py --no-probes              # compare only, skip generation probes
"""
import os, sys, json, math, gzip, random, argparse
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
from rich.panel import Panel

ROOT      = Path(__file__).parent
TEST_FILE = ROOT / "data_finetune" / "dataset_test.jsonl"
TEST_GZ   = Path(str(TEST_FILE) + ".gz")
CKPT_DIR  = ROOT / "RhizomeML-finetuned"

STOP_MARKERS = ["<|im_end|>", "<|end|>", "<|endoftext|>", "</s>", "[/INST]"]
TEXT_MARKERS = ["User:", "Assistant:", "Human:", "\nUser\n", "\nAssistant\n"]

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RhizomeML checkpoints")
    p.add_argument("--checkpoint", "-c", default=None,
                   help="Path to a single checkpoint (default: compare all in RhizomeML-finetuned/)")
    p.add_argument("--samples", "-n", type=int, default=60,
                   help="Number of test samples for perplexity (default: 60)")
    p.add_argument("--no-quant", action="store_true",
                   help="Disable 4-bit quantization (use bf16)")
    p.add_argument("--no-probes", action="store_true",
                   help="Skip qualitative generation probes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_samples(n, seed):
    test_file = TEST_FILE if TEST_FILE.exists() else TEST_GZ
    opener = gzip.open if test_file.suffix == ".gz" else open
    with opener(test_file, "rt", encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
    random.seed(seed)
    return random.sample(all_records, min(n, len(all_records)))


def load_checkpoint(ckpt_path, no_quant=False):
    ckpt_path = Path(ckpt_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()

    if is_peft:
        cfg = json.loads((ckpt_path / "adapter_config.json").read_text())
        base_name = cfg["base_model_name_or_path"]
    else:
        base_name = str(ckpt_path)

    use_quant = not no_quant and torch.cuda.is_available()
    quant_kw  = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16} if use_quant else {}

    with console.status(f"Loading {Path(base_name).name} + {ckpt_path.name}...", spinner="dots"):
        base = AutoModelForCausalLM.from_pretrained(
            base_name, device_map="auto",
            torch_dtype=None if use_quant else torch.bfloat16,
            trust_remote_code=True, **quant_kw,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(ckpt_path)) if is_peft else base

    model.eval()
    return model, tokenizer, base_name, use_quant


def compute_perplexity(model, tokenizer, samples):
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

    if not total_tok:
        return None, None
    ce  = total_loss / total_tok
    ppl = math.exp(ce)
    return ce, ppl


def run_probes(model, tokenizer):
    probes = [
        "What's your view on consciousness?",
        "Tell me about your framework.",
        "What makes a well-designed system?",
        "How do you tackle a problem you've never seen before?",
        "What do you think about AI and creativity?",
    ]
    system = "You are a helpful, thoughtful assistant."
    device = next(model.parameters()).device

    stop_ids = [tokenizer.eos_token_id]
    for marker in STOP_MARKERS:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in stop_ids:
            stop_ids.append(ids[0])

    console.print(Rule("[bold cyan]Qualitative Probes"))
    for q in probes:
        console.print(f"\n[bold yellow]▶  {q}[/]")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"System: {system}\nUser: {q}\nAssistant:"

        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=200, temperature=0.7,
                top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )
        reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for marker in TEXT_MARKERS:
            idx = reply.find(marker)
            if idx != -1:
                reply = reply[:idx]
        console.print(Panel(reply.strip(), border_style="magenta", padding=(0, 1)))
    console.print(Rule())


def _free(model, tokenizer=None):
    del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_single(ckpt_path, args, samples):
    console.print(Rule(f"[bold cyan]{Path(ckpt_path).name}[/]"))
    model, tokenizer, base_name, use_quant = load_checkpoint(ckpt_path, args.no_quant)
    console.print(f"[green]Base: {base_name}  Quant: {'4-bit' if use_quant else 'bf16'}[/]")

    ce, ppl = compute_perplexity(model, tokenizer, samples)
    if ppl is None:
        console.print("[red]No valid samples.[/]")
    else:
        color = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"
        t = Table(box=box.ROUNDED, show_header=False, title="Perplexity Result")
        t.add_row("Samples",           str(len(samples)))
        t.add_row("Avg cross-entropy", f"{ce:.4f}")
        t.add_row("Perplexity",        f"[{color}]{ppl:.2f}[/]")
        t.add_row("Interpretation",
                  "[green]Great[/]"  if ppl < 25 else
                  "[yellow]Good[/]"  if ppl < 50 else
                  "[red]Needs work[/]")
        console.print(t)

    if not args.no_probes:
        run_probes(model, tokenizer)

    _free(model, tokenizer)


def compare_all(args, samples):
    checkpoints = sorted(
        [p for p in CKPT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoints:
        console.print(f"[red]No checkpoints found in {CKPT_DIR}[/]")
        sys.exit(1)
    console.print(f"[cyan]Checkpoints found: {len(checkpoints)}[/]\n")

    results = []
    for ckpt in checkpoints:
        console.print(Rule(f"[bold cyan]{ckpt.name}[/]"))
        model, tokenizer, _, _ = load_checkpoint(ckpt, args.no_quant)
        ce, ppl = compute_perplexity(model, tokenizer, samples)

        if ppl is not None:
            color = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"
            console.print(f"  CE: {ce:.4f}  PPL: [{color}]{ppl:.2f}[/]")
            results.append((ckpt, ckpt.name, ce, ppl))
        else:
            console.print("[red]No valid samples.[/]")
            results.append((ckpt, ckpt.name, None, None))

        _free(model, tokenizer)

    # Summary table
    console.print()
    console.print(Rule("[bold white]Summary[/]"))
    t = Table(box=box.ROUNDED, title="Checkpoint Perplexity Comparison")
    t.add_column("Checkpoint",    style="cyan")
    t.add_column("Cross-Entropy", justify="right")
    t.add_column("Perplexity",    justify="right")
    t.add_column("",              justify="left")

    valid    = [(ck, n, ce, ppl) for ck, n, ce, ppl in results if ppl is not None]
    best_ppl = min(r[3] for r in valid) if valid else None

    for _, name, ce, ppl in results:
        if ppl is None:
            t.add_row(name, "-", "-", "")
        else:
            color  = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"
            marker = " ◀ best" if ppl == best_ppl else ""
            t.add_row(name, f"{ce:.4f}", f"[{color}]{ppl:.2f}[/]", marker)
    console.print(t)

    # Run probes on best checkpoint
    if not args.no_probes and valid:
        best_ckpt = min(valid, key=lambda r: r[3])[0]
        console.print(f"\n[cyan]Running qualitative probes on best: {best_ckpt.name}[/]")
        model, tokenizer, _, _ = load_checkpoint(best_ckpt, args.no_quant)
        run_probes(model, tokenizer)
        _free(model, tokenizer)


def main():
    args = parse_args()

    test_file = TEST_FILE if TEST_FILE.exists() else TEST_GZ
    if not test_file.exists():
        console.print("[red]Test file not found. Run data_formatter.py first.[/]")
        sys.exit(1)

    samples = load_samples(args.samples, args.seed)
    console.print(f"[cyan]Test samples loaded: {len(samples)}[/]")

    if args.checkpoint:
        eval_single(args.checkpoint, args, samples)
    else:
        compare_all(args, samples)


if __name__ == "__main__":
    main()
