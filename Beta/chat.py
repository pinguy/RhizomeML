"""
RhizomeML Chat — terminal interface for the fine-tuned model.

Usage:
    python chat.py [--model PATH] [--base BASE_MODEL] [--no-quant] [--max-new 512]

Commands inside chat:
    /clear   — reset conversation history
    /eval    — run perplexity + qualitative eval on the test set
    /info    — show model & generation settings
    /quit    — exit
"""

import os, sys, time, json, math, argparse
from pathlib import Path
from threading import Thread

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

ROOT        = Path(__file__).parent
ADAPTER_DIR = ROOT / "RhizomeML-finetuned"
TEST_FILE   = ROOT / "data_finetune" / "dataset_test.jsonl"

console = Console()

# ── args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default=str(ADAPTER_DIR))
    p.add_argument("--base",     default=None)
    p.add_argument("--no-quant", action="store_true")
    p.add_argument("--max-new",  type=int,   default=512)
    p.add_argument("--temp",     type=float, default=0.7)
    p.add_argument("--top-p",    type=float, default=0.9)
    p.add_argument("--rep-pen",  type=float, default=1.1)
    return p.parse_args()

# ── model ──────────────────────────────────────────────────────────────────────
def load_model(args):
    adapter_path = Path(args.model)
    is_peft = (adapter_path / "adapter_config.json").exists()

    if is_peft and not args.base:
        cfg       = json.loads((adapter_path / "adapter_config.json").read_text())
        base_name = cfg["base_model_name_or_path"]
    else:
        base_name = args.base or args.model

    use_quant = not args.no_quant and torch.cuda.is_available()
    quant_kw  = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16} if use_quant else {}

    with console.status(f"[bold cyan]Loading [white]{base_name}[/]...", spinner="dots"):
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            torch_dtype=None if use_quant else torch.bfloat16,
            trust_remote_code=True,
            **quant_kw,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path) if is_peft else base_name,
            trust_remote_code=True,
        )

    if is_peft:
        with console.status("[bold cyan]Merging LoRA adapter...", spinner="dots"):
            model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base

    model.eval()
    return model, tokenizer, base_name, use_quant

# ── prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(tokenizer, history, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + history
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages]
        parts.append("Assistant:")
        return "\n".join(parts)

# ── streaming generation ───────────────────────────────────────────────────────
def stream_reply(model, tokenizer, prompt, args):
    device  = next(model.parameters()).device
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    Thread(target=model.generate, kwargs=dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.rep_pen,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    ), daemon=True).start()

    t0, n_tok, buf = time.time(), 0, ""
    for chunk in streamer:
        buf   += chunk
        n_tok += 1
        yield chunk
    elapsed = time.time() - t0
    yield f"\n\n[dim]({n_tok} tok · {elapsed:.1f}s · {n_tok/max(elapsed,0.01):.1f} tok/s)[/dim]"
    return buf

# ── eval ───────────────────────────────────────────────────────────────────────
def run_eval(model, tokenizer, args, n_samples=60):
    if not TEST_FILE.exists():
        console.print("[yellow]Test file not found.[/]")
        return

    console.print(Rule("[bold cyan]Evaluation"))
    import random
    random.seed(42)

    records = [json.loads(l) for l in TEST_FILE.read_text().splitlines() if l.strip()]
    samples = random.sample(records, min(n_samples, len(records)))

    device = next(model.parameters()).device
    total_loss = total_tok = 0

    with console.status(f"[cyan]Perplexity on {len(samples)} samples...", spinner="dots"):
        for rec in samples:
            text = rec.get("text") or rec.get("content") or rec.get("prompt", "")
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            n   = ids["input_ids"].shape[1]
            if n < 4:
                continue
            with torch.no_grad():
                loss = model(**ids, labels=ids["input_ids"]).loss.item()
            total_loss += loss * n
            total_tok  += n

    if not total_tok:
        console.print("[red]No valid samples.[/]")
        return

    ppl = math.exp(total_loss / total_tok)
    color = "green" if ppl < 25 else "yellow" if ppl < 50 else "red"

    t = Table(box=box.ROUNDED, show_header=False, title="Perplexity Result")
    t.add_row("Samples",              str(len(samples)))
    t.add_row("Avg cross-entropy",    f"{total_loss/total_tok:.4f}")
    t.add_row("Perplexity",           f"[{color}]{ppl:.2f}[/]")
    t.add_row("Interpretation",
              "[green]Great[/]"   if ppl < 25 else
              "[yellow]Good[/]"   if ppl < 50 else
              "[red]Needs work[/]")
    console.print(t)

    # Qualitative probes
    probes = [
        "What's your view on consciousness?",
        "Tell me about your framework.",
        "What makes a well-designed system?",
        "How do you tackle a problem you've never seen before?",
        "What do you think about AI and creativity?",
    ]
    system = "You are a helpful, thoughtful assistant."
    console.print(Rule("[bold cyan]Qualitative Probes"))

    for q in probes:
        console.print(f"\n[bold yellow]▶  {q}[/]")
        prompt = build_prompt(tokenizer, [{"role": "user", "content": q}], system=system)
        ids    = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=200, temperature=0.7,
                top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        console.print(Panel(reply, border_style="magenta", padding=(0, 1)))

    console.print(Rule())

# ── banner ─────────────────────────────────────────────────────────────────────
BANNER = """\
[bold cyan]
  ██████╗ ██╗  ██╗██╗███████╗ ██████╗ ███╗   ███╗███████╗
  ██╔══██╗██║  ██║██║╚══███╔╝██╔═══██╗████╗ ████║██╔════╝
  ██████╔╝███████║██║  ███╔╝ ██║   ██║██╔████╔██║█████╗
  ██╔══██╗██╔══██║██║ ███╔╝  ██║   ██║██║╚██╔╝██║██╔══╝
  ██║  ██║██║  ██║██║███████╗╚██████╔╝██║ ╚═╝ ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝[/bold cyan]
[dim]  Fine-tuned on your own conversations  ·  /clear  /eval  /info  /quit[/dim]
"""

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    console.print(BANNER)

    model, tokenizer, base_name, use_quant = load_model(args)
    device   = next(model.parameters()).device
    quant_str = "4-bit" if use_quant else "bf16"

    console.print(Panel(
        f"[cyan]Base:[/]     {base_name}\n"
        f"[cyan]Adapter:[/]  {args.model}\n"
        f"[cyan]Device:[/]   {device}   [cyan]Quant:[/] {quant_str}\n"
        f"[cyan]Temp:[/] {args.temp}  [cyan]Top-p:[/] {args.top_p}  "
        f"[cyan]Rep-pen:[/] {args.rep_pen}  [cyan]Max-new:[/] {args.max_new}",
        title="[bold]Ready[/]", border_style="green", padding=(0, 1)
    ))

    SYSTEM = (
        "You are an intelligent assistant fine-tuned on the user's own conversations. "
        "Be direct, insightful, and speak naturally."
    )
    history = []

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]You ▶[/]  ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Bye.[/]")
            break

        if cmd in ("/clear", "/reset"):
            history.clear()
            console.print("[dim]History cleared.[/]")
            continue

        if cmd == "/eval":
            run_eval(model, tokenizer, args)
            continue

        if cmd in ("/info", "/settings"):
            console.print(Panel(
                f"Base: {base_name}\nDevice: {device} | Quant: {quant_str}\n"
                f"Temp: {args.temp} | Top-p: {args.top_p} | Rep-pen: {args.rep_pen} | Max-new: {args.max_new}\n"
                f"History turns: {len(history)//2}",
                title="Settings", border_style="cyan"
            ))
            continue

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(tokenizer, history, system=SYSTEM)

        console.print()
        console.print("[bold magenta]Rhizome ▶[/]  ", end="")

        chunks = []
        for chunk in stream_reply(model, tokenizer, prompt, args):
            console.print(chunk, end="", markup=True)
            chunks.append(chunk)
        console.print()

        # strip stats line, store clean reply in history
        full = "".join(chunks)
        clean = full.rsplit("\n\n", 1)[0].strip()
        if clean:
            history.append({"role": "assistant", "content": clean})


if __name__ == "__main__":
    main()
