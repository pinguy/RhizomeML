"""Standalone eval runner — runs chat.py's eval against the latest checkpoint."""
import sys
sys.argv = ["run_eval.py", "--model", "RhizomeML-finetuned/checkpoint-15900"]

from chat import parse_args, load_model, run_eval
from rich.console import Console
console = Console()

args = parse_args()
console.print(f"[cyan]Loading checkpoint: {args.model}[/]")
model, tokenizer, base_name, use_quant = load_model(args)
console.print(f"[green]Model loaded. Base: {base_name}  Quant: {'4-bit' if use_quant else 'bf16'}[/]")
run_eval(model, tokenizer, args)
