#!/usr/bin/env python3
"""
build_dataset.py — one-command dataset builder for RhizomeML.

Usage:
    python build_dataset.py [--force] [--skip-embed] [--skip-conv] [--skip-pdfs] [--no-gzip]

Steps:
    1. batch_embedder.py      conversations → memory_texts.npy / memory.index
    2. data_formatter.py      memory        → data_finetune/conv_{train,validation,test}.jsonl
    3. adaptive_semantic.py   each PDF      → data_finetune/pdf_{name}_qa_{split}.jsonl
    4. Merge all splits       → data_finetune/dataset_{train,validation,test}.jsonl
    5. Print summary
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data_finetune"
PDF_DIR  = ROOT / "PDFs"
PYTHON   = sys.executable

SPLITS = ["train", "validation", "test"]


# ── helpers ────────────────────────────────────────────────────────────────────

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)


def safe_name(pdf_path: Path) -> str:
    """Filesystem-safe short name derived from PDF stem."""
    s = pdf_path.stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:60]


def open_jsonl(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def read_jsonl(path: Path):
    records = []
    try:
        with open_jsonl(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as exc:
        print(f"  [WARN] Could not read {path.name}: {exc}")
    return records


def find_file(directory: Path, name: str):
    """Return path if name or name+'.gz' exists, else None."""
    p = directory / name
    if p.exists():
        return p
    pg = directory / (name + ".gz")
    if pg.exists():
        return pg
    return None


# ── merge ──────────────────────────────────────────────────────────────────────

def merge_splits(no_gzip: bool):
    print(f"\n{'='*60}")
    print("  Step 4 · Merging splits → dataset_{{train,validation,test}}.jsonl")
    print(f"{'='*60}\n")

    total_written = {}

    for split in SPLITS:
        # Locate conv file produced by data_formatter
        conv_file = find_file(DATA_DIR, f"conv_{split}.jsonl")

        # Locate pdf QA files produced by adaptive_semantic
        pdf_qa_files = sorted(DATA_DIR.glob(f"pdf_*_qa_{split}.jsonl*"))

        all_files = ([conv_file] if conv_file else []) + pdf_qa_files

        if not all_files:
            print(f"  [{split}] No source files found — skipping.")
            total_written[split] = 0
            continue

        out_name = f"dataset_{split}.jsonl" + ("" if no_gzip else ".gz")
        out_path = DATA_DIR / out_name

        opener = (
            gzip.open(out_path, "wt", encoding="utf-8")
            if not no_gzip
            else open(out_path, "w", encoding="utf-8")
        )

        n_written = 0
        with opener as fout:
            for src in all_files:
                label = src.name.replace(".gz", "").replace(".jsonl", "")
                # strip trailing split suffix for cleaner label
                for s in SPLITS:
                    label = label.removesuffix(f"_{s}")
                records = read_jsonl(src)
                for rec in records:
                    fout.write(json.dumps(rec) + "\n")
                n_written += len(records)
                print(f"  [{split}] {label:<50} {len(records):>7,} records")

        total_written[split] = n_written
        print(f"  [{split}] {'TOTAL':<50} {n_written:>7,} → {out_path.name}\n")

    return total_written


# ── summary ────────────────────────────────────────────────────────────────────

def print_summary(total_written):
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    grand = 0
    for split in SPLITS:
        n = total_written.get(split, 0)
        grand += n
        bar = "█" * min(40, n // max(1, max(total_written.values()) // 40))
        print(f"  {split:<12} {n:>8,}  {bar}")
    print(f"  {'─'*54}")
    print(f"  {'TOTAL':<12} {grand:>8,}")
    print(f"\n  Output: {DATA_DIR}/")
    print(f"{'='*60}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build RhizomeML fine-tune dataset end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--force",       action="store_true", help="Re-process everything, ignore cached outputs")
    ap.add_argument("--skip-embed",  action="store_true", help="Skip batch_embedder.py")
    ap.add_argument("--skip-conv",   action="store_true", help="Skip data_formatter.py")
    ap.add_argument("--skip-pdfs",   action="store_true", help="Skip PDF processing")
    ap.add_argument("--skip-merge",  action="store_true", help="Skip final merge step")
    ap.add_argument("--no-gzip",     action="store_true", help="Write plain .jsonl instead of .jsonl.gz")
    ap.add_argument("--pdf-workers", type=int, default=None, help="Worker threads for adaptive_semantic.py")
    ap.add_argument("--pdf",         metavar="NAME", default=None,
                    help="Process only this PDF filename (basename), e.g. 'mybook.pdf'")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: embed conversations ───────────────────────────────────────────
    # NOTE: adaptive_semantic.py overwrites memory_texts.npy / memory.index on
    # each PDF run, so we must always re-embed conversations BEFORE formatting,
    # even if the files already exist.  Skip only when the user explicitly asks.
    if args.skip_embed:
        print("[skip] batch_embedder.py  (--skip-embed)")
    else:
        run([PYTHON, "batch_embedder.py"], "Step 1 · batch_embedder.py")

    # ── Step 2: format conversations ──────────────────────────────────────────
    conv_done = (
        find_file(DATA_DIR, "conv_train.jsonl") is not None
        or find_file(DATA_DIR, "conv_train.jsonl.gz") is not None
    )

    if args.skip_conv:
        print("[skip] data_formatter.py  (--skip-conv)")
    elif conv_done and not args.force:
        print("[skip] data_formatter.py  — conv_train.jsonl already exists  (--force to redo)")
    else:
        cmd = [
            PYTHON, "data_formatter.py",
            "--output-dir",    str(DATA_DIR),
            "--output-prefix", "conv",
            "--enable-semantic-labeling",
            "--semantic-mode",   "normal",
            "--semantic-method", "hybrid",
            "--extract-keyphrases",
        ]
        if args.no_gzip:
            cmd.append("--no-gzip")
        run(cmd, "Step 2 · data_formatter.py")

    # ── Step 3: PDF Q&A ───────────────────────────────────────────────────────
    if args.skip_pdfs:
        print("[skip] PDF processing  (--skip-pdfs)")
    else:
        if args.pdf:
            pdf_files = [PDF_DIR / args.pdf]
            if not pdf_files[0].exists():
                print(f"[ERROR] --pdf '{args.pdf}' not found in {PDF_DIR}")
                sys.exit(1)
        else:
            pdf_files = sorted(PDF_DIR.glob("*.pdf"))

        if not pdf_files:
            print(f"[warn] No PDFs found in {PDF_DIR}")
        else:
            print(f"\nFound {len(pdf_files)} PDF(s) in {PDF_DIR}")

            for pdf in pdf_files:
                sname    = safe_name(pdf)
                qa_train = find_file(DATA_DIR, f"pdf_{sname}_qa_train.jsonl")
                done     = qa_train is not None

                if done and not args.force:
                    print(f"[skip] {pdf.name}  — output already exists  (--force to redo)")
                    continue

                prefix = str(DATA_DIR / f"pdf_{sname}")
                cmd = [
                    PYTHON, str(ROOT / "PDFs" / "adaptive_semantic.py"),
                    "--pdf-dir",               str(PDF_DIR),
                    "--pdf-file",              pdf.name,
                    "--output-prefix",         prefix,
                    "--enable-semantic-labeling",
                    "--semantic-mode",         "normal",
                    "--semantic-method",       "hybrid",
                    "--extract-keyphrases",
                    "--qa-max-pairs-per-source", "20000",
                ]
                if args.pdf_workers:
                    cmd += ["--workers", str(args.pdf_workers)]
                if args.no_gzip:
                    cmd.append("--no-gzip")

                run(cmd, f"Step 3 · {pdf.name}")

    # ── Step 4: merge ─────────────────────────────────────────────────────────
    if args.skip_merge:
        print("[skip] merge step  (--skip-merge)")
        return

    total_written = merge_splits(args.no_gzip)

    # ── Step 5: summary ───────────────────────────────────────────────────────
    print_summary(total_written)


if __name__ == "__main__":
    main()
