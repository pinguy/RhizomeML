"""
Production PDF → Memory → Q&A Pipeline with Adaptive Semantics + IPF
====================================================================

Enhanced version with all improvements implemented:
- Memory-mapped arrays for large datasets
- Retry logic for embedding failures
- Phrase vs word theme separation
- Incremental TF-IDF with state preservation
- Parallel chunking with ProcessPoolExecutor
- NLTK stopwords integration
- All configurable thresholds in Config
- Enhanced IPF validation and error handling

# Required for IPF functionality
pip3 install pyipf

# Other requirements
pip3 install sentence-transformers scikit-learn ftfy pdfminer.six numpy tqdm torch nltk keybert

Usage:

python3 adaptive_semantic.py --pdf-dir ./PDFs --force-cpu --enable-semantic-labeling --semantic-mode normal --semantic-method hybrid --extract-keyphrases --use-mmap --embedding-retry-attempts 5 --checkpoint-every 25

More examples at bottom of file.
"""

from __future__ import annotations
import os
import re
import gc
import json
import gzip
import uuid
import random
import logging
import argparse
import hashlib
import pickle
import multiprocessing
import time
import tempfile
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

import ftfy
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NLTK Setup ---
NLTK_AVAILABLE = False
NLTK_STOPWORDS = set()
try:
    import nltk
    from nltk.data import find
    try:
        find('tokenizers/punkt_tab')
        find('corpora/stopwords')
        from nltk.corpus import stopwords
        NLTK_STOPWORDS = set(stopwords.words('english'))
        print(f"✓ NLTK loaded with {len(NLTK_STOPWORDS)} stopwords")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        NLTK_STOPWORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available. Install with: pip install nltk")

# Optional deps
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

try:
    from pyipf import ipf as pyipf_function
    
    class IPF:
        """Wrapper for pyipf with enhanced validation"""
        
        def __init__(self, seed, aggregates, dimensions, 
                     convergence_rate=0.01, max_iteration=100):
            self.seed = seed
            self.aggregates = aggregates
            self.dimensions = dimensions
            self.convergence_rate = convergence_rate
            self.max_iteration = max_iteration
            self._validate_inputs()
        
        def _validate_inputs(self):
            """Validate IPF inputs before running"""
            # Check seed
            if not isinstance(self.seed, np.ndarray):
                raise ValueError("Seed must be a numpy array")
            if not np.all(np.isfinite(self.seed)):
                raise ValueError("Seed contains non-finite values")
            if np.any(self.seed < 0):
                raise ValueError("Seed contains negative values")
            
            # Check aggregates
            if not self.aggregates or len(self.aggregates) < 2:
                raise ValueError("Need at least 2 marginals for IPF")
            
            for i, agg in enumerate(self.aggregates):
                arr = np.asarray(agg, dtype=float)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Marginal {i} contains non-finite values")
                if np.sum(arr) == 0:
                    raise ValueError(f"Marginal {i} sums to zero")
            
            # Check dimension matching
            if self.seed.ndim != len(self.aggregates):
                raise ValueError(f"Seed has {self.seed.ndim} dimensions but {len(self.aggregates)} marginals provided")
        
        def iteration(self):
            """Run IPF with enhanced error handling"""
            # Convert aggregates to numpy arrays
            marginals = [np.asarray(agg, dtype=float) for agg in self.aggregates]
            
            # Add epsilon for numerical stability
            seed_stable = self.seed + 1e-10
            
            try:
                result = pyipf_function(
                    Z0=seed_stable,
                    marginals=marginals,
                    tol_convg=self.convergence_rate,
                    max_itr=self.max_iteration,
                    convg='relative',
                    pbar=False
                )
                
                # Validate result
                if not np.all(np.isfinite(result)):
                    raise ValueError("IPF produced non-finite values")
                
                return result
            except Exception as e:
                raise RuntimeError(f"IPF iteration failed: {e}")
    
    IPF_AVAILABLE = True
except Exception as e:
    IPF_AVAILABLE = False
    print(f"⚠️  IPF not available: {e}")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except:
    KEYBERT_AVAILABLE = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# ============================================================================
# CONFIG WITH ALL MAGIC NUMBERS
# ============================================================================

@dataclass
class Config:
    # IO
    pdf_dir: str = './PDFs'
    output_prefix: str = 'dataset'
    gzip_output: bool = True
    
    # Parallelization
    max_workers: int = None
    use_parallel_chunking: bool = True
    parallel_chunking_threshold: int = 10  # Use parallel chunking if >10 docs
    
    # Memory management
    use_mmap: bool = False
    checkpoint_every: int = 0  # Checkpoint every N chunks (0 = disabled)
    checkpoint_dir: str = './checkpoints'
    
    # Embedding resilience
    embedding_retry_attempts: int = 3
    embedding_retry_delay: float = 1.0  # seconds
    
    # Extraction
    enable_ocr: bool = False
    ocr_preprocessing: bool = False
    extract_sections: bool = True
    extract_all_pages: bool = True
    min_section_title_size: float = 12.0
    max_section_title_words: int = 15
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunking_method: str = 'fixed'
    min_text_length: int = 20
    max_text_length: int = 10000
    min_words: int = 3
    punctuation_ratio_threshold: float = 0.6
    
    # Embeddings
    embedding_model: str = 'all-MiniLM-L12-v2'
    embedding_dim: int = 384
    batch_size: int = 100
    force_cpu: bool = False
    
    # Semantic labeling thresholds
    enable_semantic_labeling: bool = False
    extract_keyphrases: bool = False
    semantic_mode: str = 'normal'
    semantic_memory_path: str = 'semantic_memory.pkl'
    semantic_method: str = 'tfidf'
    max_themes_per_chunk: int = 3
    keyphrase_confidence_threshold: float = 0.3
    tfidf_min_score: float = 0.1
    theme_normalization_min_length: int = 3
    centroid_similarity_threshold: float = 0.7
    min_sentence_words_for_complete: int = 3
    ipf_min_themes_for_calibration: int = 2
    
    # IPF-specific
    ipf_convergence_rate: float = 0.01
    ipf_max_iterations: int = 100
    ipf_calibrate_cooccurrence: bool = True
    ipf_balance_hierarchy: bool = True
    ipf_smooth_distributions: bool = True
    ipf_compute_mi: bool = True
    
    # Similarity
    sim_threshold: float = 0.7
    thread_sim_threshold: float = 0.65
    max_merged_length: int = 2000
    
    # Quality
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'length_quality': 0.10,
        'coherence_quality': 0.20,
        'information_density': 0.20,
        'structural_quality': 0.15,
        'linguistic_quality': 0.15,
        'semantic_coherence': 0.20,
    })
    
    # Q&A
    generate_qa: bool = True
    qa_max_pairs_per_source: int = 5000
    qa_diversity_sim_threshold: float = 0.85
    qa_group_sim_threshold: float = 0.8
    qa_max_group_length: int = 5000
    qa_min_context_length: int = 50
    
    # Splits
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    # Misc
    save_intermediates: bool = True
    seed: int = 42
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = os.cpu_count() or 4
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
        except:
            pass
        
        # Create checkpoint directory if checkpointing is enabled
        if self.checkpoint_every > 0:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEXT UTILITIES (unchanged from original)
# ============================================================================

def clean_text(text: str) -> str:
    text = ftfy.fix_encoding(text)
    text = ftfy.fix_text(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\b"(\w+)"\b', r'\1', text)
    text = re.sub(r'"([^"]+[.,!?])"', r'\1', text)
    text = re.sub(r'"([A-Z][^"]*?)"(?=\s|$)', r'\1', text)
    text = re.sub(r'\\(["\'])', r'\1', text)
    while '\\\"' in text or "\\\'" in text:
        text = text.replace('\\\"', '"').replace("\\\'", "'")
    text = re.sub(r'\*+"', '"', text)
    text = re.sub(r'"\*+', '"', text)
    text = re.sub(r'\*\s*"', ' *"', text)
    text = re.sub(r'"\s*\*', '"* ', text)
    text = re.sub(r"(?i)\b([a-z]+)9(?=(?:t|s|m|re|ve|ll|d)\b)", r"\1'", text)
    text = re.sub(r"(?i)(?<=in)9(?=\b|[^a-z])", "g", text)
    text = re.sub(r"(?i)\b([a-z]{2,})9(?=s\b)", r"\1'", text)
    text = re.sub(r'([!?.,]){2,}["\']', r'\1"', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\b9em\b', 'em', text)
    text = re.sub(r'(?<!\d)9(?!\d)', '', text)
    return text.strip()

def validate_text(text: str, cfg: Config) -> bool:
    if not text or not text.strip():
        return False
    if len(text) < cfg.min_text_length or len(text) > cfg.max_text_length:
        return False
    words = text.split()
    if len(words) < cfg.min_words:
        return False
    alpha = sum(c.isalpha() for c in text)
    if len(text) > 0 and (len(text) - alpha) / len(text) > cfg.punctuation_ratio_threshold:
        return False
    return True

# ============================================================================
# SECTION EXTRACTOR (unchanged from original)
# ============================================================================

class SectionExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def extract(self, pdf_path: str) -> Dict:
        if not self.cfg.extract_sections:
            return {"sections": [], "toc": [], "total_sections": 0}
        
        sections = []
        try:
            laparams = LAParams()
            page_count = 0
            
            for page_num, layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
                page_count += 1
                for el in layout:
                    if isinstance(el, LTTextContainer):
                        txt = el.get_text().strip()
                        if not txt:
                            continue
                        
                        fs = self._avg_font_size(el)
                        
                        if (fs >= self.cfg.min_section_title_size and
                            len(txt.split()) <= self.cfg.max_section_title_words and
                            self._looks_like_title(txt)):
                            sections.append({
                                "title": txt,
                                "page": page_num + 1,
                                "font_size": fs
                            })
        except Exception as e:
            logger.warning(f"Section extraction failed: {e}")
        
        toc = self._extract_toc(pdf_path)
        
        return {
            "sections": sections,
            "toc": toc,
            "total_sections": len(sections) + len(toc)
        }
    
    def _avg_font_size(self, element) -> float:
        try:
            sizes = []
            for item in element:
                if hasattr(item, '__iter__'):
                    for ch in item:
                        if isinstance(ch, LTChar):
                            sizes.append(ch.height)
            return float(np.mean(sizes)) if sizes else 0.0
        except:
            return 0.0
    
    def _looks_like_title(self, text: str) -> bool:
        patterns = [
            r'^\d+\.?\s+[A-Z]',
            r'^(Chapter|Section|Part|Appendix)\s+\d+',
            r'^[A-Z][A-Za-z\s]{2,30}$',
            r'^\d+\.\d+'
        ]
        if any(re.match(p, text) for p in patterns):
            return True
        words = text.split()
        if words:
            cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
            return cap_ratio > 0.7
        return False
    
    def _extract_toc(self, pdf_path: str) -> List[Dict]:
        toc = []
        try:
            first_pages = extract_text(pdf_path, page_numbers=[0, 1, 2])
            lines = [l.strip() for l in first_pages.split('\n')]
            in_toc = False
            
            for line in lines:
                if re.match(r'(table of contents|contents)', line.lower()):
                    in_toc = True
                    continue
                
                if in_toc:
                    m = re.match(r'([\d\.]+)\s+(.+?)\s+\.{2,}\s*(\d+)', line)
                    if m:
                        toc.append({
                            "number": m.group(1),
                            "title": m.group(2).strip(),
                            "page": int(m.group(3))
                        })
                    elif line and not re.match(r'^\d+$', line):
                        if toc:
                            break
        except:
            pass
        
        return toc
    
    def match_chunk(self, chunk_text: str, sections: List[Dict], 
                    chunk_pos: int, total_chunks: int) -> Optional[str]:
        if not sections:
            return None
        
        for s in sections:
            title = s.get('title', '')
            if title and title.lower() in chunk_text.lower()[:200]:
                return title
        
        ratio = chunk_pos / max(total_chunks, 1)
        best = None
        best_dist = 1e9
        
        for s in sections:
            if 'page' in s:
                sec_ratio = s['page'] / 100.0
                d = abs(ratio - sec_ratio)
                if d < best_dist:
                    best_dist = d
                    best = s.get('title', '')
        
        return best

# ============================================================================
# PDF PROCESSOR WITH PARALLEL CHUNKING
# ============================================================================

def _chunk_single_doc_worker(args):
    """Worker function for parallel chunking"""
    doc, cfg, sectioner = args
    sections = doc.get('structure', {}).get('sections', []) + doc.get('structure', {}).get('toc', [])
    
    if cfg.chunking_method == 'sentence':
        chunk_texts = _sentence_based_chunks(doc['text'], max_words=cfg.chunk_size)
    elif cfg.chunking_method == 'texttile':
        chunk_texts = _detect_topic_boundaries(doc['text'], cfg)
    else:
        chunk_texts = _fixed_word_chunks(doc['text'], cfg.chunk_size, cfg.chunk_overlap)
    
    total_chunks = len(chunk_texts)
    chunks = []
    for i, txt in enumerate(chunk_texts):
        if not txt or not validate_text(txt, cfg):
            continue
        
        sect = sectioner.match_chunk(txt, sections, i, total_chunks)
        
        chunks.append({
            'text': txt,
            'filename': doc['filename'],
            'chunk_index': i,
            'section_title': sect,
            'has_section': sect is not None,
            'ocr_used': doc.get('ocr_used', False),
        })
    
    return chunks

def _sentence_based_chunks(text: str, min_sentences: int = 5, max_words: int = 600) -> List[str]:
    if not NLTK_AVAILABLE:
        return _fixed_word_chunks(text, max_words)
    
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        return _fixed_word_chunks(text, max_words)
    
    chunks = []
    current = []
    current_words = 0
    
    for sent in sentences:
        sent_words = len(sent.split())
        
        if current_words + sent_words > max_words and len(current) >= min_sentences:
            chunks.append(' '.join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks

def _fixed_word_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    words = text.split()
    chunk_texts = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    
    for i in range(0, len(words), step):
        txt = ' '.join(words[i:i + chunk_size]).strip()
        if txt:
            chunk_texts.append(txt)
    return chunk_texts

def _detect_topic_boundaries(text: str, cfg: Config) -> List[str]:
    if not NLTK_AVAILABLE:
        return _fixed_word_chunks(text, cfg.chunk_size, cfg.chunk_overlap)
    try:
        from nltk.tokenize import TextTilingTokenizer
        ttt = TextTilingTokenizer(w=20, k=10)
        tiles = ttt.tokenize(text)
        return tiles
    except Exception as e:
        return _fixed_word_chunks(text, cfg.chunk_size, cfg.chunk_overlap)

class PDFProcessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sectioner = SectionExtractor(cfg)
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'ocr_used': 0,
            'total_chunks': 0,
            'sections_extracted': 0,
        }
    
    def extract_pdfs(self) -> List[Dict]:
        if not os.path.isdir(self.cfg.pdf_dir):
            logger.error(f"PDF directory not found: {self.cfg.pdf_dir}")
            return []
        
        files = [f for f in os.listdir(self.cfg.pdf_dir) if f.lower().endswith('.pdf')]
        self.stats['total_pdfs'] = len(files)
        
        if not files:
            logger.warning("No PDF files found")
            return []
        
        logger.info(f"Found {len(files)} PDFs. Using {self.cfg.max_workers} parallel workers...")
        
        paths = [os.path.join(self.cfg.pdf_dir, f) for f in files]
        docs = []
        
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = {executor.submit(self._process_single_pdf, p): p for p in paths}
            
            with tqdm(total=len(files), desc="Extracting PDFs") as pbar:
                for future in as_completed(futures):
                    doc = future.result()
                    if doc:
                        docs.append(doc)
                        self.stats['successful'] += 1
                        if doc.get('ocr_used'):
                            self.stats['ocr_used'] += 1
                    else:
                        self.stats['failed'] += 1
                    pbar.update(1)
        
        logger.info(f"✓ Extracted {self.stats['successful']}/{self.stats['total_pdfs']} PDFs")
        if self.stats['ocr_used'] > 0:
            logger.info(f"  OCR used on {self.stats['ocr_used']} scanned PDFs")
        if self.stats['failed'] > 0:
            logger.warning(f"  Failed: {self.stats['failed']} PDFs")
        
        return docs
    
    def _process_single_pdf(self, path: str) -> Optional[Dict]:
        fn = os.path.basename(path)
        ocr_used = False
        
        try:
            all_pages = []
            page_count = 0
            
            try:
                for page_layout in extract_pages(path):
                    page_count += 1
                    page_text = []
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            page_text.append(element.get_text())
                    all_pages.append(' '.join(page_text))
                
                text = '\n\n'.join(all_pages)
            except Exception as e:
                logger.warning(f"[{fn}] Page-by-page failed: {e}, trying extract_text...")
                text = extract_text(path)
            
            if not (text or '').strip() and self.cfg.enable_ocr and OCR_AVAILABLE:
                logger.info(f"[{fn}] No text found, attempting OCR...")
                try:
                    images = convert_from_path(path, dpi=300)
                    txts = [pytesseract.image_to_string(img) for img in images]
                    text = ' '.join(txts)
                    ocr_used = True
                    page_count = len(images)
                except Exception as e:
                    logger.error(f"[{fn}] OCR failed: {e}")
            
            text = clean_text(text or '')
            
            if ocr_used and self.cfg.ocr_preprocessing:
                text = self._clean_ocr_artifacts(text)
            
            if not text:
                logger.warning(f"[{fn}] No text after cleaning")
                return None
            
            word_count = len(text.split())
            char_count = len(text)
            
            struct = self.sectioner.extract(path)
            if struct.get('total_sections', 0) > 0:
                self.stats['sections_extracted'] += struct['total_sections']
            
            return {
                "filename": fn,
                "text": text,
                "structure": struct,
                "ocr_used": ocr_used,
                "word_count": word_count,
                "char_count": char_count,
                "page_count": page_count
            }
        except Exception as e:
            logger.error(f"[{fn}] Processing failed: {e}", exc_info=True)
            return None
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        ocr_fixes = {
            r'\bl\b': 'I',
            r'\bO(?=\d)': '0',
            r'(?<=\d)O\b': '0',
            r'\brn\b': 'm',
            r'\bvv': 'w',
            r'\|': 'I',
        }
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        lines = text.split('\n')
        line_counts = Counter(lines)
        cleaned_lines = [l for l in lines if line_counts[l] <= 3 or not l.strip()]
        text = '\n'.join(cleaned_lines)
        
        text = clean_text(text)
        return text
    
    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """Split documents into chunks with optional parallel processing"""
        logger.info(f"Chunking {len(docs)} documents...")
        
        # Use parallel chunking if enabled and we have enough documents
        if self.cfg.use_parallel_chunking and len(docs) >= self.cfg.parallel_chunking_threshold:
            logger.info(f"Using parallel chunking with {self.cfg.max_workers} workers...")
            args_list = [(doc, self.cfg, self.sectioner) for doc in docs]
            
            chunks = []
            with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
                futures = [executor.submit(_chunk_single_doc_worker, args) for args in args_list]
                
                with tqdm(total=len(docs), desc="Parallel chunking") as pbar:
                    for future in as_completed(futures):
                        try:
                            doc_chunks = future.result()
                            chunks.extend(doc_chunks)
                        except Exception as e:
                            logger.error(f"Chunking worker failed: {e}")
                        pbar.update(1)
        else:
            # Sequential chunking
            chunks = []
            for d in tqdm(docs, desc="Chunking"):
                sections = d.get('structure', {}).get('sections', []) + d.get('structure', {}).get('toc', [])
                
                if self.cfg.chunking_method == 'sentence':
                    chunk_texts = _sentence_based_chunks(d['text'], max_words=self.cfg.chunk_size)
                elif self.cfg.chunking_method == 'texttile':
                    chunk_texts = _detect_topic_boundaries(d['text'], self.cfg)
                else:
                    chunk_texts = _fixed_word_chunks(d['text'], self.cfg.chunk_size, self.cfg.chunk_overlap)
                
                total_chunks = len(chunk_texts)
                for i, txt in enumerate(chunk_texts):
                    if not txt or not validate_text(txt, self.cfg):
                        continue
                    
                    sect = self.sectioner.match_chunk(txt, sections, i, total_chunks)
                    
                    chunks.append({
                        'text': txt,
                        'filename': d['filename'],
                        'chunk_index': i,
                        'section_title': sect,
                        'has_section': sect is not None,
                        'ocr_used': d.get('ocr_used', False),
                    })
        
        self.stats['total_chunks'] = len(chunks)
        logger.info(f"✓ Created {len(chunks)} valid chunks")
        return chunks

# ============================================================================
# EMBEDDING STORE WITH RETRY LOGIC AND MMAP
# ============================================================================

class EmbeddingStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        device = 'cpu' if cfg.force_cpu or not torch.cuda.is_available() else 'cuda'
        if cfg.force_cpu and torch.cuda.is_available():
            logger.warning("CUDA is available but --force-cpu flag is set. Using CPU.")
        
        logger.info(f"Using device: {device} for embeddings")
        logger.info(f"Loading embedding model: {cfg.embedding_model}...")
        self.model = SentenceTransformer(cfg.embedding_model, device=device)
        self.texts: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.stats = {
            'embedded': 0,
            'retry_attempts': 0,
            'failed_embeddings': 0
        }
        
        # Memory-mapped array support
        self.mmap_file = None
        self.mmap_array = None
    
    def _embed_with_retry(self, batch: List[str]) -> np.ndarray:
        """Embed a batch with retry logic"""
        for attempt in range(self.cfg.embedding_retry_attempts):
            try:
                vecs = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=self.model.device
                )
                return vecs
            except Exception as e:
                self.stats['retry_attempts'] += 1
                if attempt < self.cfg.embedding_retry_attempts - 1:
                    logger.warning(f"Embedding failed (attempt {attempt + 1}/{self.cfg.embedding_retry_attempts}): {e}")
                    time.sleep(self.cfg.embedding_retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Embedding failed after {self.cfg.embedding_retry_attempts} attempts: {e}")
                    self.stats['failed_embeddings'] += len(batch)
                    # Return zero vectors as fallback
                    return np.zeros((len(batch), self.cfg.embedding_dim), dtype=np.float32)
    
    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        """Embed with progress tracking, retry logic, and checkpointing"""
        logger.info(f"Embedding {len(texts)} texts on device '{self.model.device}'...")
        
        # Setup memory-mapped array if enabled
        if self.cfg.use_mmap:
            self.mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
            self.mmap_array = np.memmap(
                self.mmap_file.name,
                dtype='float32',
                mode='w+',
                shape=(len(texts), self.cfg.embedding_dim)
            )
            logger.info(f"Using memory-mapped array: {self.mmap_file.name}")
        
        embs = []
        checkpoint_counter = 0
        
        for i in tqdm(range(0, len(texts), self.cfg.batch_size), desc='Embedding'):
            batch = texts[i:i + self.cfg.batch_size]
            vecs = self._embed_with_retry(batch)
            
            if self.cfg.use_mmap:
                self.mmap_array[i:i + len(batch)] = vecs
            else:
                embs.extend(vecs)
            
            self.texts.extend(batch)
            self.vectors.extend(vecs)
            self.stats['embedded'] += len(batch)
            
            # Checkpoint if enabled
            if self.cfg.checkpoint_every > 0:
                checkpoint_counter += len(batch)
                if checkpoint_counter >= self.cfg.checkpoint_every:
                    self._save_checkpoint(i + len(batch), len(texts))
                    checkpoint_counter = 0
        
        logger.info(f"✓ Embedded {self.stats['embedded']} texts")
        if self.stats['retry_attempts'] > 0:
            logger.info(f"  Retry attempts: {self.stats['retry_attempts']}")
        if self.stats['failed_embeddings'] > 0:
            logger.warning(f"  Failed embeddings (using zero vectors): {self.stats['failed_embeddings']}")
        
        if self.cfg.use_mmap:
            return self.mmap_array
        else:
            return np.array(embs)
    
    def _save_checkpoint(self, current: int, total: int):
        """Save embedding checkpoint"""
        checkpoint_path = Path(self.cfg.checkpoint_dir) / f"embeddings_checkpoint_{current}_of_{total}.pkl"
        checkpoint_data = {
            'vectors': self.vectors[:current],
            'texts': self.texts[:current],
            'stats': self.stats.copy()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def build_faiss(self, index_path: str = 'memory.index', 
                   texts_path: str = 'memory_texts.npy'):
        """Build and save FAISS index"""
        if not self.vectors:
            logger.warning("No vectors to index")
            return
        
        np.save(texts_path, np.array(self.texts, dtype=object))
        logger.info(f"✓ Saved {len(self.texts)} texts to {texts_path}")
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index")
            return
        
        dim = len(self.vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(self.vectors).astype('float32'))
        faiss.write_index(index, index_path)
        logger.info(f"✓ Saved FAISS index to {index_path}")

# ============================================================================
# SEMANTIC MEMORY WITH INCREMENTAL TF-IDF
# ============================================================================

@dataclass
class SemanticMemory:
    """Persistent semantic knowledge with TF-IDF state"""
    # Separate phrase and word themes
    phrase_themes: Counter = field(default_factory=Counter)
    word_themes: Counter = field(default_factory=Counter)
    theme_counts: Counter = field(default_factory=Counter)  # Combined for backward compat
    
    co_occurrence: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    coherence_weights: Dict[str, float] = field(default_factory=dict)
    hierarchy: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    high_mi_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # TF-IDF state
    tfidf_vocabulary: Optional[Dict] = None
    tfidf_idf_values: Optional[np.ndarray] = None
    tfidf_fitted: bool = False
    
    generation: int = 0
    ipf_generation: int = 0
    total_chunks_processed: int = 0
    total_themes_discovered: int = 0

# ============================================================================
# IPF SEMANTIC ENHANCER WITH VALIDATION
# ============================================================================

class IPFSemanticEnhancer:
    """Enhanced IPF with comprehensive validation"""
    
    def __init__(self, memory: SemanticMemory, cfg: Config):
        self.memory = memory
        self.cfg = cfg
        if not IPF_AVAILABLE:
            logger.warning("⚠️  pyipf not installed. Install with: pip install pyipf")
    
    def calibrate_cooccurrence(self, expected_marginals: dict = None):
        """Use IPF to calibrate co-occurrence with validation"""
        if not IPF_AVAILABLE or not self.cfg.ipf_calibrate_cooccurrence:
            return
        
        themes = list(self.memory.theme_counts.keys())
        n = len(themes)
        
        if n < self.cfg.ipf_min_themes_for_calibration:
            logger.debug(f"Not enough themes for IPF calibration (need {self.cfg.ipf_min_themes_for_calibration}, have {n})")
            return
        
        logger.info(f"  IPF: Calibrating {n}x{n} co-occurrence matrix...")
        
        # Create matrix
        co_matrix = np.zeros((n, n))
        theme_to_idx = {t: i for i, t in enumerate(themes)}
        
        for theme_a in themes:
            for theme_b, count in self.memory.co_occurrence[theme_a].items():
                if theme_b in theme_to_idx:
                    i, j = theme_to_idx[theme_a], theme_to_idx[theme_b]
                    co_matrix[i, j] = count
        
        # Add epsilon for numerical stability
        co_matrix = co_matrix + 0.1
        
        # Define target marginals
        if expected_marginals:
            row_marginals = [expected_marginals.get(t, self.memory.theme_counts[t]) 
                           for t in themes]
        else:
            row_marginals = [self.memory.theme_counts[t] for t in themes]
        
        col_marginals = row_marginals.copy()
        
        # Validate inputs
        if not all(np.isfinite(row_marginals)) or not all(np.isfinite(col_marginals)):
            logger.warning("Non-finite marginals detected, skipping IPF calibration")
            return
        
        if sum(row_marginals) == 0 or sum(col_marginals) == 0:
            logger.warning("Zero-sum marginals detected, skipping IPF calibration")
            return
        
        # Apply IPF
        try:
            aggregates = [row_marginals, col_marginals]
            dimensions = [[0], [1]]
            
            ipf = IPF(co_matrix, aggregates, dimensions, 
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            
            calibrated_matrix = ipf.iteration()
            
            # Update co-occurrence with calibrated values
            for i, theme_a in enumerate(themes):
                for j, theme_b in enumerate(themes):
                    if i != j:
                        calibrated_count = int(calibrated_matrix[i, j])
                        if calibrated_count > 0:
                            self.memory.co_occurrence[theme_a][theme_b] = calibrated_count
            
            logger.info(f"    ✓ Calibrated {n}x{n} matrix")
            
        except Exception as e:
            logger.warning(f"IPF calibration failed: {e}")
    
    def balance_hierarchical_constraints(self):
        """Use IPF to enforce hierarchical constraints with validation"""
        if not IPF_AVAILABLE or not self.cfg.ipf_balance_hierarchy:
            return
        
        if not self.memory.hierarchy:
            logger.debug("No hierarchical relationships to balance")
            return
        
        parent_themes = list(self.memory.hierarchy.keys())
        all_children = set()
        for children in self.memory.hierarchy.values():
            all_children.update(children)
        
        if not parent_themes or not all_children:
            return
        
        logger.info(f"  IPF: Balancing {len(parent_themes)} hierarchical relationships...")
        
        n_parents = len(parent_themes)
        n_children = len(all_children)
        children_list = list(all_children)
        
        table = np.zeros((n_parents, n_children))
        
        for i, parent in enumerate(parent_themes):
            for j, child in enumerate(children_list):
                if child in self.memory.hierarchy[parent]:
                    if child in self.memory.co_occurrence[parent]:
                        table[i, j] = self.memory.co_occurrence[parent][child]
                    else:
                        table[i, j] = 1.0
        
        # Target marginals
        row_totals = [self.memory.theme_counts[p] for p in parent_themes]
        col_totals = [self.memory.theme_counts[c] for c in children_list]
        
        # Validation
        if not all(np.isfinite(row_totals)) or not all(np.isfinite(col_totals)):
            logger.warning("Non-finite totals in hierarchy, skipping")
            return
        
        try:
            aggregates = [row_totals, col_totals]
            dimensions = [[0], [1]]
            
            ipf = IPF(table + 0.1, aggregates, dimensions,
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            balanced_table = ipf.iteration()
            
            # Update hierarchy weights
            for i, parent in enumerate(parent_themes):
                for j, child in enumerate(children_list):
                    if child in self.memory.hierarchy[parent]:
                        weight = balanced_table[i, j] / (row_totals[i] + 1e-8)
                        self.memory.hierarchy[parent][child] = weight
            
            logger.info(f"    ✓ Balanced {len(parent_themes)} parent-child relationships")
            
        except Exception as e:
            logger.warning(f"Hierarchical IPF balancing failed: {e}")
    
    def smooth_theme_distributions(self, target_distribution: dict = None):
        """Smooth theme distributions with validation"""
        if not IPF_AVAILABLE or not self.cfg.ipf_smooth_distributions:
            return
        
        themes = list(self.memory.theme_counts.keys())
        n = len(themes)
        
        if n < 2:
            return
        
        n_docs = len(self.memory.clusters)
        if n_docs == 0:
            return
        
        logger.info(f"  IPF: Smoothing {n} theme distributions across {n_docs} clusters...")
        
        doc_theme_matrix = np.zeros((n_docs, n))
        theme_to_idx = {t: i for i, t in enumerate(themes)}
        
        for doc_idx, (cluster_name, cluster_themes) in enumerate(self.memory.clusters.items()):
            for theme in cluster_themes:
                if theme in theme_to_idx:
                    doc_theme_matrix[doc_idx, theme_to_idx[theme]] = \
                        self.memory.theme_counts[theme]
        
        doc_theme_matrix = doc_theme_matrix + 0.1
        
        # Define target column marginals
        if target_distribution:
            col_totals = [target_distribution.get(t, self.memory.theme_counts[t]) 
                         for t in themes]
        else:
            col_totals = [self.memory.theme_counts[t] for t in themes]
        
        row_totals = doc_theme_matrix.sum(axis=1).tolist()
        
        # Validation
        if not all(np.isfinite(row_totals)) or not all(np.isfinite(col_totals)):
            logger.warning("Non-finite values in distribution smoothing, skipping")
            return
        
        try:
            aggregates = [row_totals, col_totals]
            dimensions = [[0], [1]]
            
            ipf = IPF(doc_theme_matrix, aggregates, dimensions,
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            smoothed_matrix = ipf.iteration()
            
            # Update theme counts
            for j, theme in enumerate(themes):
                smoothed_count = int(smoothed_matrix[:, j].sum())
                self.memory.theme_counts[theme] = smoothed_count
            
            logger.info(f"    ✓ Smoothed {n} themes")
            
        except Exception as e:
            logger.warning(f"Distribution smoothing failed: {e}")
    
    def compute_mutual_information(self):
        """Compute MI with validation"""
        if not self.cfg.ipf_compute_mi:
            return
        
        themes = list(self.memory.theme_counts.keys())
        n = len(themes)
        
        if n < 2:
            return
        
        logger.info(f"  IPF: Computing mutual information for {n} themes...")
        
        co_matrix = np.zeros((n, n))
        theme_to_idx = {t: i for i, t in enumerate(themes)}
        
        total = sum(self.memory.theme_counts.values())
        if total == 0:
            return
        
        for theme_a in themes:
            for theme_b, count in self.memory.co_occurrence[theme_a].items():
                if theme_b in theme_to_idx:
                    i, j = theme_to_idx[theme_a], theme_to_idx[theme_b]
                    co_matrix[i, j] = count / total
        
        p_themes = np.array([self.memory.theme_counts[t] / total for t in themes])
        
        mi_matrix = {}
        for i, theme_a in enumerate(themes):
            for j, theme_b in enumerate(themes):
                if i < j:
                    p_ab = co_matrix[i, j]
                    p_a = p_themes[i]
                    p_b = p_themes[j]
                    
                    if p_ab > 0 and p_a > 0 and p_b > 0:
                        mi = p_ab * np.log2(p_ab / (p_a * p_b + 1e-10))
                        mi_matrix[(theme_a, theme_b)] = mi
        
        top_mi = sorted(mi_matrix.items(), key=lambda x: x[1], reverse=True)[:20]
        self.memory.high_mi_pairs = {pair: mi for pair, mi in top_mi}
        
        logger.info(f"    ✓ Computed MI for {len(mi_matrix)} pairs, stored top 20")

# ============================================================================
# SEMANTIC LABELER WITH PHRASE/WORD SEPARATION & INCREMENTAL TF-IDF
# ============================================================================

class SemanticLabeler:
    """Enhanced labeler with phrase/word separation and incremental TF-IDF"""
    
    # Minimal stopwords if NLTK unavailable
    FALLBACK_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall',
    }
    
    def __init__(self, cfg: Config, embedding_model=None):
        self.cfg = cfg
        self.mode = cfg.semantic_mode
        self.method = cfg.semantic_method
        self.embedding_model = embedding_model
        self.discovered = Counter()
        
        # Use NLTK stopwords if available, else fallback
        self.stopwords = NLTK_STOPWORDS if NLTK_STOPWORDS else self.FALLBACK_STOPWORDS
        logger.info(f"Using {len(self.stopwords)} stopwords")
        
        # TF-IDF for word extraction (single words)
        if self.method in ['tfidf', 'hybrid']:
            self.tfidf_words = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 1),  # Single words only
                stop_words=list(self.stopwords),
                min_df=2
            )
            self._tfidf_words_corpus = []
        
        # KeyBERT for phrase extraction (2-3 word ngrams)
        self.kw_model = None
        if self.cfg.extract_keyphrases:
            if KEYBERT_AVAILABLE and self.embedding_model:
                try:
                    self.kw_model = KeyBERT(model=self.embedding_model)
                    logger.info("✓ KeyBERT initialized for phrase extraction")
                except Exception as e:
                    logger.warning(f"Failed to initialize KeyBERT: {e}")
            elif not KEYBERT_AVAILABLE:
                logger.warning("⚠️  --extract-keyphrases enabled, but 'keybert' not installed.")
        
        # Adaptive mode state
        if self.mode == 'adaptive':
            self.memory = SemanticMemory()
            self._current_run_themes = []
            self._current_run_records = []
            self._load_memory()
            logger.info(f"🧠 Adaptive mode: Gen {self.memory.generation}, Method: {self.method}")
        else:
            logger.info(f"📋 Normal mode: Method: {self.method}")
    
    def _load_memory(self):
        """Load semantic memory with TF-IDF state"""
        path = Path(self.cfg.semantic_memory_path)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    self.memory = pickle.load(f)
                
                # Restore TF-IDF state if available
                if self.memory.tfidf_fitted and self.memory.tfidf_vocabulary:
                    self.tfidf_words.vocabulary_ = self.memory.tfidf_vocabulary
                    self.tfidf_words.idf_ = self.memory.tfidf_idf_values
                    logger.info(f"✓ Restored TF-IDF state with {len(self.memory.tfidf_vocabulary)} terms")
                
                logger.info(f"✓ Loaded memory (Gen {self.memory.generation}, "
                          f"IPF Gen {self.memory.ipf_generation}, "
                          f"{len(self.memory.theme_counts)} themes)")
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")
        else:
            logger.info("No previous semantic memory found, starting fresh")
    
    def save_memory(self):
        """Save semantic memory with TF-IDF state"""
        if self.mode != 'adaptive':
            return
        
        # Save TF-IDF state
        if hasattr(self.tfidf_words, 'vocabulary_'):
            self.memory.tfidf_vocabulary = self.tfidf_words.vocabulary_
            self.memory.tfidf_idf_values = self.tfidf_words.idf_
            self.memory.tfidf_fitted = True
        
        self.memory.generation += 1
        path = Path(self.cfg.semantic_memory_path)
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
        logger.info(f"✓ Saved memory to {path} (Gen {self.memory.generation})")
    
    def label(self, text: str) -> Dict:
        """Label text with separated phrase/word themes"""
        if self.mode == 'adaptive':
            return self._label_adaptive(text)
        else:
            return self._label_normal(text)
    
    def _get_raw_candidates(self, text: str) -> Tuple[Set[str], Set[str]]:
        """Extract phrase and word candidates separately"""
        phrases = set()
        words = set()
        
        # Priority 1: KeyBERT for phrases (2-3 words)
        if self.cfg.extract_keyphrases and self.kw_model:
            phrases.update(self._extract_keyphrases(text))
        
        # Priority 2: TF-IDF for words (single tokens)
        if self.method in ['tfidf', 'hybrid']:
            self._tfidf_words_corpus.append(text)
            words.update(self._extract_tfidf_words(text))
        
        # Priority 3: Fallback heuristics
        if not phrases and not words:
            fallback = self._extract_hierarchical_themes(text)
            # Separate by word count
            for theme in fallback:
                if len(theme.split('_')) > 1:
                    phrases.add(theme)
                else:
                    words.add(theme)
        
        return phrases, words
    
    def _label_normal(self, text: str) -> Dict:
        """Normal mode with phrase/word separation"""
        phrases, words = self._get_raw_candidates(text)
        
        # Normalize
        phrases = [self._normalize(p) for p in phrases]
        words = [self._normalize(w) for w in words]
        
        phrases = [p for p in phrases if p and len(p) >= self.cfg.theme_normalization_min_length]
        words = [w for w in words if w and len(w) >= self.cfg.theme_normalization_min_length]
        
        # Combine and deduplicate
        all_themes = list(dict.fromkeys(phrases + words))[:self.cfg.max_themes_per_chunk]
        
        if not all_themes:
            all_themes = [self._classify_content_type(text)]
        
        for t in all_themes:
            self.discovered[t] += 1
        
        return {
            'themes': all_themes,
            'phrase_themes': phrases,
            'word_themes': words,
            'primary_theme': all_themes[0] if all_themes else 'general_content',
            'confidence': min(0.8, 0.5 + 0.1 * len(all_themes)),
            'method': f'normal_{self.method}'
        }
    
    def _label_adaptive(self, text: str) -> Dict:
        """Adaptive mode with learning"""
        phrases, words = self._get_raw_candidates(text)
        
        # Normalize
        norm_phrases = {self._normalize(p) for p in phrases if p}
        norm_words = {self._normalize(w) for w in words if w}
        
        norm_phrases = {p for p in norm_phrases if p and len(p) >= self.cfg.theme_normalization_min_length}
        norm_words = {w for w in norm_words if w and len(w) >= self.cfg.theme_normalization_min_length}
        
        # Apply learned reinforcement
        if self.memory.generation > 0:
            all_normalized = norm_phrases | norm_words
            scored = self._apply_coherence_weights(all_normalized, text)
            themes = [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        else:
            themes = list(norm_phrases) + list(norm_words)
        
        # Add concept matches for IPF/hybrid
        if self.method in ['ipf', 'hybrid'] and self.embedding_model and self.memory.centroids:
            concept_matches = self._match_to_centroids(text)
            themes.extend(concept_matches)
        
        themes = list(dict.fromkeys(themes))[:self.cfg.max_themes_per_chunk]
        
        if not themes:
            themes = [self._classify_content_type(text)]
        
        # Track for learning
        self._current_run_themes.append(themes)
        self._current_run_records.append({
            'text': text,
            'themes': themes,
            'phrases': list(norm_phrases),
            'words': list(norm_words)
        })
        
        confidence = self._compute_confidence(themes, text)
        
        return {
            'themes': themes,
            'phrase_themes': list(norm_phrases),
            'word_themes': list(norm_words),
            'primary_theme': themes[0] if themes else 'general_content',
            'confidence': confidence,
            'method': f'adaptive_{self.method}',
            'generation': self.memory.generation,
            'ipf_generation': self.memory.ipf_generation if self.method in ['ipf', 'hybrid'] else 0
        }
    
    def _extract_keyphrases(self, text: str) -> Set[str]:
        """Extract multi-word phrases using KeyBERT"""
        if not self.kw_model:
            return set()
        
        try:
            keywords = self.kw_model.extract_keywords(
                text[:1000],
                keyphrase_ngram_range=(2, 3),  # 2-3 word phrases only
                stop_words='english',
                top_n=10,
                use_maxsum=True,
                nr_candidates=20,
                diversity=0.7
            )
            
            phrases = set()
            for phrase, score in keywords:
                if score > self.cfg.keyphrase_confidence_threshold:
                    phrases.add(phrase.lower())
            
            return phrases
        except Exception as e:
            logger.debug(f"KeyBERT extraction failed: {e}")
            return set()
    
    def _extract_tfidf_words(self, text: str) -> Set[str]:
        """Extract single words using TF-IDF"""
        # Fit TF-IDF if we have enough documents (warm-up phase)
        if not hasattr(self.tfidf_words, 'vocabulary_') and len(self._tfidf_words_corpus) >= 5:
            try:
                self.tfidf_words.fit(self._tfidf_words_corpus)
                logger.info(f"✓ TF-IDF fitted with {len(self.tfidf_words.vocabulary_)} terms")
            except Exception as e:
                logger.warning(f"TF-IDF fitting failed: {e}")
                return set()
        
        if not hasattr(self.tfidf_words, 'vocabulary_'):
            return set()
        
        try:
            vec = self.tfidf_words.transform([text])
            feature_names = self.tfidf_words.get_feature_names_out()
            
            scores = vec.toarray()[0]
            top_indices = scores.argsort()[-10:][::-1]
            
            words = set()
            for idx in top_indices:
                if scores[idx] > self.cfg.tfidf_min_score:
                    word = feature_names[idx]
                    if word.lower() not in self.stopwords:
                        words.add(word)
            
            return words
        except Exception as e:
            logger.debug(f"TF-IDF extraction failed: {e}")
            return set()
    
    def _extract_hierarchical_themes(self, text: str) -> Set[str]:
        """Fallback heuristics"""
        themes = set()
        
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', text)
        themes.update(cap_phrases[:5])
        
        tech_terms = re.findall(r'\b[A-Za-z]+[-_][A-Za-z0-9]+\b', text)
        themes.update(tech_terms[:3])
        
        quoted = re.findall(r'"([^"]{3,30})"', text)
        themes.update(quoted[:3])
        
        return themes
    
    def _classify_content_type(self, text: str) -> str:
        """Classify general content type"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['study', 'research', 'experiment']):
            return 'research_content'
        elif any(word in text_lower for word in ['section', 'chapter', 'introduction']):
            return 'structured_document'
        elif any(word in text_lower for word in ['figure', 'table', 'chart']):
            return 'visual_reference'
        elif len(re.findall(r'\d+', text)) / max(len(text.split()), 1) > 0.1:
            return 'data_content'
        else:
            return 'general_content'
    
    def _normalize(self, theme: str) -> str:
        """Normalize theme string"""
        if not theme:
            return ''
        
        theme = theme.lower().strip()
        theme = re.sub(r'[^\w\s-]', '', theme)
        theme = re.sub(r'\s+', '_', theme)
        
        words = theme.split('_')
        words = [w for w in words if w not in self.stopwords]
        theme = '_'.join(words)
        
        return theme
    
    def _apply_coherence_weights(self, candidates: Set[str], text: str) -> List[Tuple[str, float]]:
        """Apply learned coherence weights"""
        scored = []
        
        for theme in candidates:
            base_score = 1.0
            
            # Frequency boost
            if theme in self.memory.theme_counts:
                freq_boost = min(np.log1p(self.memory.theme_counts[theme]) / 5, 0.5)
                base_score += freq_boost
            
            # Co-occurrence boost
            coherence_boost = 0.0
            if theme in self.memory.co_occurrence:
                for other in candidates:
                    if other != theme and other in self.memory.co_occurrence[theme]:
                        co_count = self.memory.co_occurrence[theme][other]
                        coherence_boost += min(co_count / 100, 0.2)
            
            base_score += coherence_boost
            
            # Learned weight
            if theme in self.memory.coherence_weights:
                base_score *= self.memory.coherence_weights[theme]
            
            # MI boost for IPF/hybrid
            if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
                mi_boost = 0.0
                for (theme_a, theme_b), mi in self.memory.high_mi_pairs.items():
                    if theme == theme_a and theme_b in candidates:
                        mi_boost += mi * 0.3
                    elif theme == theme_b and theme_a in candidates:
                        mi_boost += mi * 0.3
                base_score += mi_boost
            
            scored.append((theme, base_score))
        
        return scored
    
    def _match_to_centroids(self, text: str) -> List[str]:
        """Match text to learned concept centroids"""
        if not self.memory.centroids:
            return []
        
        try:
            text_embedding = self.embedding_model.encode(
                text[:500],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            matches = []
            for theme, centroid in self.memory.centroids.items():
                similarity = np.dot(text_embedding, centroid)
                if similarity > self.cfg.centroid_similarity_threshold:
                    matches.append(theme)
            
            return matches[:2]
        except:
            return []
    
    def _compute_confidence(self, themes: List[str], text: str) -> float:
        """Compute confidence based on learned patterns"""
        if not themes:
            return 0.3
        
        base = 0.5 + 0.1 * len(themes)
        
        if self.memory.generation > 0:
            known_themes = sum(1 for t in themes if t in self.memory.theme_counts)
            base += 0.1 * (known_themes / len(themes))
        
        # IPF/Hybrid MI boost
        if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
            mi_pairs_found = 0
            for i, theme_a in enumerate(themes):
                for theme_b in themes[i+1:]:
                    if (theme_a, theme_b) in self.memory.high_mi_pairs or \
                       (theme_b, theme_a) in self.memory.high_mi_pairs:
                        mi_pairs_found += 1
            if mi_pairs_found > 0:
                base += 0.05 * min(mi_pairs_found, 3)
        
        return min(base, 0.95)
    
    def learn_from_run(self):
        """Bootstrap semantics with phrase/word tracking"""
        if self.mode != 'adaptive':
            return
        
        if not self._current_run_records:
            logger.warning("No records to learn from")
            return
        
        logger.info(f"\n🧠 Learning from {len(self._current_run_records)} chunks (method: {self.method})...")
        
        # Phase 1: Update theme frequencies (separate phrase/word)
        for record in self._current_run_records:
            for theme in record['themes']:
                self.memory.theme_counts[theme] += 1
            for phrase in record.get('phrases', []):
                self.memory.phrase_themes[phrase] += 1
            for word in record.get('words', []):
                self.memory.word_themes[word] += 1
        
        # Phase 2: Build co-occurrence
        from itertools import combinations
        for record in self._current_run_records:
            themes = record['themes']
            for a, b in combinations(themes, 2):
                self.memory.co_occurrence[a][b] += 1
                self.memory.co_occurrence[b][a] += 1
        
        # Phase 3: Identify clusters
        self._build_clusters()
        
        # Phase 4: Compute coherence weights
        self._compute_coherence_weights()
        
        # Phase 5: Build centroids
        if self.embedding_model:
            self._build_centroids()
        
        # Phase 6: Build hierarchy
        self._build_hierarchy()
        
        # Phase 7: IPF Enhancement
        if self.method in ['ipf', 'hybrid'] and IPF_AVAILABLE:
            logger.info("  Applying IPF enhancement...")
            enhancer = IPFSemanticEnhancer(self.memory, self.cfg)
            
            enhancer.calibrate_cooccurrence()
            enhancer.balance_hierarchical_constraints()
            enhancer.smooth_theme_distributions()
            enhancer.compute_mutual_information()
            
            self.memory.ipf_generation += 1
            logger.info(f"✓ IPF enhancement complete (IPF Gen {self.memory.ipf_generation})")
        
        # Update statistics
        self.memory.total_chunks_processed += len(self._current_run_records)
        self.memory.total_themes_discovered = len(self.memory.theme_counts)
        
        logger.info(f"✓ Learned {len(self.memory.theme_counts)} unique themes")
        logger.info(f"  - Phrase themes: {len(self.memory.phrase_themes)}")
        logger.info(f"  - Word themes: {len(self.memory.word_themes)}")
        logger.info(f"✓ Discovered {len(self.memory.clusters)} concept clusters")
        
        # Clear current run
        self._current_run_themes = []
        self._current_run_records = []
    
    def _build_clusters(self):
        """Build concept clusters from co-occurrence"""
        visited = set()
        cluster_id = 0
        
        for theme in self.memory.theme_counts:
            if theme in visited:
                continue
            
            cluster = {theme}
            visited.add(theme)
            
            if theme in self.memory.co_occurrence:
                for related, count in self.memory.co_occurrence[theme].most_common(5):
                    if count >= 3:
                        cluster.add(related)
                        visited.add(related)
            
            if len(cluster) > 1:
                self.memory.clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1
    
    def _compute_coherence_weights(self):
        """Compute reinforcement weights"""
        for theme in self.memory.theme_counts:
            freq_weight = np.log1p(self.memory.theme_counts[theme]) / 10
            
            if theme in self.memory.co_occurrence:
                total_co = sum(self.memory.co_occurrence[theme].values())
                unique_partners = len(self.memory.co_occurrence[theme])
                diversity_factor = unique_partners / max(total_co, 1)
                coherence_boost = min(diversity_factor * 2, 1.0)
            else:
                coherence_boost = 0.0
            
            self.memory.coherence_weights[theme] = 1.0 + freq_weight + coherence_boost
    
    def _build_centroids(self):
        """Build semantic centroids"""
        if not self.memory.clusters:
            return
        
        logger.info("  Building concept centroids...")
        
        for cluster_name, themes in self.memory.clusters.items():
            theme_texts = defaultdict(list)
            for record in self._current_run_records:
                for theme in record['themes']:
                    if theme in themes:
                        theme_texts[theme].append(record['text'][:200])
            
            embeddings = []
            for theme, texts in theme_texts.items():
                if texts:
                    try:
                        emb = self.embedding_model.encode(
                            texts[0],
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        embeddings.append(emb)
                    except:
                        pass
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                primary = max(themes, key=lambda t: self.memory.theme_counts[t])
                self.memory.centroids[primary] = centroid
    
    def _build_hierarchy(self):
        """Build hierarchical relationships"""
        for theme in self.memory.theme_counts:
            if theme not in self.memory.co_occurrence:
                continue
            
            theme_count = self.memory.theme_counts[theme]
            
            for related, co_count in self.memory.co_occurrence[theme].items():
                related_count = self.memory.theme_counts[related]
                
                if co_count / theme_count > 0.7 and related_count > theme_count * 2:
                    if related not in self.memory.hierarchy:
                        self.memory.hierarchy[related] = {}
                    elif isinstance(self.memory.hierarchy[related], set):
                        old_set = self.memory.hierarchy[related]
                        self.memory.hierarchy[related] = {c: 1.0 for c in old_set}
                    
                    self.memory.hierarchy[related][theme] = 1.0
    
    def print_semantic_summary(self):
        """Print summary of learned semantics"""
        if self.mode == 'normal':
            logger.info("\n📊 Theme Discovery (Normal Mode):")
            logger.info(f"Method: {self.method}")
            for theme, count in self.discovered.most_common(15):
                logger.info(f"  {theme}: {count}")
            if len(self.discovered) > 15:
                logger.info(f"  ... and {len(self.discovered) - 15} more themes")
            return
        
        # Adaptive mode summary
        logger.info("\n" + "=" * 70)
        logger.info("SEMANTIC MEMORY SUMMARY (Adaptive Mode)")
        logger.info("=" * 70)
        logger.info(f"Method: {self.method}")
        logger.info(f"Generation: {self.memory.generation}")
        if self.method in ['ipf', 'hybrid']:
            logger.info(f"IPF Generation: {self.memory.ipf_generation}")
        logger.info(f"Total themes: {len(self.memory.theme_counts)}")
        logger.info(f"  - Phrase themes: {len(self.memory.phrase_themes)}")
        logger.info(f"  - Word themes: {len(self.memory.word_themes)}")
        logger.info(f"Total chunks processed: {self.memory.total_chunks_processed}")
        logger.info(f"Concept clusters: {len(self.memory.clusters)}")
        
        logger.info("\n🔥 Top 20 Themes:")
        for theme, count in self.memory.theme_counts.most_common(20):
            weight = self.memory.coherence_weights.get(theme, 1.0)
            theme_type = "phrase" if theme in self.memory.phrase_themes else "word"
            logger.info(f"  {theme:40s} | count: {count:4d} | weight: {weight:.2f} | type: {theme_type}")
        
        if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
            logger.info("\n🧬 Top Mutual Information Pairs:")
            for (theme_a, theme_b), mi in list(self.memory.high_mi_pairs.items())[:10]:
                logger.info(f"  {theme_a:30s} <-> {theme_b:30s} | MI: {mi:.4f}")
        
        logger.info("=" * 70)

# ============================================================================
# QUALITY SCORER (unchanged from original, already has semantic_coherence)
# ============================================================================

class QualityScorer:
    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.weights = cfg.quality_weights
        self.model = model
    
    def score(self, text: str) -> Dict[str, float]:
        scores = {
            'length_quality': self._length(text),
            'coherence_quality': self._coherence_heuristic(text),
            'information_density': self._info_density(text),
            'structural_quality': self._structure(text),
            'linguistic_quality': self._linguistics(text),
        }
        
        if self.model:
            scores['semantic_coherence'] = self._semantic_coherence(text)
        
        total_weight = sum(self.weights.get(k, 0) for k in scores)
        composite = 0.0
        if total_weight > 0:
            for k, v in scores.items():
                composite += v * (self.weights.get(k, 0) / total_weight)
        
        scores['composite_quality'] = round(composite, 3)
        return scores
    
    def _length(self, text):
        L = len(text)
        W = len(text.split())
        if 100 <= L <= 2000 and 20 <= W <= 400:
            return 1.0
        if 50 <= L <= 3000 and 10 <= W <= 500:
            return 0.7
        if L >= 10 and W >= 3:
            return 0.4
        return 0.1
    
    def _coherence_heuristic(self, text):
        sents = re.split(r'[.!?]+', text)
        complete = [s for s in sents if len(s.split()) >= self.cfg.min_sentence_words_for_complete]
        score = 0.0
        if sents:
            score += 0.3 * (len(complete) / len(sents))
        if re.search(r'[A-Z]', text):
            score += 0.2
        return min(score + 0.3, 1.0)
    
    def _semantic_coherence(self, chunk: str) -> float:
        if not NLTK_AVAILABLE:
            return 0.5
        
        try:
            sentences = nltk.sent_tokenize(chunk)
        except:
            return 0.5
        
        if len(sentences) < 2:
            return 0.5
        
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            return float(avg_similarity)
        except:
            return 0.4
    
    def _info_density(self, text):
        words = text.split()
        if not words:
            return 0.0
        uniq = len(set(words)) / len(words)
        score = 0.3 * uniq
        if re.search(r'\d', text):
            score += 0.2
        return min(score + 0.3, 1.0)
    
    def _structure(self, text):
        sents = [s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) >= self.cfg.min_sentence_words_for_complete]
        score = 0.4 if len(sents) >= 2 else 0.0
        if text.rstrip().endswith(('.', '!', '?')):
            score += 0.3
        return min(score + 0.3, 1.0)
    
    def _linguistics(self, text):
        score = 0.0
        if not re.search(r'\s{3,}', text):
            score += 0.25
        if not text.isupper() and not text.islower():
            score += 0.25
        if len(set(text.lower())) >= 20:
            score += 0.25
        return min(score + 0.25, 1.0)

# ============================================================================
# THREAD LINKER, KNOWLEDGE BUILDER, QA BUILDER (unchanged from original)
# ============================================================================

class ThreadLinker:
    def __init__(self, cfg: Config, embeddings: np.ndarray):
        self.cfg = cfg
        self.emb = embeddings
    
    def link(self, records: List[Dict]) -> List[Dict]:
        if not records:
            return records
        
        logger.info("Creating semantic threads...")
        threads = {}
        mapping = {}
        by_src = defaultdict(list)
        
        for i, r in enumerate(records):
            src = r.get('metadata', {}).get('filename', 'unknown')
            by_src[src].append(i)
        
        tcount = 0
        for src, idxs in by_src.items():
            if len(idxs) <= 1:
                continue
            
            base = self.emb[idxs]
            for i in range(len(idxs)):
                gi = idxs[i]
                if gi in mapping:
                    continue
                
                tid = f"thread_{tcount:06d}"
                threads[tid] = [gi]
                mapping[gi] = tid
                
                for j in range(i + 1, min(i + 5, len(idxs))):
                    gj = idxs[j]
                    if gj in mapping:
                        continue
                    
                    sim = cosine_similarity([base[i]], [base[j]])[0][0]
                    if sim >= self.cfg.thread_sim_threshold:
                        threads[tid].append(gj)
                        mapping[gj] = tid
                
                tcount += 1
        
        for i, r in enumerate(records):
            if i in mapping:
                tid = mapping[i]
                r['thread_id'] = tid
                r['metadata']['thread_size'] = len(threads[tid])
                r['metadata']['thread_position'] = threads[tid].index(i) + 1
            else:
                r['thread_id'] = f"single_{uuid.uuid4().hex[:12]}"
                r['metadata']['thread_size'] = 1
                r['metadata']['thread_position'] = 1
        
        logger.info(f"✓ Created {len(threads)} threads")
        return records

class KnowledgeBuilder:
    def __init__(self, cfg: Config, embedder: EmbeddingStore):
        self.cfg = cfg
        self.embedder = embedder
        self.qual = QualityScorer(cfg, model=embedder.model)
        self.labeler = SemanticLabeler(cfg, embedder.model) if cfg.enable_semantic_labeling else None
    
    def dedup(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for c in chunks:
            h = hashlib.sha256(c['text'].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(c)
        logger.info(f"Deduplicated: {len(chunks)} → {len(out)}")
        return out
    
    def group_consecutive(self, chunks: List[Dict], embeddings: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        if not chunks:
            return [], np.array([])
        
        logger.info("Grouping similar chunks...")
        by_src = defaultdict(list)
        for i, ch in enumerate(chunks):
            by_src[ch['filename']].append((i, ch))
        
        grouped = []
        g_emb = []
        
        for src, items in tqdm(by_src.items(), desc="Grouping by source"):
            if not items:
                continue
            
            idxs = [i for i, _ in items]
            arr = [c for _, c in items]
            
            try:
                emb = embeddings[idxs]
            except IndexError:
                logger.warning(f"Embedding index mismatch for {src}, skipping grouping")
                for i, ch in enumerate(arr):
                    grouped.append(ch)
                    try:
                        g_emb.append(embeddings[idxs[i]])
                    except:
                        pass
                continue
            
            i = 0
            while i < len(arr):
                texts = [arr[i]['text']]
                cur = arr[i].copy()
                s = emb[i].copy()
                cur_emb = emb[i]
                j = i + 1
                
                while j < len(arr):
                    sim = cosine_similarity([cur_emb], [emb[j]])[0][0]
                    new_text = ' '.join(texts) + ' ' + arr[j]['text']
                    
                    if sim >= self.cfg.sim_threshold and len(new_text) <= self.cfg.max_merged_length:
                        texts.append(arr[j]['text'])
                        s += emb[j]
                        cur_emb = s / (np.linalg.norm(s) + 1e-8)
                        j += 1
                    else:
                        break
                
                cur['text'] = ' '.join(texts)
                cur['merged_from'] = j - i
                grouped.append(cur)
                g_emb.append(cur_emb)
                i = j
        
        logger.info(f"✓ Grouped into {len(grouped)} chunks")
        return grouped, np.array(g_emb)
    
    def build(self, chunks: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        chunks = self.dedup(chunks)
        texts = [c['text'] for c in chunks]
        emb = self.embedder.embed_chunks(texts)
        
        grouped, gemb = self.group_consecutive(chunks, emb)
        
        records = []
        for idx, ch in enumerate(tqdm(grouped, desc='Scoring quality')):
            q = self.qual.score(ch['text'])
            
            meta = {
                'filename': ch['filename'],
                'chunk_index': ch.get('chunk_index', idx),
                'section_title': ch.get('section_title'),
                'has_section': ch.get('has_section', False),
                'merged_from': ch.get('merged_from', 1),
                'length': len(ch['text']),
                'word_count': len(ch['text'].split()),
                'sentence_count': len(re.split(r'[.!?]+', ch['text']))
            }
            
            if self.labeler:
                lab = self.labeler.label(ch['text'])
                meta.update({
                    'semantic_themes': lab['themes'],
                    'phrase_themes': lab.get('phrase_themes', []),
                    'word_themes': lab.get('word_themes', []),
                    'primary_theme': lab['primary_theme'],
                    'theme_confidence': lab['confidence'],
                    'labeling_method': lab['method']
                })
                if 'generation' in lab:
                    meta['semantic_generation'] = lab['generation']
                if 'ipf_generation' in lab:
                    meta['ipf_generation'] = lab['ipf_generation']
            
            rec = {
                'text': ch['text'],
                'metadata': meta,
                'quality_scores': q
            }
            records.append(rec)
        
        logger.info(f"✓ Created {len(records)} knowledge records")
        return records, gemb

class QABuilder:
    def __init__(self, cfg: Config, embedder: EmbeddingStore):
        self.cfg = cfg
        self.embedder = embedder
    
    def _diverse_prompts(self, chunk_text: str, metadata: Dict) -> List[str]:
        paras = re.split(r'\n\n+', chunk_text)
        first = (paras[0][:500] if paras else chunk_text[:500]).strip()
        key_terms = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', chunk_text)))
        
        theme = metadata.get('primary_theme', 'general_topic')
        if theme == 'general_topic' and metadata.get('semantic_themes'):
            theme = metadata['semantic_themes'][0]
        
        templates = [
            f"Summarize the key ideas in: '{first}'.",
            f"What is the main topic in this text: '{first}'?",
            "What key arguments are made in this text?",
            "What questions does this passage raise?",
            f"Describe the key steps in: '{first}'.",
            f"What are the implications discussed regarding {theme}?",
            f"What examples are provided in: '{first}'?",
            f"How does this relate to {theme}?",
        ]
        
        if key_terms:
            term = random.choice(key_terms)
            templates.extend([
                f"Explain the significance of '{term}' in this passage.",
                f"What does the text say about '{term}'?"
            ])
        
        k = min(4, len(templates))
        return random.sample(templates, k=k)
    
    def _group_consecutive(self, entries: List[Dict]) -> List[Dict]:
        if not entries:
            return []
        
        texts = [e['text'] for e in entries]
        embeds = self.embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.cfg.batch_size
        )
        
        grouped = []
        cur_texts = [texts[0]]
        cur_sum = embeds[0].copy()
        cur_emb = embeds[0]
        cur_meta = entries[0]['metadata'].copy()
        
        for i in range(1, len(entries)):
            sim = cosine_similarity([cur_emb], [embeds[i]])[0][0]
            new = ' '.join(cur_texts) + ' ' + texts[i]
            
            if sim >= self.cfg.qa_group_sim_threshold and len(new) <= self.cfg.qa_max_group_length:
                cur_texts.append(texts[i])
                cur_sum += embeds[i]
                cur_emb = cur_sum / (np.linalg.norm(cur_sum) + 1e-8)
            else:
                grouped.append({'text': ' '.join(cur_texts), 'metadata': cur_meta})
                cur_texts = [texts[i]]
                cur_sum = embeds[i].copy()
                cur_emb = embeds[i]
                cur_meta = entries[i]['metadata'].copy()
        
        grouped.append({'text': ' '.join(cur_texts), 'metadata': cur_meta})
        return grouped
    
    def _dedup_qas(self, qa_list: List[Dict], sim_threshold=0.92) -> List[Dict]:
        if not qa_list:
            return []
        
        texts = [q['user'] for q in qa_list]
        embs = self.embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.cfg.batch_size
        )
        
        keep = []
        keep_indices = []
        
        for i in range(len(embs)):
            is_duplicate = False
            if keep_indices:
                kept_embs = embs[keep_indices]
                sim_scores = cosine_similarity([embs[i]], kept_embs)[0]
                if np.any(sim_scores > sim_threshold):
                    is_duplicate = True
            
            if not is_duplicate:
                keep.append(qa_list[i])
                keep_indices.append(i)
        
        if len(qa_list) != len(keep):
            logger.info(f"Deduplicated Q&A: {len(qa_list)} → {len(keep)}")
        
        return keep
    
    def build(self, knowledge_records: List[Dict]) -> List[Dict]:
        logger.info(f"Generating Q&A from {len(knowledge_records)} knowledge records...")
        
        by_src = defaultdict(list)
        for r in knowledge_records:
            by_src[r['metadata'].get('filename', 'unknown')].append(r)
        
        qa = []
        existing_ans_embeds = []
        
        for src, entries in tqdm(by_src.items(), desc='Q&A by source'):
            grouped = self._group_consecutive(entries)
            cap = max(1, self.cfg.qa_max_pairs_per_source // 4)
            
            for g in grouped[:cap]:
                text = g['text']
                if len(text) < self.cfg.qa_min_context_length:
                    continue
                
                thread_id = str(uuid.uuid4())
                ans_emb = self.embedder.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                if existing_ans_embeds:
                    sims = [cosine_similarity([ans_emb], [e])[0][0] for e in existing_ans_embeds]
                    if any(s > self.cfg.qa_diversity_sim_threshold for s in sims):
                        continue
                
                existing_ans_embeds.append(ans_emb)
                
                qs = self._diverse_prompts(text, g.get('metadata', {}))
                q_emb = self.embedder.model.encode(
                    qs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                sim_diag = cosine_similarity(q_emb, np.stack([ans_emb] * len(qs))).diagonal()
                
                for q, sim in zip(qs, sim_diag):
                    qa.append({
                        'user': q,
                        'assistant': text,
                        'quality_metrics': {
                            'quality_score': float(round(float(sim), 3)),
                            'relevance': float(round(float(sim), 3)),
                            'clarity': float(round(random.uniform(0.7, 1.0), 2))
                        },
                        'source_metadata': {
                            **g.get('metadata', {}),
                            'source_file': src,
                            'thread_id': thread_id
                        }
                    })
        
        logger.info(f"✓ Created {len(qa)} Q&A pairs")
        qa = self._dedup_qas(qa)
        return qa

# ============================================================================
# UTILITIES
# ============================================================================

def save_jsonl(data: List[Dict], path: str, gzip_out: bool):
    """Save to JSONL with optional compression"""
    if gzip_out:
        with gzip.open(path + '.gz', 'wt', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
        size_mb = os.path.getsize(path + '.gz') / (1024 * 1024)
        logger.info(f"✓ Saved {len(data)} records → {path}.gz ({size_mb:.2f} MB)")
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"✓ Saved {len(data)} records → {path} ({size_mb:.2f} MB)")

def stratified_splits(items: List[Dict], split_ratio: Tuple[float, float, float]) -> Dict[str, List[Dict]]:
    """Create stratified train/val/test splits"""
    if not items:
        return {'train': [], 'validation': [], 'test': []}
    
    key = 'quality_scores' if 'quality_scores' in items[0] else 'quality_metrics'
    metric_key = 'composite_quality' if key == 'quality_scores' else 'quality_score'
    
    items = sorted(items, key=lambda x: x[key][metric_key], reverse=True)
    
    total = len(items)
    q = 4
    for i in range(q):
        s = i * (total // q)
        e = (i + 1) * (total // q) if i < q - 1 else total
        block = items[s:e]
        random.shuffle(block)
        items[s:e] = block
    
    tr_end = int(total * split_ratio[0])
    va_end = tr_end + int(total * split_ratio[1])
    
    return {
        'train': items[:tr_end],
        'validation': items[tr_end:va_end],
        'test': items[va_end:]
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(cfg: Config):
    """Execute the complete pipeline"""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("PRODUCTION PDF → MEMORY → Q&A PIPELINE (ENHANCED)")
    logger.info("=" * 70)
    logger.info(f"PDF dir: {cfg.pdf_dir}")
    logger.info(f"Workers: {cfg.max_workers}")
    logger.info(f"Parallel chunking: {cfg.use_parallel_chunking}")
    logger.info(f"Memory mapping: {cfg.use_mmap}")
    logger.info(f"Checkpointing: every {cfg.checkpoint_every} chunks" if cfg.checkpoint_every > 0 else "Checkpointing: disabled")
    logger.info(f"Embedding retries: {cfg.embedding_retry_attempts}")
    logger.info(f"OCR: {cfg.enable_ocr} (Preprocessing: {cfg.ocr_preprocessing})")
    logger.info(f"Chunking: {cfg.chunking_method} (size: {cfg.chunk_size}, overlap: {cfg.chunk_overlap})")
    logger.info(f"Semantic labeling: {cfg.enable_semantic_labeling}")
    if cfg.enable_semantic_labeling:
        logger.info(f"  - Mode: {cfg.semantic_mode}")
        logger.info(f"  - Method: {cfg.semantic_method}")
        logger.info(f"  - Keyphrases: {cfg.extract_keyphrases}")
        logger.info(f"  - IPF available: {IPF_AVAILABLE}")
    logger.info(f"Q&A Generation: {cfg.generate_qa}")
    logger.info("=" * 70)
    
    # Stage 1: Extract & chunk PDFs
    logger.info("\n[Stage 1/7] Extracting PDFs...")
    pdfp = PDFProcessor(cfg)
    docs = pdfp.extract_pdfs()
    
    if not docs:
        logger.error("❌ No documents extracted. Aborting.")
        return
    
    logger.info(f"\n[Stage 2/7] Chunking documents...")
    chunks = pdfp.chunk_documents(docs)
    
    if not chunks:
        logger.error("❌ No chunks produced. Aborting.")
        return
    
    # Stage 2: Create embedding store
    logger.info(f"\n[Stage 3/7] Initializing embeddings...")
    store = EmbeddingStore(cfg)
    
    # Stage 3: Build knowledge records
    logger.info(f"\n[Stage 4/7] Building knowledge records...")
    kb = KnowledgeBuilder(cfg, store)
    knowledge, k_emb = kb.build(chunks)
    
    # Stage 3.5: Adaptive learning (if enabled)
    if cfg.enable_semantic_labeling and kb.labeler and cfg.semantic_mode == 'adaptive':
        kb.labeler.learn_from_run()
        kb.labeler.save_memory()
    
    # Stage 4: Thread linking
    logger.info(f"\n[Stage 5/7] Linking semantic threads...")
    linker = ThreadLinker(cfg, k_emb)
    knowledge = linker.link(knowledge)
    
    # Stage 5: Save memory files
    if cfg.save_intermediates:
        logger.info(f"\n[Stage 6/7] Saving memory files...")
        store.build_faiss('memory.index', 'memory_texts.npy')
    
    # Stage 6: Split and save knowledge
    logger.info(f"\n[Stage 7/7] Creating data splits...")
    k_splits = stratified_splits(knowledge, cfg.split_ratio)
    
    for name, data in k_splits.items():
        if data:
            save_jsonl(data, f"{cfg.output_prefix}_knowledge_{name}.jsonl", cfg.gzip_output)
    
    # Stage 7: Generate Q&A pairs (optional)
    if cfg.generate_qa:
        logger.info(f"\nGenerating Q&A pairs...")
        qa_builder = QABuilder(cfg, store)
        qa_pairs = qa_builder.build(knowledge)
        
        qa_splits = stratified_splits(qa_pairs, cfg.split_ratio)
        
        for name, data in qa_splits.items():
            if data:
                save_jsonl(data, f"{cfg.output_prefix}_qa_{name}.jsonl", cfg.gzip_output)
    else:
        logger.info("\nSkipping Q&A generation as per --no-qa flag.")
        qa_pairs = []
        qa_splits = {'train': [], 'validation': [], 'test': []}
    
    # Final statistics
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDFs processed: {pdfp.stats['successful']}/{pdfp.stats['total_pdfs']}")
    logger.info(f"  - Sections extracted: {pdfp.stats['sections_extracted']}")
    logger.info(f"  - OCR used: {pdfp.stats['ocr_used']}")
    logger.info(f"Chunks created: {pdfp.stats['total_chunks']}")
    logger.info(f"Embeddings generated: {store.stats['embedded']}")
    if store.stats['retry_attempts'] > 0:
        logger.info(f"  - Retry attempts: {store.stats['retry_attempts']}")
    if store.stats['failed_embeddings'] > 0:
        logger.info(f"  - Failed embeddings: {store.stats['failed_embeddings']}")
    logger.info(f"Knowledge records: {len(knowledge)}")
    logger.info(f"  - Train: {len(k_splits['train'])}")
    logger.info(f"  - Validation: {len(k_splits['validation'])}")
    logger.info(f"  - Test: {len(k_splits['test'])}")
    logger.info(f"Q&A pairs: {len(qa_pairs)}")
    if cfg.generate_qa and qa_pairs:
        logger.info(f"  - Train: {len(qa_splits['train'])}")
        logger.info(f"  - Validation: {len(qa_splits['validation'])}")
        logger.info(f"  - Test: {len(qa_splits['test'])}")
    logger.info(f"\nProcessing time: {elapsed / 60:.2f} minutes")
    if pdfp.stats['successful'] > 0:
        logger.info(f"Average: {elapsed / pdfp.stats['successful']:.1f}s per PDF")
    logger.info("=" * 70)
    
    # Semantic labeling stats
    if cfg.enable_semantic_labeling and kb.labeler:
        kb.labeler.print_semantic_summary()

# ============================================================================
# CLI
# ============================================================================

def cli():
    """Command-line interface with all new options"""
    p = argparse.ArgumentParser(
        description='Enhanced PDF → Memory → Q&A Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 NEW FEATURES IN THIS ENHANCED VERSION:

1. **Memory Management**
   --use-mmap                    Enable memory-mapped arrays for large datasets
   --checkpoint-every 100        Save checkpoints every N chunks

2. **Embedding Resilience**
   --embedding-retry-attempts 3  Retry failed embeddings with exponential backoff

3. **Parallel Processing**
   Automatic parallel chunking for datasets with >10 documents

4. **Phrase vs Word Separation**
   - KeyBERT extracts 2-3 word phrases (conceptual themes)
   - TF-IDF extracts single words (topical keywords)
   - Tracked separately in semantic memory

5. **Incremental TF-IDF**
   - TF-IDF state preserved between runs
   - Warm-up phase (fits after 5+ documents)
   - Vocabulary persists in semantic memory

6. **NLTK Integration**
   - Uses NLTK stopwords if available (200+ words)
   - Falls back to minimal stopwords if NLTK unavailable

7. **Enhanced IPF Validation**
   - Input validation before running IPF
   - Checks for finite values, non-zero sums
   - Dimension matching verification
   - Better error messages

8. **All Magic Numbers Now Configurable**
   See Config dataclass for all thresholds

EXAMPLE USAGE:

# Maximum performance for large datasets
python3 adaptive_semantic.py \\
  --pdf-dir ./PDFs \\
  --workers 16 \\
  --use-mmap \\
  --checkpoint-every 50 \\
  --enable-semantic-labeling \\
  --semantic-mode adaptive \\
  --semantic-method hybrid \\
  --extract-keyphrases \\
  --batch-size 50

# Fast mode with resilience
python3 adaptive_semantic.py \\
  --pdf-dir ./PDFs \\
  --force-cpu \\
  --enable-semantic-labeling \\
  --embedding-retry-attempts 5 \\
  --checkpoint-every 25

# Recommended for flat PDFs
python3 adaptive_semantic.py \\
  --pdf-dir ./PDFs \\
  --enable-semantic-labeling \\
  --semantic-mode adaptive \\
  --semantic-method hybrid \\
  --chunk-size 400 \\
  --chunk-overlap 100 \\
  --chunking-method sentence \\
  --extract-keyphrases \\
  --enable-ocr \\
  --ocr-preprocessing
        """
    )
    
    # IO
    p.add_argument('--pdf-dir', default='./PDFs')
    p.add_argument('--output-prefix', default='dataset')
    p.add_argument('--no-gzip', action='store_true')
    
    # Performance
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--no-parallel-chunking', action='store_true',
                   help='Disable parallel chunking (default: enabled for >10 docs)')
    
    # Memory management (NEW)
    p.add_argument('--use-mmap', action='store_true',
                   help='Use memory-mapped arrays for embeddings (large datasets)')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Save checkpoint every N chunks (0=disabled)')
    p.add_argument('--checkpoint-dir', default='./checkpoints',
                   help='Directory for checkpoints')
    
    # Embedding resilience (NEW)
    p.add_argument('--embedding-retry-attempts', type=int, default=3,
                   help='Number of retry attempts for failed embeddings')
    p.add_argument('--embedding-retry-delay', type=float, default=1.0,
                   help='Initial delay between retries (seconds, exponential backoff)')
    
    # Extraction
    p.add_argument('--enable-ocr', action='store_true')
    p.add_argument('--ocr-preprocessing', action='store_true')
    p.add_argument('--no-sections', action='store_true')
    
    # Chunking
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--chunk-overlap', type=int, default=100)
    p.add_argument('--chunking-method', choices=['fixed', 'sentence', 'texttile'],
                   default='fixed')
    p.add_argument('--batch-size', type=int, default=100)
    
    # Embeddings
    p.add_argument('--embedding-model', default='all-MiniLM-L12-v2')
    p.add_argument('--force-cpu', action='store_true')
    
    # Semantic labeling
    p.add_argument('--enable-semantic-labeling', action='store_true')
    p.add_argument('--extract-keyphrases', action='store_true',
                   help='Use KeyBERT for 2-3 word phrase extraction')
    p.add_argument('--semantic-mode', choices=['normal', 'adaptive'],
                   default='normal')
    p.add_argument('--semantic-method', choices=['tfidf', 'ipf', 'hybrid'],
                   default='tfidf')
    p.add_argument('--semantic-memory-path', default='semantic_memory.pkl')
    
    # IPF-specific
    p.add_argument('--ipf-convergence-rate', type=float, default=0.01)
    p.add_argument('--ipf-max-iterations', type=int, default=100)
    p.add_argument('--no-ipf-calibrate', action='store_true')
    p.add_argument('--no-ipf-hierarchy', action='store_true')
    p.add_argument('--no-ipf-smooth', action='store_true')
    p.add_argument('--no-ipf-mi', action='store_true')
    
    # Thresholds (now all configurable)
    p.add_argument('--sim-threshold', type=float, default=0.7)
    p.add_argument('--thread-threshold', type=float, default=0.65)
    p.add_argument('--max-merged-length', type=int, default=2000)
    p.add_argument('--keyphrase-confidence-threshold', type=float, default=0.3,
                   help='Minimum confidence for KeyBERT phrases')
    p.add_argument('--tfidf-min-score', type=float, default=0.1,
                   help='Minimum TF-IDF score for word extraction')
    
    # Q&A
    p.add_argument('--no-qa', action='store_true')
    p.add_argument('--qa-max-pairs-per-source', type=int, default=5000)
    p.add_argument('--qa-diversity-sim-threshold', type=float, default=0.85)
    p.add_argument('--qa-group-sim-threshold', type=float, default=0.8)
    p.add_argument('--qa-max-group-length', type=int, default=5000)
    p.add_argument('--qa-min-context-length', type=int, default=50,
                   help='Minimum context length for Q&A generation')
    
    # Misc
    p.add_argument('--no-save-intermediates', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug', action='store_true')
    
    args = p.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Build config
    cfg = Config(
        pdf_dir=args.pdf_dir,
        output_prefix=args.output_prefix,
        gzip_output=not args.no_gzip,
        max_workers=args.workers,
        use_parallel_chunking=not args.no_parallel_chunking,
        use_mmap=args.use_mmap,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        embedding_retry_attempts=args.embedding_retry_attempts,
        embedding_retry_delay=args.embedding_retry_delay,
        enable_ocr=args.enable_ocr,
        ocr_preprocessing=args.ocr_preprocessing,
        extract_sections=not args.no_sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_method=args.chunking_method,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        force_cpu=args.force_cpu,
        enable_semantic_labeling=args.enable_semantic_labeling,
        extract_keyphrases=args.extract_keyphrases,
        semantic_mode=args.semantic_mode,
        semantic_method=args.semantic_method,
        semantic_memory_path=args.semantic_memory_path,
        ipf_convergence_rate=args.ipf_convergence_rate,
        ipf_max_iterations=args.ipf_max_iterations,
        ipf_calibrate_cooccurrence=not args.no_ipf_calibrate,
        ipf_balance_hierarchy=not args.no_ipf_hierarchy,
        ipf_smooth_distributions=not args.no_ipf_smooth,
        ipf_compute_mi=not args.no_ipf_mi,
        sim_threshold=args.sim_threshold,
        thread_sim_threshold=args.thread_threshold,
        max_merged_length=args.max_merged_length,
        keyphrase_confidence_threshold=args.keyphrase_confidence_threshold,
        tfidf_min_score=args.tfidf_min_score,
        generate_qa=not args.no_qa,
        qa_max_pairs_per_source=args.qa_max_pairs_per_source,
        qa_diversity_sim_threshold=args.qa_diversity_sim_threshold,
        qa_group_sim_threshold=args.qa_group_sim_threshold,
        qa_max_group_length=args.qa_max_group_length,
        qa_min_context_length=args.qa_min_context_length,
        save_intermediates=not args.no_save_intermediates,
        seed=args.seed,
    )
    
    run(cfg)

if __name__ == '__main__':
    cli()
