# Run with this for CPU only: python3 data_formatter.py --force-cpu --enable-semantic-labeling --semantic-mode normal --semantic-method hybrid 
# Enabling --extract-keyphrases runs very slow but improves semantic themes


from __future__ import annotations

import numpy as np
import json
import ftfy
import re
import random
import hashlib
import pickle
import gzip
import logging
import argparse
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true" if os.cpu_count() > 6 else "false"
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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

# --- Core ML/Data Libs ---
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# --- Optional Libs (IPF, KeyBERT) ---
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
            if not isinstance(self.seed, np.ndarray):
                raise ValueError("Seed must be a numpy array")
            if not np.all(np.isfinite(self.seed)):
                raise ValueError("Seed contains non-finite values")
            if np.any(self.seed < 0):
                raise ValueError("Seed contains negative values")
            if not self.aggregates or len(self.aggregates) < 2:
                raise ValueError("Need at least 2 marginals for IPF")
            for i, agg in enumerate(self.aggregates):
                arr = np.asarray(agg, dtype=float)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Marginal {i} contains non-finite values")
                if np.sum(arr) == 0:
                    raise ValueError(f"Marginal {i} sums to zero")
            if self.seed.ndim != len(self.aggregates):
                raise ValueError(f"Seed has {self.seed.ndim} dimensions but {len(self.aggregates)} marginals provided")
        
        def iteration(self):
            marginals = [np.asarray(agg, dtype=float) for agg in self.aggregates]
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
                if not np.all(np.isfinite(result)):
                    raise ValueError("IPF produced non-finite values")
                return result
            except Exception as e:
                raise RuntimeError(f"IPF iteration failed: {e}")
    
    IPF_AVAILABLE = True
except Exception as e:
    IPF_AVAILABLE = False
    print(f"⚠️  IPF not available: {e}. Install with 'pip install pyipf'")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except:
    KEYBERT_AVAILABLE = False
    print("⚠️  KeyBERT not available. Install with 'pip install keybert'")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY: METADATA TYPE NORMALIZATION (PATCH)
# ============================================================================

def normalize_metadata_types(metadata: Dict) -> Dict:
    """
    Normalize metadata types to ensure schema consistency for PyArrow.
    Converts all IDs and timestamps to strings to avoid type conflicts.
    """
    if not isinstance(metadata, dict):
        return metadata
    
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            normalized[key] = None
        elif key in ['conversation_id', 'message_id', 'user_id', 'timestamp']:
            # Convert IDs and timestamps to strings for consistency
            normalized[key] = str(value)
        elif isinstance(value, dict):
            # Recursively normalize nested dicts
            normalized[key] = normalize_metadata_types(value)
        elif isinstance(value, list):
            # Normalize list items if they're dicts
            normalized[key] = [normalize_metadata_types(item) if isinstance(item, dict) else item 
                             for item in value]
        else:
            normalized[key] = value
    
    return normalized

# ============================================================================
# UNIFIED CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Unified configuration for data processing, scoring, and semantics."""
    
    # --- IO and General ---
    output_dir: str = 'data_finetune'
    output_prefix: str = 'dataset'
    gzip_output: bool = True
    input_memory_texts_path: str = 'memory_texts.npy'
    input_memory_metadata_path: str = 'memory_metadata.pkl'
    
    # --- Parallelization and Performance ---
    max_workers: int = None
    batch_size: int = 64
    embedding_cache_size: int = 50000
    
    # --- Text Cleaning & Validation ---
    min_text_length: int = 20
    max_text_length: int = 10000
    min_words: int = 3
    punctuation_ratio_threshold: float = 0.6
    
    # --- Embeddings ---
    embedding_model: str = 'all-MiniLM-L12-v2'
    embedding_dim: int = 384
    force_cpu: bool = False
    embedding_retry_attempts: int = 3
    embedding_retry_delay: float = 1.0
    
    # --- Semantic Labeling ---
    enable_semantic_labeling: bool = True
    extract_keyphrases: bool = True
    semantic_mode: str = 'adaptive'
    semantic_memory_path: str = 'semantic_memory.pkl'
    semantic_method: str = 'hybrid'
    max_themes_per_chunk: int = 3
    keyphrase_confidence_threshold: float = 0.3
    tfidf_min_score: float = 0.1
    theme_normalization_min_length: int = 3
    centroid_similarity_threshold: float = 0.7
    min_sentence_words_for_complete: int = 3
    ipf_min_themes_for_calibration: int = 2
    
    # --- IPF-specific ---
    ipf_convergence_rate: float = 0.01
    ipf_max_iterations: int = 100
    ipf_calibrate_cooccurrence: bool = True
    ipf_balance_hierarchy: bool = True
    ipf_smooth_distributions: bool = True
    ipf_compute_mi: bool = True
    
    # --- Chunk Quality Scoring ---
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'length_quality': 0.10,
        'coherence_quality': 0.20,
        'information_density': 0.20,
        'structural_quality': 0.15,
        'linguistic_quality': 0.15,
        'semantic_coherence': 0.20,
    })
    
    # --- Q&A Pair Generation ---
    context_window_size: int = 3
    max_pairs_per_source: int = 5000
    
    # --- Q&A Pair Quality Filtering ---
    dedup_similarity_threshold: float = 0.95
    min_semantic_similarity: float = 0.1
    max_semantic_similarity: float = 0.95
    min_length_ratio: float = 0.1
    max_length_ratio: float = 10.0
    qa_quality_score_threshold: float = 0.46
    
    # --- Splits ---
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    
    # --- Misc ---
    seed: int = 42
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8) or 4
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
        except:
            pass

# ============================================================================
# TEXT UTILITIES
# ============================================================================

def clean_text(text: str) -> str:
    """Robust text cleaning function."""
    text = ftfy.fix_encoding(text)
    text = ftfy.fix_text(text)
    #text = re.sub(r'[ \t]+', ' ', text)
    #text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    #text = re.sub(r'\s*\n\s*', ' ', text)
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
    """Robust text validation function."""
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

def _clean_text_batch(batch: List[str]) -> List[str]:
    return [clean_text(text) for text in batch]

def _validate_text_batch(texts: List[str], config: Config) -> List[bool]:
    """Validate texts in batch"""
    return [validate_text(text, config) for text in texts]

# ============================================================================
# SEMANTIC MEMORY
# ============================================================================

@dataclass
class SemanticMemory:
    """Persistent semantic knowledge with TF-IDF state"""
    phrase_themes: Counter = field(default_factory=Counter)
    word_themes: Counter = field(default_factory=Counter)
    theme_counts: Counter = field(default_factory=Counter)
    
    co_occurrence: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    coherence_weights: Dict[str, float] = field(default_factory=dict)
    hierarchy: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    high_mi_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    tfidf_vocabulary: Optional[Dict] = None
    tfidf_idf_values: Optional[np.ndarray] = None
    tfidf_fitted: bool = False
    
    generation: int = 0
    ipf_generation: int = 0
    total_chunks_processed: int = 0
    total_themes_discovered: int = 0

# ============================================================================
# IPF SEMANTIC ENHANCER
# ============================================================================

class IPFSemanticEnhancer:
    """Enhanced IPF with comprehensive validation"""
    
    def __init__(self, memory: SemanticMemory, cfg: Config):
        self.memory = memory
        self.cfg = cfg
        if not IPF_AVAILABLE:
            logger.warning("⚠️  pyipf not installed. IPF features disabled.")
    
    def calibrate_cooccurrence(self, expected_marginals: dict = None):
        if not IPF_AVAILABLE or not self.cfg.ipf_calibrate_cooccurrence:
            return
        
        themes = list(self.memory.theme_counts.keys())
        n = len(themes)
        
        if n < self.cfg.ipf_min_themes_for_calibration:
            logger.debug(f"Not enough themes for IPF (need {self.cfg.ipf_min_themes_for_calibration}, have {n})")
            return
        
        logger.info(f"  IPF: Calibrating {n}x{n} co-occurrence matrix...")
        co_matrix = np.zeros((n, n))
        theme_to_idx = {t: i for i, t in enumerate(themes)}
        
        for theme_a in themes:
            for theme_b, count in self.memory.co_occurrence[theme_a].items():
                if theme_b in theme_to_idx:
                    i, j = theme_to_idx[theme_a], theme_to_idx[theme_b]
                    co_matrix[i, j] = count
        
        co_matrix = co_matrix + 0.1
        
        if expected_marginals:
            row_marginals = [expected_marginals.get(t, self.memory.theme_counts[t]) 
                           for t in themes]
        else:
            row_marginals = [self.memory.theme_counts[t] for t in themes]
        
        col_marginals = row_marginals.copy()
        
        if not all(np.isfinite(row_marginals)) or not all(np.isfinite(col_marginals)):
            logger.warning("Non-finite marginals, skipping IPF calibration")
            return
        if sum(row_marginals) == 0 or sum(col_marginals) == 0:
            logger.warning("Zero-sum marginals, skipping IPF calibration")
            return
        
        try:
            ipf = IPF(co_matrix, [row_marginals, col_marginals], [[0], [1]], 
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            calibrated_matrix = ipf.iteration()
            
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
        if not IPF_AVAILABLE or not self.cfg.ipf_balance_hierarchy or not self.memory.hierarchy:
            return
        
        logger.info(f"  IPF: Balancing {len(self.memory.hierarchy)} hierarchical relationships...")
        parent_themes = list(self.memory.hierarchy.keys())
        all_children = set(c for children in self.memory.hierarchy.values() for c in children)
        
        if not parent_themes or not all_children:
            return
        
        n_parents = len(parent_themes)
        n_children = len(all_children)
        children_list = list(all_children)
        table = np.zeros((n_parents, n_children))
        
        for i, parent in enumerate(parent_themes):
            for j, child in enumerate(children_list):
                if child in self.memory.hierarchy[parent]:
                    table[i, j] = self.memory.co_occurrence[parent].get(child, 1.0)
        
        row_totals = [self.memory.theme_counts[p] for p in parent_themes]
        col_totals = [self.memory.theme_counts[c] for c in children_list]
        
        if not all(np.isfinite(row_totals)) or not all(np.isfinite(col_totals)):
            logger.warning("Non-finite totals in hierarchy, skipping")
            return
        
        try:
            ipf = IPF(table + 0.1, [row_totals, col_totals], [[0], [1]],
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            balanced_table = ipf.iteration()
            
            for i, parent in enumerate(parent_themes):
                for j, child in enumerate(children_list):
                    if child in self.memory.hierarchy[parent]:
                        weight = balanced_table[i, j] / (row_totals[i] + 1e-8)
                        self.memory.hierarchy[parent][child] = weight
            logger.info(f"    ✓ Balanced {len(parent_themes)} relationships")
        except Exception as e:
            logger.warning(f"Hierarchical IPF balancing failed: {e}")

    def smooth_theme_distributions(self, target_distribution: dict = None):
        if not IPF_AVAILABLE or not self.cfg.ipf_smooth_distributions or not self.memory.clusters:
            return
        
        themes = list(self.memory.theme_counts.keys())
        n = len(themes)
        n_docs = len(self.memory.clusters)
        if n < 2 or n_docs == 0:
            return
        
        logger.info(f"  IPF: Smoothing {n} theme distributions across {n_docs} clusters...")
        doc_theme_matrix = np.zeros((n_docs, n))
        theme_to_idx = {t: i for i, t in enumerate(themes)}
        
        for doc_idx, (cluster_name, cluster_themes) in enumerate(self.memory.clusters.items()):
            for theme in cluster_themes:
                if theme in theme_to_idx:
                    doc_theme_matrix[doc_idx, theme_to_idx[theme]] = self.memory.theme_counts[theme]
        
        doc_theme_matrix = doc_theme_matrix + 0.1
        
        if target_distribution:
            col_totals = [target_distribution.get(t, self.memory.theme_counts[t]) for t in themes]
        else:
            col_totals = [self.memory.theme_counts[t] for t in themes]
        
        row_totals = doc_theme_matrix.sum(axis=1).tolist()
        
        if not all(np.isfinite(row_totals)) or not all(np.isfinite(col_totals)):
            logger.warning("Non-finite values in distribution smoothing, skipping")
            return
        
        try:
            ipf = IPF(doc_theme_matrix, [row_totals, col_totals], [[0], [1]],
                     convergence_rate=self.cfg.ipf_convergence_rate,
                     max_iteration=self.cfg.ipf_max_iterations)
            smoothed_matrix = ipf.iteration()
            
            for j, theme in enumerate(themes):
                smoothed_count = int(smoothed_matrix[:, j].sum())
                self.memory.theme_counts[theme] = smoothed_count
            logger.info(f"    ✓ Smoothed {n} themes")
        except Exception as e:
            logger.warning(f"Distribution smoothing failed: {e}")

    def compute_mutual_information(self):
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
# SEMANTIC LABELER
# ============================================================================

class SemanticLabeler:
    """Enhanced labeler with phrase/word separation and incremental TF-IDF"""
    
    FALLBACK_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall'
    }
    
    def __init__(self, cfg: Config, embedding_model=None):
        self.cfg = cfg
        self.mode = cfg.semantic_mode
        self.method = cfg.semantic_method
        self.embedding_model = embedding_model
        self.discovered = Counter()
        
        self.stopwords = NLTK_STOPWORDS if NLTK_STOPWORDS else self.FALLBACK_STOPWORDS
        logger.info(f"SemanticLabeler: Using {len(self.stopwords)} stopwords")
        
        if self.method in ['tfidf', 'hybrid']:
            self.tfidf_words = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 1),
                stop_words=list(self.stopwords),
                min_df=2
            )
            self._tfidf_words_corpus = []
        
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
        
        if self.mode == 'adaptive':
            self.memory = SemanticMemory()
            self._current_run_records = []
            self._load_memory()
            logger.info(f"🧠 SemanticLabeler: Adaptive mode (Gen {self.memory.generation}, Method: {self.method})")
        else:
            logger.info(f"📋 SemanticLabeler: Normal mode (Method: {self.method})")
    
    def _load_memory(self):
        path = Path(self.cfg.semantic_memory_path)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    self.memory = pickle.load(f)
                if self.memory.tfidf_fitted and self.memory.tfidf_vocabulary:
                    self.tfidf_words.vocabulary_ = self.memory.tfidf_vocabulary
                    self.tfidf_words.idf_ = self.memory.tfidf_idf_values
                    logger.info(f"✓ Restored TF-IDF state with {len(self.memory.tfidf_vocabulary)} terms")
                logger.info(f"✓ Loaded memory (Gen {self.memory.generation}, {len(self.memory.theme_counts)} themes)")
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")
        else:
            logger.info("No previous semantic memory found, starting fresh")
    
    def save_memory(self):
        if self.mode != 'adaptive':
            return
        
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
        if self.mode == 'adaptive':
            return self._label_adaptive(text)
        else:
            return self._label_normal(text)
    
    def _get_raw_candidates(self, text: str) -> Tuple[Set[str], Set[str]]:
        phrases = set()
        words = set()
        
        if self.cfg.extract_keyphrases and self.kw_model:
            phrases.update(self._extract_keyphrases(text))
        
        if self.method in ['tfidf', 'hybrid']:
            self._tfidf_words_corpus.append(text)
            words.update(self._extract_tfidf_words(text))
        
        if not phrases and not words:
            fallback = self._extract_hierarchical_themes(text)
            for theme in fallback:
                if len(theme.split('_')) > 1:
                    phrases.add(theme)
                else:
                    words.add(theme)
        
        return phrases, words
    
    def _label_normal(self, text: str) -> Dict:
        phrases, words = self._get_raw_candidates(text)
        
        phrases = [self._normalize(p) for p in phrases if p and len(p) >= self.cfg.theme_normalization_min_length]
        words = [self._normalize(w) for w in words if w and len(w) >= self.cfg.theme_normalization_min_length]
        
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
        phrases, words = self._get_raw_candidates(text)
        
        norm_phrases = {self._normalize(p) for p in phrases if p and len(p) >= self.cfg.theme_normalization_min_length}
        norm_words = {self._normalize(w) for w in words if w and len(w) >= self.cfg.theme_normalization_min_length}
        
        all_normalized = norm_phrases | norm_words
        if self.memory.generation > 0:
            scored = self._apply_coherence_weights(all_normalized, text)
            themes = [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        else:
            themes = list(all_normalized)
        
        if self.method in ['ipf', 'hybrid'] and self.embedding_model and self.memory.centroids:
            concept_matches = self._match_to_centroids(text)
            themes.extend(concept_matches)
        
        themes = list(dict.fromkeys(themes))[:self.cfg.max_themes_per_chunk]
        if not themes:
            themes = [self._classify_content_type(text)]
        
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
        if not self.kw_model: return set()
        try:
            keywords = self.kw_model.extract_keywords(
                text[:1000], keyphrase_ngram_range=(2, 3), stop_words='english',
                top_n=10, use_maxsum=True, nr_candidates=20, diversity=0.7
            )
            return {phrase.lower() for phrase, score in keywords if score > self.cfg.keyphrase_confidence_threshold}
        except Exception as e:
            logger.debug(f"KeyBERT extraction failed: {e}")
            return set()
    
    def _extract_tfidf_words(self, text: str) -> Set[str]:
        if not hasattr(self.tfidf_words, 'vocabulary_') and len(self._tfidf_words_corpus) >= 5:
            try:
                self.tfidf_words.fit(self._tfidf_words_corpus)
                logger.info(f"✓ TF-IDF fitted with {len(self.tfidf_words.vocabulary_)} terms")
            except Exception as e:
                logger.warning(f"TF-IDF fitting failed: {e}")
                return set()
        
        if not hasattr(self.tfidf_words, 'vocabulary_'): return set()
        
        try:
            vec = self.tfidf_words.transform([text])
            feature_names = self.tfidf_words.get_feature_names_out()
            scores = vec.toarray()[0]
            top_indices = scores.argsort()[-10:][::-1]
            
            return {feature_names[idx] for idx in top_indices 
                    if scores[idx] > self.cfg.tfidf_min_score and feature_names[idx].lower() not in self.stopwords}
        except Exception as e:
            logger.debug(f"TF-IDF extraction failed: {e}")
            return set()
    
    def _extract_hierarchical_themes(self, text: str) -> Set[str]:
        themes = set()
        themes.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', text)[:5])
        themes.update(re.findall(r'\b[A-Za-z]+[-_][A-Za-z0-9]+\b', text)[:3])
        themes.update(re.findall(r'"([^"]{3,30})"', text)[:3])
        return themes
    
    def _classify_content_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ['study', 'research', 'experiment']): return 'research_content'
        if any(w in text_lower for w in ['section', 'chapter', 'introduction']): return 'structured_document'
        if any(w in text_lower for w in ['figure', 'table', 'chart']): return 'visual_reference'
        if len(re.findall(r'\d+', text)) / max(len(text.split()), 1) > 0.1: return 'data_content'
        return 'general_content'
    
    def _normalize(self, theme: str) -> str:
        if not theme: return ''
        theme = theme.lower().strip()
        theme = re.sub(r'[^\w\s-]', '', theme)
        theme = re.sub(r'\s+', '_', theme)
        words = [w for w in theme.split('_') if w not in self.stopwords]
        return '_'.join(words)
    
    def _apply_coherence_weights(self, candidates: Set[str], text: str) -> List[Tuple[str, float]]:
        scored = []
        for theme in candidates:
            base_score = 1.0
            if theme in self.memory.theme_counts:
                base_score += min(np.log1p(self.memory.theme_counts[theme]) / 5, 0.5)
            if theme in self.memory.co_occurrence:
                for other in candidates:
                    if other != theme and other in self.memory.co_occurrence[theme]:
                        base_score += min(self.memory.co_occurrence[theme][other] / 100, 0.2)
            if theme in self.memory.coherence_weights:
                base_score *= self.memory.coherence_weights[theme]
            if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
                for (a, b), mi in self.memory.high_mi_pairs.items():
                    if (theme == a and b in candidates) or (theme == b and a in candidates):
                        base_score += mi * 0.3
            scored.append((theme, base_score))
        return scored
    
    def _match_to_centroids(self, text: str) -> List[str]:
        if not self.memory.centroids or not self.embedding_model: return []
        try:
            emb = self.embedding_model.encode(text[:500], convert_to_numpy=True, normalize_embeddings=True)
            matches = []
            for theme, centroid in self.memory.centroids.items():
                if np.dot(emb, centroid) > self.cfg.centroid_similarity_threshold:
                    matches.append(theme)
            return matches[:2]
        except:
            return []
    
    def _compute_confidence(self, themes: List[str], text: str) -> float:
        if not themes: return 0.3
        base = 0.5 + 0.1 * len(themes)
        if self.memory.generation > 0:
            known = sum(1 for t in themes if t in self.memory.theme_counts)
            base += 0.1 * (known / len(themes))
        if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
            mi_pairs = 0
            for i, a in enumerate(themes):
                for b in themes[i+1:]:
                    if (a, b) in self.memory.high_mi_pairs or (b, a) in self.memory.high_mi_pairs:
                        mi_pairs += 1
            base += 0.05 * min(mi_pairs, 3)
        return min(base, 0.95)
    
    def learn_from_run(self):
        if self.mode != 'adaptive' or not self._current_run_records:
            return
        
        logger.info(f"\n🧠 Learning from {len(self._current_run_records)} chunks (method: {self.method})...")
        
        for record in self._current_run_records:
            for theme in record['themes']:
                self.memory.theme_counts[theme] += 1
            for phrase in record.get('phrases', []):
                self.memory.phrase_themes[phrase] += 1
            for word in record.get('words', []):
                self.memory.word_themes[word] += 1
        
        from itertools import combinations
        for record in self._current_run_records:
            for a, b in combinations(record['themes'], 2):
                self.memory.co_occurrence[a][b] += 1
                self.memory.co_occurrence[b][a] += 1
        
        self._build_clusters()
        self._compute_coherence_weights()
        if self.embedding_model: self._build_centroids()
        self._build_hierarchy()
        
        if self.method in ['ipf', 'hybrid'] and IPF_AVAILABLE:
            logger.info("  Applying IPF enhancement...")
            enhancer = IPFSemanticEnhancer(self.memory, self.cfg)
            enhancer.calibrate_cooccurrence()
            enhancer.balance_hierarchical_constraints()
            enhancer.smooth_theme_distributions()
            enhancer.compute_mutual_information()
            self.memory.ipf_generation += 1
            logger.info(f"✓ IPF enhancement complete (IPF Gen {self.memory.ipf_generation})")
        
        self.memory.total_chunks_processed += len(self._current_run_records)
        self.memory.total_themes_discovered = len(self.memory.theme_counts)
        logger.info(f"✓ Learned {len(self.memory.theme_counts)} unique themes")
        
        self._current_run_records = []
    
    def _build_clusters(self):
        visited, cluster_id = set(), 0
        for theme in self.memory.theme_counts:
            if theme in visited: continue
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
        for theme in self.memory.theme_counts:
            freq_w = np.log1p(self.memory.theme_counts[theme]) / 10
            co_boost = 0.0
            if theme in self.memory.co_occurrence:
                total_co = sum(self.memory.co_occurrence[theme].values())
                div_factor = len(self.memory.co_occurrence[theme]) / max(total_co, 1)
                co_boost = min(div_factor * 2, 1.0)
            self.memory.coherence_weights[theme] = 1.0 + freq_w + co_boost
    
    def _build_centroids(self):
        if not self.memory.clusters: return
        logger.info("  Building concept centroids...")
        for cluster_name, themes in self.memory.clusters.items():
            theme_texts = defaultdict(list)
            for r in self._current_run_records:
                for t in r['themes']:
                    if t in themes: theme_texts[t].append(r['text'][:200])
            
            embeddings = []
            for theme, texts in theme_texts.items():
                if texts:
                    try:
                        embeddings.append(self.embedding_model.encode(texts[0], convert_to_numpy=True, normalize_embeddings=True))
                    except: pass
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid /= (np.linalg.norm(centroid) + 1e-8)
                primary = max(themes, key=lambda t: self.memory.theme_counts[t])
                self.memory.centroids[primary] = centroid
    
    def _build_hierarchy(self):
        for theme in self.memory.theme_counts:
            if theme not in self.memory.co_occurrence: continue
            theme_count = self.memory.theme_counts[theme]
            for related, co_count in self.memory.co_occurrence[theme].items():
                related_count = self.memory.theme_counts[related]
                if co_count / theme_count > 0.7 and related_count > theme_count * 2:
                    if related not in self.memory.hierarchy: 
                        self.memory.hierarchy[related] = {}
                    elif isinstance(self.memory.hierarchy[related], set):
                        self.memory.hierarchy[related] = {c: 1.0 for c in self.memory.hierarchy[related]}
                    self.memory.hierarchy[related][theme] = 1.0
    
    def print_semantic_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("SEMANTIC SUMMARY")
        logger.info("=" * 70)
        
        if self.mode == 'normal':
            logger.info("Mode: Normal")
            logger.info(f"Method: {self.method}")
            for theme, count in self.discovered.most_common(15):
                logger.info(f"  {theme}: {count}")
            if len(self.discovered) > 15:
                logger.info(f"  ... and {len(self.discovered) - 15} more themes")
            return
        
        logger.info(f"Mode: Adaptive (Method: {self.method})")
        logger.info(f"Generation: {self.memory.generation}")
        if self.method in ['ipf', 'hybrid']:
            logger.info(f"IPF Generation: {self.memory.ipf_generation}")
        logger.info(f"Total themes: {len(self.memory.theme_counts)}")
        logger.info(f"  - Phrase themes: {len(self.memory.phrase_themes)}")
        logger.info(f"  - Word themes: {len(self.memory.word_themes)}")
        logger.info(f"Total chunks processed: {self.memory.total_chunks_processed}")
        
        logger.info("\n🔥 Top 20 Themes:")
        for theme, count in self.memory.theme_counts.most_common(20):
            weight = self.memory.coherence_weights.get(theme, 1.0)
            ttype = "phrase" if theme in self.memory.phrase_themes else "word"
            logger.info(f"  {theme:40s} | count: {count:4d} | weight: {weight:.2f} | type: {ttype}")
        
        if self.method in ['ipf', 'hybrid'] and self.memory.high_mi_pairs:
            logger.info("\n🧬 Top Mutual Information Pairs:")
            for (a, b), mi in list(self.memory.high_mi_pairs.items())[:10]:
                logger.info(f"  {a:30s} <-> {b:30s} | MI: {mi:.4f}")
        logger.info("=" * 70)

# ============================================================================
# QUALITY SCORER
# ============================================================================

class QualityScorer:
    """Scores the quality of a *single* text chunk."""
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
            'semantic_coherence': 0.5
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
        L, W = len(text), len(text.split())
        if 100 <= L <= 2000 and 20 <= W <= 400: return 1.0
        if 50 <= L <= 3000 and 10 <= W <= 500: return 0.7
        if L >= 10 and W >= 3: return 0.4
        return 0.1
    
    def _coherence_heuristic(self, text):
        sents = re.split(r'[.!?]+', text)
        complete = [s for s in sents if len(s.split()) >= self.cfg.min_sentence_words_for_complete]
        score = 0.3 * (len(complete) / len(sents)) if sents else 0.0
        if re.search(r'[A-Z]', text): score += 0.2
        return min(score + 0.3, 1.0)
    
    def _semantic_coherence(self, chunk: str) -> float:
        if not NLTK_AVAILABLE: return 0.5
        try:
            sentences = nltk.sent_tokenize(chunk)
        except:
            return 0.5
        if len(sentences) < 2: return 0.5
        
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
            sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings) - 1)]
            return float(np.mean(sims))
        except:
            return 0.4
    
    def _info_density(self, text):
        words = text.split()
        if not words: return 0.0
        uniq = len(set(words)) / len(words)
        score = 0.3 * uniq
        if re.search(r'\d', text): score += 0.2
        return min(score + 0.3, 1.0)
    
    def _structure(self, text):
        sents = [s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) >= self.cfg.min_sentence_words_for_complete]
        score = 0.4 if len(sents) >= 2 else 0.0
        if text.rstrip().endswith(('.', '!', '?')): score += 0.3
        return min(score + 0.3, 1.0)
    
    def _linguistics(self, text):
        score = 0.0
        if not re.search(r'\s{3,}', text): score += 0.25
        if not text.isupper() and not text.islower(): score += 0.25
        if len(set(text.lower())) >= 20: score += 0.25
        return min(score + 0.25, 1.0)

# ============================================================================
# Q&A BUILDER
# ============================================================================

class QABuilder:
    """Generates diverse Q&A prompts from chunks."""
    def __init__(self, cfg: Config, model: SentenceTransformer):
        self.cfg = cfg
        self.model = model
    
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

# ============================================================================
# OPTIMIZED DATA PROCESSOR
# ============================================================================

class OptimizedDataProcessor:
    def __init__(self, config: Config, use_semantic_filtering: bool = True):
        self.config = config
        self.use_semantic_filtering = use_semantic_filtering
        self.model: Optional[SentenceTransformer] = None
        
        if self.use_semantic_filtering:
            self._load_embedding_model()
            
        self.text_hashes: Set[str] = set()
        
        # Initialize components
        self.qual_scorer = QualityScorer(config, self.model)
        self.labeler = SemanticLabeler(config, self.model) if config.enable_semantic_labeling else None

    def _load_embedding_model(self) -> None:
        logger.info("Loading SentenceTransformer model for semantic filtering...")
        try:
            device = 'cpu' if self.config.force_cpu or not torch.cuda.is_available() else 'cuda'
            self.model = SentenceTransformer(self.config.embedding_model, device=device)
            self.model.eval()
            logger.info(f"✓ Model {self.config.embedding_model} loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            self.use_semantic_filtering = False
            logger.warning("Disabling semantic filtering due to model loading failure.")

    def _compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings in strict batches for GPU efficiency"""
        if not self.use_semantic_filtering or not self.model:
            return np.array([])
        
        if not texts:
            return np.array([])

        all_embeddings = []
        
        # Disable tokenizers parallelism to avoid deadlocks in some environments
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Process in chunks to respect batch size and GPU memory
        total_texts = len(texts)
        logger.info(f"Computing embeddings for {total_texts} texts in batches of {self.config.batch_size}...")
        
        for i in tqdm(range(0, total_texts, self.config.batch_size), desc="Embedding Batches"):
            batch = texts[i : i + self.config.batch_size]
            try:
                # This will use the GPU if self.model.device is 'cuda'
                embeddings = self.model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=len(batch)
                )
                all_embeddings.append(embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i // self.config.batch_size}: {e}")
                # Fallback: return zero vectors for this batch to maintain alignment
                all_embeddings.append(np.zeros((len(batch), self.config.embedding_dim)))

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    # ========================================================================
    # LOAD MEMORY TEXTS
    # ========================================================================

    def load_memory_texts(self) -> List[Dict]:
        """Load memory texts and metadata"""
        texts_path = Path(self.config.input_memory_texts_path)
        metadata_path = Path(self.config.input_memory_metadata_path)
        
        if not texts_path.exists() or not metadata_path.exists():
            logger.warning(f"Memory files not found:")
            if not texts_path.exists():
                logger.warning(f"  - {texts_path}")
            if not metadata_path.exists():
                logger.warning(f"  - {metadata_path}")
            logger.warning("Skipping memory texts source.")
            return []
        
        try:
            embedded_texts = np.load(texts_path, allow_pickle=True)
            with open(metadata_path, 'rb') as f:
                embedded_metadata = pickle.load(f)
            
            logger.info(f"✓ Loaded {len(embedded_texts)} memory texts")
            
            entries = [
                {'text': text, 'metadata': meta} 
                for text, meta in zip(embedded_texts, embedded_metadata)
            ]
            
            return entries
        
        except Exception as e:
            logger.error(f"Failed to load memory texts: {e}")
            return []

    # ========================================================================
    # DEDUPLICATE AND CLEAN
    # ========================================================================

    def deduplicate_and_clean_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized deduplication and cleaning with parallel processing, scoring, and labeling."""
        logger.info(f"Starting deduplication and cleaning of {len(entries)} entries...")
        
        texts = [entry.get('text', '') for entry in entries]
        
        logger.info("Cleaning texts in parallel...")
        chunk_size = max(1, len(texts) // self.config.max_workers)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        cleaned_texts_flat = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            for chunk in tqdm(executor.map(_clean_text_batch, text_chunks), total=len(text_chunks), desc="Cleaning Chunks"):
                cleaned_texts_flat.extend(chunk)
        
        logger.info("Validating texts in parallel...")
        config_for_workers = self.config
        validate_func = partial(_validate_text_batch, config=config_for_workers)
        
        validations_flat = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            for chunk in tqdm(executor.map(validate_func, text_chunks), total=len(text_chunks), desc="Validating Chunks"):
                validations_flat.extend(chunk)
        
        valid_entries, valid_texts = [], []
        current_text_hashes = set()
        
        for i, (entry, cleaned_text, is_valid) in enumerate(zip(entries, cleaned_texts_flat, validations_flat)):
            if not is_valid or not cleaned_text:
                continue
            
            text_hash = hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
            if text_hash in current_text_hashes:
                continue
            current_text_hashes.add(text_hash)
            
            entry['cleaned_text'] = cleaned_text
            valid_entries.append(entry)
            valid_texts.append(cleaned_text)
        
        logger.info(f"After basic filtering (length, patterns, exact duplicates): {len(valid_entries)} entries remain")
        
        if not self.use_semantic_filtering:
            return valid_entries
        
        logger.info("Computing embeddings for semantic deduplication...")
        # Now uses the optimized batch function
        embeddings = self._compute_embeddings_batch(valid_texts)
        
        if embeddings.size == 0 or len(embeddings) != len(valid_texts):
            logger.warning("No embeddings computed or mismatch. Skipping semantic deduplication.")
            keep_indices = list(range(len(valid_entries)))
        else:
            logger.info("Performing semantic deduplication...")
            keep_indices = self._efficient_deduplication(embeddings)
        
        final_entries = [valid_entries[i] for i in keep_indices]
        
        # Scoring and Labeling
        logger.info("Scoring and labeling final entries...")
        for entry in tqdm(final_entries, desc="Scoring & Labeling"):
            entry['quality_scores'] = self.qual_scorer.score(entry['cleaned_text'])
            if self.labeler:
                labels = self.labeler.label(entry['cleaned_text'])
                entry['semantic_labels'] = labels
                if 'metadata' not in entry:
                    entry['metadata'] = {}
                entry['metadata']['semantic_themes'] = labels.get('themes')
                entry['metadata']['primary_theme'] = labels.get('primary_theme')
        
        logger.info(f"Final count after semantic deduplication & labeling: {len(final_entries)} entries")
        return final_entries

    def _efficient_deduplication(self, embeddings: np.ndarray, threshold: float = None) -> List[int]:
        """Efficient semantic deduplication using vectorized operations"""
        if embeddings.size == 0:
            return list(range(len(embeddings)))
        
        threshold = threshold or self.config.dedup_similarity_threshold
        keep_indices = []
        chunk_size = 1000
        
        for i in range(0, len(embeddings), chunk_size):
            end_idx = min(i + chunk_size, len(embeddings))
            current_chunk = embeddings[i:end_idx]
            
            if keep_indices:
                kept_embeddings = embeddings[keep_indices]
                similarities = cosine_similarity(current_chunk, kept_embeddings)
                max_similarities = np.max(similarities, axis=1)
                
                chunk_keep = [j for j, sim in enumerate(max_similarities) if sim < threshold]
                keep_indices.extend([i + j for j in chunk_keep])
            else:
                if len(current_chunk) > 1:
                    internal_similarities = cosine_similarity(current_chunk)
                    np.fill_diagonal(internal_similarities, 0)
                    to_remove_in_chunk = set()
                    for r in range(current_chunk.shape[0]):
                        for c in range(r + 1, current_chunk.shape[0]):
                            if internal_similarities[r, c] >= threshold:
                                to_remove_in_chunk.add(c)
                    for j in range(current_chunk.shape[0]):
                        if j not in to_remove_in_chunk:
                            keep_indices.append(i + j)
                else:
                    keep_indices.extend(list(range(i, end_idx)))
        
        return keep_indices

    # ========================================================================
    # CREATE CONVERSATIONAL PAIRS
    # ========================================================================

    def create_conversational_pairs(self, convo_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized conversational pair creation"""
        pairs = []
        conversations: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for entry in convo_entries:
            convo_id = entry['metadata'].get('conversation_id')
            if convo_id is not None:
                conversations[convo_id].append(entry)
        
        logger.info(f"Processing {len(conversations)} conversations...")
        all_pairs_from_conversations = []
        
        for convo_id, msgs in tqdm(conversations.items(), desc="Creating conversational pairs"):
            msgs.sort(key=lambda x: x['metadata'].get('timestamp') or 0)
            
            current_user_msg = None
            conversation_context = []
            
            batch_user_texts, batch_assistant_texts, batch_metadata = [], [], []
            
            for msg in msgs:
                author, text = msg['metadata'].get('author'), msg.get('cleaned_text')
                if not text: continue
                
                if author == 'user':
                    current_user_msg = msg
                    context_text = " ".join(conversation_context[-self.config.context_window_size:])
                    current_user_msg['context'] = context_text
                    
                elif author == 'assistant' and current_user_msg:
                    # Extract ONLY the final user message (without context prefix)
                    user_text_raw = current_user_msg['cleaned_text']
                    
                    # If there's a "User:" prefix, extract just the final message
                    user_match = re.search(r'User:\s*(.+?)(?:\s*Assistant:|$)', user_text_raw, re.DOTALL)
                    if user_match:
                        final_user_text = user_match.group(1).strip()
                    else:
                        final_user_text = user_text_raw
                    
                    batch_user_texts.append(final_user_text)
                    batch_assistant_texts.append(text)
                    
                    # PATCH: Normalize metadata types for schema consistency
                    batch_metadata.append({
                        'user_msg': normalize_metadata_types(current_user_msg['metadata']),
                        'assistant_msg': normalize_metadata_types(msg['metadata']),
                        'source_file': str(current_user_msg['metadata'].get('source_file', 'conversation')),
                        'themes': msg.get('semantic_labels', {}).get('themes', []),
                        'source': 'conversation'
                    })
                    
                    conversation_context.extend([f"User: {final_user_text}", f"Assistant: {text}"])
                    current_user_msg = None
                else:
                    current_user_msg = None

            if batch_user_texts:
                quality_metrics_batch = self._assess_pair_quality_batch(batch_user_texts, batch_assistant_texts)
                
                for user, assistant, meta, quality in zip(
                    batch_user_texts, batch_assistant_texts, batch_metadata, quality_metrics_batch
                ):
                    if quality["quality_score"] >= self.config.qa_quality_score_threshold:
                        all_pairs_from_conversations.append({
                            'text': f"<|user|>{user}<|assistant|>{assistant}<|endoftext|>",
                            'user': user, 
                            'assistant': assistant,
                            'quality_metrics': quality, 
                            'source_metadata': meta
                        })
        
        logger.info(f"Created {len(all_pairs_from_conversations)} conversational pairs")
        return all_pairs_from_conversations

    def _assess_pair_quality_batch(self, user_texts: List[str], assistant_texts: List[str]) -> List[Dict[str, Any]]:
        """Assess quality for multiple Q&A pairs at once"""
        results = []
        all_texts = user_texts + assistant_texts
        
        if self.use_semantic_filtering:
            all_embeddings = self._compute_embeddings_batch(all_texts)
            user_embeddings = all_embeddings[:len(user_texts)]
            assistant_embeddings = all_embeddings[len(user_texts):]
        else:
            user_embeddings = assistant_embeddings = None
        
        for i, (user_text, assistant_text) in enumerate(zip(user_texts, assistant_texts)):
            metrics = {
                "user_len": len(user_text), "assistant_len": len(assistant_text),
                "user_words": len(user_text.split()), "assistant_words": len(assistant_text.split()),
                "semantic_similarity": 0.0, "length_ratio": 0.0, "quality_score": 0.0,
                "readability_score": 0.0, "information_density": 0.0
            }

            if metrics["user_len"] > 0:
                metrics["length_ratio"] = metrics["assistant_len"] / metrics["user_len"]

            if self.use_semantic_filtering and user_embeddings is not None and user_embeddings.size > 0:
                try:
                    similarity = cosine_similarity([user_embeddings[i]], [assistant_embeddings[i]])[0][0]
                    metrics["semantic_similarity"] = float(similarity)
                except Exception as e:
                    metrics["semantic_similarity"] = 0.0

            sentences = len(re.split(r'[.!?]+', assistant_text))
            avg_sentence_len = metrics["assistant_words"] / max(sentences, 1)
            metrics["readability_score"] = min(1.0, 20 / max(avg_sentence_len, 1))

            words = assistant_text.lower().split()
            metrics["information_density"] = len(set(words)) / max(len(words), 1)

            score = 1.0
            if not (self.config.min_length_ratio <= metrics["length_ratio"] <= self.config.max_length_ratio):
                score *= 0.5
            if not (self.config.min_semantic_similarity <= metrics["semantic_similarity"] <= self.config.max_semantic_similarity):
                score *= 0.7
            score *= (0.7 + 0.3 * metrics["readability_score"])
            score *= (0.8 + 0.2 * metrics["information_density"])

            metrics["quality_score"] = score
            results.append(metrics)
        
        return results

    # ========================================================================
    # CREATE PDF Q&A PAIRS
    # ========================================================================

    def create_pdf_qa_pairs(self, pdf_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized PDF Q&A pair creation with QABuilder and QualityScorer"""
        if not self.use_semantic_filtering:
            logger.warning("Semantic filtering disabled, cannot create PDF Q&A pairs.")
            return []
            
        qa_pairs = []
        qa_builder = QABuilder(self.config, self.model)
        logger.info(f"Generating Q&A pairs from {len(pdf_entries)} PDF chunks...")
        
        by_source = defaultdict(list)
        for entry in pdf_entries:
            by_source[entry['metadata'].get('filename', 'unknown_pdf')].append(entry)
        
        for source, entries in tqdm(by_source.items(), desc="Processing PDF sources"):
            max_entries = min(len(entries), self.config.max_pairs_per_source // 3)
            selected_entries = random.sample(entries, max_entries) if len(entries) > max_entries else entries
            
            batch_questions, batch_answers, batch_metadata = [], [], []
            
            for entry in selected_entries:
                # Filter by chunk quality
                chunk_quality = entry.get('quality_scores', {}).get('composite_quality', 0)
                if chunk_quality < 0.4:
                    continue
                
                chunk_text, metadata = entry.get('cleaned_text'), entry.get('metadata', {})
                if not chunk_text: continue
                
                # Use QABuilder
                questions = qa_builder._diverse_prompts(chunk_text, metadata)
                
                for question in questions:
                    batch_questions.append(question)
                    batch_answers.append(chunk_text)
                    
                    # PATCH: Normalize metadata types for schema consistency
                    normalized_meta = normalize_metadata_types(metadata)
                    normalized_meta.update({
                        'source_file': str(source),
                        'source': 'pdf'
                    })
                    batch_metadata.append(normalized_meta)
            
            if batch_questions:
                quality_metrics_batch = self._assess_pair_quality_batch(batch_questions, batch_answers)
                
                for q, a, meta, quality in zip(batch_questions, batch_answers, batch_metadata, quality_metrics_batch):
                    if quality["quality_score"] >= self.config.qa_quality_score_threshold:
                        qa_pairs.append({
                            'text': f"<|user|>{q}<|assistant|>{a}<|endoftext|>",
                            'user': q, 
                            'assistant': a,
                            'quality_metrics': quality, 
                            'source_metadata': meta
                        })
        
        logger.info(f"Created {len(qa_pairs)} PDF Q&A pairs")
        return qa_pairs

    # ========================================================================
    # LEARNING AND SAVING
    # ========================================================================

    def learn_and_save_semantics(self):
        """If in adaptive mode, run the learning step and save the memory."""
        if self.labeler and self.config.semantic_mode == 'adaptive':
            logger.info("Running adaptive semantic learning step...")
            self.labeler.learn_from_run()
            self.labeler.save_memory()
            self.labeler.print_semantic_summary()

    # ========================================================================
    # CREATE SPLITS AND SAVE
    # ========================================================================

    def create_data_splits(self, all_pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Creates stratified train, validation, and test splits based on quality."""
        logger.info(f"Splitting {len(all_pairs)} pairs into train/val/test...")
        if not all_pairs:
            return {"train": [], "validation": [], "test": []}

        all_pairs.sort(key=lambda x: x['quality_metrics']['quality_score'], reverse=True)
        
        # Stratify by shuffling within quartiles
        q = 4
        total = len(all_pairs)
        for i in range(q):
            s = i * (total // q)
            e = (i + 1) * (total // q) if i < q - 1 else total
            block = all_pairs[s:e]
            random.shuffle(block)
            all_pairs[s:e] = block

        tr_end = int(total * self.config.split_ratio[0])
        va_end = tr_end + int(total * self.config.split_ratio[1])

        splits = {
            "train": all_pairs[:tr_end],
            "validation": all_pairs[tr_end:va_end],
            "test": all_pairs[va_end:]
        }

        logger.info(f"Splits: Train={len(splits['train'])}, Validation={len(splits['validation'])}, Test={len(splits['test'])}")
        return splits

    def save_datasets(self, splits: Dict[str, List[Dict[str, Any]]]) -> None:
        """Saves the splits to .jsonl files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_summary = {
            'total_pairs': sum(len(data) for data in splits.values()),
            'splits': {name: len(data) for name, data in splits.items()},
            'quality_stats': {},
            'theme_distribution': defaultdict(int),
            'source_distribution': defaultdict(int)
        }
        
        for split_name, data in splits.items():
            if not data: continue
            
            qualities = [p['quality_metrics']['quality_score'] for p in data]
            metadata_summary['quality_stats'][split_name] = {
                'mean': float(np.mean(qualities)), 
                'std': float(np.std(qualities)),
                'min': float(np.min(qualities)), 
                'max': float(np.max(qualities))
            }
            
            for pair in data:
                themes = pair.get('source_metadata', {}).get('themes', ['general'])
                for theme in themes:
                    metadata_summary['theme_distribution'][theme] += 1
                
                source = pair.get('source_metadata', {}).get('source', 'unknown')
                metadata_summary['source_distribution'][source] += 1
            
            # PATCH: Save formatted for training with "text" AND "source_metadata"
            path = output_dir / f"{self.config.output_prefix}_{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    # Include source_metadata with normalized types for theme-weighted sampling
                    formatted = {
                        "text": item['text'],
                        "source_metadata": normalize_metadata_types(item.get('source_metadata', {}))
                    }
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            
            # Save detailed for analysis
            det_path = output_dir / f"{self.config.output_prefix}_{split_name}_detailed.jsonl"
            with open(det_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"Saved {len(data)} items to {path}")
        
        with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dataset metadata to {output_dir / 'dataset_metadata.json'}")

# ============================================================================
# CLI AND MAIN EXECUTION
# ============================================================================

def parse_args() -> Config:
    """Command-line interface to build Config object"""
    p = argparse.ArgumentParser(description='Memory Texts Data Formatter - Process conversation and PDF data')
    
    # --- IO and General ---
    p.add_argument('--input-memory-texts', default='memory_texts.npy',
                   help='Input memory texts numpy file')
    p.add_argument('--input-memory-metadata', default='memory_metadata.pkl',
                   help='Input memory metadata pickle file')
    p.add_argument('--output-dir', default='data_finetune', help='Output directory for datasets')
    p.add_argument('--output-prefix', default='dataset', help='Prefix for output .jsonl files')
    p.add_argument('--no-gzip', action='store_true', help='Disable gzip compression')
    
    # --- Performance ---
    p.add_argument('--workers', type=int, default=None, help='Max parallel workers (default: auto)')
    p.add_argument('--batch-size', type=int, default=64, help='Batch size for embeddings')
    p.add_argument('--embedding-cache-size', type=int, default=50000, help='Size of LRU cache for embeddings')
    
    # --- Embeddings ---
    p.add_argument('--embedding-model', default='all-MiniLM-L12-v2', help='SentenceTransformer model')
    p.add_argument('--force-cpu', action='store_true', help='Force CPU for embeddings')
    
    # --- Semantic Labeling ---
    p.add_argument('--enable-semantic-labeling', action='store_true', help='Enable semantic labeling')
    p.add_argument('--extract-keyphrases', action='store_true', help='Use KeyBERT for phrase extraction')
    p.add_argument('--semantic-mode', choices=['normal', 'adaptive'], default='normal', help='Semantic mode')
    p.add_argument('--semantic-method', choices=['tfidf', 'ipf', 'hybrid'], default='hybrid', help='Semantic method')
    p.add_argument('--semantic-memory-path', default='semantic_memory.pkl', help='Path to semantic memory file')
    
    # --- Q&A Pair Quality ---
    p.add_argument('--dedup-similarity-threshold', type=float, default=0.95, help='Similarity to deduplicate source chunks')
    p.add_argument('--min-semantic-similarity', type=float, default=0.1, help='Min Q/A relevance')
    p.add_argument('--max-semantic-similarity', type=float, default=0.95, help='Max Q/A relevance (avoid identical)')
    p.add_argument('--qa-quality-score-threshold', type=float, default=0.46, help='Min composite score for a Q&A pair')
    
    # --- Misc ---
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = p.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Build config from args
    cfg = Config(
        input_memory_texts_path=args.input_memory_texts,
        input_memory_metadata_path=args.input_memory_metadata,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        gzip_output=not args.no_gzip,
        max_workers=args.workers,
        batch_size=args.batch_size,
        embedding_cache_size=args.embedding_cache_size,
        embedding_model=args.embedding_model,
        force_cpu=args.force_cpu,
        enable_semantic_labeling=args.enable_semantic_labeling,
        extract_keyphrases=args.extract_keyphrases,
        semantic_mode=args.semantic_mode,
        semantic_method=args.semantic_method,
        semantic_memory_path=args.semantic_memory_path,
        dedup_similarity_threshold=args.dedup_similarity_threshold,
        min_semantic_similarity=args.min_semantic_similarity,
        max_semantic_similarity=args.max_semantic_similarity,
        qa_quality_score_threshold=args.qa_quality_score_threshold,
        seed=args.seed
    )
    
    return cfg

def main(cfg: Config):
    """Main processing pipeline - Memory Texts Only"""
    
    # Auto-detect GPU availability (Added logic)
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        if cfg.force_cpu:
             logger.info(f"🚀 Compatible GPU detected: {torch.cuda.get_device_name(0)}, but --force-cpu is set. Using CPU.")
        else:
             logger.info(f"🚀 Compatible GPU detected: {torch.cuda.get_device_name(0)}. Enabling GPU acceleration.")
    else:
        logger.info("⚠️ No compatible GPU detected. Falling back to CPU.")
    
    logger.info("=" * 70)
    logger.info("MEMORY TEXTS DATA FORMATTER")
    logger.info("=" * 70)
    logger.info(f"Memory Texts: {cfg.input_memory_texts_path}")
    logger.info(f"Memory Metadata: {cfg.input_memory_metadata_path}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Semantic labeling: {cfg.enable_semantic_labeling}")
    if cfg.enable_semantic_labeling:
        logger.info(f"  - Mode: {cfg.semantic_mode}")
        logger.info(f"  - Method: {cfg.semantic_method}")
        logger.info(f"  - Keyphrases: {cfg.extract_keyphrases}")
    logger.info("=" * 70)
    
    # Initialize processor with all components
    processor = OptimizedDataProcessor(
        config=cfg,
        use_semantic_filtering=True
    )
    
    # ========================================================================
    # LOAD MEMORY DATA
    # ========================================================================
    
    logger.info("\n🔥 LOADING MEMORY DATA...")
    
    # Load memory texts and metadata
    memory_entries_raw = processor.load_memory_texts()
    
    total_raw = len(memory_entries_raw)
    logger.info(f"\n✓ Total raw entries loaded: {total_raw}")
    
    if total_raw == 0:
        logger.error("No data to process! Exiting.")
        return
    
    # ========================================================================
    # PROCESS MEMORY ENTRIES (Clean, deduplicate, score, label)
    # ========================================================================
    
    logger.info("\n🔧 PROCESSING MEMORY ENTRIES...")
    cleaned_memory_entries = processor.deduplicate_and_clean_entries(memory_entries_raw)
    
    # Separate by source type
    convo_entries = [e for e in cleaned_memory_entries if e.get('metadata', {}).get('source') == 'conversation']
    pdf_entries = [e for e in cleaned_memory_entries if e.get('metadata', {}).get('source') == 'pdf']
    
    logger.info(f"✓ Memory data distribution:")
    logger.info(f"  - Conversation entries: {len(convo_entries)}")
    logger.info(f"  - PDF entries: {len(pdf_entries)}")
    
    # ========================================================================
    # CREATE PAIRS FROM MEMORY ENTRIES
    # ========================================================================
    
    logger.info("\n🔄 CREATING PAIRS FROM MEMORY ENTRIES...")
    
    # Create conversational pairs
    conversational_pairs = processor.create_conversational_pairs(convo_entries)
    
    # Create PDF Q&A pairs
    pdf_qa_pairs = processor.create_pdf_qa_pairs(pdf_entries)
    
    # ========================================================================
    # MERGE ALL PAIRS
    # ========================================================================
    
    logger.info("\n🔀 MERGING ALL DATA SOURCES...")
    
    all_final_pairs = conversational_pairs + pdf_qa_pairs
    
    logger.info(f"✓ Total merged pairs: {len(all_final_pairs)}")
    logger.info(f"  - From conversations: {len(conversational_pairs)}")
    logger.info(f"  - From PDFs: {len(pdf_qa_pairs)}")
    
    if not all_final_pairs:
        logger.error("No quality dialogue pairs created! Consider adjusting quality thresholds.")
        return
    
    # ========================================================================
    # CREATE SPLITS AND SAVE
    # ========================================================================
    
    # Create train/val/test splits
    splits = processor.create_data_splits(all_final_pairs)
    
    # Save datasets
    processor.save_datasets(splits)
    
    # Run learning step
    processor.learn_and_save_semantics()
    
    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    
    total_pairs = len(all_final_pairs)
    avg_quality = np.mean([pair['quality_metrics']['quality_score'] for pair in all_final_pairs])
    
    logger.info(f"\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total quality pairs processed: {total_pairs}")
    logger.info(f"Average quality score: {avg_quality:.3f}")
    logger.info(f"\nData Sources:")
    logger.info(f"  - Conversational pairs: {len(conversational_pairs)}")
    logger.info(f"  - PDF Q&A pairs: {len(pdf_qa_pairs)}")
    logger.info(f"\nSplits:")
    logger.info(f"  - Train: {len(splits['train'])}")
    logger.info(f"  - Validation: {len(splits['validation'])}")
    logger.info(f"  - Test: {len(splits['test'])}")
    logger.info(f"\nGenerated files in: {cfg.output_dir}")
    logger.info("=" * 70)
    
    # Print theme statistics
    all_themes = []
    source_counts = defaultdict(int)
    
    for entry in all_final_pairs:
        all_themes.extend(entry.get('source_metadata', {}).get('themes', []))
        source = entry.get('source_metadata', {}).get('source', 'unknown')
        source_counts[source] += 1
    
    if all_themes:
        theme_counts = Counter(all_themes)
        logger.info("\n📊 THEME DISTRIBUTION:")
        logger.info(f"Unique themes: {len(theme_counts)}")
        logger.info("Top 20 themes:")
        for theme, count in theme_counts.most_common(20):
            logger.info(f"  {theme}: {count}")
        if len(theme_counts) > 20:
            logger.info(f"  ... and {len(theme_counts) - 20} more themes")
    
    logger.info("\n📈 SOURCE DISTRIBUTION:")
    for source, count in source_counts.items():
        percentage = (count / total_pairs) * 100
        logger.info(f"  {source}: {count} ({percentage:.1f}%)")
    
    logger.info("\n✅ All datasets saved successfully!")
    logger.info(f"   Training format files: {cfg.output_prefix}_{{train,validation,test}}.jsonl")
    logger.info(f"   Detailed files: {cfg.output_prefix}_{{train,validation,test}}_detailed.jsonl")
    logger.info(f"   Metadata: dataset_metadata.json")
    
    if cfg.semantic_mode == 'adaptive':
        logger.info(f"   Semantic memory: {cfg.semantic_memory_path}")

if __name__ == "__main__":
    config = parse_args()
    main(config)
