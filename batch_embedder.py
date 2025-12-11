import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
from typing import List, Generator, Optional, Dict, Any
from dataclasses import dataclass, field
import hashlib
import pickle
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Reverted to INFO
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process"""
    model_name: str = 'all-MiniLM-L12-v2'
    batch_size: int = 32
    chunk_size: int = 512
    max_chunk_overlap: int = 50
    min_text_length: int = 15  # Relaxed from 20/30
    max_text_length: int = 2000 # This now acts more as a max chunk length, not total message length
    index_type: str = 'flat'
    use_gpu: bool = False
    save_incremental: bool = True
    deduplication: bool = True
    min_sentence_length: int = 5   # Relaxed from 10
    max_non_alpha_ratio: float = 0.6 # Relaxed from 0.5/0.4
    filter_common_patterns: bool = True
    num_cpu_threads: Optional[int] = None  # None = auto-detect
    enable_parallel_processing: bool = True
    parallel_workers: Optional[int] = None  # None = auto-detect

class ImprovedBatchEmbedder:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config: EmbeddingConfig = config or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[Any] = None
        self.memory_texts: List[str] = []
        self.memory_vectors: List[np.ndarray] = []
        self.memory_metadata: List[Dict[str, Any]] = []
        self.text_hashes: set = set()
        self.total_embedded: int = 0
        self.device: str = "cpu"
        
        self._setup_cpu_threading()
        self._load_model()
        self._setup_device()
    
    def _setup_cpu_threading(self) -> None:
        """Configure CPU threading for optimal performance"""
        if not self.config.use_gpu:
            if self.config.num_cpu_threads is None:
                num_threads = os.cpu_count()
                logger.info(f"Auto-detected {num_threads} CPU cores")
            else:
                num_threads = self.config.num_cpu_threads
                
            torch.set_num_threads(num_threads)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"🔧 Configured CPU threading: {num_threads} threads")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model"""
        try:
            logger.info(f"📥 Loading model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise e
    
    def _setup_device(self) -> None:
        """Setup device (CPU/GPU) for the model"""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            if self.model:
                self.model = self.model.to(self.device)
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("🔧 Using CPU for embeddings")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            return np.array([])
    
    def _embed_batch_parallel(self, texts: List[str]) -> np.ndarray:
        """Embed batches with parallel processing"""
        if not texts or not self.config.enable_parallel_processing:
            return self._embed_batch(texts)
        
        if self.config.parallel_workers is None:
            num_workers = min(os.cpu_count(), 4)
        else:
            num_workers = self.config.parallel_workers
            
        if len(texts) < num_workers * 2:
            return self._embed_batch(texts)
        
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._embed_batch, chunk) for chunk in text_chunks]
            
            for future in tqdm(futures, desc="Parallel Embedding Chunks"):
                try:
                    chunk_embeddings = future.result()
                    if len(chunk_embeddings) > 0:
                        embeddings.extend(chunk_embeddings)
                except Exception as e:
                    logger.error(f"Parallel embedding chunk failed: {e}")
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _clean_and_validate_text(self, text: str) -> str:
        """Clean and validate text before processing"""
        if not isinstance(text, str):
            logger.debug(f"Skipping text: Not a string (type: {type(text)}).")
            return ""
        
        original_text_snippet = text[:100].replace('\n', '\\n') + ('...' if len(text) > 100 else '')
        text = text.strip()
        
        if len(text) < self.config.min_text_length:
            logger.debug(f"Skipping text: Too short ({len(text)} chars < {self.config.min_text_length}). Text: '{original_text_snippet}'")
            return ""
        
        # Removed the hard truncation here. _chunk_text will handle splitting long texts.
        # if len(text) > self.config.max_text_length:
        #     logger.debug(f"Truncating text: Too long ({len(text)} chars > {self.config.max_text_length}). Text: '{original_text_snippet}'")
        #     text = text[:self.config.max_text_length]
        
        words = text.split()
        if len(words) < self.config.min_sentence_length:
            logger.debug(f"Skipping text: Too few words ({len(words)} words < {self.config.min_sentence_length}). Text: '{original_text_snippet}'")
            return ""
            
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_count / len(text) < (1 - self.config.max_non_alpha_ratio):
            logger.debug(f"Skipping text: Too many non-alpha chars (ratio: {alpha_count / len(text):.2f} < {1 - self.config.max_non_alpha_ratio:.2f}). Text: '{original_text_snippet}'")
            return ""
        
        if self.config.filter_common_patterns:
            if re.match(r'^[\s\d\W]*$', text):
                logger.debug(f"Skipping text: Matches common low-quality pattern. Text: '{original_text_snippet}'")
                return ""
        
        return text
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata. Ensures chunks are not too long."""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        # If the text is shorter than or equal to the desired chunk size, return it as a single chunk.
        if len(words) <= self.config.chunk_size and len(text) <= self.config.max_text_length:
            return [{
                'text': text,
                'metadata': {**metadata, 'chunk_id': 0, 'total_chunks': 1}
            }]
        
        # Otherwise, proceed with chunking
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.config.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Ensure the chunk itself doesn't exceed max_text_length (character count)
            # This is a secondary check, as chunk_size is based on words.
            if len(chunk_text) > self.config.max_text_length:
                # If a single word makes it too long, or if a chunk is too long,
                # we might need to adjust chunk_size or max_text_length config.
                # For now, we'll truncate the chunk to max_text_length if it exceeds it.
                # This is a fallback if word-based chunking results in very long char chunks.
                chunk_text = chunk_text[:self.config.max_text_length]
                logger.debug(f"Chunk text truncated to {self.config.max_text_length} chars to fit max_text_length config. Original chunk length: {len(chunk_text)} chars.")

            if len(chunk_text.strip()) < self.config.min_text_length:
                # Skip very short chunks that might result from aggressive chunking
                i += max(1, self.config.chunk_size - self.config.max_chunk_overlap)
                continue
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': len(chunks),
                    'total_chunks': -1 # Will be updated after all chunks are known
                }
            })
            
            i += max(1, self.config.chunk_size - self.config.max_chunk_overlap)
        
        # Update total_chunks metadata for all generated chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_texts_batch(self, text_data: List[Dict[str, Any]]) -> int:
        """Enhanced version with parallel processing option"""
        if not text_data:
            return 0
        
        processed_items = []
        for item in text_data:
            original_text = item.get('text', '')
            metadata = item.get('metadata', {})

            cleaned_text = self._clean_and_validate_text(original_text)
            
            if cleaned_text:
                # _chunk_text now handles splitting long messages into multiple chunks
                chunks = self._chunk_text(cleaned_text, metadata)
                processed_items.extend(chunks)
            else:
                # Debug message already in _clean_and_validate_text
                pass
        
        if not processed_items:
            logger.debug("No items left after initial cleaning and chunking for this batch.")
            return 0
        
        texts_to_embed = []
        metadata_for_embed = []

        if self.config.deduplication:
            for item in processed_items:
                text_hash = self._get_text_hash(item['text'])
                if text_hash not in self.text_hashes:
                    self.text_hashes.add(text_hash)
                    texts_to_embed.append(item['text'])
                    metadata_for_embed.append(item['metadata'])
                else:
                    logger.debug(f"Skipping text due to deduplication: '{item['text'][:50]}...'")
        else:
            texts_to_embed = [item['text'] for item in processed_items]
            metadata_for_embed = [item['metadata'] for item in processed_items]

        if not texts_to_embed:
            logger.debug("No unique items left after deduplication for this batch.")
            return 0

        if self.config.enable_parallel_processing and not self.config.use_gpu:
            embeddings = self._embed_batch_parallel(texts_to_embed)
        else:
            embeddings = self._embed_batch(texts_to_embed)
        
        added_count_current_batch = 0
        for i, emb in enumerate(embeddings):
            if emb is not None and not np.all(emb == 0):
                self.memory_texts.append(texts_to_embed[i])
                self.memory_vectors.append(emb)
                self.memory_metadata.append(metadata_for_embed[i])
                added_count_current_batch += 1
            else:
                logger.warning(f"Skipping embedding for text due to zero vector: '{texts_to_embed[i][:50]}...'")
        
        self.total_embedded += added_count_current_batch
        
        return added_count_current_batch
    
    def stream_conversations(self, path: str, strict_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Enhanced conversation streaming with better Claude chat format support.
        Handles various conversation formats including Claude chat exports.
        """
        if not os.path.exists(path):
            logger.warning(f"Conversations file not found: {path}")
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.debug(f"Conversations data type: {type(data)}")
            
            # Handle different conversation formats
            conversations = []
            
            if isinstance(data, list):
                # This is the primary path for the user's conversations2.json (list of conversation objects)
                conversations = data
            elif isinstance(data, dict):
                # Check for various dictionary formats
                if "conversations" in data:
                    # Format: {"conversations": [...]}
                    conversations = data["conversations"]
                elif "mapping" in data:
                    # Single ChatGPT conversation
                    conversations = [data]
                elif "messages" in data:
                    # Single conversation with messages array
                    conversations = [data]
                elif all(isinstance(v, dict) for v in data.values()):
                    # Dictionary of conversations (keyed by ID)
                    conversations = list(data.values())
                elif all(isinstance(v, list) for v in data.values()):
                    # Dictionary where values are lists of messages
                    for conv_id, messages in data.items():
                        conversations.append({"messages": messages, "id": conv_id})
                else:
                    # Try to treat as single conversation
                    conversations = [data]
            
            for convo_idx, convo in enumerate(conversations):
                if not isinstance(convo, dict):
                    if strict_mode:
                        raise ValueError(f"Conversation entry at index {convo_idx} is not a dictionary.")
                    logger.debug(f"Skipping non-dict conversation entry at index {convo_idx}")
                    continue
                
                # --- START OF LOGIC FOR CLAUDE EXPORT FORMAT (integrated) ---
                # Check if it's a conversation object containing 'chat_messages'
                if "chat_messages" in convo and isinstance(convo["chat_messages"], list):
                    conversation_uuid = convo.get("uuid", f"convo_{convo_idx}") # Get conversation UUID
                    messages = convo["chat_messages"]
                    logger.debug(f"Processing conversation '{conversation_uuid}' with {len(messages)} messages.")
                    for msg_idx, msg in enumerate(messages):
                        try:
                            if isinstance(msg, dict):
                                content = self._extract_message_content(msg)
                                if content:
                                    author = self._extract_author(msg)
                                    timestamp = msg.get("created_at") or msg.get("updated_at") or msg.get("timestamp") or msg.get("time")
                                    
                                    yield {
                                        'text': content,
                                        'metadata': {
                                            'source': 'conversation',
                                            'conversation_id': conversation_uuid, # Use conversation UUID
                                            'message_id': msg.get("uuid", msg_idx), # Use message UUID if available
                                            'author': author,
                                            'timestamp': timestamp
                                        }
                                    }
                            elif strict_mode:
                                raise ValueError(f"Message at index {msg_idx} in conversation {conversation_uuid} is not a dictionary.")
                        except Exception as e:
                            if strict_mode:
                                raise ValueError(f"Error processing message at index {msg_idx} in conversation {conversation_uuid}: {e}") from e
                            logger.debug(f"Skipping message at index {msg_idx} in conversation {conversation_uuid}: {e}")
                # --- END OF LOGIC FOR CLAUDE EXPORT FORMAT ---

                # Handle ChatGPT mapping format (original logic, will be skipped if chat_messages found)
                elif "mapping" in convo:
                    yield from self._process_mapping_format(convo, convo_idx, strict_mode)
                
                # Handle messages array format (original logic, will be skipped if chat_messages found)
                elif "messages" in convo:
                    messages = convo["messages"]
                    if isinstance(messages, list):
                        for msg_idx, msg in enumerate(messages):
                            try:
                                if isinstance(msg, dict):
                                    content = self._extract_message_content(msg)
                                    if content:
                                        author = self._extract_author(msg)
                                        timestamp = msg.get("timestamp") or msg.get("created_at") or msg.get("time")
                                        
                                        yield {
                                            'text': content,
                                            'metadata': {
                                                'source': 'conversation',
                                                'conversation_id': convo_idx,
                                                'message_id': msg_idx,
                                                'author': author,
                                                'timestamp': timestamp
                                            }
                                        }
                                elif strict_mode:
                                    raise ValueError(f"Message at index {msg_idx} in conversation {convo_idx} is not a dictionary.")
                            except Exception as e:
                                if strict_mode:
                                    raise ValueError(f"Error processing message at index {msg_idx} in conversation {convo_idx}: {e}") from e
                                logger.debug(f"Skipping message at index {msg_idx} in conversation {convo_idx}: {e}")
                    else:
                        if strict_mode:
                            raise ValueError(f"'messages' in conversation {convo_idx} is not a list.")
                        logger.debug(f"'messages' in conversation {convo_idx} is not a list, skipping")
                
                # Handle direct message format (e.g., older Claude chat exports or simple lists of messages)
                # This path is less likely if the above 'chat_messages' logic handles the primary format
                else:
                    content = self._extract_message_content(convo)
                    if content:
                        author = self._extract_author(convo)
                        timestamp = convo.get("timestamp") or convo.get("created_at")
                        
                        yield {
                            'text': content,
                            'metadata': {
                                'source': 'conversation',
                                'conversation_id': convo_idx, # This will be the message's index in the top-level list
                                'message_id': convo.get("uuid", convo_idx), # Use UUID if available, else index
                                'author': author,
                                'timestamp': timestamp
                            }
                        }
                            
        except json.JSONDecodeError as e:
            if strict_mode:
                raise ValueError(f"Invalid JSON in conversations file {path}: {e}") from e
            logger.error(f"Invalid JSON in conversations file {path}: {e}")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"Error reading conversations file {path}: {e}") from e
            logger.error(f"Error reading conversations file {path}: {e}")
    
    def _process_mapping_format(self, convo: dict, convo_idx: int, strict_mode: bool) -> Generator[Dict[str, Any], None, None]:
        """Process ChatGPT mapping format"""
        mapping = convo["mapping"]
        for msg_id, msg_data in mapping.items():
            try:
                if not isinstance(msg_data, dict):
                    if strict_mode:
                        raise ValueError(f"Message data for ID {msg_id} is not a dictionary.")
                    continue
                    
                message = msg_data.get("message")
                if not message or not isinstance(message, dict):
                    if strict_mode:
                        raise ValueError(f"Message for ID {msg_id} is missing or not a dictionary.")
                    continue
                    
                content = self._extract_message_content(message)
                if content:
                    author = self._extract_author(message)
                    timestamp = message.get("create_time") or message.get("timestamp")
                    
                    yield {
                        'text': content,
                        'metadata': {
                            'source': 'conversation',
                            'conversation_id': convo_idx,
                            'message_id': msg_id,
                            'author': author,
                            'timestamp': timestamp
                        }
                    }
                    
            except Exception as e:
                if strict_mode:
                    raise ValueError(f"Error processing message ID {msg_id}: {e}") from e
                logger.debug(f"Skipping malformed message {msg_id}: {e}")
    
    def _extract_message_content(self, message: dict) -> str:
        """Extract content from various message formats"""
        if not isinstance(message, dict):
            return ""
        
        # Prioritize 'text' field, then 'content'
        content = message.get("text") or message.get("content") or message.get("message")
        
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, dict):
            # Handle nested content structures like {"parts": ["text"]} or {"text": "..."}
            if "parts" in content and isinstance(content["parts"], list):
                parts = content["parts"]
                if parts and isinstance(parts[0], str):
                    return parts[0].strip()
            elif "text" in content:
                return str(content["text"]).strip()
        elif isinstance(content, list):
            # Handle content as array (e.g., list of strings or list of content objects)
            if content:
                # Try to join string parts if it's a list of strings
                if all(isinstance(item, str) for item in content):
                    return " ".join(item.strip() for item in content).strip() # Ensure stripping each part
                # If it's a list of dictionaries, try to extract text from each
                elif all(isinstance(item, dict) for item in content):
                    extracted_parts = []
                    for item in content:
                        part_text = item.get("text") or item.get("content")
                        if isinstance(part_text, str):
                            extracted_parts.append(part_text.strip()) # Ensure stripping each part
                    return " ".join(extracted_parts).strip()
        
        return ""
    
    def _extract_author(self, message: dict) -> str:
        """Extract author/role from various message formats"""
        if not isinstance(message, dict):
            return "unknown"
        
        # Try different author field names
        author_info = message.get("author") or message.get("role") or message.get("sender")
        
        if isinstance(author_info, dict):
            role = author_info.get("role") or author_info.get("name") or "unknown"
        elif isinstance(author_info, str):
            role = author_info
        else:
            role = "unknown"
        
        # Normalize common role names
        role = role.lower()
        if role in ["human", "user"]:
            return "user"
        elif role in ["assistant", "ai", "claude"]:
            return "assistant"
        else:
            return role
    
    def stream_pdf_chunks(self, json_path: str, strict_mode: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Stream PDF chunks from a JSON file"""
        if not os.path.exists(json_path):
            logger.warning(f"PDF JSON file not found: {json_path}")
            return
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            
            if not isinstance(entries, list):
                if strict_mode:
                    raise ValueError(f"Expected list of PDF entries, got {type(entries)}")
                logger.error(f"Expected list of PDF entries, got {type(entries)}")
                return
            
            for doc_idx, doc in enumerate(entries):
                if not isinstance(doc, dict):
                    if strict_mode:
                        raise ValueError(f"PDF entry at index {doc_idx} is not a dictionary.")
                    continue
                
                text = ""
                for text_field in ["text", "content", "total_text", "body"]:
                    if text_field in doc and isinstance(doc[text_field], str):
                        text = doc[text_field].strip()
                        break
                
                if not text:
                    if strict_mode:
                        raise ValueError(f"Document {doc_idx} has no valid text field.")
                    continue
                
                filename = doc.get("filename", f"document_{doc_idx}.pdf")
                
                source_info = {
                    'source': 'pdf',
                    'filename': filename,
                    'document_id': doc_idx
                }
                
                chunks = self._chunk_text(text, source_info)
                for chunk in chunks:
                    yield chunk
                    
        except json.JSONDecodeError as e:
            if strict_mode:
                raise ValueError(f"Invalid JSON in PDF file {json_path}: {e}") from e
            logger.error(f"Invalid JSON in PDF file {json_path}: {e}")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"Error reading PDF file {json_path}: {e}") from e
            logger.error(f"Error reading PDF file {json_path}: {e}")
    
    def load_and_embed_all(self, 
                          convo_path: Optional[str] = None,
                          convo_path2: Optional[str] = None,
                          pdf_json_path: Optional[str] = None,
                          custom_data: Optional[List[Dict[str, Any]]] = None,
                          strict_mode: bool = False) -> None:
        """Load and embed all data sources"""
        
        logger.info("🟢 Starting comprehensive embedding process...")
        start_time = time.time()
        
        # Estimate total items for progress bar
        total_items_to_process = 0
        
        def estimate_convo_items(path):
            count = 0
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        # For Claude format, estimate total messages by summing chat_messages lengths
                        # For other list formats, count top-level items
                        is_claude_format = False
                        if data and isinstance(data[0], dict) and "chat_messages" in data[0]:
                            is_claude_format = True
                        
                        if is_claude_format:
                            for convo_obj in data:
                                if isinstance(convo_obj, dict) and "chat_messages" in convo_obj and isinstance(convo_obj["chat_messages"], list):
                                    count += len(convo_obj["chat_messages"])
                                else:
                                    count += 1 # Fallback for unexpected items in a Claude list
                        else:
                            count = len(data)
                    elif isinstance(data, dict):
                        if "conversations" in data:
                            count = len(data["conversations"])
                        elif "mapping" in data:
                            count = len(data["mapping"])
                        elif "messages" in data:
                            count = len(data["messages"])
                        else:
                            count = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in data.values())
                except Exception as e:
                    logger.warning(f"Could not estimate count for {path}: {e}")
            return count

        total_items_to_process += estimate_convo_items(convo_path)
        total_items_to_process += estimate_convo_items(convo_path2)

        if pdf_json_path and os.path.exists(pdf_json_path):
            try:
                with open(pdf_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_items_to_process += len(data)
            except Exception as e:
                logger.warning(f"Could not estimate count for PDF file: {e}")

        if custom_data:
            total_items_to_process += len(custom_data)

        pbar = tqdm(total=max(total_items_to_process, 1), desc="Total Embedding Progress", unit="items")
        
        def process_convo_file(path, pbar_instance):
            if path:
                logger.info(f"📥 Processing conversations from: {path}")
                batch = []
                try:
                    for msg_data in self.stream_conversations(path, strict_mode=strict_mode):
                        batch.append(msg_data)
                        if len(batch) >= self.config.batch_size:
                            added = self.add_texts_batch(batch)
                            pbar_instance.update(len(batch))
                            batch = []
                            if self.config.save_incremental and self.total_embedded % 1000 == 0:
                                self._save_checkpoint()
                    if batch:
                        added = self.add_texts_batch(batch)
                        pbar_instance.update(len(batch))
                    logger.info(f"✅ Finished processing conversations from {path}")
                except Exception as e:
                    logger.error(f"Error processing conversations from {path}: {e}")
                    if strict_mode:
                        raise e

        process_convo_file(convo_path, pbar)
        process_convo_file(convo_path2, pbar)
        
        if pdf_json_path:
            logger.info(f"📚 Processing PDF chunks from: {pdf_json_path}")
            batch = []
            
            try:
                for chunk_data in self.stream_pdf_chunks(pdf_json_path, strict_mode=strict_mode):
                    batch.append(chunk_data)
                    
                    if len(batch) >= self.config.batch_size:
                        added = self.add_texts_batch(batch)
                        pbar.update(len(batch))
                        batch = []
                        if self.config.save_incremental and self.total_embedded % 1000 == 0:
                            self._save_checkpoint()
                
                if batch:
                    added = self.add_texts_batch(batch)
                    pbar.update(len(batch))
                
                logger.info(f"✅ Finished processing PDF chunks")
                
            except Exception as e:
                logger.error(f"Error processing PDFs: {e}")
                if strict_mode:
                    raise e
        
        if custom_data:
            logger.info(f"🔧 Processing custom data: {len(custom_data)} items")
            try:
                for i in range(0, len(custom_data), self.config.batch_size):
                    batch = custom_data[i:i + self.config.batch_size]
                    added = self.add_texts_batch(batch)
                    pbar.update(len(batch))
                    if self.config.save_incremental and self.total_embedded % 1000 == 0:
                        self._save_checkpoint()
            except Exception as e:
                logger.error(f"Error processing custom data: {e}")
                if strict_mode:
                    raise e
        
        pbar.close()
        
        if self.total_embedded > 0:
            self.build_and_save_index()
        else:
            logger.warning("No data was embedded. Check your input files and structure.")
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 Embedding complete! Total: {self.total_embedded} items in {elapsed:.2f}s")
        if elapsed > 0:
            logger.info(f"⚡ Rate: {self.total_embedded/elapsed:.2f} items/second")
    
    def _save_checkpoint(self) -> None:
        """Save incremental checkpoint"""
        checkpoint_path = "embedding_checkpoint.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'memory_texts': self.memory_texts,
                    'memory_vectors': self.memory_vectors,
                    'memory_metadata': self.memory_metadata,
                    'text_hashes': self.text_hashes,
                    'total_embedded': self.total_embedded
                }, f)
            logger.debug(f"💾 Checkpoint saved: {self.total_embedded} items")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str = "embedding_checkpoint.pkl") -> bool:
        """Load from checkpoint"""
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            self.memory_texts = data['memory_texts']
            self.memory_vectors = data['memory_vectors']
            self.memory_metadata = data['memory_metadata']
            self.text_hashes = data.get('text_hashes', set())
            self.total_embedded = data['total_embedded']
            
            logger.info(f"📥 Loaded checkpoint: {self.total_embedded} items")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def build_and_save_index(self, output_dir: str = ".") -> None:
        """Build and save FAISS index"""
        if not self.memory_vectors:
            logger.warning("No vectors to index!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"🔨 Building FAISS index with {len(self.memory_vectors)} vectors...")
        
        vectors = np.array(self.memory_vectors).astype('float32')
        dimension = vectors.shape[1]
        
        if self.config.index_type == 'flat' or len(vectors) < 1000:
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.index_type == 'ivf':
            nlist = min(int(np.sqrt(len(vectors))), 100)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            self.index.train(vectors)
        elif self.config.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = 40
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(vectors)
        
        # Save all components
        index_path = output_path / "memory.index"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"💾 FAISS index saved to: {index_path}")
        
        texts_path = output_path / "memory_texts.npy"
        np.save(texts_path, np.array(self.memory_texts, dtype=object))
        logger.info(f"📚 Memory texts saved to: {texts_path}")
        
        metadata_path = output_path / "memory_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.memory_metadata, f)
        logger.info(f"🏷️ Metadata saved to: {metadata_path}")
        
        config_path = output_path / "embedding_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        logger.info(f"⚙️ Configuration saved to: {config_path}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the index for similar texts"""
        if not self.index:
            logger.error("Index not built! Call build_and_save_index() first or load a checkpoint.")
            return []
        
        query_vector = self.model.encode([query], normalize_embeddings=True)
        
        if self.config.index_type == 'hnsw' and hasattr(self.index.hnsw, 'efSearch'):
            self.index.hnsw.efSearch = max(k, self.index.hnsw.efConstruction)
        
        scores, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.memory_texts):
                results.append({
                    'text': self.memory_texts[idx],
                    'metadata': self.memory_metadata[idx],
                    'similarity_score': float(score),
                    'rank': i + 1
                })
        
        return results

    def diagnose_json_files(self, convo_path: Optional[str] = None, convo_path2: Optional[str] = None, pdf_path: Optional[str] = None) -> None:
        """Diagnose JSON file structures for debugging"""
        logger.info("🔍 Diagnosing JSON file structures...")
        
        def _diagnose_single_convo_file(path):
            if not os.path.exists(path):
                logger.warning(f"Conversations file not found: {path}")
                return
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"📄 Conversations file ({path}):")
                logger.info(f"  - Type: {type(data)}")
                logger.info(f"  - Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    logger.info(f"  - First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        logger.info(f"  - First item keys: {list(first_item.keys())}")
                        if 'chat_messages' in first_item and isinstance(first_item['chat_messages'], list):
                            logger.info("  - Appears to be a list of Claude conversation objects (contains 'chat_messages').")
                            if first_item['chat_messages']:
                                first_chat_message = first_item['chat_messages'][0]
                                logger.info(f"    - First chat_message keys: {list(first_chat_message.keys())}")
                                logger.info(f"    - First chat_message sender: {first_chat_message.get('sender')}")
                                logger.info(f"    - First chat_message text snippet: '{first_chat_message.get('text', '')[:50]}...'")
                                if isinstance(first_chat_message.get('content'), list) and len(first_chat_message['content']) > 0:
                                    logger.info(f"    - First chat_message content[0] keys: {list(first_chat_message['content'][0].keys())}")
                                    logger.info(f"    - First chat_message content[0] text snippet: '{first_chat_message['content'][0].get('text', '')[:50]}...'")
                            else:
                                logger.info("  - 'chat_messages' list is empty in the first conversation object.")
                        elif 'mapping' in first_item:
                            logger.info("  - Appears to be ChatGPT 'mapping' format (list of conversations).")
                            if first_item['mapping']:
                                first_map_id = list(first_item['mapping'].keys())[0]
                                first_map_msg = first_item['mapping'][first_map_id]
                                logger.info(f"    - First mapping message keys: {list(first_map_msg.keys())}")
                                if 'message' in first_map_msg and isinstance(first_map_msg['message'], dict):
                                    author_role = first_map_msg['message'].get('author', {}).get('role', 'N/A')
                                    content_keys = list(first_map_msg['message'].get('content', {}).keys())
                                    logger.info(f"      - 'message' content keys: {content_keys}")
                                    logger.info(f"      - Example author role: '{author_role}'")
                        elif 'messages' in first_item and isinstance(first_item['messages'], list):
                            logger.info("  - Appears to be 'messages' array format (list of conversations).")
                            if first_item['messages']:
                                logger.info(f"    - First message in 'messages' keys: {list(first_item['messages'][0].keys())}")
                                author_role = first_item['messages'][0].get('role', 'N/A')
                                logger.info(f"    - Example author role: '{author_role}'")
                        elif 'sender' in first_item and 'text' in first_item and 'content' in first_item:
                            logger.info("  - Appears to be a direct list of message objects (e.g., simple chat export).")
                            logger.info(f"    - First message sender: {first_item.get('sender')}")
                            logger.info(f"    - First message text snippet: '{first_item.get('text', '')[:50]}...'")
                            if isinstance(first_item.get('content'), list) and len(first_item['content']) > 0:
                                logger.info(f"    - First message content[0] keys: {list(first_item['content'][0].keys())}")
                                logger.info(f"    - First message content[0] text snippet: '{first_item['content'][0].get('text', '')[:50]}...'")
                        else:
                            logger.info("  - Does NOT appear to be a recognized conversation format at the top level.")
                            logger.info(f"    - First item structure: {json.dumps(first_item, indent=2)[:500]}...") # Print more of the structure
                    else:
                        logger.info(f"  - First item is not a dictionary, type: {type(first_item)}")
                
                elif isinstance(data, dict):
                    logger.info(f"  - Root keys: {list(data.keys())}")
                    if 'mapping' in data:
                        logger.info("  - Appears to be a single ChatGPT 'mapping' conversation at root.")
                        if data['mapping']:
                            first_map_id = list(data['mapping'].keys())[0]
                            first_map_msg = data['mapping'][first_map_id]
                            logger.info(f"    - First mapping message keys: {list(first_map_msg.keys())}")
                            if 'message' in first_map_msg and isinstance(first_map_msg['message'], dict):
                                author_role = first_map_msg['message'].get('author', {}).get('role', 'N/A')
                                content_keys = list(first_map_msg['message'].get('content', {}).keys())
                                logger.info(f"      - 'message' content keys: {content_keys}")
                                logger.info(f"      - Example author role: '{author_role}'")
                    elif 'messages' in data:
                        logger.info("  - Appears to be a single conversation with 'messages' array at root.")
                        if data['messages'] and isinstance(data['messages'], list):
                            logger.info(f"    - First message in 'messages' keys: {list(data['messages'][0].keys())}")
                            author_role = data['messages'][0].get('role', 'N/A')
                            logger.info(f"    - Example author role: '{author_role}'")
                    else:
                        logger.info("  - Does NOT appear to be a recognized conversation format at the root level.")
                        logger.info(f"    - Full root structure: {json.dumps(data, indent=2)[:500]}...") # Print more of the structure
            except Exception as e:
                logger.error(f"Error diagnosing conversations file {path}: {e}")

        _diagnose_single_convo_file(convo_path)
        _diagnose_single_convo_file(convo_path2) # Diagnose the new conversation file
        
        if pdf_path:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
            else:
                try:
                    with open(pdf_path, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                    
                    logger.info(f"📚 PDF file ({pdf_path}):")
                    logger.info(f"  - Type: {type(data)}")
                    logger.info(f"  - Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        logger.info(f"  - First item type: {type(first_item)}")
                        if isinstance(first_item, dict):
                            logger.info(f"  - First item keys: {list(first_item.keys())}")
                            found_text_field = False
                            for text_field in ["text", "content", "total_text", "body"]:
                                if text_field in first_item:
                                    logger.info(f"  - Contains '{text_field}' field.")
                                    if isinstance(first_item[text_field], str):
                                        logger.info(f"    - '{text_field}' type: string, length: {len(first_item[text_field])}")
                                    else:
                                        logger.info(f"    - '{text_field}' type: {type(first_item[text_field])}")
                                    found_text_field = True
                            if not found_text_field:
                                logger.warning("  - No common text fields ('text', 'content', 'total_text', 'body') found in first item.")
                                
                except Exception as e:
                    logger.error(f"Error diagnosing PDF file {pdf_path}: {e}")

def main():
    """Main execution function with CPU/GPU optimization and progress indication"""
    config = EmbeddingConfig(
        batch_size=64,
        chunk_size=400,
        max_chunk_overlap=50,
        min_text_length=15, # Adjusted
        use_gpu=False,  # Set to True if you have a compatible GPU (CUDA)
        index_type='flat',
        deduplication=True,
        min_sentence_length=5, # Adjusted
        max_non_alpha_ratio=0.6, # Adjusted
        filter_common_patterns=True,
        num_cpu_threads=None,
        enable_parallel_processing=True,
        parallel_workers=None
    )
    
    embedder = ImprovedBatchEmbedder(config)
    
    # Diagnose both conversation files
    embedder.diagnose_json_files("conversations.json", "conversations2.json", "pdf_texts.json")
    
    if embedder.load_checkpoint():
        logger.info("Resuming from checkpoint...")
    
    embedder.load_and_embed_all(
        convo_path="conversations.json",
        convo_path2="conversations2.json", # Pass the new conversation file path
        pdf_json_path="pdf_texts.json",
        strict_mode=False # Keep this as False to avoid crashing on minor issues, but check logs
    )
    
    if embedder.total_embedded > 0:
        logger.info("\n🔍 Testing search functionality:")
        query_texts = [
            "What are the main findings of the study?",
            "Can you explain the concept of consciousness?",
            "Tell me about the project timeline and key milestones."
        ]
        for query in query_texts:
            print(f"\nQuery: '{query}'")
            results = embedder.search(query, k=3)
            if results:
                for result in results:
                    print(f"  Rank {result['rank']}: Score={result['similarity_score']:.4f}")
                    print(f"    Text: {result['text'][:150]}...")
                    print(f"    Metadata: {result['metadata']}")
            else:
                print("  No results found.")
    else:
        logger.warning("No data was embedded. Check your input files and their structure, or try running with strict_mode=True in load_and_embed_all for detailed errors.")

if __name__ == "__main__":
    main()
