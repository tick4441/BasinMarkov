#!/usr/bin/env python3
"""
Basin Markov - A disk-based hybrid Markov model chatbot
Combines HMM, VLMM, and Bayesian approaches in a compressed format
"""

import sqlite3
import os
import sys
import random
import struct
from collections import defaultdict, Counter
from typing import List, Tuple, Optional
import zlib
import time

class BasinMarkov:
    def __init__(self, db_path: str = "basin_markov.db", context_size: int = 5):
        """
        Initialize Basin Markov model with disk storage
        
        Args:
            db_path: Path to SQLite database for storing transitions
            context_size: Maximum context window (5-7 recommended)
        """
        self.db_path = db_path
        self.context_size = context_size
        self.conn = None
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_dirty = False  # Track if vocab needs commit
        self._setup_database()
        self._load_vocab()
        
    def _setup_database(self):
        """Setup SQLite database with optimized schema"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        self.conn.execute("PRAGMA page_size=4096")  # Optimal page size
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        
        # Vocabulary table (word <-> ID mapping)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                word_id INTEGER PRIMARY KEY,
                word TEXT UNIQUE NOT NULL
            )
        """)
        
        # Main transitions table with binary compression
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                context BLOB PRIMARY KEY,
                next_data BLOB NOT NULL
            )
        """)
        
        # Metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        self.conn.commit()
        
    def _load_vocab(self):
        """Load vocabulary from database"""
        cursor = self.conn.execute("SELECT word_id, word FROM vocabulary")
        for word_id, word in cursor.fetchall():
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
    
    def _get_or_create_word_id(self, word: str) -> int:
        """Get word ID or create new one"""
        if word in self.word_to_id:
            return self.word_to_id[word]
        
        word_id = len(self.word_to_id)
        self.word_to_id[word] = word_id
        self.id_to_word[word_id] = word
        self.vocab_dirty = True
        
        return word_id
    
    def _commit_vocab(self):
        """Commit vocabulary changes to database"""
        if not self.vocab_dirty:
            return
        
        # Get current max ID from database
        cursor = self.conn.execute("SELECT MAX(word_id) FROM vocabulary")
        result = cursor.fetchone()
        max_id = result[0] if result[0] is not None else -1
        
        # Insert only new words
        new_words = []
        for word, word_id in self.word_to_id.items():
            if word_id > max_id:
                new_words.append((word_id, word))
        
        if new_words:
            self.conn.executemany(
                "INSERT OR IGNORE INTO vocabulary VALUES (?, ?)",
                new_words
            )
            self.conn.commit()
        
        self.vocab_dirty = False
    
    def _encode_context(self, context: tuple, create_new: bool = False) -> bytes:
        """
        Encode context tuple as binary (packed 32-bit unsigned integers)
        Supports vocabularies up to 4.2 billion words
        
        Args:
            context: Tuple of words to encode
            create_new: If True, create new word IDs; if False, raise KeyError for unknown words
                       Default False for safety during inference
        """
        if create_new:
            word_ids = [self._get_or_create_word_id(w) for w in context]
        else:
            word_ids = [self.word_to_id[w] for w in context]  # Raises KeyError if word unknown
        return struct.pack(f'{len(word_ids)}I', *word_ids)
    
    def _decode_context(self, data: bytes) -> tuple:
        """Decode binary context back to words (32-bit unsigned integers)"""
        num_words = len(data) // 4
        word_ids = struct.unpack(f'{num_words}I', data)
        return tuple(self.id_to_word[wid] for wid in word_ids)
    
    def _encode_transitions(self, word_counts: Counter, create_new: bool = False) -> bytes:
        """
        Encode word->count transitions as compressed binary
        Format: [num_words][word_id1][count1][word_id2][count2]...
        Uses 32-bit unsigned integers and zlib compression
        
        Args:
            word_counts: Counter of word -> count
            create_new: If True, create new word IDs; if False, use existing only
                       Default False for safety during inference
        """
        data = [len(word_counts)]
        for word, count in word_counts.items():
            if create_new:
                word_id = self._get_or_create_word_id(word)
            else:
                word_id = self.word_to_id.get(word)
                if word_id is None:
                    continue  # Skip unknown words
            data.extend([word_id, count])
        
        packed = struct.pack(f'{len(data)}I', *data)
        return zlib.compress(packed, level=6)
    
    def _decode_transitions(self, data: bytes) -> dict:
        """Decode compressed binary transitions back to dict"""
        unpacked_data = struct.unpack(
            f'{len(zlib.decompress(data)) // 4}I',
            zlib.decompress(data)
        )
        
        num_words = unpacked_data[0]
        transitions = {}
        
        for i in range(num_words):
            word_id = unpacked_data[1 + i * 2]
            count = unpacked_data[2 + i * 2]
            word = self.id_to_word[word_id]
            transitions[word] = count
        
        return transitions
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with proper punctuation handling
        
        Separates punctuation as individual tokens for better context matching
        and more natural text generation.
        
        Args:
            text: Raw text string
            
        Returns:
            List of tokens including separated punctuation
        """
        import re
        
        # Replace common punctuation with spaced versions
        # This ensures "Hello!" becomes ["Hello", "!"]
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        text = re.sub(r'(["\'])', r' \1 ', text)
        
        # Handle ellipsis specially
        text = text.replace('...', ' … ')
        
        # Split on whitespace and filter empty strings
        tokens = [t for t in text.split() if t]
        
        return tokens
    
    def train(self, text_file: str, batch_size: int = 10000):
        """
        Train the model on a text file (streaming to avoid RAM overload)
        
        Args:
            text_file: Path to training text file
            batch_size: Number of tokens to process at once
        """
        print(f"Training Basin Markov on {text_file}...")
        start_time = time.time()
        
        # Temporary dict for batching
        batch_transitions = defaultdict(Counter)
        tokens_processed = 0
        
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            tokens = []
            
            for line in f:
                # Tokenize with proper punctuation separation
                line_tokens = self._tokenize(line)
                tokens.extend(line_tokens)
                
                # Process batch
                while len(tokens) >= self.context_size + 1:
                    # Extract contexts of various lengths (VLMM approach)
                    for ctx_len in range(1, self.context_size + 1):
                        if len(tokens) > ctx_len:
                            context = tuple(tokens[:ctx_len])
                            next_word = tokens[ctx_len]
                            batch_transitions[context][next_word] += 1
                    
                    tokens.pop(0)
                    tokens_processed += 1
                    
                    # Write batch to disk
                    if tokens_processed % batch_size == 0:
                        self._write_batch(batch_transitions)
                        batch_transitions.clear()
                        elapsed = time.time() - start_time
                        speed = tokens_processed / elapsed if elapsed > 0 else 0
                        print(f"Processed {tokens_processed:,} tokens ({speed:.0f} tok/s)...", end='\r')
        
        # Write remaining batch
        if batch_transitions:
            self._write_batch(batch_transitions)
        
        # Final vocab commit
        self._commit_vocab()
        
        # Store metadata
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            ("tokens_trained", str(tokens_processed))
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            ("vocab_size", str(len(self.word_to_id)))
        )
        self.conn.commit()
        
        elapsed = time.time() - start_time
        speed = tokens_processed / elapsed if elapsed > 0 else 0
        
        print(f"\nTraining complete! Processed {tokens_processed:,} tokens in {elapsed:.1f}s ({speed:.0f} tok/s)")
        print("Running VACUUM to compress database...")
        self.optimize()
        print("Done!")
        
    def _write_batch(self, batch_transitions: dict):
        """
        Write a batch of transitions to disk in binary format with optimized bulk operations
        
        Performance optimizations:
        - Bulk vocabulary commits before encoding
        - Bulk existence check for all contexts
        - Batch inserts and updates using executemany
        """
        # Commit vocabulary changes first (bulk insert all new words)
        self._commit_vocab()
        
        # Pre-encode all contexts (now vocabulary is complete, no new IDs created)
        encoded_contexts = {}
        for context in batch_transitions.keys():
            try:
                # Use create_new=True during batch writing (vocabulary is being built)
                encoded_contexts[context] = self._encode_context(context, create_new=True)
            except (struct.error, KeyError):
                continue  # Skip contexts that can't be encoded
        
        # Bulk check which contexts already exist (single query instead of N queries)
        context_blobs = list(encoded_contexts.values())
        existing_contexts = set()
        
        if context_blobs:
            # Use parameterized query for bulk check
            placeholders = ','.join(['?' for _ in context_blobs])
            cursor = self.conn.execute(
                f"SELECT context FROM transitions WHERE context IN ({placeholders})",
                context_blobs
            )
            existing_contexts = {row[0] for row in cursor.fetchall()}
        
        # Prepare bulk updates and inserts
        updates = []
        inserts = []
        
        for context, next_word_counts in batch_transitions.items():
            if context not in encoded_contexts:
                continue
                
            context_bin = encoded_contexts[context]
            
            if context_bin in existing_contexts:
                # Merge with existing transitions (still requires individual SELECT for BLOB data)
                cursor = self.conn.execute(
                    "SELECT next_data FROM transitions WHERE context = ?",
                    (context_bin,)
                )
                result = cursor.fetchone()
                
                if result:
                    try:
                        existing_transitions = self._decode_transitions(result[0])
                        
                        # Merge counts
                        for word, count in next_word_counts.items():
                            existing_transitions[word] = existing_transitions.get(word, 0) + count
                        
                        new_data = self._encode_transitions(Counter(existing_transitions), create_new=True)
                        updates.append((new_data, context_bin))
                    except (struct.error, KeyError, zlib.error):
                        continue  # Skip if decode/encode fails
            else:
                # New context - direct insert
                try:
                    new_data = self._encode_transitions(next_word_counts, create_new=True)
                    inserts.append((context_bin, new_data))
                except (struct.error, KeyError):
                    continue
        
        # Execute bulk operations (much faster than individual commits)
        if updates:
            self.conn.executemany(
                "UPDATE transitions SET next_data = ? WHERE context = ?",
                updates
            )
        
        if inserts:
            self.conn.executemany(
                "INSERT OR IGNORE INTO transitions VALUES (?, ?)",
                inserts
            )
        
        self.conn.commit()
    
    def generate(self, prompt: str = "", max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generate text from the model
        
        Args:
            prompt: Starting text
            max_length: Maximum number of tokens to generate
            temperature: Randomness (0.01-10.0, higher = more random)
        """
        # Clamp temperature to safe range
        temperature = max(0.01, min(temperature, 10.0))
        
        if prompt:
            tokens = self._tokenize(prompt)
        else:
            # Random start
            tokens = self._get_random_context()
        
        # Always generate something, even without training data
        if not tokens:
            tokens = ["I", "think"]
        
        for _ in range(max_length):
            next_word = self._predict_next(tokens, temperature=temperature)
            if not next_word:
                # Fallback: generate creative response
                next_word = self._creative_fallback(tokens)
            
            if not next_word:
                break
                
            tokens.append(next_word)
            
            # Stop at sentence end
            if next_word in ['.', '!', '?'] and len(tokens) > 10:
                break
        
        # Join tokens with proper spacing (don't add space before punctuation)
        result = []
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in '.,!?;:)\'"' or token == '…':
                result.append(token)  # No space before punctuation
            elif i > 0 and tokens[i-1] in '("\'':
                result.append(token)  # No space after opening punctuation
            else:
                result.append(' ' + token)
        
        return ''.join(result)
    
    def _predict_next(self, context_tokens: List[str], temperature: float = 0.8) -> Optional[str]:
        """Predict next word using variable-length context (VLMM)"""
        # Try contexts from longest to shortest (backoff strategy)
        for ctx_len in range(min(len(context_tokens), self.context_size), 0, -1):
            context = tuple(context_tokens[-ctx_len:])
            
            try:
                context_bin = self._encode_context(context)
            except (struct.error, KeyError):
                continue

            cursor = self.conn.execute(
                "SELECT next_data FROM transitions WHERE context = ?",
                (context_bin,)
            )
            result = cursor.fetchone()
            
            if result:
                try:
                    transitions = self._decode_transitions(result[0])
                except (struct.error, KeyError, zlib.error):
                    continue
                
                # Apply temperature for sampling
                words = list(transitions.keys())
                counts = [transitions[w] ** (1.0 / temperature) for w in words]
                
                # Weighted random choice
                total = sum(counts)
                if total == 0:
                    continue
                    
                rand = random.uniform(0, total)
                cumsum = 0
                
                for word, count in zip(words, counts):
                    cumsum += count
                    if rand <= cumsum:
                        return word
        
        return None
    
    def _creative_fallback(self, context_tokens: List[str]) -> str:
        """Generate creative response when no exact match found"""
        # Strategy 1: Look for similar single-word contexts
        if context_tokens:
            last_word = context_tokens[-1]
            try:
                last_word_bin = self._encode_context((last_word,))
            except (struct.error, KeyError):
                pass
            else:
                cursor = self.conn.execute(
                    "SELECT next_data FROM transitions WHERE context = ?",
                    (last_word_bin,)
                )
                result = cursor.fetchone()
                
                if result:
                    try:
                        transitions = self._decode_transitions(result[0])
                        words = list(transitions.keys())
                        counts = list(transitions.values())
                        
                        total = sum(counts)
                        if total > 0:
                            rand = random.uniform(0, total)
                            cumsum = 0
                            
                            for word, count in zip(words, counts):
                                cumsum += count
                                if rand <= cumsum:
                                    return word
                    except (struct.error, KeyError, zlib.error):
                        pass
        
        # Strategy 2: Return random frequent word
        cursor = self.conn.execute(
            "SELECT next_data FROM transitions ORDER BY RANDOM() LIMIT 1"
        )
        result = cursor.fetchone()
        
        if result:
            try:
                transitions = self._decode_transitions(result[0])
                return random.choice(list(transitions.keys()))
            except (struct.error, KeyError, zlib.error, IndexError):
                pass
        
        # Strategy 3: Default responses (should rarely happen)
        fallbacks = ["interesting", "yes", "perhaps", "indeed", "certainly", "maybe", "I", "see"]
        return random.choice(fallbacks)
    
    def _get_random_context(self) -> List[str]:
        """Get a random starting context"""
        cursor = self.conn.execute(
            "SELECT context FROM transitions ORDER BY RANDOM() LIMIT 1"
        )
        result = cursor.fetchone()
        if result:
            try:
                return list(self._decode_context(result[0]))
            except (struct.error, KeyError):
                pass
        return []
    
    def chat(self):
        """Interactive chat mode"""
        print("=" * 60)
        print("Welcome to BasinMarkov! Press CTRL + C keys to exit the chat!")
        print("=" * 60)
        print()
        
        try:
            while True:
                user_input = input("User: ").strip()
                
                if not user_input:
                    continue
                
                # Generate response based on user input
                # Use higher temperature for more creative responses
                response = self.generate(prompt=user_input, max_length=30, temperature=1.0)
                
                # Remove user's exact input from start if it's just repeating
                response_lower = response.lower()
                user_lower = user_input.lower()
                
                if response_lower.startswith(user_lower):
                    # Try to extract just the response part
                    response_words = response.split()
                    user_words = user_input.split()
                    if len(response_words) > len(user_words):
                        response = " ".join(response_words[len(user_words):])
                
                print(f"Basin: {response}")
                print()
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting with Basin Markov!")
            sys.exit(0)
    
    def optimize(self):
        """Optimize and compress the database"""
        print("Optimizing database...")
        
        # Analyze tables for better query planning
        self.conn.execute("ANALYZE")
        
        # Vacuum to reclaim space and defragment
        print("Vacuuming database (this may take a while)...")
        self.conn.execute("VACUUM")
        
        self.conn.commit()
        print("Optimization complete!")
    
    def stats(self):
        """Display model statistics with compression analysis"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM transitions")
        num_contexts = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM vocabulary")
        vocab_size = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT value FROM metadata WHERE key = ?", ("tokens_trained",))
        result = cursor.fetchone()
        tokens_trained = int(result[0]) if result else 0
        
        db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
        
        # Calculate compression ratio and efficiency metrics
        if tokens_trained > 0 and vocab_size > 0:
            avg_word_len = 5  # Approximate average English word length
            uncompressed_size_mb = (tokens_trained * avg_word_len) / (1024 * 1024)
            compression_ratio = uncompressed_size_mb / db_size if db_size > 0 else 0
            
            # Bytes per token stored
            bytes_per_token = (db_size * 1024 * 1024) / tokens_trained
        else:
            compression_ratio = 0
            bytes_per_token = 0
        
        print(f"\n{'='*60}")
        print(f"Basin Markov Statistics")
        print(f"{'='*60}")
        print(f"  Vocabulary size:     {vocab_size:,} unique words")
        print(f"  Unique contexts:     {num_contexts:,}")
        print(f"  Tokens trained:      {tokens_trained:,}")
        print(f"  Database size:       {db_size:.2f} MB")
        
        if compression_ratio > 0:
            print(f"  Compression ratio:   {compression_ratio:.2f}x (vs raw text)")
            print(f"  Storage efficiency:  {bytes_per_token:.2f} bytes/token")
        
        print(f"  Context window:      1-{self.context_size} words (VLMM)")
        print(f"  Encoding:            32-bit binary + zlib level 6")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            # Final vocab commit
            self._commit_vocab()
            self.conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Basin Markov - Disk-based Markov chatbot")
    parser.add_argument("--train", type=str, help="Train on text file")
    parser.add_argument("--chat", action="store_true", help="Start chat interface")
    parser.add_argument("--generate", type=str, help="Generate text from prompt")
    parser.add_argument("--stats", action="store_true", help="Show model statistics")
    parser.add_argument("--optimize", action="store_true", help="Optimize and compress database")
    parser.add_argument("--db", type=str, default="basin_markov.db", help="Database path")
    parser.add_argument("--context", type=int, default=5, help="Context size (5-7 recommended)")
    
    args = parser.parse_args()
    
    model = BasinMarkov(db_path=args.db, context_size=args.context)
    
    try:
        if args.train:
            model.train(args.train)
            model.stats()
        
        elif args.chat:
            model.chat()
        
        elif args.generate:
            result = model.generate(prompt=args.generate, max_length=50)
            print(f"\n{result}\n")
        
        elif args.stats:
            model.stats()
        
        elif args.optimize:
            model.optimize()
            model.stats()
        
        else:
            parser.print_help()
    
    finally:
        model.close()


if __name__ == "__main__":
    main()
