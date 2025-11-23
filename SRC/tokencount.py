#!/usr/bin/env python3
"""
Token Counter - Count tokens in text files using BasinMarkov's tokenization
"""

import sys
import re
import time
from typing import List

def tokenize(text: str) -> List[str]:
    """
    Tokenize text with proper punctuation handling
    (Same method as BasinMarkov uses)
    """
    # Replace common punctuation with spaced versions
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    text = re.sub(r'(["\'])', r' \1 ', text)
    
    # Handle ellipsis specially
    text = text.replace('...', ' … ')
    
    # Split on whitespace and filter empty strings
    tokens = [t for t in text.split() if t]
    
    return tokens

def count_tokens(file_path: str, show_progress: bool = True):
    """
    Count tokens in a file with streaming to avoid memory issues
    
    Args:
        file_path: Path to text file
        show_progress: Show progress updates
    """
    print(f"Counting tokens in: {file_path}")
    print("=" * 60)
    
    start_time = time.time()
    total_tokens = 0
    total_chars = 0
    total_lines = 0
    unique_tokens = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_tokens = tokenize(line)
                total_tokens += len(line_tokens)
                total_chars += len(line)
                total_lines += 1
                
                # Track unique tokens (careful with memory on huge files)
                if total_tokens < 10_000_000:  # Only track first 10M tokens
                    unique_tokens.update(line_tokens)
                
                # Progress update every 100k lines
                if show_progress and line_num % 100_000 == 0:
                    elapsed = time.time() - start_time
                    speed = total_tokens / elapsed if elapsed > 0 else 0
                    print(f"Lines: {line_num:,} | Tokens: {total_tokens:,} | Speed: {speed:,.0f} tok/s", end='\r')
        
        elapsed = time.time() - start_time
        speed = total_tokens / elapsed if elapsed > 0 else 0
        
        # Final results
        print("\n" + "=" * 60)
        print("Token Count Results:")
        print("=" * 60)
        print(f"  Total tokens:        {total_tokens:,}")
        print(f"  Total characters:    {total_chars:,}")
        print(f"  Total lines:         {total_lines:,}")
        
        if len(unique_tokens) > 0:
            print(f"  Unique tokens:       {len(unique_tokens):,}")
            print(f"  Vocabulary coverage: {len(unique_tokens) / total_tokens * 100:.2f}%")
        else:
            print(f"  Unique tokens:       (file too large to track)")
        
        print(f"  Average tokens/line: {total_tokens / total_lines:.1f}")
        print(f"  Processing time:     {elapsed:.2f}s")
        print(f"  Processing speed:    {speed:,.0f} tokens/second")
        print("=" * 60)
        
        # Size estimates
        print("\nEstimated BasinMarkov Database Size:")
        print("-" * 60)
        
        # Conservative estimate: 10-15 bytes per token after compression
        estimated_size_mb = (total_tokens * 12) / (1024 * 1024)
        print(f"  Estimated DB size:   {estimated_size_mb:.1f} MB - {estimated_size_mb * 1.5:.1f} MB")
        
        # Training time estimate at 50k tok/s
        training_time_sec = total_tokens / 50000
        if training_time_sec < 60:
            print(f"  Est. training time:  {training_time_sec:.0f} seconds (@ 50k tok/s)")
        elif training_time_sec < 3600:
            print(f"  Est. training time:  {training_time_sec / 60:.1f} minutes (@ 50k tok/s)")
        else:
            print(f"  Est. training time:  {training_time_sec / 3600:.1f} hours (@ 50k tok/s)")
        
        print("=" * 60)
        
        return total_tokens
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Count tokens in text files")
    parser.add_argument("file", type=str, help="Path to text file")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress updates")
    parser.add_argument("--simple", action="store_true", help="Simple output (just token count)")
    
    args = parser.parse_args()
    
    if args.simple:
        # Quick count without details
        total = 0
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                total += len(tokenize(line))
        print(total)
    else:
        count_tokens(args.file, show_progress=not args.no_progress)

if __name__ == "__main__":
    main()

