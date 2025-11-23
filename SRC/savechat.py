#!/usr/bin/env python3
"""
BasinMarkov Chat Logger - Logs chat sessions without interfering with the main program
Captures input/output and saves to text file without prefixes
"""

import subprocess
import sys
import os
import time
from datetime import datetime

class ChatLogger:
    def __init__(self, log_file: str = "chat_log.txt", append: bool = True):
        """
        Initialize chat logger
        
        Args:
            log_file: Path to save chat logs
            append: If True, append to existing log; if False, overwrite
        """
        self.log_file = log_file
        self.append = append
        
    def log_session(self, basin_script: str = "basin.py", db_path: str = None):
        """
        Run BasinMarkov chat and log the session
        
        Args:
            basin_script: Path to basin.py
            db_path: Optional database path to use
        """
        # Prepare command
        cmd = ["python3", basin_script, "--chat"]
        if db_path:
            cmd.extend(["--db", db_path])
        
        # Open log file
        mode = 'a' if self.append else 'w'
        
        print("=" * 60)
        print("BasinMarkov Chat Logger")
        print("=" * 60)
        print(f"Log file: {self.log_file}")
        print(f"Mode: {'Append' if self.append else 'Overwrite'}")
        print(f"Starting BasinMarkov chat session...")
        print("=" * 60)
        print()
        
        try:
            with open(self.log_file, mode, encoding='utf-8') as log:
                # Write session header
                log.write("\n" + "="*60 + "\n")
                log.write(f"Chat Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("="*60 + "\n\n")
                log.flush()
                
                # Start BasinMarkov process
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Print welcome messages from BasinMarkov
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    print(line, end='')
                    
                    # Stop after welcome message (when it shows "User:")
                    if "User:" in line:
                        break
                
                # Interactive chat loop
                while True:
                    try:
                        # Get user input
                        user_input = input()
                        
                        # Log user input (without "User:" prefix)
                        log.write(user_input + "\n")
                        log.flush()
                        
                        # Send to BasinMarkov
                        process.stdin.write(user_input + "\n")
                        process.stdin.flush()
                        
                        # Read BasinMarkov's response
                        # Skip the "Basin:" prefix line
                        response_line = process.stdout.readline()
                        
                        if "Basin:" in response_line:
                            # Extract just the response text
                            response = response_line.split("Basin:", 1)[1].strip()
                            print(f"Basin: {response}")
                            
                            # Log response (without "Basin:" prefix)
                            log.write(response + "\n")
                            log.flush()
                        
                        # Read empty line after response
                        process.stdout.readline()
                        
                        # Print "User:" prompt for next input
                        next_prompt = process.stdout.readline()
                        if next_prompt:
                            print(next_prompt, end='')
                        
                    except KeyboardInterrupt:
                        # User pressed CTRL+C
                        print("\n\nSaving chat log and exiting...")
                        process.terminate()
                        break
                    except Exception as e:
                        print(f"\nError: {e}")
                        process.terminate()
                        break
                
                # Write session footer
                log.write("\n" + "="*60 + "\n")
                log.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("="*60 + "\n\n")
                
                print(f"\nChat log saved to: {self.log_file}")
                
        except FileNotFoundError:
            print(f"Error: Could not find '{basin_script}'")
            print("Make sure basin.py is in the same directory")
        except Exception as e:
            print(f"Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Log BasinMarkov chat sessions")
    parser.add_argument("--log", type=str, default="chat_log.txt", help="Log file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite log file instead of appending")
    parser.add_argument("--basin", type=str, default="basin.py", help="Path to basin.py")
    parser.add_argument("--db", type=str, help="Database path for BasinMarkov")
    
    args = parser.parse_args()
    
    logger = ChatLogger(log_file=args.log, append=not args.overwrite)
    logger.log_session(basin_script=args.basin, db_path=args.db)

if __name__ == "__main__":
    main()
