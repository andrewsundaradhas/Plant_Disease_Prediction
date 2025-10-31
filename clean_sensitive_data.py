"""
Script to clean sensitive data before committing to GitHub.
Run this before committing to ensure no sensitive information is included.
"""
import os
import re
from pathlib import Path

def clean_file(file_path):
    """Remove sensitive data from a file."""
    sensitive_patterns = [
        r'(?i)password\s*[=:].*',
        r'(?i)secret\s*[=:].*',
        r'(?i)key\s*[=:].*',
        r'(?i)token\s*[=:].*',
        r'(?i)credential\s*[=:].*',
        r'(?i)api[_\-]?key\s*[=:].*',
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace sensitive patterns
        for pattern in sensitive_patterns:
            content = re.sub(pattern, r'\g<0>  # [REDACTED]', content, flags=re.MULTILINE)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Cleaned: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Files to clean
    extensions = ['.py', '.json', '.yaml', '.yml', '.env', '.sh']
    exclude_dirs = ['venv', 'env', '.git', 'node_modules', '__pycache__']
    
    for root, dirs, files in os.walk('.'):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = Path(root) / file
                clean_file(file_path)

if __name__ == "__main__":
    print("Cleaning sensitive data...")
    main()
    print("Done cleaning. Please review changes before committing.")
