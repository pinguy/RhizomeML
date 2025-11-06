#!/usr/bin/env python3
"""
Script to fix type inconsistencies in JSONL files.
Normalizes conversation_id and other fields to consistent types.
"""

import json
from pathlib import Path
from typing import Any, Dict
import shutil

def normalize_conversation_ids(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert all conversation_id fields to strings."""
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            if key == 'conversation_id':
                # Convert to string regardless of original type
                normalized[key] = str(value)
            elif isinstance(value, dict):
                normalized[key] = normalize_conversation_ids(value)
            elif isinstance(value, list):
                normalized[key] = [normalize_conversation_ids(item) if isinstance(item, dict) else item 
                                   for item in value]
            else:
                normalized[key] = value
        return normalized
    return data

def fix_jsonl_file(input_file: str, output_file: str = None, backup: bool = True):
    """
    Fix type inconsistencies in a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output file (if None, overwrites input with backup)
        backup: Whether to create a .backup file
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_file}")
        return False
    
    # Determine output path
    if output_file is None:
        output_path = input_path
        if backup:
            backup_path = input_path.with_suffix(input_path.suffix + '.backup')
            shutil.copy2(input_path, backup_path)
            print(f"📦 Created backup: {backup_path}")
    else:
        output_path = Path(output_file)
    
    # Process the file
    fixed_count = 0
    error_count = 0
    total_lines = 0
    
    temp_output = output_path.with_suffix('.tmp')
    
    print(f"🔧 Processing {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(temp_output, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line)
                    
                    # Check if normalization is needed
                    original_json = json.dumps(data, sort_keys=True)
                    normalized_data = normalize_conversation_ids(data)
                    normalized_json = json.dumps(normalized_data, sort_keys=True)
                    
                    if original_json != normalized_json:
                        fixed_count += 1
                        if fixed_count <= 5:  # Show first 5 fixes
                            print(f"  ✏️  Fixed line {line_num}")
                    
                    # Write normalized data
                    outfile.write(json.dumps(normalized_data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"  ⚠️  Skipping malformed JSON at line {line_num}: {e}")
                except Exception as e:
                    error_count += 1
                    print(f"  ⚠️  Error at line {line_num}: {e}")
        
        # Replace original with fixed version
        if temp_output.exists():
            temp_output.replace(output_path)
            print(f"\n✅ Processing complete!")
            print(f"   📊 Total lines: {total_lines}")
            print(f"   🔧 Fixed lines: {fixed_count}")
            print(f"   ⚠️  Errors: {error_count}")
            print(f"   💾 Output: {output_path}")
            return True
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if temp_output.exists():
            temp_output.unlink()
        return False

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix type inconsistencies in JSONL files (especially conversation_id)"
    )
    parser.add_argument("input_file", help="Input JSONL file to fix")
    parser.add_argument("-o", "--output", help="Output file (default: overwrite input)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup file")
    
    args = parser.parse_args()
    
    success = fix_jsonl_file(
        args.input_file,
        args.output,
        backup=not args.no_backup
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
