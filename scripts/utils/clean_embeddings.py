
import json
import sys
from pathlib import Path
import shutil

def remove_embeddings(obj):
    """Recursively remove 'embedding' keys from a dictionary or list."""
    if isinstance(obj, dict):
        return {
            k: remove_embeddings(v) 
            for k, v in obj.items() 
            if k != "embedding"
        }
    elif isinstance(obj, list):
        return [remove_embeddings(item) for item in obj]
    else:
        return obj

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_embeddings.py <json_file>")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
        
    # Backup original
    backup_path = file_path.with_name(file_path.stem + "_full" + file_path.suffix)
    shutil.copy(file_path, backup_path)
    print(f"✓ Backed up full data to {backup_path.name}")
    
    # Read and clean
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    cleaned_data = remove_embeddings(data)
    
    # Save cleaned
    with open(file_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
        
    print(f"✓ Removed embeddings from {file_path.name}")

if __name__ == "__main__":
    main()
