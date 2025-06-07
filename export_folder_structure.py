import os
from pathlib import Path

def load_gitignore(root_dir):
    gitignore_path = os.path.join(root_dir, '.gitignore')
    ignored = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle directory patterns
                if line.endswith('/'):
                    ignored.add(line.rstrip('/'))
                else:
                    ignored.add(line)
    return ignored

def write_folder_structure(root_dir, output_file):
    root_dir = os.path.abspath(root_dir)
    ignored = load_gitignore(root_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        for dirpath, dirnames, _ in os.walk(root_dir):
            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                rel_path = ''
            parts = rel_path.split(os.sep) if rel_path else []
            # Skip if any part of the path is in .gitignore
            if any(part in ignored for part in parts):
                dirnames[:] = []  # Don't recurse further
                continue
            indent = '    ' * len(parts)
            f.write(f'{indent}{os.path.basename(dirpath) or os.path.basename(root_dir)}/\n')
            # Remove ignored subdirectories from walking
            dirnames[:] = [d for d in dirnames if d not in ignored]

# Example usage:

root_directory = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao'
output_txt_file = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/folder_structure3.txt'

write_folder_structure(root_directory, output_txt_file)