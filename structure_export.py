import os
import sys

# --- Fix Windows Unicode output ---
sys.stdout.reconfigure(encoding='utf-8')

# --- Ignore patterns ---
IGNORE_DIRS = {'__pycache__'}
IGNORE_EXTS = {'.pyc', '.log', '.jpg', '.gitkeep'}

def print_tree(root, indent="", file=None):
    """Recursively print/write project structure, ignoring unwanted files/folders."""
    try:
        items = [i for i in sorted(os.listdir(root)) if i not in IGNORE_DIRS]
    except PermissionError:
        return

    for i, item in enumerate(items):
        # Skip all dotfiles/folders
        if item.startswith('.'):
            continue

        path = os.path.join(root, item)
        connector = "└── " if i == len(items) - 1 else "├── "

        # Skip ignored files by extension
        if os.path.isfile(path) and any(item.endswith(ext) for ext in IGNORE_EXTS):
            continue

        line = indent + connector + (item + "/" if os.path.isdir(path) else item)

        # Print to console
        print(line)
        # Write to file
        if file:
            print(line, file=file)

        # Recurse into directories
        if os.path.isdir(path):
            print_tree(path, indent + ("    " if i == len(items) - 1 else "│   "), file=file)

if __name__ == "__main__":
    with open("project_structure.txt", "w", encoding="utf-8") as f:
        print_tree(".", file=f)
