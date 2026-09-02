import os

def print_tree(root_dir, prefix="", excluded=None):
    if excluded is None:
        excluded = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules"
        }

    items = sorted(
        item for item in os.listdir(root_dir)
        if item not in excluded
    )

    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = i == len(items) - 1

        print(prefix + ("└── " if is_last else "├── ") + item)

        if os.path.isdir(path):
            print_tree(
                path,
                prefix + ("    " if is_last else "│   "),
                excluded
            )

print_tree(".")