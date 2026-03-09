#!/usr/bin/env bash
# This script links the .env file in the current directory into each subfolder.
# If a subfolder already contains .env (file or symlink), it will be renamed to .env.bak.

set -euo pipefail

ROOT_DIR="$(pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found in $ROOT_DIR"
    exit 1
fi

# Loop through each subdirectory
for dir in */; do
    [[ -d "$dir" ]] || continue  # Skip if not a directory

    target="${dir%.}/.env"

    # If .env or symlink exists, rename it to .env.bak
    if [[ -e "$target" || -L "$target" ]]; then
        echo "Backing up existing $target -> ${target}.bak"
        mv -f "$target" "${target}.bak"
    fi

    # Create symlink
    ln -s "../.env" "$target"
    echo "Created symlink in $dir"
done

echo "Done."
