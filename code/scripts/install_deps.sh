#!/bin/bash

# ==========================================================
# Intelligent Dependency Installer for TSF
# Description:
#   Installs packages from requirements.txt only if they 
#   are missing from the current environment. 
#   If a package is already installed, it skips installation 
#   to honor the existing environment state.
#
# How to use:
#   bash scripts/install_deps.sh
#
# ==========================================================

REQ_FILE="requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "[ERROR] $REQ_FILE not found!"
    exit 1
fi

echo ">>> Checking project dependencies from $REQ_FILE..."

# Loop through each line in requirements.txt
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # Extract package name (everything before ==)
    package=$(echo "$line" | cut -d'=' -f1)
    
    # Handle special import names
    import_name=$(echo "$package" | tr '-' '_')
    [[ "$package" == "scikit-learn" ]] && import_name="sklearn"

    # Check if package is installed in current python environment
    if python -c "import $import_name" > /dev/null 2>&1; then
        current_ver=$(python -c "import $import_name; print($import_name.__version__)" 2>/dev/null || echo "unknown")
        echo "[PASS] $package is already installed (Detected version: $current_ver). Skipping."
    else
        echo "[INSTALL] $package not found. Installing $line..."
        pip install "$line"
    fi
done < "$REQ_FILE"

echo ">>> Dependency check complete."
