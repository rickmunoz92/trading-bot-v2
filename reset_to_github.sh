#!/usr/bin/env bash
set -euo pipefail

# reset_to_github.sh
# This script will force your local repo folder to exactly match GitHub (origin/main).
# ⚠️ WARNING: This will delete ALL local changes, untracked files, and ignored files!

echo "➡ Fetching latest from GitHub..."
git fetch origin

echo "➡ Resetting local branch to origin/main..."
git reset --hard origin/main

echo "➡ Cleaning untracked and ignored files..."
git clean -fdx

echo "✔ Done. Your local folder now exactly matches GitHub (origin/main)."
