#!/usr/bin/env bash
set -euo pipefail
#
# reset_to_github.sh (safe)
# Force local repo to match origin/main, but PRESERVE local env + venv + copilot memory.
#
# Keeps: .env, .env.* , *.env , .venv/ , copilot_memory*.json
# Everything else (including ignored files) will be cleaned.

echo "➡ Fetching latest from GitHub..."
git fetch origin

echo "➡ Resetting local branch to origin/main..."
git reset --hard origin/main

echo "➡ Cleaning untracked/ignored files, preserving env/venv/memory..."
# -x : also remove ignored files
# -e : exclude patterns to KEEP (repeatable)
git clean -fdx \  -e .env \  -e .env.* \  -e "*.env" \  -e .venv/ \  -e "copilot_memory*.json"

echo "✔ Done. Local folder matches origin/main (env/venv preserved)."
