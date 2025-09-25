#!/usr/bin/env bash
# Safe push helper: prevents committing .env/.venv and common secrets, then pushes current branch.
# Usage: ./gitpush.sh "Commit message"
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "❌ Please provide a commit message."
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi
COMMIT_MSG="$1"

# Ensure we're inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not a git repository. Run this inside your project folder."
  exit 1
fi

# Ensure .gitignore exists and contains critical entries
touch .gitignore
ensure_ignore() {
  local p="$1"
  grep -qxF "$p" .gitignore 2>/dev/null || echo "$p" >> .gitignore
}
ensure_ignore "# Local env"
ensure_ignore ".env"
ensure_ignore ".env.*"
ensure_ignore "*.env"
ensure_ignore ""
ensure_ignore "# Python cache"
ensure_ignore "__pycache__/"
ensure_ignore "*.pyc"
ensure_ignore ""
ensure_ignore "# Virtualenv"
ensure_ignore ".venv/"
ensure_ignore ""
ensure_ignore "# macOS"
ensure_ignore ".DS_Store"

# If these paths were ever tracked, remove from index (keep files locally)
git rm -r --cached --ignore-unmatch .env .env.* *.env .venv >/dev/null 2>&1 || true

# Stage all changes
git add -A

# Just-in-case: unstage sensitive paths if they slipped in
git reset -q HEAD -- .env .env.* >/dev/null 2>&1 || true
git reset -q HEAD -- '*.env' >/dev/null 2>&1 || true
git reset -q HEAD -- .venv >/dev/null 2>&1 || true
git reset -q HEAD -- '**/.env' '**/.env.*' '**/*.env' '**/.venv/**' >/dev/null 2>&1 || true

# Guard 1: block if .env or .venv is still staged
if git diff --cached --name-only | grep -E '(^|/)\.env($|\.|/)|(^|/)\.venv/|(^|/)[^/]*\.env$' >/dev/null; then
  echo "🚫 Refusing to commit: .env or .venv is staged."
  echo "   Update .gitignore and try again."
  exit 1
fi

# Guard 2: basic secret scan in staged diff (OpenAI-style keys)
if git diff --cached -U0 | grep -E 'sk-[A-Za-z0-9_-]{20,}' >/dev/null; then
  echo "🚫 Detected potential OpenAI API key (sk-...) in staged changes."
  echo "   Remove/redact it before pushing."
  exit 1
fi

# Always include updated .gitignore if it changed
git add .gitignore || true

# Commit only if there are staged changes
if git diff --cached --quiet; then
  echo "ℹ️ Nothing to commit. Pushing existing commits..."
else
  git commit -m "$COMMIT_MSG"
fi

# Determine current branch
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Ensure origin exists
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "❌ No 'origin' remote configured. Add it, e.g.:"
  echo "   git remote add origin https://github.com/<user>/<repo>.git"
  exit 1
fi

# Push, setting upstream if needed
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push origin "$BRANCH"
else
  git push -u origin "$BRANCH"
fi
