#!/bin/bash
# Usage: ./gitpush.sh "Commit message"

if [ -z "$1" ]; then
  echo "❌ Please provide a commit message."
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

COMMIT_MSG="$1"

# Ensure we’re in a git repo
if [ ! -d ".git" ]; then
  echo "❌ Not a git repository. Run this inside your project folder."
  exit 1
fi

# Stage all changes
git add .

# Commit with your message
git commit -m "$COMMIT_MSG"

# Push to main branch
git push origin main
