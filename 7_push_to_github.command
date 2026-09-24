#!/bin/bash
# Step 7: push the project to GitHub (code, reports, .dvc pointer files, and tags; data stays in DVC)
cd "$(dirname "$0")" || exit 1
echo "Create an EMPTY repository on github.com first (no README), then paste its URL here."
echo "Example: https://github.com/your-name/ai4i-failure-warning.git"
read -r -p "Repository URL: " URL
[ -z "$URL" ] && { echo "No URL entered."; read -r -p "Press Return to close..."; exit 1; }
git add -A && git commit -q -m "docs + demo scripts" 2>/dev/null
if git remote | grep -q '^origin$'; then git remote set-url origin "$URL"; else git remote add origin "$URL"; fi
echo "When Git asks for a password, paste a GitHub personal access token (not your account password)."
git push -u origin main --tags && echo "=== PUSHED TO GITHUB ===" || echo "=== PUSH FAILED - see the message above ==="
read -r -p "Press Return to close this window..."
