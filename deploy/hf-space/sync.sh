#!/usr/bin/env bash
# Push the backend to a Hugging Face Space.
#   1. Create a Space at https://huggingface.co/new-space  (SDK: Docker, CPU basic)
#   2. ./deploy/hf-space/sync.sh <your-hf-username> <space-name>
set -euo pipefail

USER_NAME="${1:?usage: sync.sh <hf-username> <space-name>}"
SPACE_NAME="${2:?usage: sync.sh <hf-username> <space-name>}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "→ cloning Space huggingface.co/spaces/$USER_NAME/$SPACE_NAME"
git clone "https://huggingface.co/spaces/$USER_NAME/$SPACE_NAME" "$WORK/space"

cd "$WORK/space"
rm -rf app scripts eval requirements.txt Dockerfile README.md
cp -r "$REPO_ROOT/backend/app" .
cp -r "$REPO_ROOT/backend/scripts" .
cp -r "$REPO_ROOT/backend/eval" .
cp "$REPO_ROOT/backend/requirements.txt" .
cp "$REPO_ROOT/deploy/hf-space/Dockerfile" .
cp "$REPO_ROOT/deploy/hf-space/README.md" .
find . -name '__pycache__' -type d -prune -exec rm -rf {} +

git add -A
if git diff --cached --quiet; then echo "→ nothing changed"; exit 0; fi
git commit -m "deploy: sync backend from attendance-face-recognition"
git push

echo
echo "✓ pushed. Build: https://huggingface.co/spaces/$USER_NAME/$SPACE_NAME"
echo "  API base:    https://$USER_NAME-$SPACE_NAME.hf.space"
