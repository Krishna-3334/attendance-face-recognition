#!/usr/bin/env bash
# ./deploy/cloudrun/deploy.sh <gcp-project-id> [region] [service-name]
set -euo pipefail

PROJECT="${1:?usage: deploy.sh <gcp-project-id> [region] [service-name]}"
REGION="${2:-asia-south1}"
SERVICE="${3:-bioaccess-api}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "→ staging build context"
cp -r "$REPO_ROOT/backend/app" "$WORK/"
cp -r "$REPO_ROOT/backend/scripts" "$WORK/"
cp "$REPO_ROOT/backend/requirements.txt" "$WORK/"
cp "$REPO_ROOT/deploy/cloudrun/Dockerfile" "$WORK/"
find "$WORK" -name '__pycache__' -type d -prune -exec rm -rf {} +

echo "→ deploying $SERVICE to $REGION in project $PROJECT"
echo "  (first build takes ~10 minutes: it bakes in ~300 MB of model weights)"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --source "$WORK" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 4 \
  --timeout 120 \
  --min-instances 0 \
  --max-instances 1 \
  --cpu-boost

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
        --region "$REGION" --format 'value(status.url)')"

echo
echo "✓ deployed: $URL"
echo "  API docs:  $URL/docs"
