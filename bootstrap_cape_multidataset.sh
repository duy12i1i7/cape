#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/duy12i1i7/cape.git"
REPO_DIR="cape"
BRANCH=""
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --all --prune
  if [[ -n "$BRANCH" ]]; then
    git -C "$REPO_DIR" checkout "$BRANCH"
  fi
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
  if [[ -n "$BRANCH" ]]; then
    git -C "$REPO_DIR" checkout "$BRANCH"
  fi
fi

exec bash "$REPO_DIR/run_cape_multidataset.sh" "${RUN_ARGS[@]}"
