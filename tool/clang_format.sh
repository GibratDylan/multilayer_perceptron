#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  tools/clang_format.sh [options] [files...]

Runs clang-format using the repo's .clang-format config.

Options:
  --all              Run on all project sources (src/**/*.cpp, include/**/*.hpp).
  --changed          Run on modified files (git diff) limited to src/include.
  --fix              Apply formatting changes in-place (-i).
  --no-werror        Do not treat formatting differences as errors (check mode).
  -h, --help         Show help.

Examples:
  ./tools/clang_format.sh --changed
  ./tools/clang_format.sh --changed --fix
  ./tools/clang_format.sh --all
  ./tools/clang_format.sh src/main.cpp include/NeuronalLayer.hpp
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

collect_all_files() {
  (cd "$ROOT_DIR" && find src include -not -path "*/Eigen/*"\
    -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
    -print)
}

collect_changed_files() {
  if ! need_cmd git; then
    echo "[clang-format] git not available; cannot use --changed" >&2
    return 2
  fi

  (cd "$ROOT_DIR" && git diff --name-only --diff-filter=ACMR | \
    grep -E '^(src|include)/.*\.(cpp|hpp|h)$' || true)
}

MODE=""
DO_FIX=0
DO_WERROR=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODE="all"
      shift
      ;;
    --changed)
      MODE="changed"
      shift
      ;;
    --fix)
      DO_FIX=1
      shift
      ;;
    --no-werror)
      DO_WERROR=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if ! need_cmd clang-format; then
  echo "[clang-format] clang-format not found in PATH." >&2
  echo "[clang-format] Install it (ex: sudo apt-get install clang-format)" >&2
  exit 2
fi

FILES=()
if [[ $# -gt 0 ]]; then
  for f in "$@"; do
    if [[ "$f" == /* ]]; then
      FILES+=("$f")
    else
      FILES+=("$ROOT_DIR/$f")
    fi
  done
else
  case "$MODE" in
    changed)
      mapfile -t rels < <(collect_changed_files)
      if [[ ${#rels[@]} -eq 0 ]]; then
        echo "[clang-format] No changed src/include .cpp/.hpp/.h files" >&2
        exit 0
      fi
      for r in "${rels[@]}"; do
        FILES+=("$ROOT_DIR/$r")
      done
      ;;
    all|"")
      mapfile -t rels < <(collect_all_files)
      for r in "${rels[@]}"; do
        FILES+=("$ROOT_DIR/$r")
      done
      ;;
    *)
      echo "[clang-format] Unknown mode: $MODE" >&2
      exit 2
      ;;
  esac
fi

FORMAT_ARGS=()
if [[ "$DO_FIX" == "1" ]]; then
  FORMAT_ARGS+=("-i")
else
  FORMAT_ARGS+=("--dry-run")
  if [[ "$DO_WERROR" == "1" ]]; then
    FORMAT_ARGS+=("-Werror")
  fi
fi

echo "[clang-format] Running on ${#FILES[@]} file(s)" >&2

EXIT_CODE=0
for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[clang-format] Skip missing: $file" >&2
    continue
  fi
  clang-format "${FORMAT_ARGS[@]}" "$file" || EXIT_CODE=$?
done

exit "$EXIT_CODE"
