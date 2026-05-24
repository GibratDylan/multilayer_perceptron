#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="$ROOT_DIR/include"
EIGEN_DIR="$INCLUDE_DIR/Eigen"
EIGEN_VERSION="3.4.0"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz"
TEMP_FILE="/tmp/eigen-$EIGEN_VERSION.tar.gz"

echo "[eigen-download] Eigen 3.4.0 setup"
echo "[eigen-download] Root directory: $ROOT_DIR"

# Check if Eigen already exists
if [ -d "$EIGEN_DIR" ]; then
  echo "[eigen-download] Eigen is already present at $EIGEN_DIR"
  exit 0
fi

echo "[eigen-download] Eigen not found. Downloading Eigen $EIGEN_VERSION..."

# Check for required tools
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
  echo "[eigen-download] ERROR: Neither wget nor curl is available" >&2
  exit 1
fi

# Download Eigen
if command -v wget &> /dev/null; then
  wget -q -O "$TEMP_FILE" "$EIGEN_URL"
elif command -v curl &> /dev/null; then
  curl -fsSL -o "$TEMP_FILE" "$EIGEN_URL"
fi

if [ ! -f "$TEMP_FILE" ]; then
  echo "[eigen-download] ERROR: Failed to download Eigen" >&2
  exit 1
fi

echo "[eigen-download] Download complete. Extracting..."

# Extract the tar.gz file
# The archive has a structure like eigen-3.4.0/Eigen, we need just the Eigen directory
tar -xzf "$TEMP_FILE" -C "$INCLUDE_DIR" --strip-components=1 "eigen-$EIGEN_VERSION/Eigen"

# Verify extraction
if [ ! -d "$EIGEN_DIR" ]; then
  echo "[eigen-download] ERROR: Failed to extract Eigen" >&2
  rm -f "$TEMP_FILE"
  exit 1
fi

# Clean up
rm -f "$TEMP_FILE"

echo "[eigen-download] Eigen $EIGEN_VERSION successfully installed to $EIGEN_DIR"
