#!/usr/bin/env bash
# fg_gate build script
# ====================
# Compiles the Rust/PyO3 extension and installs it into the current
# Python environment.  Run from the fg_gate_rs/ directory.
#
# Usage:
#   ./build.sh              # release build + install
#   ./build.sh --dev        # debug build (faster compile, slower runtime)
#   ./build.sh --test-only  # run Rust + Python tests without rebuilding
#   ./build.sh --bench      # build + run latency benchmark
#   ./build.sh --clean      # remove build artifacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source Rust if installed via rustup
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

# Verify dependencies
check_dep() {
    command -v "$1" &>/dev/null || { echo "ERROR: $1 not found. Install with: $2"; exit 1; }
}
check_dep rustc  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
check_dep cargo  "see above"
check_dep python3 "system package manager"
check_dep maturin "pip install maturin"

echo "fg_gate build"
echo "  rustc:   $(rustc --version)"
echo "  maturin: $(maturin --version)"
echo "  python:  $(python3 --version)"
echo ""

MODE="${1:-}"

if [ "$MODE" == "--clean" ]; then
    echo "Cleaning build artifacts..."
    cargo clean
    rm -f target/wheels/*.whl
    echo "Done."
    exit 0
fi

if [ "$MODE" == "--test-only" ]; then
    echo "[1/2] Rust unit tests..."
    cargo test
    echo ""
    echo "[2/2] Python integration tests..."
    python3 tests/test_integration.py
    exit $?
fi

if [ "$MODE" == "--dev" ]; then
    echo "[1/2] Building (dev mode)..."
    maturin build
    WHEEL=$(ls target/wheels/*.whl | head -1)
else
    echo "[1/2] Building (release mode)..."
    maturin build --release
    WHEEL=$(ls target/wheels/*.whl | sort -t- -k4 -r | head -1)
fi

echo ""
echo "[2/2] Installing wheel: $WHEEL"
pip install "$WHEEL" --force-reinstall -q

echo ""
echo "Installation complete."
python3 -c "from fg_gate._fg_gate import fg_version; print('  Loaded:', fg_version())"

if [ "$MODE" == "--bench" ]; then
    echo ""
    echo "[3/3] Running latency benchmark..."
    python3 tests/test_integration.py
fi

echo ""
echo "Done. Run tests with:  ./build.sh --test-only"
echo "Run benchmark with:    ./build.sh --bench"
