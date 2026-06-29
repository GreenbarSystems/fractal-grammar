#!/usr/bin/env bash
# fg-sync installer
# Usage: curl -fsSL https://raw.githubusercontent.com/ryandmoore1976/fractal-grammar/main/fg-sync/install.sh | bash
#
# Installs fg-sync and its dependencies into a local virtualenv at ~/.fg-sync/venv
# and symlinks the `fg-sync` binary to /usr/local/bin (or ~/.local/bin).

set -euo pipefail

FG_HOME="${HOME}/.fg-sync"
VENV="${FG_HOME}/venv"
REPO="https://github.com/ryandmoore1976/fractal-grammar"
VERSION="0.1.0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[fg-sync]${NC} $*"; }
warn() { echo -e "${YELLOW}[fg-sync]${NC} $*"; }
err()  { echo -e "${RED}[fg-sync]${NC} $*" >&2; exit 1; }

# ---- Check Python ----
if ! command -v python3 &>/dev/null; then
    err "Python 3.11+ is required. Install from https://python.org"
fi

PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
    err "Python 3.11+ required (found $(python3 --version))"
fi

log "Python $(python3 --version) found"

# ---- Create fg-sync home ----
mkdir -p "${FG_HOME}/logs"
log "fg-sync home: ${FG_HOME}"

# ---- Create virtualenv ----
if [[ ! -d "${VENV}" ]]; then
    log "Creating virtualenv at ${VENV}..."
    python3 -m venv "${VENV}"
fi

# ---- Install fg-sync ----
log "Installing fg-sync ${VERSION}..."
"${VENV}/bin/pip" install --quiet --upgrade pip

# Install from PyPI (if published) or from local source
if pip show fg-sync &>/dev/null 2>&1; then
    "${VENV}/bin/pip" install --quiet "fg-sync[pipeline]==${VERSION}"
else
    # Fallback: install from GitHub release tarball
    warn "fg-sync not on PyPI yet — installing from GitHub..."
    "${VENV}/bin/pip" install --quiet \
        "git+${REPO}.git#subdirectory=fg-sync"
fi

# Also install fractal-grammar (companion library)
log "Installing fractal-grammar pipeline dependencies..."
"${VENV}/bin/pip" install --quiet \
    scikit-learn hdbscan datasketch numpy rich click httpx uvicorn starlette aiofiles apscheduler

# ---- Symlink binary ----
BINARY="${VENV}/bin/fg-sync"
if [[ ! -f "${BINARY}" ]]; then
    # Create a wrapper script if the binary isn't in venv
    cat > "${BINARY}" <<'EOF'
#!/usr/bin/env bash
exec "$(dirname "$0")/../venv/bin/python" -m fg_sync.cli "$@"
EOF
    chmod +x "${BINARY}"
fi

# Try /usr/local/bin first, fall back to ~/.local/bin
LINK_TARGET=""
if [[ -w "/usr/local/bin" ]]; then
    LINK_TARGET="/usr/local/bin/fg-sync"
else
    mkdir -p "${HOME}/.local/bin"
    LINK_TARGET="${HOME}/.local/bin/fg-sync"
    # Ensure ~/.local/bin is on PATH
    if [[ ":${PATH}:" != *":${HOME}/.local/bin:"* ]]; then
        warn "Add ~/.local/bin to your PATH:"
        warn '  echo '"'"'export PATH="$HOME/.local/bin:$PATH"'"'"' >> ~/.bashrc'
        warn "  source ~/.bashrc"
    fi
fi

ln -sf "${BINARY}" "${LINK_TARGET}"
log "Linked: ${LINK_TARGET} → ${BINARY}"

# ---- Initialize config ----
CONFIG_PATH="${FG_HOME}/fg-sync.toml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    log "Generating default config at ${CONFIG_PATH}..."
    "${VENV}/bin/fg-sync" init 2>/dev/null || true
fi

# ---- Done ----
echo ""
log "fg-sync ${VERSION} installed successfully!"
echo ""
echo "  Quick start:"
echo "    fg-sync run              # start proxy + scheduler"
echo "    fg-sync sync             # one-shot pipeline run"
echo "    fg-sync status           # check state"
echo "    fg-sync metrics compare  # view M1-M5 metrics"
echo ""
echo "  Config: ${CONFIG_PATH}"
echo "  Point your Ollama client at: http://localhost:11435"
echo ""
echo "  Docs: ${REPO}"
