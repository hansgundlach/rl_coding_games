#!/bin/bash

# Simple Claude Code Installation Script
# Fast installation with minimal dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

echo "🚀 Installing Claude Code (minimal setup)..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "Don't run as root. Run as regular user."
fi

# Only install curl if not present
if ! command -v curl &> /dev/null; then
    log "Installing curl..."
    sudo apt update -qq
    sudo apt install -y curl
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [[ $NODE_VERSION -ge 18 ]]; then
        log "Node.js $(node --version) is already installed ✓"
    else
        warn "Node.js version is too old, installing latest..."
        INSTALL_NODE=true
    fi
else
    log "Installing Node.js..."
    INSTALL_NODE=true
fi

# Install Node.js only if needed
if [[ $INSTALL_NODE == true ]]; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - > /dev/null
    sudo apt install -y nodejs
fi

# Configure npm to avoid permission issues
NPM_GLOBAL_DIR="$HOME/.npm-global"
mkdir -p "$NPM_GLOBAL_DIR"
npm config set prefix "$NPM_GLOBAL_DIR"

# Add to PATH if not already there
if ! grep -q "\.npm-global/bin" ~/.bashrc; then
    echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
fi

export PATH="$NPM_GLOBAL_DIR/bin:$PATH"

# Install Claude Code
log "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Verify installation
if command -v claude &> /dev/null; then
    log "✅ Claude Code installed successfully!"
    log "Version: $(claude --version)"
else
    error "❌ Installation failed"
fi

echo ""
echo "🎉 Done! Claude Code is ready to use."
echo ""
echo "To get started:"
echo "1. Restart terminal or run: source ~/.bashrc"
echo "2. Run: claude"
echo "3. Follow authentication prompts"
echo "" 