#!/usr/bin/env bash
# scripts/setup_terminal_colors.sh
#
# Apply a professional color scheme to the SSH session on the GPU droplet.
# Usage: ssh root@<ip> 'bash -s' < scripts/setup_terminal_colors.sh

set -euo pipefail

MARKER="# === retail-concierge demo colors (do not edit below) ==="

COLOR_BLOCK=$(cat <<'BLOCK_EOF'
# === retail-concierge demo colors (do not edit below) ===
export TERM=xterm-256color
export CLICOLOR=1
export LS_COLORS='di=1;34:ln=1;36:ex=1;32:*.py=1;33:*.sh=1;32:*.json=1;33:*.csv=1;35:*.log=1;31'
alias ls='ls --color=auto'
alias ll='ls -alF --color=auto'
alias grep='grep --color=auto'
PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '
# === end retail-concierge demo colors ===
BLOCK_EOF
)

if ! grep -qF "$MARKER" ~/.bashrc 2>/dev/null; then
    printf '\n%s\n' "$COLOR_BLOCK" >> ~/.bashrc
    echo "[colors] appended color block to ~/.bashrc"
else
    echo "[colors] ~/.bashrc already has the color block (no change)"
fi

eval "$COLOR_BLOCK"
echo "[colors] applied to current session"
