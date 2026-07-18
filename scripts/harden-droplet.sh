#!/usr/bin/env bash
# Lock down the GPU droplet's local firewall as a second layer
# behind the DO Cloud Firewall. Idempotent.
#
# Run as root on the droplet:
#   sudo bash scripts/harden-droplet.sh
#   sudo bash scripts/harden-droplet.sh 1.2.3.4

set -euo pipefail

# Get the IP you want to allow (passed as arg, or detect from SSH session)
ALLOWED_IP="${1:-${SUDO_USER_IP:-}}"
if [ -z "$ALLOWED_IP" ]; then
    # Last resort: take the IP that just SSH'd in
    ALLOWED_IP=$(who --ips am i 2>/dev/null | awk '{print $NF}' | tr -d '()')
fi
if [ -z "$ALLOWED_IP" ] || [ "$ALLOWED_IP" = "-" ]; then
    echo "Could not auto-detect your IP. Pass it explicitly:"
    echo "  sudo bash scripts/harden-droplet.sh 1.2.3.4"
    exit 1
fi

echo "==> Allowing SSH + vLLM (port 8000) from $ALLOWED_IP only"

# Reset to a known-clean state
iptables -F
iptables -X
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Loopback always allowed
iptables -A INPUT -i lo -j ACCEPT

# Established/related connections (keeps your existing SSH session alive)
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# SSH from your IP only
iptables -A INPUT -p tcp --dport 22 -s "$ALLOWED_IP" -j ACCEPT

# vLLM API from your IP only
iptables -A INPUT -p tcp --dport 8000 -s "$ALLOWED_IP" -j ACCEPT

# Persist (Ubuntu 24.04 uses netfilter-persistent; install if missing)
if ! command -v netfilter-persistent >/dev/null 2>&1; then
    apt-get install -y iptables-persistent >/dev/null 2>&1
fi
netfilter-persistent save

echo ""
echo "==> Active INPUT rules:"
iptables -L INPUT -n --line-numbers

echo ""
echo "==> Verify from your laptop:"
echo "    curl http://<droplet-ip>:8000/health        # should work"
echo "    curl http://<droplet-ip>:8000/v1/models     # should work"
echo ""
echo "To remove these rules later:"
echo "    iptables -F && netfilter-persistent save"