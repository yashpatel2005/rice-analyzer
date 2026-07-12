#!/bin/bash
# ============================================================================
# Deployment script for dnw-testing server (100.99.188.73)
# Deploys Rice Analyzer backend with Cloudflare Tunnel
# ============================================================================

set -e

PROJECT_DIR="$HOME/Rice_Analyzer"
SERVICE_NAME="rice-analyzer"
TUNNEL_NAME="rice-analyzer-backend"
DOMAIN="api.yash-patel.in"

echo "============================================================"
echo "  Rice Analyzer Backend Deployment on dnw-testing"
echo "============================================================"
echo ""

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev gcc git curl

# 2. Install cloudflared
echo "[2/6] Installing cloudflared..."
if ! command -v cloudflared &> /dev/null; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
    sudo chmod +x /usr/local/bin/cloudflared
    echo "cloudflared installed"
else
    echo "cloudflared already installed"
fi

# 3. Clone or update project
echo "[3/6] Setting up project..."
if [ ! -d "$PROJECT_DIR" ]; then
    git clone <REPO_URL> "$PROJECT_DIR"
else
    cd "$PROJECT_DIR" && git pull
fi
cd "$PROJECT_DIR"

# 4. Python virtual environment
echo "[4/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

# 5. Create systemd service for Flask/Gunicorn
echo "[5/6] Creating systemd service..."
sudo tee /etc/systemd/system/rice-analyzer.service > /dev/null << EOF
[Unit]
Description=Rice Analyzer Flask Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment="FLASK_HOST=0.0.0.0"
Environment="FLASK_PORT=5050"
Environment="FLASK_DEBUG=false"
ExecStart=$PROJECT_DIR/venv/bin/gunicorn --bind 0.0.0.0:5050 --workers 2 --timeout 120 "app:app"
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for cloudflared tunnel
sudo tee /etc/systemd/system/rice-analyzer-tunnel.service > /dev/null << EOF
[Unit]
Description=Cloudflare Tunnel for Rice Analyzer
After=network.target rice-analyzer.service

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/cloudflared tunnel run $TUNNEL_NAME
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 6. Enable and start services
echo "[6/6] Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable --now rice-analyzer
sudo systemctl enable --now rice-analyzer-tunnel

echo ""
echo "============================================================"
echo "  Deployment Complete!"
echo "============================================================"
echo ""
echo "Backend should be running at: https://$DOMAIN"
echo ""
echo "Check status:"
echo "  sudo systemctl status rice-analyzer"
echo "  sudo systemctl status rice-analyzer-tunnel"
echo ""
echo "View logs:"
echo "  sudo journalctl -u rice-analyzer -f"
echo "  sudo journalctl -u rice-analyzer-tunnel -f"
echo ""
echo "If tunnel not created yet, run:"
echo "  cloudflared tunnel login"
echo "  cloudflared tunnel create $TUNNEL_NAME"
echo "  cloudflared tunnel route dns $TUNNEL_NAME $DOMAIN"
