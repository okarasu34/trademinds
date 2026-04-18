#!/bin/bash
# ─────────────────────────────────────────────────────────
# TradeMinds — Sunucuda Çalıştırma Scripti
# Kullanım: bash run.sh
# ─────────────────────────────────────────────────────────

set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$APP_DIR/backend"
FRONTEND_DIR="$APP_DIR/frontend"
VENV="$BACKEND_DIR/venv"
LOG_DIR="/var/log/trademinds"

echo "══════════════════════════════════════════"
echo "  TradeMinds — Başlatılıyor"
echo "  Dizin: $APP_DIR"
echo "══════════════════════════════════════════"

# ─── 1. .env kontrolü ───
if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo "❌ .env dosyası bulunamadı!"
  echo "   cp $BACKEND_DIR/.env.example $BACKEND_DIR/.env"
  echo "   Sonra ANTHROPIC_API_KEY ve diğer değerleri doldur."
  exit 1
fi

# ─── 2. Python venv ───
if [ ! -d "$VENV" ]; then
  echo "[1/5] Python sanal ortam oluşturuluyor..."
  python3 -m venv "$VENV"
fi

echo "[2/5] Bağımlılıklar yükleniyor..."
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$BACKEND_DIR/requirements.txt" -q

# ─── 3. Log dizini ───
mkdir -p "$LOG_DIR"

# ─── 4. DB migration ───
echo "[3/5] Veritabanı migration çalıştırılıyor..."
cd "$BACKEND_DIR"
"$VENV/bin/alembic" upgrade head

# ─── 5. Frontend build ───
if [ -d "$FRONTEND_DIR" ] && [ -f "$FRONTEND_DIR/package.json" ]; then
  echo "[4/5] Frontend build ediliyor..."
  cd "$FRONTEND_DIR"
  npm install --silent
  VITE_API_URL=/api/v1 npm run build
else
  echo "[4/5] Frontend dizini bulunamadı, atlanıyor..."
fi

# ─── 6. Backend başlat ───
echo "[5/5] Backend başlatılıyor..."
echo ""
echo "  API:      http://localhost:8000"
echo "  Docs:     http://localhost:8000/api/docs  (DEBUG=true ise)"
echo "  Loglar:   $LOG_DIR/backend.log"
echo ""
echo "  Durdurmak için: Ctrl+C"
echo "══════════════════════════════════════════"

cd "$BACKEND_DIR"
"$VENV/bin/uvicorn" main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --loop asyncio \
  --log-level info
