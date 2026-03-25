#!/bin/bash
# ─────────────────────────────────────────────────────────
# TradeMinds — VPS Auto Setup Script
# Tested on: Ubuntu 22.04 LTS (Vultr High Frequency)
# Run as root: bash setup.sh yourdomain.com
# ─────────────────────────────────────────────────────────

set -e  # Exit on any error

DOMAIN=${1:-"yourdomain.com"}
APP_USER="trademinds"
APP_DIR="/home/$APP_USER/trademinds"
DB_PASS=$(openssl rand -hex 16)
REDIS_PASS=$(openssl rand -hex 16)
SECRET_KEY=$(openssl rand -hex 64)
JWT_KEY=$(openssl rand -hex 64)
ENCRYPT_KEY=$(openssl rand -hex 32)

echo "══════════════════════════════════════════"
echo "  TradeMinds VPS Setup"
echo "  Domain: $DOMAIN"
echo "══════════════════════════════════════════"

# ─── 1. System update ───
echo "[1/12] Updating system..."
apt update -qq && apt upgrade -y -qq
apt install -y -qq \
  python3.11 python3.11-pip python3.11-venv \
  nodejs npm nginx \
  postgresql-14 redis-server \
  git supervisor ufw certbot python3-certbot-nginx \
  build-essential libpq-dev \
  htop curl wget vim unzip

# ─── 2. Firewall ───
echo "[2/12] Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP"
ufw allow 443/tcp comment "HTTPS"
ufw --force enable

# ─── 3. App user ───
echo "[3/12] Creating app user..."
id -u $APP_USER &>/dev/null || useradd -m -s /bin/bash $APP_USER

# ─── 4. PostgreSQL ───
echo "[4/12] Setting up PostgreSQL..."
systemctl start postgresql
systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER $APP_USER WITH PASSWORD '$DB_PASS';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE trademinds OWNER $APP_USER;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trademinds TO $APP_USER;" 2>/dev/null || true

# ─── 5. Redis ───
echo "[5/12] Setting up Redis..."
sed -i "s/# requirepass foobared/requirepass $REDIS_PASS/" /etc/redis/redis.conf
sed -i "s/bind 127.0.0.1/bind 127.0.0.1/" /etc/redis/redis.conf
systemctl restart redis
systemctl enable redis

# ─── 6. Clone / copy project ───
echo "[6/12] Setting up application..."
mkdir -p $APP_DIR
# If you have a git repo:
# git clone https://github.com/yourrepo/trademinds.git $APP_DIR
# For now, copy from current directory:
cp -r . $APP_DIR/ 2>/dev/null || true
chown -R $APP_USER:$APP_USER /home/$APP_USER

# ─── 7. Python virtual env ───
echo "[7/12] Installing Python dependencies..."
sudo -u $APP_USER bash -c "
  python3.11 -m venv $APP_DIR/backend/venv
  $APP_DIR/backend/venv/bin/pip install --upgrade pip -q
  $APP_DIR/backend/venv/bin/pip install -r $APP_DIR/backend/requirements.txt -q
"

# ─── 8. Environment file ───
echo "[8/12] Creating environment configuration..."
cat > $APP_DIR/backend/.env << EOF
APP_NAME=TradeMinds
DEBUG=false
ALLOWED_ORIGINS=["https://$DOMAIN"]

DATABASE_URL=postgresql+asyncpg://$APP_USER:$DB_PASS@localhost:5432/trademinds
REDIS_URL=redis://:$REDIS_PASS@localhost:6379/0

SECRET_KEY=$SECRET_KEY
JWT_SECRET_KEY=$JWT_KEY
BROKER_ENCRYPTION_KEY=$ENCRYPT_KEY

# ── Fill these in manually ──
ANTHROPIC_API_KEY=sk-ant-FILL_ME
MYFXBOOK_EMAIL=FILL_ME
MYFXBOOK_PASSWORD=FILL_ME
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
NOTIFICATION_EMAIL=

BASE_CURRENCY=USD
EOF
chown $APP_USER:$APP_USER $APP_DIR/backend/.env
chmod 600 $APP_DIR/backend/.env

# ─── 9. Run DB migrations ───
echo "[9/12] Running database migrations..."
sudo -u $APP_USER bash -c "
  cd $APP_DIR/backend
  source venv/bin/activate
  alembic upgrade head
"

# ─── 10. Frontend build ───
echo "[10/12] Building frontend..."
sudo -u $APP_USER bash -c "
  cd $APP_DIR/frontend
  npm install --silent
  VITE_API_URL=/api/v1 npm run build
"

# ─── 11. Supervisor ───
echo "[11/12] Configuring Supervisor..."
mkdir -p /var/log/trademinds
chown $APP_USER:$APP_USER /var/log/trademinds
cp $APP_DIR/deploy/supervisor.conf /etc/supervisor/conf.d/trademinds.conf
supervisorctl reread
supervisorctl update
supervisorctl start trademinds-backend

# ─── 12. Nginx + SSL ───
echo "[12/12] Configuring Nginx & SSL..."
cp $APP_DIR/deploy/nginx.conf /etc/nginx/sites-available/trademinds
sed -i "s/yourdomain.com/$DOMAIN/g" /etc/nginx/sites-available/trademinds
ln -sf /etc/nginx/sites-available/trademinds /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
systemctl enable nginx

# SSL certificate
certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m "admin@$DOMAIN" || \
  echo "⚠ SSL setup skipped — configure DNS first, then run: certbot --nginx -d $DOMAIN"

# ─── DB Backup cron ───
(crontab -l 2>/dev/null; echo "0 2 * * 0 pg_dump trademinds | gzip > /var/backups/trademinds/weekly_\$(date +\%Y\%m\%d).sql.gz") | crontab -
mkdir -p /var/backups/trademinds

# ─── Summary ───
echo ""
echo "══════════════════════════════════════════"
echo "  ✅ TradeMinds Setup Complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  URL:      https://$DOMAIN"
echo "  API Docs: https://$DOMAIN/api/docs"
echo ""
echo "  ⚠️  IMPORTANT: Edit the .env file and add:"
echo "     $APP_DIR/backend/.env"
echo "     - ANTHROPIC_API_KEY"
echo "     - MYFXBOOK_EMAIL / PASSWORD"
echo "     - SMTP credentials (for email alerts)"
echo "     - TELEGRAM_BOT_TOKEN (optional)"
echo ""
echo "  After editing .env:"
echo "     supervisorctl restart trademinds-backend"
echo ""
echo "  Logs:"
echo "     tail -f /var/log/trademinds/backend.log"
echo ""
echo "  DB Password:    $DB_PASS"
echo "  Redis Password: $REDIS_PASS"
echo "  (Save these somewhere safe!)"
echo "══════════════════════════════════════════"
