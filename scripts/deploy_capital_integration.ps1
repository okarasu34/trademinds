# Deploy Capital.com integration to server (PowerShell)
# Usage: .\scripts\deploy_capital_integration.ps1

$SERVER = "root@your-server-ip"
$REMOTE_PATH = "/root/trademinds"

Write-Host "🚀 Deploying Capital.com integration to server..." -ForegroundColor Green

# Copy Capital.com adapter
Write-Host "📦 Copying capital_adapter.py..." -ForegroundColor Cyan
scp backend/brokers/capital_adapter.py "${SERVER}:${REMOTE_PATH}/backend/brokers/"

# Copy updated base adapter
Write-Host "📦 Copying base_adapter.py..." -ForegroundColor Cyan
scp backend/brokers/base_adapter.py "${SERVER}:${REMOTE_PATH}/backend/brokers/"

# Copy updated trading bot
Write-Host "📦 Copying trading_bot.py..." -ForegroundColor Cyan
scp backend/bot/trading_bot.py "${SERVER}:${REMOTE_PATH}/backend/bot/"

# Copy watchlist update script
Write-Host "📦 Copying update_capital_watchlist.py..." -ForegroundColor Cyan
ssh $SERVER "mkdir -p ${REMOTE_PATH}/scripts"
scp scripts/update_capital_watchlist.py "${SERVER}:${REMOTE_PATH}/scripts/"

# Copy documentation
Write-Host "📦 Copying documentation..." -ForegroundColor Cyan
ssh $SERVER "mkdir -p ${REMOTE_PATH}/docs"
scp docs/CAPITAL_COM_INTEGRATION.md "${SERVER}:${REMOTE_PATH}/docs/"

Write-Host "✅ Files copied successfully!" -ForegroundColor Green

# Restart backend
Write-Host "🔄 Restarting backend service..." -ForegroundColor Yellow
ssh $SERVER "supervisorctl restart trademinds"

Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check backend status
Write-Host "🔍 Checking backend status..." -ForegroundColor Cyan
ssh $SERVER "curl -s http://localhost:8001/health || echo 'Backend not responding yet'"

Write-Host ""
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create 'TradeMinds' watchlist on Capital.com"
Write-Host "2. Add symbols to the watchlist"
Write-Host "3. Run: ssh $SERVER 'cd $REMOTE_PATH && python3 scripts/update_capital_watchlist.py'"
Write-Host "4. Start the bot via API or frontend"
Write-Host ""
Write-Host "Monitor logs: ssh $SERVER 'tail -f /tmp/trademinds.log | grep -i capital'" -ForegroundColor Cyan
