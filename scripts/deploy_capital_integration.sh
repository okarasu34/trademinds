#!/bin/bash
# Deploy Capital.com integration to server
# Usage: ./scripts/deploy_capital_integration.sh

set -e

SERVER="root@your-server-ip"
REMOTE_PATH="/root/trademinds"

echo "🚀 Deploying Capital.com integration to server..."

# Copy Capital.com adapter
echo "📦 Copying capital_adapter.py..."
scp backend/brokers/capital_adapter.py $SERVER:$REMOTE_PATH/backend/brokers/

# Copy updated base adapter
echo "📦 Copying base_adapter.py..."
scp backend/brokers/base_adapter.py $SERVER:$REMOTE_PATH/backend/brokers/

# Copy updated trading bot
echo "📦 Copying trading_bot.py..."
scp backend/bot/trading_bot.py $SERVER:$REMOTE_PATH/backend/bot/

# Copy watchlist update script
echo "📦 Copying update_capital_watchlist.py..."
ssh $SERVER "mkdir -p $REMOTE_PATH/scripts"
scp scripts/update_capital_watchlist.py $SERVER:$REMOTE_PATH/scripts/

# Copy documentation
echo "📦 Copying documentation..."
ssh $SERVER "mkdir -p $REMOTE_PATH/docs"
scp docs/CAPITAL_COM_INTEGRATION.md $SERVER:$REMOTE_PATH/docs/

echo "✅ Files copied successfully!"

# Restart backend
echo "🔄 Restarting backend service..."
ssh $SERVER "supervisorctl restart trademinds"

echo "⏳ Waiting for backend to start..."
sleep 10

# Check backend status
echo "🔍 Checking backend status..."
ssh $SERVER "curl -s http://localhost:8001/health || echo 'Backend not responding yet'"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Create 'TradeMinds' watchlist on Capital.com"
echo "2. Add symbols to the watchlist"
echo "3. Run: ssh $SERVER 'cd $REMOTE_PATH && python3 scripts/update_capital_watchlist.py'"
echo "4. Start the bot via API or frontend"
echo ""
echo "Monitor logs: ssh $SERVER 'tail -f /tmp/trademinds.log | grep -i capital'"
