#!/usr/bin/env python3
"""Battle-Bot Dashboard - Live markets + Trading activity."""

import asyncio
import json
import os
from datetime import datetime
from aiohttp import web
import aiohttp
import httpx
from dotenv import load_dotenv

load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"


class BattleBotDashboard:
    """Combined dashboard: live markets + trading activity."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._app = None
        self._runner = None
        self._websockets: set[web.WebSocketResponse] = set()
        
        # Market data
        self._markets: dict[str, dict] = {}
        
        # Trading data
        self._positions: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._analyses: list[dict] = []
        self._stats = {
            'bankroll': float(os.getenv('INITIAL_BANKROLL', 1000)),
            'starting_bankroll': float(os.getenv('INITIAL_BANKROLL', 1000)),
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'dry_run': os.getenv('DRY_RUN', 'true').lower() == 'true',
            'running': False,
        }
        
        self._running = False
        
    async def start(self):
        """Start the dashboard server."""
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/markets', self._handle_markets)
        self._app.router.add_get('/api/state', self._handle_state)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, '0.0.0.0', self.port)
        await site.start()
        
        self._running = True
        self._stats['running'] = True
        
        print(f"\n{'='*50}")
        print("BATTLE-BOT DASHBOARD")
        print(f"{'='*50}")
        print(f"\nOpen http://localhost:{self.port} in your browser")
        print("Press Ctrl+C to stop\n")
        
        # Start market refresh loop
        asyncio.create_task(self._refresh_loop())
    
    async def stop(self):
        """Stop the dashboard."""
        self._running = False
        self._stats['running'] = False
        for ws in self._websockets:
            await ws.close()
        if self._runner:
            await self._runner.cleanup()
    
    async def _refresh_loop(self):
        """Fetch markets periodically."""
        while self._running:
            try:
                await self._fetch_markets()
                await self._broadcast({
                    'type': 'update',
                    'markets': list(self._markets.values()),
                    'stats': self._stats,
                    'positions': list(self._positions.values()),
                    'trades': self._trades[-50:],
                })
            except Exception as e:
                print(f"Error: {e}")
            await asyncio.sleep(30)
    
    async def _fetch_markets(self):
        """Fetch live markets from Polymarket."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{GAMMA_API}/markets", params={
                'limit': 100,
                'active': 'true',
                'closed': 'false',
            })
            response.raise_for_status()
            raw_markets = response.json()
        
        markets = {}
        for m in raw_markets:
            try:
                market = self._parse_market(m)
                if market and market['volume_24h'] >= 1000:
                    markets[market['id']] = market
            except:
                pass
        
        sorted_markets = sorted(markets.values(), key=lambda x: x['volume_24h'], reverse=True)
        self._markets = {m['id']: m for m in sorted_markets[:50]}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(self._markets)} markets")
    
    def _parse_market(self, m: dict) -> dict:
        """Parse market data."""
        price = 0.5
        if 'outcomePrices' in m:
            prices = m['outcomePrices']
            if isinstance(prices, list) and len(prices) > 0:
                try:
                    price = float(prices[0])
                except:
                    pass
        
        volume = float(m.get('volume', 0) or m.get('volume24hr', 0) or 0)
        
        end_date = None
        for field in ['endDate', 'end_date']:
            if field in m and m[field]:
                try:
                    end_date = m[field][:10]
                except:
                    pass
        
        question = m.get('question', '').lower()
        if any(w in question for w in ['bitcoin', 'btc', 'eth', 'crypto', 'solana']):
            category = 'crypto'
        elif any(w in question for w in ['trump', 'biden', 'election', 'president']):
            category = 'politics'
        elif any(w in question for w in ['nfl', 'nba', 'mlb', 'super bowl']):
            category = 'sports'
        elif any(w in question for w in ['fed', 'rate', 'inflation', 'gdp']):
            category = 'economics'
        elif any(w in question for w in ['ai', 'openai', 'google', 'apple']):
            category = 'tech'
        else:
            category = 'other'
        
        slug = m.get('slug', '')
        
        return {
            'id': m.get('conditionId', m.get('condition_id', '')),
            'question': m.get('question', 'Unknown'),
            'price': price,
            'price_pct': int(price * 100),
            'volume_24h': volume,
            'volume_display': f"${volume/1000:.1f}k" if volume < 1000000 else f"${volume/1000000:.2f}M",
            'liquidity': float(m.get('liquidity', 0) or 0),
            'end_date': end_date,
            'category': category,
            'url': f"https://polymarket.com/event/{slug}" if slug else "",
            'image': m.get('image', ''),
        }
    
    # Public methods for bot integration
    async def add_position(self, position: dict):
        """Add a new position."""
        self._positions[position['id']] = position
        self._stats['total_trades'] += 1
        
        trade = {**position, 'action': 'ENTRY', 'timestamp': datetime.utcnow().isoformat()}
        self._trades.append(trade)
        
        await self._broadcast({'type': 'trade_entry', 'trade': trade, 'stats': self._stats})
    
    async def close_position(self, position_id: str, exit_price: float, pnl: float):
        """Close a position."""
        position = self._positions.pop(position_id, None)
        if not position:
            return
        
        # Update stats
        self._stats['total_pnl'] += pnl
        self._stats['daily_pnl'] += pnl
        self._stats['bankroll'] += pnl
        
        if pnl > 0:
            self._stats['winning_trades'] += 1
            self._stats['best_trade'] = max(self._stats['best_trade'], pnl)
        else:
            self._stats['losing_trades'] += 1
            self._stats['worst_trade'] = min(self._stats['worst_trade'], pnl)
        
        total = self._stats['winning_trades'] + self._stats['losing_trades']
        self._stats['win_rate'] = self._stats['winning_trades'] / total if total > 0 else 0
        
        trade = {
            **position,
            'action': 'EXIT',
            'exit_price': exit_price,
            'pnl': pnl,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.append(trade)
        
        await self._broadcast({'type': 'trade_exit', 'trade': trade, 'stats': self._stats})
    
    async def log_analysis(self, analysis: dict):
        """Log an AI analysis."""
        self._analyses.append({**analysis, 'timestamp': datetime.utcnow().isoformat()})
        self._analyses = self._analyses[-100:]
        await self._broadcast({'type': 'analysis', 'analysis': analysis})
    
    async def update_stats(self, stats: dict):
        """Update stats."""
        self._stats.update(stats)
        await self._broadcast({'type': 'stats', 'stats': self._stats})
    
    async def _handle_index(self, request):
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_markets(self, request):
        return web.json_response(list(self._markets.values()))
    
    async def _handle_state(self, request):
        return web.json_response({
            'markets': list(self._markets.values()),
            'positions': list(self._positions.values()),
            'trades': self._trades[-50:],
            'stats': self._stats,
        })
    
    async def _handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)
        
        # Send current state
        await ws.send_str(json.dumps({
            'type': 'init',
            'markets': list(self._markets.values()),
            'positions': list(self._positions.values()),
            'trades': self._trades[-50:],
            'stats': self._stats,
        }))
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self._websockets.discard(ws)
        
        return ws
    
    async def _broadcast(self, data: dict):
        if not self._websockets:
            return
        msg = json.dumps(data)
        dead = set()
        for ws in self._websockets:
            try:
                await ws.send_str(msg)
            except:
                dead.add(ws)
        self._websockets -= dead


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battle-Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            padding: 16px 24px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo { font-size: 22px; font-weight: 700; }
        .logo .battle { color: #58a6ff; }
        .logo .bot { color: #f0883e; }
        
        .header-right { display: flex; align-items: center; gap: 16px; }
        
        .status-badge {
            padding: 6px 14px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-badge.live { background: #238636; }
        .status-badge.dry-run { background: #9e6a03; }
        
        .live-dot {
            width: 8px; height: 8px;
            background: #3fb950;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .tabs {
            display: flex;
            gap: 4px;
            padding: 12px 24px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
        }
        
        .tab {
            padding: 10px 20px;
            border: none;
            background: transparent;
            color: #8b949e;
            cursor: pointer;
            font-size: 14px;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .tab:hover { background: #21262d; color: #e6edf3; }
        .tab.active { background: #58a6ff; color: #fff; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Stats Bar */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            padding: 16px 24px;
            background: #0d1117;
        }
        
        .stat-card {
            background: #161b22;
            padding: 14px 16px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
        .stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
        .stat-value.positive { color: #3fb950; }
        .stat-value.negative { color: #f85149; }
        .stat-value.neutral { color: #58a6ff; }
        
        /* Markets Grid */
        .markets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 12px;
            padding: 16px 24px;
        }
        
        .market-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.2s;
            cursor: pointer;
        }
        .market-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
        
        .market-header { padding: 14px; display: flex; gap: 10px; }
        .market-image { width: 44px; height: 44px; border-radius: 8px; background: #30363d; object-fit: cover; }
        .market-info { flex: 1; min-width: 0; }
        .market-question { font-size: 13px; font-weight: 500; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .market-meta { display: flex; gap: 10px; margin-top: 6px; font-size: 11px; color: #8b949e; }
        
        .market-footer {
            padding: 10px 14px;
            background: #0d1117;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .price-bar { flex: 1; margin-right: 12px; }
        .price-labels { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; }
        .price-yes { color: #3fb950; font-weight: 600; }
        .price-no { color: #f85149; }
        .price-track { height: 6px; background: #f8514920; border-radius: 3px; overflow: hidden; }
        .price-fill { height: 100%; background: linear-gradient(90deg, #3fb950, #2ea043); border-radius: 3px; }
        
        .volume-badge { font-size: 11px; color: #8b949e; }
        .volume-badge strong { color: #e6edf3; }
        
        /* Positions & Trades */
        .section { padding: 16px 24px; }
        .section-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        .section-title .count { background: #30363d; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #8b949e; }
        
        .positions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 12px;
        }
        
        .position-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 14px;
        }
        .position-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px; }
        .position-question { font-size: 13px; flex: 1; margin-right: 10px; }
        .position-side { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .position-side.yes { background: #238636; }
        .position-side.no { background: #da3633; }
        
        .position-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .position-stat-label { font-size: 10px; color: #8b949e; }
        .position-stat-value { font-size: 14px; font-weight: 600; }
        
        .trades-list { display: flex; flex-direction: column; gap: 8px; }
        
        .trade-item {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .trade-info { flex: 1; }
        .trade-question { font-size: 13px; margin-bottom: 4px; }
        .trade-details { font-size: 11px; color: #8b949e; }
        .trade-badge { padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; }
        .trade-badge.entry { background: #238636; }
        .trade-badge.profit { background: #3fb950; color: #000; }
        .trade-badge.loss { background: #f85149; }
        
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #8b949e;
        }
        .empty-state .icon { font-size: 40px; margin-bottom: 12px; }
        
        .category-badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .category-crypto { background: #f7931a20; color: #f7931a; }
        .category-politics { background: #da363320; color: #f85149; }
        .category-sports { background: #3fb95020; color: #3fb950; }
        .category-economics { background: #a371f720; color: #a371f7; }
        .category-tech { background: #58a6ff20; color: #58a6ff; }
        .category-other { background: #8b949e20; color: #8b949e; }
        
        .filters {
            display: flex;
            gap: 8px;
            padding: 0 24px 12px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 6px 12px;
            border: 1px solid #30363d;
            border-radius: 16px;
            background: transparent;
            color: #8b949e;
            cursor: pointer;
            font-size: 12px;
        }
        .filter-btn:hover { border-color: #58a6ff; color: #58a6ff; }
        .filter-btn.active { background: #58a6ff; color: #fff; border-color: #58a6ff; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo"><span class="battle">Battle</span><span class="bot">Bot</span></div>
        <div class="header-right">
            <div class="live-dot"></div>
            <span id="updateTime" style="font-size: 12px; color: #8b949e;">Connecting...</span>
            <div id="statusBadge" class="status-badge dry-run">DRY RUN</div>
        </div>
    </div>
    
    <div class="tabs">
        <button class="tab active" data-tab="portfolio">Portfolio</button>
        <button class="tab" data-tab="markets">Markets</button>
        <button class="tab" data-tab="trades">Trade History</button>
    </div>
    
    <!-- PORTFOLIO TAB -->
    <div id="portfolio" class="tab-content active">
        <div class="stats-bar">
            <div class="stat-card">
                <div class="stat-label">Bankroll</div>
                <div id="bankroll" class="stat-value neutral">$1,000</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div id="totalPnl" class="stat-value neutral">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Today's P&L</div>
                <div id="dailyPnl" class="stat-value neutral">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div id="winRate" class="stat-value neutral">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trades</div>
                <div id="totalTrades" class="stat-value neutral">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Trade</div>
                <div id="bestTrade" class="stat-value positive">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Worst Trade</div>
                <div id="worstTrade" class="stat-value negative">$0.00</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Active Positions <span id="positionsCount" class="count">0</span></div>
            <div id="positionsList" class="positions-grid">
                <div class="empty-state">
                    <div class="icon">📭</div>
                    <div>No active positions</div>
                    <div style="font-size: 12px; margin-top: 8px;">Positions will appear here when the bot enters trades</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Recent Trades <span id="recentTradesCount" class="count">0</span></div>
            <div id="recentTradesList" class="trades-list">
                <div class="empty-state">
                    <div class="icon">📜</div>
                    <div>No trades yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- MARKETS TAB -->
    <div id="markets" class="tab-content">
        <div class="filters">
            <button class="filter-btn active" data-filter="all">All</button>
            <button class="filter-btn" data-filter="crypto">Crypto</button>
            <button class="filter-btn" data-filter="politics">Politics</button>
            <button class="filter-btn" data-filter="sports">Sports</button>
            <button class="filter-btn" data-filter="economics">Economics</button>
            <button class="filter-btn" data-filter="tech">Tech</button>
        </div>
        <div id="marketsGrid" class="markets-grid"></div>
    </div>
    
    <!-- TRADES TAB -->
    <div id="trades" class="tab-content">
        <div class="section">
            <div class="section-title">All Trades <span id="allTradesCount" class="count">0</span></div>
            <div id="allTradesList" class="trades-list">
                <div class="empty-state">
                    <div class="icon">📜</div>
                    <div>No trades yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let markets = [];
        let positions = [];
        let trades = [];
        let stats = {};
        let currentFilter = 'all';
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('updateTime').textContent = 'Connected';
            };
            
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            };
            
            ws.onclose = () => {
                document.getElementById('updateTime').textContent = 'Reconnecting...';
                setTimeout(connect, 3000);
            };
        }
        
        function handleMessage(msg) {
            switch(msg.type) {
                case 'init':
                case 'update':
                    if (msg.markets) markets = msg.markets;
                    if (msg.positions) positions = msg.positions;
                    if (msg.trades) trades = msg.trades;
                    if (msg.stats) stats = msg.stats;
                    renderAll();
                    break;
                case 'trade_entry':
                    trades.unshift(msg.trade);
                    positions = [...positions.filter(p => p.id !== msg.trade.id), msg.trade];
                    if (msg.stats) stats = msg.stats;
                    renderAll();
                    break;
                case 'trade_exit':
                    trades.unshift(msg.trade);
                    positions = positions.filter(p => p.id !== msg.trade.id);
                    if (msg.stats) stats = msg.stats;
                    renderAll();
                    break;
                case 'stats':
                    stats = msg.stats;
                    renderStats();
                    break;
            }
        }
        
        function renderAll() {
            renderStats();
            renderPositions();
            renderMarkets();
            renderTrades();
        }
        
        function renderStats() {
            document.getElementById('bankroll').textContent = `$${(stats.bankroll || 1000).toLocaleString()}`;
            
            const totalPnl = stats.total_pnl || 0;
            const el1 = document.getElementById('totalPnl');
            el1.textContent = `${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`;
            el1.className = `stat-value ${totalPnl > 0 ? 'positive' : totalPnl < 0 ? 'negative' : 'neutral'}`;
            
            const dailyPnl = stats.daily_pnl || 0;
            const el2 = document.getElementById('dailyPnl');
            el2.textContent = `${dailyPnl >= 0 ? '+' : ''}$${dailyPnl.toFixed(2)}`;
            el2.className = `stat-value ${dailyPnl > 0 ? 'positive' : dailyPnl < 0 ? 'negative' : 'neutral'}`;
            
            document.getElementById('winRate').textContent = `${((stats.win_rate || 0) * 100).toFixed(0)}%`;
            document.getElementById('totalTrades').textContent = stats.total_trades || 0;
            document.getElementById('bestTrade').textContent = `+$${(stats.best_trade || 0).toFixed(2)}`;
            document.getElementById('worstTrade').textContent = `$${(stats.worst_trade || 0).toFixed(2)}`;
            
            const badge = document.getElementById('statusBadge');
            badge.textContent = stats.dry_run ? 'DRY RUN' : 'LIVE';
            badge.className = `status-badge ${stats.dry_run ? 'dry-run' : 'live'}`;
            
            document.getElementById('updateTime').textContent = new Date().toLocaleTimeString();
        }
        
        function renderPositions() {
            const list = document.getElementById('positionsList');
            document.getElementById('positionsCount').textContent = positions.length;
            
            if (positions.length === 0) {
                list.innerHTML = `<div class="empty-state"><div class="icon">📭</div><div>No active positions</div></div>`;
                return;
            }
            
            list.innerHTML = positions.map(p => `
                <div class="position-card">
                    <div class="position-header">
                        <div class="position-question">${escapeHtml((p.question || 'Unknown').substring(0, 60))}...</div>
                        <span class="position-side ${(p.side || 'yes').toLowerCase()}">${(p.side || 'YES').toUpperCase()}</span>
                    </div>
                    <div class="position-stats">
                        <div><div class="position-stat-label">Entry</div><div class="position-stat-value">${((p.entry_price || p.price || 0) * 100).toFixed(0)}¢</div></div>
                        <div><div class="position-stat-label">Size</div><div class="position-stat-value">$${(p.size || 0).toFixed(0)}</div></div>
                        <div><div class="position-stat-label">P&L</div><div class="position-stat-value" style="color: ${(p.pnl || 0) >= 0 ? '#3fb950' : '#f85149'}">${(p.pnl || 0) >= 0 ? '+' : ''}$${(p.pnl || 0).toFixed(2)}</div></div>
                    </div>
                </div>
            `).join('');
        }
        
        function renderMarkets() {
            const grid = document.getElementById('marketsGrid');
            const filtered = currentFilter === 'all' ? markets : markets.filter(m => m.category === currentFilter);
            
            if (filtered.length === 0) {
                grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div class="icon">🔍</div><div>Loading markets...</div></div>`;
                return;
            }
            
            grid.innerHTML = filtered.map(m => `
                <div class="market-card" onclick="window.open('${m.url}', '_blank')">
                    <div class="market-header">
                        ${m.image ? `<img class="market-image" src="${m.image}" alt="">` : '<div class="market-image"></div>'}
                        <div class="market-info">
                            <div class="market-question">${escapeHtml(m.question)}</div>
                            <div class="market-meta">
                                <span class="category-badge category-${m.category}">${m.category}</span>
                                ${m.end_date ? `<span>Ends ${m.end_date}</span>` : ''}
                            </div>
                        </div>
                    </div>
                    <div class="market-footer">
                        <div class="price-bar">
                            <div class="price-labels">
                                <span class="price-yes">${m.price_pct}% Yes</span>
                                <span class="price-no">${100 - m.price_pct}% No</span>
                            </div>
                            <div class="price-track"><div class="price-fill" style="width:${m.price_pct}%"></div></div>
                        </div>
                        <div class="volume-badge"><strong>${m.volume_display}</strong> vol</div>
                    </div>
                </div>
            `).join('');
        }
        
        function renderTrades() {
            const recentList = document.getElementById('recentTradesList');
            const allList = document.getElementById('allTradesList');
            const recent = trades.slice(0, 5);
            
            document.getElementById('recentTradesCount').textContent = recent.length;
            document.getElementById('allTradesCount').textContent = trades.length;
            
            const renderTradeItem = t => {
                const isEntry = t.action === 'ENTRY';
                const pnl = t.pnl || 0;
                return `
                    <div class="trade-item">
                        <div class="trade-info">
                            <div class="trade-question">${escapeHtml((t.question || 'Unknown').substring(0, 50))}...</div>
                            <div class="trade-details">${new Date(t.timestamp).toLocaleString()} · ${(t.side || 'YES').toUpperCase()} @ ${((t.price || t.entry_price || 0) * 100).toFixed(0)}¢ · $${(t.size || 0).toFixed(0)}</div>
                        </div>
                        <span class="trade-badge ${isEntry ? 'entry' : pnl >= 0 ? 'profit' : 'loss'}">${isEntry ? 'ENTRY' : (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2)}</span>
                    </div>
                `;
            };
            
            if (recent.length === 0) {
                recentList.innerHTML = `<div class="empty-state"><div class="icon">📜</div><div>No trades yet</div></div>`;
            } else {
                recentList.innerHTML = recent.map(renderTradeItem).join('');
            }
            
            if (trades.length === 0) {
                allList.innerHTML = `<div class="empty-state"><div class="icon">📜</div><div>No trades yet</div></div>`;
            } else {
                allList.innerHTML = trades.map(renderTradeItem).join('');
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });
        
        // Market filters
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentFilter = btn.dataset.filter;
                renderMarkets();
            });
        });
        
        connect();
    </script>
</body>
</html>
'''


async def main():
    dashboard = BattleBotDashboard(port=8080)
    await dashboard.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await dashboard.stop()


if __name__ == "__main__":
    asyncio.run(main())
