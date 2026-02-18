"""Real-time web dashboard for Battle-Bot."""

import asyncio
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from aiohttp import web
import aiohttp
from loguru import logger


@dataclass
class DashboardEvent:
    """Event to send to dashboard."""
    type: str  # 'market_added', 'market_removed', 'trade_entry', 'trade_exit', 'status', 'analysis'
    timestamp: str
    data: dict
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Dashboard:
    """Real-time web dashboard server."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._websockets: set[web.WebSocketResponse] = set()
        
        # State
        self._markets: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}
        self._recent_trades: list[dict] = []
        self._recent_analyses: list[dict] = []
        self._stats: dict = {
            'bankroll': 1000,
            'daily_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'active_positions': 0,
            'markets_monitored': 0,
            'dry_run': True,
            'running': False,
        }
        
    async def start(self) -> None:
        """Start the dashboard server."""
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/state', self._handle_state)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Dashboard running at http://localhost:{self.port}")
    
    async def stop(self) -> None:
        """Stop the dashboard server."""
        for ws in self._websockets:
            await ws.close()
        self._websockets.clear()
        
        if self._runner:
            await self._runner.cleanup()
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the dashboard HTML."""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_state(self, request: web.Request) -> web.Response:
        """Return current state as JSON."""
        state = {
            'stats': self._stats,
            'markets': list(self._markets.values()),
            'positions': list(self._positions.values()),
            'recent_trades': self._recent_trades[-50:],
            'recent_analyses': self._recent_analyses[-20:],
        }
        return web.json_response(state)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self._websockets.add(ws)
        logger.debug(f"Dashboard client connected ({len(self._websockets)} total)")
        
        # Send current state
        await ws.send_str(json.dumps({
            'type': 'init',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'stats': self._stats,
                'markets': list(self._markets.values()),
                'positions': list(self._positions.values()),
                'recent_trades': self._recent_trades[-50:],
            }
        }))
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self._websockets.discard(ws)
            logger.debug(f"Dashboard client disconnected ({len(self._websockets)} total)")
        
        return ws
    
    async def _broadcast(self, event: DashboardEvent) -> None:
        """Broadcast event to all connected clients."""
        if not self._websockets:
            return
        
        message = event.to_json()
        dead = set()
        
        for ws in self._websockets:
            try:
                await ws.send_str(message)
            except Exception:
                dead.add(ws)
        
        self._websockets -= dead
    
    # Public methods for bot to call
    
    async def update_stats(self, stats: dict) -> None:
        """Update overall stats."""
        self._stats.update(stats)
        await self._broadcast(DashboardEvent(
            type='stats',
            timestamp=datetime.utcnow().isoformat(),
            data=self._stats,
        ))
    
    async def market_added(self, market: dict) -> None:
        """Called when a market is added."""
        self._markets[market['token_id']] = market
        self._stats['markets_monitored'] = len(self._markets)
        
        await self._broadcast(DashboardEvent(
            type='market_added',
            timestamp=datetime.utcnow().isoformat(),
            data=market,
        ))
    
    async def market_removed(self, token_id: str, reason: str = "") -> None:
        """Called when a market is removed."""
        market = self._markets.pop(token_id, None)
        self._stats['markets_monitored'] = len(self._markets)
        
        await self._broadcast(DashboardEvent(
            type='market_removed',
            timestamp=datetime.utcnow().isoformat(),
            data={'token_id': token_id, 'reason': reason, 'market': market},
        ))
    
    async def analysis_complete(self, market_id: str, question: str, result: dict) -> None:
        """Called when AI analysis completes."""
        analysis = {
            'market_id': market_id,
            'question': question[:80],
            'timestamp': datetime.utcnow().isoformat(),
            **result,
        }
        self._recent_analyses.append(analysis)
        self._recent_analyses = self._recent_analyses[-100:]
        
        await self._broadcast(DashboardEvent(
            type='analysis',
            timestamp=datetime.utcnow().isoformat(),
            data=analysis,
        ))
    
    async def trade_entry(self, trade: dict) -> None:
        """Called when a trade is entered."""
        self._positions[trade['token_id']] = trade
        self._stats['active_positions'] = len(self._positions)
        self._stats['total_trades'] += 1
        
        trade_record = {**trade, 'action': 'ENTRY', 'timestamp': datetime.utcnow().isoformat()}
        self._recent_trades.append(trade_record)
        self._recent_trades = self._recent_trades[-200:]
        
        await self._broadcast(DashboardEvent(
            type='trade_entry',
            timestamp=datetime.utcnow().isoformat(),
            data=trade_record,
        ))
    
    async def trade_exit(self, trade: dict) -> None:
        """Called when a trade is exited."""
        self._positions.pop(trade.get('token_id'), None)
        self._stats['active_positions'] = len(self._positions)
        
        trade_record = {**trade, 'action': 'EXIT', 'timestamp': datetime.utcnow().isoformat()}
        self._recent_trades.append(trade_record)
        self._recent_trades = self._recent_trades[-200:]
        
        await self._broadcast(DashboardEvent(
            type='trade_exit',
            timestamp=datetime.utcnow().isoformat(),
            data=trade_record,
        ))
    
    async def log_event(self, event_type: str, data: dict) -> None:
        """Log a generic event."""
        await self._broadcast(DashboardEvent(
            type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
        ))


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battle-Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0a0e17;
            color: #e4e6eb;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
            padding: 20px 30px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: #58a6ff;
        }
        
        .logo span { color: #f0883e; }
        
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .status-badge.running { background: #238636; color: #fff; }
        .status-badge.dry-run { background: #9e6a03; color: #fff; }
        .status-badge.stopped { background: #da3633; color: #fff; }
        
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            padding: 20px 30px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
        }
        
        .stat-card {
            background: #21262d;
            padding: 15px 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
        
        .stat-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            margin-top: 5px;
        }
        
        .stat-value.positive { color: #3fb950; }
        .stat-value.negative { color: #f85149; }
        .stat-value.neutral { color: #58a6ff; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px 30px;
        }
        
        @media (max-width: 1200px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        .panel {
            background: #161b22;
            border-radius: 12px;
            border: 1px solid #30363d;
            overflow: hidden;
        }
        
        .panel-header {
            padding: 15px 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-title {
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .panel-title .icon { font-size: 18px; }
        
        .panel-count {
            background: #30363d;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            color: #8b949e;
        }
        
        .panel-content {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .market-item, .trade-item, .analysis-item {
            padding: 15px 20px;
            border-bottom: 1px solid #21262d;
            transition: background 0.2s;
        }
        
        .market-item:hover, .trade-item:hover { background: #1c2128; }
        
        .market-question {
            font-size: 14px;
            color: #e4e6eb;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        
        .market-meta {
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #8b949e;
        }
        
        .market-price {
            font-weight: 600;
            color: #58a6ff;
        }
        
        .trade-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .trade-info { flex: 1; }
        
        .trade-question {
            font-size: 13px;
            color: #e4e6eb;
            margin-bottom: 4px;
        }
        
        .trade-details {
            font-size: 12px;
            color: #8b949e;
        }
        
        .trade-badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .trade-badge.entry { background: #238636; color: #fff; }
        .trade-badge.exit { background: #8957e5; color: #fff; }
        .trade-badge.profit { background: #3fb950; color: #000; }
        .trade-badge.loss { background: #f85149; color: #fff; }
        
        .analysis-item {
            background: #1c2128;
            margin: 10px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        
        .analysis-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .analysis-question {
            font-size: 13px;
            color: #e4e6eb;
        }
        
        .analysis-time {
            font-size: 11px;
            color: #8b949e;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        
        .analysis-stat {
            text-align: center;
            padding: 8px;
            background: #21262d;
            border-radius: 6px;
        }
        
        .analysis-stat-label {
            font-size: 10px;
            color: #8b949e;
            text-transform: uppercase;
        }
        
        .analysis-stat-value {
            font-size: 16px;
            font-weight: 600;
            margin-top: 2px;
        }
        
        .analysis-decision {
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .analysis-decision.trade { background: #238636; color: #fff; }
        .analysis-decision.no-trade { background: #30363d; color: #8b949e; }
        
        .empty-state {
            padding: 40px 20px;
            text-align: center;
            color: #8b949e;
        }
        
        .empty-state .icon { font-size: 40px; margin-bottom: 10px; }
        
        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #8b949e;
        }
        
        .live-dot {
            width: 8px;
            height: 8px;
            background: #3fb950;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .positions-panel { grid-column: 1 / -1; }
        
        .position-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            padding: 15px;
        }
        
        .position-card {
            background: #21262d;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #30363d;
        }
        
        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 10px;
        }
        
        .position-question {
            font-size: 13px;
            color: #e4e6eb;
            flex: 1;
            margin-right: 10px;
        }
        
        .position-side {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .position-side.yes { background: #238636; color: #fff; }
        .position-side.no { background: #da3633; color: #fff; }
        
        .position-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        
        .position-stat-label {
            font-size: 10px;
            color: #8b949e;
        }
        
        .position-stat-value {
            font-size: 14px;
            font-weight: 600;
        }
        
        .toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .toast {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideIn 0.3s ease;
            max-width: 350px;
        }
        
        .toast.entry { border-left: 4px solid #238636; }
        .toast.exit { border-left: 4px solid #8957e5; }
        .toast.analysis { border-left: 4px solid #58a6ff; }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .scrollbar::-webkit-scrollbar { width: 8px; }
        .scrollbar::-webkit-scrollbar-track { background: #161b22; }
        .scrollbar::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
        .scrollbar::-webkit-scrollbar-thumb:hover { background: #484f58; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">Battle<span>Bot</span> Dashboard</div>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div class="live-indicator">
                <div class="live-dot"></div>
                <span>Live</span>
            </div>
            <div id="statusBadge" class="status-badge dry-run">DRY RUN</div>
        </div>
    </div>
    
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-label">Bankroll</div>
            <div id="bankroll" class="stat-value neutral">$1,000</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Daily P&L</div>
            <div id="dailyPnl" class="stat-value neutral">$0.00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Active Positions</div>
            <div id="positions" class="stat-value neutral">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Markets Monitored</div>
            <div id="markets" class="stat-value neutral">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Trades</div>
            <div id="totalTrades" class="stat-value neutral">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div id="winRate" class="stat-value neutral">0%</div>
        </div>
    </div>
    
    <div class="main-grid">
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📊</span> Monitored Markets</div>
                <div id="marketsCount" class="panel-count">0</div>
            </div>
            <div id="marketsList" class="panel-content scrollbar">
                <div class="empty-state">
                    <div class="icon">🔍</div>
                    <div>Discovering markets...</div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🤖</span> AI Analyses</div>
                <div id="analysesCount" class="panel-count">0</div>
            </div>
            <div id="analysesList" class="panel-content scrollbar">
                <div class="empty-state">
                    <div class="icon">🧠</div>
                    <div>Waiting for price movements...</div>
                </div>
            </div>
        </div>
        
        <div class="panel positions-panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">💼</span> Active Positions</div>
                <div id="positionsCount" class="panel-count">0</div>
            </div>
            <div id="positionsList" class="panel-content scrollbar">
                <div class="empty-state">
                    <div class="icon">📭</div>
                    <div>No active positions</div>
                </div>
            </div>
        </div>
        
        <div class="panel" style="grid-column: 1 / -1;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📜</span> Trade History</div>
                <div id="tradesCount" class="panel-count">0</div>
            </div>
            <div id="tradesList" class="panel-content scrollbar">
                <div class="empty-state">
                    <div class="icon">⏳</div>
                    <div>No trades yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="toastContainer" class="toast-container"></div>
    
    <script>
        let ws;
        let reconnectAttempts = 0;
        
        const state = {
            markets: {},
            positions: {},
            trades: [],
            analyses: [],
            stats: {}
        };
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to dashboard');
                reconnectAttempts = 0;
            };
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
            
            ws.onclose = () => {
                console.log('Disconnected, reconnecting...');
                setTimeout(() => {
                    reconnectAttempts++;
                    connect();
                }, Math.min(1000 * reconnectAttempts, 10000));
            };
        }
        
        function handleMessage(msg) {
            switch (msg.type) {
                case 'init':
                    state.stats = msg.data.stats || {};
                    state.markets = {};
                    (msg.data.markets || []).forEach(m => state.markets[m.token_id] = m);
                    state.positions = {};
                    (msg.data.positions || []).forEach(p => state.positions[p.token_id] = p);
                    state.trades = msg.data.recent_trades || [];
                    updateAll();
                    break;
                    
                case 'stats':
                    state.stats = msg.data;
                    updateStats();
                    break;
                    
                case 'market_added':
                    state.markets[msg.data.token_id] = msg.data;
                    updateMarkets();
                    showToast('analysis', `Market added: ${msg.data.question?.substring(0, 50)}...`);
                    break;
                    
                case 'market_removed':
                    delete state.markets[msg.data.token_id];
                    updateMarkets();
                    break;
                    
                case 'analysis':
                    state.analyses.unshift(msg.data);
                    state.analyses = state.analyses.slice(0, 50);
                    updateAnalyses();
                    if (msg.data.decision === 'TRADE') {
                        showToast('analysis', `Trade signal: ${msg.data.question}`);
                    }
                    break;
                    
                case 'trade_entry':
                    state.positions[msg.data.token_id] = msg.data;
                    state.trades.unshift(msg.data);
                    updatePositions();
                    updateTrades();
                    showToast('entry', `Entered: ${msg.data.question?.substring(0, 40)}... @ ${msg.data.price}`);
                    break;
                    
                case 'trade_exit':
                    delete state.positions[msg.data.token_id];
                    state.trades.unshift(msg.data);
                    updatePositions();
                    updateTrades();
                    const pnl = msg.data.pnl || 0;
                    showToast('exit', `Exited: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} P&L`);
                    break;
            }
        }
        
        function updateAll() {
            updateStats();
            updateMarkets();
            updatePositions();
            updateTrades();
            updateAnalyses();
        }
        
        function updateStats() {
            const s = state.stats;
            document.getElementById('bankroll').textContent = `$${(s.bankroll || 1000).toLocaleString()}`;
            
            const pnl = s.daily_pnl || 0;
            const pnlEl = document.getElementById('dailyPnl');
            pnlEl.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
            pnlEl.className = `stat-value ${pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral'}`;
            
            document.getElementById('positions').textContent = s.active_positions || 0;
            document.getElementById('markets').textContent = s.markets_monitored || Object.keys(state.markets).length;
            document.getElementById('totalTrades').textContent = s.total_trades || 0;
            document.getElementById('winRate').textContent = `${((s.win_rate || 0) * 100).toFixed(0)}%`;
            
            const badge = document.getElementById('statusBadge');
            if (s.dry_run) {
                badge.textContent = 'DRY RUN';
                badge.className = 'status-badge dry-run';
            } else if (s.running) {
                badge.textContent = 'LIVE';
                badge.className = 'status-badge running';
            } else {
                badge.textContent = 'STOPPED';
                badge.className = 'status-badge stopped';
            }
        }
        
        function updateMarkets() {
            const list = document.getElementById('marketsList');
            const markets = Object.values(state.markets);
            document.getElementById('marketsCount').textContent = markets.length;
            
            if (markets.length === 0) {
                list.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><div>Discovering markets...</div></div>';
                return;
            }
            
            list.innerHTML = markets.map(m => `
                <div class="market-item">
                    <div class="market-question">${escapeHtml(m.question || 'Unknown')}</div>
                    <div class="market-meta">
                        <span class="market-price">${((m.current_price || 0.5) * 100).toFixed(0)}¢</span>
                        <span>Vol: $${((m.volume_24h || 0) / 1000).toFixed(1)}k</span>
                        <span>${m.category || 'other'}</span>
                    </div>
                </div>
            `).join('');
        }
        
        function updatePositions() {
            const list = document.getElementById('positionsList');
            const positions = Object.values(state.positions);
            document.getElementById('positionsCount').textContent = positions.length;
            
            if (positions.length === 0) {
                list.innerHTML = '<div class="empty-state"><div class="icon">📭</div><div>No active positions</div></div>';
                return;
            }
            
            list.innerHTML = '<div class="position-grid">' + positions.map(p => `
                <div class="position-card">
                    <div class="position-header">
                        <div class="position-question">${escapeHtml((p.question || 'Unknown').substring(0, 60))}...</div>
                        <span class="position-side ${(p.side || 'yes').toLowerCase()}">${(p.side || 'YES').toUpperCase()}</span>
                    </div>
                    <div class="position-stats">
                        <div>
                            <div class="position-stat-label">Entry</div>
                            <div class="position-stat-value">${((p.entry_price || p.price || 0) * 100).toFixed(0)}¢</div>
                        </div>
                        <div>
                            <div class="position-stat-label">Size</div>
                            <div class="position-stat-value">$${(p.size || 0).toFixed(0)}</div>
                        </div>
                        <div>
                            <div class="position-stat-label">P&L</div>
                            <div class="position-stat-value" style="color: ${(p.unrealized_pnl || 0) >= 0 ? '#3fb950' : '#f85149'}">
                                ${(p.unrealized_pnl || 0) >= 0 ? '+' : ''}$${(p.unrealized_pnl || 0).toFixed(2)}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('') + '</div>';
        }
        
        function updateTrades() {
            const list = document.getElementById('tradesList');
            document.getElementById('tradesCount').textContent = state.trades.length;
            
            if (state.trades.length === 0) {
                list.innerHTML = '<div class="empty-state"><div class="icon">⏳</div><div>No trades yet</div></div>';
                return;
            }
            
            list.innerHTML = state.trades.slice(0, 50).map(t => {
                const isEntry = t.action === 'ENTRY';
                const pnl = t.pnl || 0;
                return `
                    <div class="trade-item">
                        <div class="trade-info">
                            <div class="trade-question">${escapeHtml((t.question || 'Unknown').substring(0, 60))}...</div>
                            <div class="trade-details">
                                ${new Date(t.timestamp).toLocaleTimeString()} · 
                                ${(t.side || 'YES').toUpperCase()} @ ${((t.price || 0) * 100).toFixed(0)}¢ · 
                                $${(t.size || 0).toFixed(0)}
                            </div>
                        </div>
                        <span class="trade-badge ${isEntry ? 'entry' : pnl >= 0 ? 'profit' : 'loss'}">
                            ${isEntry ? 'ENTRY' : (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2)}
                        </span>
                    </div>
                `;
            }).join('');
        }
        
        function updateAnalyses() {
            const list = document.getElementById('analysesList');
            document.getElementById('analysesCount').textContent = state.analyses.length;
            
            if (state.analyses.length === 0) {
                list.innerHTML = '<div class="empty-state"><div class="icon">🧠</div><div>Waiting for price movements...</div></div>';
                return;
            }
            
            list.innerHTML = state.analyses.slice(0, 20).map(a => `
                <div class="analysis-item">
                    <div class="analysis-header">
                        <div class="analysis-question">${escapeHtml((a.question || 'Unknown').substring(0, 60))}...</div>
                        <div class="analysis-time">${new Date(a.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div class="analysis-grid">
                        <div class="analysis-stat">
                            <div class="analysis-stat-label">AI Prob</div>
                            <div class="analysis-stat-value">${((a.raw_prob || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div class="analysis-stat">
                            <div class="analysis-stat-label">Market</div>
                            <div class="analysis-stat-value">${((a.market_price || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div class="analysis-stat">
                            <div class="analysis-stat-label">Edge</div>
                            <div class="analysis-stat-value" style="color: ${(a.edge || 0) > 0 ? '#3fb950' : '#8b949e'}">
                                ${((a.edge || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div class="analysis-stat">
                            <div class="analysis-stat-label">Conf</div>
                            <div class="analysis-stat-value">${((a.confidence || 0) * 100).toFixed(0)}%</div>
                        </div>
                    </div>
                    <div class="analysis-decision ${a.decision === 'TRADE' ? 'trade' : 'no-trade'}">
                        ${a.decision || 'NO_TRADE'} ${a.reason ? '· ' + a.reason : ''}
                    </div>
                </div>
            `).join('');
        }
        
        function showToast(type, message) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }
        
        connect();
    </script>
</body>
</html>
'''
