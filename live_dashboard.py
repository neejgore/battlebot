#!/usr/bin/env python3
"""Live Polymarket Dashboard - Real market data, no bot."""

import asyncio
import json
from datetime import datetime, timedelta
from aiohttp import web
import aiohttp
import httpx

GAMMA_API = "https://gamma-api.polymarket.com"


class LiveDashboard:
    """Real-time Polymarket market dashboard."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._app = None
        self._runner = None
        self._websockets: set[web.WebSocketResponse] = set()
        self._markets: dict[str, dict] = {}
        self._running = False
        
    async def start(self):
        """Start the dashboard server."""
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/markets', self._handle_markets)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, '0.0.0.0', self.port)
        await site.start()
        
        self._running = True
        print(f"\n{'='*50}")
        print("POLYMARKET LIVE DASHBOARD")
        print(f"{'='*50}")
        print(f"\nOpen http://localhost:{self.port} in your browser")
        print("Press Ctrl+C to stop\n")
        
        # Start market refresh loop
        asyncio.create_task(self._refresh_loop())
    
    async def stop(self):
        """Stop the dashboard."""
        self._running = False
        for ws in self._websockets:
            await ws.close()
        if self._runner:
            await self._runner.cleanup()
    
    async def _refresh_loop(self):
        """Fetch markets periodically."""
        while self._running:
            try:
                await self._fetch_markets()
                await self._broadcast({'type': 'markets', 'data': list(self._markets.values())})
            except Exception as e:
                print(f"Error fetching markets: {e}")
            await asyncio.sleep(30)  # Refresh every 30 seconds
    
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
        
        # Sort by volume
        sorted_markets = sorted(markets.values(), key=lambda x: x['volume_24h'], reverse=True)
        self._markets = {m['id']: m for m in sorted_markets[:50]}
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(self._markets)} markets")
    
    def _parse_market(self, m: dict) -> dict:
        """Parse market data."""
        # Get price
        price = 0.5
        if 'outcomePrices' in m:
            prices = m['outcomePrices']
            if isinstance(prices, list) and len(prices) > 0:
                try:
                    price = float(prices[0])
                except:
                    pass
        
        # Get volume
        volume = float(m.get('volume', 0) or m.get('volume24hr', 0) or 0)
        
        # Get end date
        end_date = None
        for field in ['endDate', 'end_date']:
            if field in m and m[field]:
                try:
                    end_date = m[field][:10]
                except:
                    pass
        
        # Category
        question = m.get('question', '').lower()
        if any(w in question for w in ['bitcoin', 'btc', 'eth', 'crypto', 'solana']):
            category = 'crypto'
        elif any(w in question for w in ['trump', 'biden', 'election', 'president', 'democrat', 'republican']):
            category = 'politics'
        elif any(w in question for w in ['nfl', 'nba', 'mlb', 'super bowl', 'game', 'win']):
            category = 'sports'
        elif any(w in question for w in ['fed', 'rate', 'inflation', 'gdp', 'economy']):
            category = 'economics'
        elif any(w in question for w in ['ai', 'openai', 'google', 'apple', 'microsoft']):
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
    
    async def _handle_index(self, request):
        """Serve the dashboard."""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_markets(self, request):
        """Return markets as JSON."""
        return web.json_response(list(self._markets.values()))
    
    async def _handle_websocket(self, request):
        """Handle WebSocket for live updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)
        
        # Send current markets
        await ws.send_str(json.dumps({
            'type': 'markets',
            'data': list(self._markets.values())
        }))
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self._websockets.discard(ws)
        
        return ws
    
    async def _broadcast(self, data: dict):
        """Broadcast to all connected clients."""
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
    <title>Polymarket Live</title>
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
            padding: 20px 30px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
        }
        
        .logo .poly { color: #58a6ff; }
        .logo .market { color: #f0883e; }
        
        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #8b949e;
        }
        
        .live-dot {
            width: 10px;
            height: 10px;
            background: #3fb950;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        .filters {
            display: flex;
            gap: 10px;
            padding: 15px 30px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            padding: 8px 16px;
            border: 1px solid #30363d;
            border-radius: 20px;
            background: transparent;
            color: #8b949e;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        
        .filter-btn:hover { border-color: #58a6ff; color: #58a6ff; }
        .filter-btn.active { background: #58a6ff; color: #fff; border-color: #58a6ff; }
        
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            padding: 20px 30px;
            background: #0d1117;
        }
        
        .stat-card {
            background: #161b22;
            padding: 15px 20px;
            border-radius: 12px;
            border: 1px solid #30363d;
        }
        
        .stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
        .stat-value { font-size: 24px; font-weight: 700; color: #58a6ff; margin-top: 4px; }
        
        .markets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 16px;
            padding: 20px 30px;
        }
        
        .market-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.2s;
            cursor: pointer;
        }
        
        .market-card:hover {
            border-color: #58a6ff;
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        
        .market-header {
            padding: 16px;
            display: flex;
            gap: 12px;
        }
        
        .market-image {
            width: 48px;
            height: 48px;
            border-radius: 8px;
            background: #30363d;
            flex-shrink: 0;
            object-fit: cover;
        }
        
        .market-info { flex: 1; min-width: 0; }
        
        .market-question {
            font-size: 14px;
            font-weight: 500;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        .market-meta {
            display: flex;
            gap: 12px;
            margin-top: 8px;
            font-size: 12px;
            color: #8b949e;
        }
        
        .market-category {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 600;
        }
        
        .category-crypto { background: #f7931a20; color: #f7931a; }
        .category-politics { background: #da363320; color: #f85149; }
        .category-sports { background: #3fb95020; color: #3fb950; }
        .category-economics { background: #a371f720; color: #a371f7; }
        .category-tech { background: #58a6ff20; color: #58a6ff; }
        .category-other { background: #8b949e20; color: #8b949e; }
        
        .market-footer {
            padding: 12px 16px;
            background: #0d1117;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .price-bar {
            flex: 1;
            margin-right: 16px;
        }
        
        .price-labels {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 12px;
        }
        
        .price-yes { color: #3fb950; font-weight: 600; }
        .price-no { color: #f85149; }
        
        .price-track {
            height: 8px;
            background: #f8514920;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .price-fill {
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #2ea043);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .volume-badge {
            font-size: 12px;
            color: #8b949e;
            white-space: nowrap;
        }
        
        .volume-badge strong { color: #e6edf3; }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #8b949e;
        }
        
        .empty-state .icon { font-size: 48px; margin-bottom: 16px; }
        
        .loading {
            display: flex;
            justify-content: center;
            padding: 40px;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .update-flash {
            animation: flash 0.5s ease;
        }
        
        @keyframes flash {
            0%, 100% { background: #161b22; }
            50% { background: #1f6feb20; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo"><span class="poly">Poly</span><span class="market">market</span> Live</div>
        <div class="live-badge">
            <div class="live-dot"></div>
            <span id="updateTime">Connecting...</span>
        </div>
    </div>
    
    <div class="filters">
        <button class="filter-btn active" data-filter="all">All Markets</button>
        <button class="filter-btn" data-filter="crypto">Crypto</button>
        <button class="filter-btn" data-filter="politics">Politics</button>
        <button class="filter-btn" data-filter="sports">Sports</button>
        <button class="filter-btn" data-filter="economics">Economics</button>
        <button class="filter-btn" data-filter="tech">Tech</button>
    </div>
    
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-label">Markets</div>
            <div id="totalMarkets" class="stat-value">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Volume</div>
            <div id="totalVolume" class="stat-value">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Price</div>
            <div id="avgPrice" class="stat-value">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Last Update</div>
            <div id="lastUpdate" class="stat-value">-</div>
        </div>
    </div>
    
    <div id="marketsGrid" class="markets-grid">
        <div class="loading"><div class="spinner"></div></div>
    </div>
    
    <script>
        let markets = [];
        let currentFilter = 'all';
        let ws;
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('updateTime').textContent = 'Connected';
            };
            
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'markets') {
                    markets = msg.data;
                    render();
                    updateStats();
                }
            };
            
            ws.onclose = () => {
                document.getElementById('updateTime').textContent = 'Reconnecting...';
                setTimeout(connect, 3000);
            };
        }
        
        function updateStats() {
            const filtered = currentFilter === 'all' ? markets : markets.filter(m => m.category === currentFilter);
            
            document.getElementById('totalMarkets').textContent = filtered.length;
            
            const totalVol = filtered.reduce((sum, m) => sum + m.volume_24h, 0);
            document.getElementById('totalVolume').textContent = totalVol > 1000000 
                ? `$${(totalVol/1000000).toFixed(1)}M` 
                : `$${(totalVol/1000).toFixed(0)}k`;
            
            const avgPrice = filtered.length > 0 
                ? Math.round(filtered.reduce((sum, m) => sum + m.price_pct, 0) / filtered.length)
                : 0;
            document.getElementById('avgPrice').textContent = `${avgPrice}%`;
            
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }
        
        function render() {
            const grid = document.getElementById('marketsGrid');
            const filtered = currentFilter === 'all' ? markets : markets.filter(m => m.category === currentFilter);
            
            if (filtered.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state" style="grid-column: 1/-1;">
                        <div class="icon">📊</div>
                        <div>No markets found</div>
                    </div>`;
                return;
            }
            
            grid.innerHTML = filtered.map(m => `
                <div class="market-card" onclick="window.open('${m.url}', '_blank')">
                    <div class="market-header">
                        ${m.image ? `<img class="market-image" src="${m.image}" alt="">` : '<div class="market-image"></div>'}
                        <div class="market-info">
                            <div class="market-question">${escapeHtml(m.question)}</div>
                            <div class="market-meta">
                                <span class="market-category category-${m.category}">${m.category}</span>
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
                            <div class="price-track">
                                <div class="price-fill" style="width: ${m.price_pct}%"></div>
                            </div>
                        </div>
                        <div class="volume-badge">
                            <strong>${m.volume_display}</strong> vol
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentFilter = btn.dataset.filter;
                render();
                updateStats();
            });
        });
        
        connect();
    </script>
</body>
</html>
'''


async def main():
    dashboard = LiveDashboard(port=8080)
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
