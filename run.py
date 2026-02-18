#!/usr/bin/env python3
"""Battle-Bot - Integrated trading bot with live dashboard."""

import asyncio
import json
import os
from datetime import datetime
from aiohttp import web
import aiohttp
import httpx
import websockets
from dotenv import load_dotenv

from logic.ai_signal import AISignalGenerator, AISignalResult
from logic.calibration import CalibrationEngine, CalibrationResult
from logic.risk_engine import RiskEngine, RiskLimits
from data.database import TelemetryDB

load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class BattleBot:
    def __init__(self, port: int = 8080):
        self.port = port
        self._app = None
        self._runner = None
        self._websockets: set[web.WebSocketResponse] = set()
        
        # Ensure storage directory exists (for Railway persistent volume)
        # Use 'storage' not 'data' to avoid overwriting code modules
        self._storage_dir = os.getenv('STORAGE_DIR', 'storage')
        os.makedirs(self._storage_dir, exist_ok=True)
        
        # Config from env
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.initial_bankroll = float(os.getenv('INITIAL_BANKROLL', 1000))
        self.min_edge = float(os.getenv('MIN_EDGE', 0.03))
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.5))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 50))
        self.kelly_fraction = float(os.getenv('FRACTIONAL_KELLY', 0.1))
        
        # TESTING ONLY: Simulate price movement (set to true for testing)
        self.simulate_prices = os.getenv('SIMULATE_PRICES', 'false').lower() == 'true'
        
        # Market data
        self._markets: dict[str, dict] = {}
        self._monitored: dict[str, dict] = {}
        
        # Trading state (persisted to disk)
        self._state_file = f"{self._storage_dir}/bot_state.json"
        self._positions: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._analyses: list[dict] = []
        self._load_state()  # Load previous positions on startup
        
        self._running = False
        self._last_analysis: dict[str, datetime] = {}
        self._analysis_cooldown = 300
        self._start_time = None
        
        # Real-time price feed
        self._ws_connection = None
        self._subscribed_tokens: set[str] = set()
        self._last_price_update: dict[str, datetime] = {}
        self._price_update_count = 0
        
        # AI Signal Generator (Claude)
        self._ai_generator = AISignalGenerator()
        self._ai_calls = 0
        self._ai_successes = 0
        
        # Telemetry Database
        self._db = TelemetryDB(f"{self._storage_dir}/battlebot.db")
        self._db_connected = False
        
        # Calibration Engine
        self._calibration: CalibrationEngine = None
        
        # Risk Engine (full V2.1 risk management)
        # TESTING MODE: Aggressive exits to see trades close faster
        self._risk_limits = RiskLimits(
            max_daily_drawdown=0.15,
            max_position_size=self.max_position_size,
            max_percent_bankroll_per_market=0.10,
            max_total_open_risk=0.30,
            max_positions=5,
            profit_take_pct=0.03,   # 3% profit target (was 15%)
            stop_loss_pct=0.03,     # 3% stop loss (was 10%)
            time_stop_hours=720,    # Disabled (30 days) - exit on performance only
            edge_scale=0.10,
            min_edge=self.min_edge,
        )
        self._risk_engine = RiskEngine(
            initial_bankroll=self.initial_bankroll,
            fractional_kelly=self.kelly_fraction,
            limits=self._risk_limits,
        )
    
    def _load_state(self):
        """Load positions and trades from disk."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._trades = state.get('trades', [])
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._trades)} trades")
            elif os.path.exists(self._state_file + '.backup'):
                print(f"[State] Main file missing, loading from backup...")
                with open(self._state_file + '.backup', 'r') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._trades = state.get('trades', [])
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._trades)} trades from backup")
        except json.JSONDecodeError as e:
            print(f"[State] WARNING: Corrupted state file, starting fresh: {e}")
            self._positions = {}
            self._trades = []
        except Exception as e:
            print(f"[State] Failed to load: {e}")
            self._positions = {}
            self._trades = []
    
    def _save_state(self):
        """Save positions and trades to disk with atomic write."""
        try:
            os.makedirs(os.path.dirname(self._state_file) or '.', exist_ok=True)
            
            state_data = {
                'positions': self._positions,
                'trades': self._trades[-100:],
                'saved_at': datetime.utcnow().isoformat(),
            }
            
            temp_file = self._state_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            if os.path.exists(self._state_file):
                backup_file = self._state_file + '.backup'
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self._state_file, backup_file)
            
            os.rename(temp_file, self._state_file)
            
        except Exception as e:
            print(f"[State] CRITICAL: Failed to save state: {e}")
            if self._positions:
                print(f"[State] WARNING: {len(self._positions)} positions may not be persisted!")
        
    def _get_stats(self) -> dict:
        """Calculate all stats from current state."""
        # Calculate at risk (sum of position sizes)
        at_risk = sum(p['size'] for p in self._positions.values())
        
        # Calculate available (initial - at_risk + realized_pnl)
        realized_pnl = sum(t.get('pnl', 0) for t in self._trades if t.get('action') == 'EXIT')
        available = self.initial_bankroll - at_risk + realized_pnl
        
        # Calculate total bankroll (available + at_risk + unrealized)
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in self._positions.values())
        total_value = available + at_risk + unrealized_pnl
        
        # Win/loss counts
        exits = [t for t in self._trades if t.get('action') == 'EXIT']
        winning = len([t for t in exits if t.get('pnl', 0) > 0])
        losing = len([t for t in exits if t.get('pnl', 0) < 0])
        breakeven = len([t for t in exits if t.get('pnl', 0) == 0])
        
        # Best/worst trades
        pnls = [t.get('pnl', 0) for t in exits]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        # Win rate
        total_closed = winning + losing
        win_rate = winning / total_closed if total_closed > 0 else 0
        
        # Return percentage
        return_pct = ((total_value - self.initial_bankroll) / self.initial_bankroll) * 100
        
        # Runtime
        runtime = ""
        if self._start_time:
            delta = datetime.utcnow() - self._start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime = f"{hours}h {minutes}m {seconds}s"
        
        return {
            'dry_run': self.dry_run,
            'running': self._running,
            'runtime': runtime,
            
            # Bankroll
            'initial_bankroll': self.initial_bankroll,
            'available': available,
            'at_risk': at_risk,
            'total_value': total_value,
            
            # P&L
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'return_pct': return_pct,
            
            # Trade counts
            'total_entries': len([t for t in self._trades if t.get('action') == 'ENTRY']),
            'total_exits': len(exits),
            'winning_trades': winning,
            'losing_trades': losing,
            'breakeven_trades': breakeven,
            'win_rate': win_rate,
            
            # Best/worst
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            
            # Current
            'open_positions': len(self._positions),
            'markets_monitored': len(self._monitored),
            'analyses_count': len(self._analyses),
            'ws_connected': self._ws_connection is not None,
            'price_updates': self._price_update_count,
            'ai_available': self._ai_generator.is_available,
            'ai_calls': self._ai_calls,
            'ai_successes': self._ai_successes,
            
            # Config
            'min_edge': self.min_edge,
            'min_confidence': self.min_confidence,
            'max_position_size': self.max_position_size,
            'kelly_fraction': self.kelly_fraction,
            
            # Risk Engine
            'kill_switch': self._risk_engine.daily_stats.kill_switch_triggered,
            'daily_drawdown': self._risk_engine.current_drawdown * 100,
            'exposure_ratio': self._risk_engine.exposure_ratio * 100,
            'trading_allowed': self._risk_engine.is_trading_allowed,
        }
        
    async def start(self):
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/state', self._handle_state)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, '0.0.0.0', self.port)
        await site.start()
        
        self._running = True
        self._start_time = datetime.utcnow()
        
        # Initialize database and calibration
        try:
            await self._db.connect()
            self._db_connected = True
            self._calibration = CalibrationEngine(self._db)
            print("[DB] Telemetry database connected")
        except Exception as e:
            print(f"[DB] Failed to connect: {e} (running without telemetry)")
            self._db_connected = False
        
        mode = "DRY RUN" if self.dry_run else "LIVE"
        print(f"\n{'='*50}")
        print(f"BATTLE-BOT [{mode}]")
        print(f"{'='*50}")
        print(f"Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Min Edge: {self.min_edge*100:.1f}%")
        print(f"Max Position: ${self.max_position_size:,.2f}")
        if self.simulate_prices:
            print(f"\n⚠️  SIMULATE_PRICES=true (testing mode)")
            print(f"   Set SIMULATE_PRICES=false for production!")
        print(f"\nDashboard: http://localhost:{self.port}")
        print("Press Ctrl+C to stop\n")
        
        asyncio.create_task(self._market_loop())
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._broadcast_loop())
        asyncio.create_task(self._price_feed_loop())
    
    async def stop(self):
        self._running = False
        for ws in self._websockets:
            await ws.close()
        if self._runner:
            await self._runner.cleanup()
        
        stats = self._get_stats()
        print(f"\n{'='*50}")
        print("SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"Runtime: {stats['runtime']}")
        print(f"Total P&L: ${stats['total_pnl']:+.2f} ({stats['return_pct']:+.1f}%)")
        print(f"Trades: {stats['total_entries']} entries, {stats['total_exits']} exits")
        print(f"Win Rate: {stats['win_rate']*100:.0f}% ({stats['winning_trades']}W / {stats['losing_trades']}L)")
        print(f"{'='*50}\n")
    
    async def _broadcast_loop(self):
        """Send updates every 5 seconds."""
        while self._running:
            await self._broadcast_update()
            await asyncio.sleep(5)
    
    async def _price_feed_loop(self):
        """Connect to Polymarket WebSocket for real-time prices."""
        retry_delay = 5
        
        while self._running:
            try:
                print(f"[WebSocket] Connecting to Polymarket...")
                
                async with websockets.connect(POLYMARKET_WS) as ws:
                    self._ws_connection = ws
                    print(f"[WebSocket] Connected! Real-time prices enabled.")
                    retry_delay = 5  # Reset on successful connection
                    
                    # Subscribe to monitored markets
                    await self._subscribe_to_markets()
                    
                    # Listen for price updates
                    async for message in ws:
                        if not self._running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._handle_price_update(data)
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"[WebSocket] Error handling message: {e}")
                            
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[WebSocket] Connection closed: {e}")
            except Exception as e:
                print(f"[WebSocket] Error: {e}")
            
            self._ws_connection = None
            
            if self._running:
                print(f"[WebSocket] Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60s
    
    async def _subscribe_to_markets(self):
        """Subscribe to price updates for monitored markets."""
        if not self._ws_connection or not self._monitored:
            return
        
        # Get token IDs for monitored markets
        token_ids = []
        for market in self._monitored.values():
            token_id = market.get('token_id') or market.get('id')
            if token_id and token_id not in self._subscribed_tokens:
                token_ids.append(token_id)
        
        if not token_ids:
            return
        
        # Subscribe message
        subscribe_msg = {
            "type": "subscribe",
            "channel": "market",
            "assets_ids": token_ids,
        }
        
        try:
            await self._ws_connection.send(json.dumps(subscribe_msg))
            self._subscribed_tokens.update(token_ids)
            print(f"[WebSocket] Subscribed to {len(token_ids)} markets")
        except Exception as e:
            print(f"[WebSocket] Subscribe error: {e}")
    
    async def _handle_price_update(self, data: dict):
        """Handle incoming price update from WebSocket."""
        # Handle different message formats
        if isinstance(data, list):
            for item in data:
                await self._process_price_change(item)
        elif data.get('type') == 'price_change' or 'price' in data:
            await self._process_price_change(data)
        elif 'data' in data:
            await self._handle_price_update(data['data'])
    
    async def _process_price_change(self, data: dict):
        """Process a single price change event."""
        asset_id = data.get('asset_id') or data.get('market') or data.get('id')
        
        # Try to get price from various fields
        price = None
        for field in ['price', 'best_bid', 'mid_price', 'last_price']:
            if field in data:
                try:
                    price = float(data[field])
                    break
                except:
                    pass
        
        if not asset_id or price is None:
            return
        
        # Update market price
        for market_id, market in self._markets.items():
            if market.get('id') == asset_id or market.get('token_id') == asset_id:
                old_price = market.get('price', 0)
                market['price'] = price
                market['price_pct'] = int(price * 100)
                self._last_price_update[market_id] = datetime.utcnow()
                self._price_update_count += 1
                
                # Log significant price changes
                if abs(price - old_price) >= 0.01:
                    print(f"[Price] {market['question'][:30]}... {old_price*100:.0f}¢ → {price*100:.0f}¢")
                
                # Immediately check positions for this market
                await self._check_position_for_market(market_id)
                break
    
    async def _check_position_for_market(self, market_id: str):
        """Check exit conditions for positions in a specific market."""
        for pos_id, pos in list(self._positions.items()):
            if pos.get('market_id') != market_id:
                continue
            
            market = self._markets.get(market_id)
            if not market:
                continue
            
            current_price = market['price']
            entry_price = pos['entry_price']
            side = pos['side']
            
            if side == 'YES':
                price_change = current_price - entry_price
            else:
                price_change = entry_price - current_price
            
            unrealized_pnl = price_change * pos['size']
            pos['current_price'] = current_price
            pos['unrealized_pnl'] = unrealized_pnl
            
            # Check exit conditions
            profit_target = pos['size'] * 0.15
            stop_loss = -pos['size'] * 0.10
            
            should_exit = False
            exit_reason = ""
            
            if unrealized_pnl >= profit_target:
                should_exit = True
                exit_reason = "PROFIT_TARGET"
            elif unrealized_pnl <= stop_loss:
                should_exit = True
                exit_reason = "STOP_LOSS"
            
            if should_exit:
                await self._exit_position(pos_id, current_price, unrealized_pnl, exit_reason)
    
    async def _market_loop(self):
        while self._running:
            try:
                await self._fetch_markets()
                await self._select_markets()
            except Exception as e:
                print(f"[Market Error] {e}")
            await asyncio.sleep(60)
    
    async def _trading_loop(self):
        await asyncio.sleep(5)
        while self._running:
            try:
                for market_id, market in list(self._monitored.items()):
                    if not self._running:
                        break
                    
                    last = self._last_analysis.get(market_id)
                    if last and (datetime.utcnow() - last).total_seconds() < self._analysis_cooldown:
                        continue
                    
                    await self._analyze_market(market)
                    self._last_analysis[market_id] = datetime.utcnow()
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"[Trading Error] {e}")
            await asyncio.sleep(30)
    
    async def _position_monitor_loop(self):
        """Monitor positions for exit conditions with V2.1 rules."""
        import random
        while self._running:
            try:
                for pos_id, pos in list(self._positions.items()):
                    market = self._markets.get(pos.get('market_id'))
                    if not market:
                        continue
                    
                    current_price = market['price']
                    entry_price = pos['entry_price']
                    side = pos['side']
                    
                    # TESTING: Simulate price movement (disable with SIMULATE_PRICES=false)
                    if self.simulate_prices:
                        drift = random.gauss(0, 0.02)  # ~2% std deviation
                        simulated_price = max(0.05, min(0.95, entry_price + drift))
                        current_price = simulated_price
                        market['price'] = simulated_price  # Update market too
                    
                    # Calculate P&L based on position side
                    if side == 'YES':
                        price_change = current_price - entry_price
                    else:
                        price_change = entry_price - current_price
                    
                    unrealized_pnl = price_change * pos['size']
                    pos['current_price'] = current_price
                    pos['unrealized_pnl'] = unrealized_pnl
                    
                    # Get configurable thresholds from risk limits
                    profit_take_pct = self._risk_limits.profit_take_pct
                    stop_loss_pct = self._risk_limits.stop_loss_pct
                    time_stop_hours = self._risk_limits.time_stop_hours
                    
                    profit_target = pos['size'] * profit_take_pct
                    stop_loss = -pos['size'] * stop_loss_pct
                    
                    should_exit = False
                    exit_reason = ""
                    
                    # 1. Profit target
                    if unrealized_pnl >= profit_target:
                        should_exit = True
                        exit_reason = "PROFIT_TARGET"
                    
                    # 2. Stop loss
                    elif unrealized_pnl <= stop_loss:
                        should_exit = True
                        exit_reason = "STOP_LOSS"
                    
                    # 3. Time stop (configurable, default 72 hours)
                    if not should_exit:
                        entry_time = pos.get('entry_time')
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(entry_time)
                        if entry_time:
                            hours_held = (datetime.utcnow() - entry_time).total_seconds() / 3600
                            if hours_held > time_stop_hours:
                                should_exit = True
                                exit_reason = "TIME_STOP"
                    
                    # 4. Signal flip check (every 10 minutes per position)
                    if not should_exit and pos.get('last_signal_check'):
                        last_check = pos.get('last_signal_check')
                        if isinstance(last_check, str):
                            last_check = datetime.fromisoformat(last_check)
                        
                        # Re-check signal every 10 minutes
                        if (datetime.utcnow() - last_check).total_seconds() > 600:
                            original_edge = pos.get('edge', 0)
                            original_prob = pos.get('ai_probability', current_price)
                            
                            # Compute current edge based on original probability and new price
                            if side == 'YES':
                                current_edge = original_prob - current_price
                            else:
                                current_edge = (1 - original_prob) - (1 - current_price)
                            
                            # If edge has flipped significantly negative (10% threshold)
                            if current_edge < -0.10:
                                should_exit = True
                                exit_reason = "SIGNAL_FLIP"
                            
                            pos['last_signal_check'] = datetime.utcnow().isoformat()
                    elif not pos.get('last_signal_check'):
                        pos['last_signal_check'] = datetime.utcnow().isoformat()
                    
                    if should_exit:
                        await self._exit_position(pos_id, current_price, unrealized_pnl, exit_reason)
                
            except Exception as e:
                print(f"[Monitor Error] {e}")
            await asyncio.sleep(10)
    
    async def _fetch_markets(self):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{GAMMA_API}/markets", params={
                'limit': 100, 'active': 'true', 'closed': 'false',
            })
            response.raise_for_status()
            raw = response.json()
        
        for m in raw:
            try:
                market = self._parse_market(m)
                if market:
                    self._markets[market['id']] = market
            except:
                pass
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(self._markets)} markets")
    
    async def _select_markets(self):
        """Filter markets using V2.1 eligibility criteria."""
        eligible = []
        rejection_counts = {
            'no_end_date': 0, 'too_far_out': 0, 'low_volume': 0, 'high_volume_headline': 0,
            'wide_spread': 0, 'extreme_price': 0, 'low_liquidity': 0,
        }
        
        # Get top 5 by volume to filter out "headline" markets
        all_markets = list(self._markets.values())
        all_markets.sort(key=lambda x: x['volume_24h'], reverse=True)
        headline_ids = {m['id'] for m in all_markets[:5]}
        
        for m in self._markets.values():
            # 1. Must have end date
            if not m.get('end_date'):
                rejection_counts['no_end_date'] += 1
                continue
            
            # 2. Time to resolution <= 30 days (or up to 90 for slow-moving markets)
            try:
                end_date = datetime.fromisoformat(m['end_date'].replace('Z', '+00:00'))
                days_to_resolution = (end_date.replace(tzinfo=None) - datetime.utcnow()).days
                if days_to_resolution > 30 and m['volume_24h'] < 50000:
                    rejection_counts['too_far_out'] += 1
                    continue
                if days_to_resolution > 90:
                    rejection_counts['too_far_out'] += 1
                    continue
            except:
                pass
            
            # 3. Not a headline market (top 5 by volume)
            if m['id'] in headline_ids:
                rejection_counts['high_volume_headline'] += 1
                continue
            
            # 4. Minimum volume (at least $10k 24h)
            if m['volume_24h'] < 10000:
                rejection_counts['low_volume'] += 1
                continue
            
            # 5. Maximum volume cap (avoid super liquid/efficient markets)
            if m['volume_24h'] > 500000:
                rejection_counts['high_volume_headline'] += 1
                continue
            
            # 6. Spread <= 4 cents (configurable)
            spread = m.get('spread', 0.02)
            if spread > 0.04:
                rejection_counts['wide_spread'] += 1
                continue
            
            # 7. Price not too extreme (tradeable edge)
            if m['price'] < 0.08 or m['price'] > 0.92:
                rejection_counts['extreme_price'] += 1
                continue
            
            # 8. Minimum liquidity (if available)
            liquidity = m.get('liquidity', 0)
            if liquidity > 0 and liquidity < 5000:
                rejection_counts['low_liquidity'] += 1
                continue
            
            eligible.append(m)
        
        # Sort by volume and select top 10
        eligible.sort(key=lambda x: x['volume_24h'], reverse=True)
        self._monitored = {m['id']: m for m in eligible[:10]}
        
        total_rejected = sum(rejection_counts.values())
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market eligibility: {len(eligible)} eligible / {len(self._markets)} total")
        if total_rejected > 0:
            reasons = [f"{k}={v}" for k, v in rejection_counts.items() if v > 0]
            print(f"[Filter] Rejected: {', '.join(reasons)}")
        
        # Subscribe to new markets via WebSocket
        if self._ws_connection:
            await self._subscribe_to_markets()
    
    async def _fetch_market_details(self, market_id: str) -> dict:
        """Fetch detailed market info including resolution rules."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{GAMMA_API}/markets/{market_id}")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            print(f"[API] Failed to fetch market details: {e}")
        return {}
    
    async def _analyze_market(self, market: dict):
        market_id = market['id']
        question = market['question']
        current_price = market['price']
        
        # Skip if already have position in this market
        if any(p['market_id'] == market_id for p in self._positions.values()):
            return
        
        # Get resolution rules (required for AI analysis)
        resolution_rules = market.get('description') or market.get('rules')
        
        # If no rules in cached data, try fetching detailed market info
        if not resolution_rules or len(resolution_rules) < 50:
            print(f"[Rules] Fetching resolution rules for: {question[:40]}...")
            details = await self._fetch_market_details(market_id)
            resolution_rules = (
                details.get('description') or 
                details.get('resolutionDetails') or 
                details.get('rules') or
                resolution_rules
            )
            
            # Update cached market with rules
            if resolution_rules:
                market['description'] = resolution_rules
        
        # Require resolution rules - skip if not available
        if not resolution_rules or len(resolution_rules) < 50:
            analysis = {
                'market_id': market_id,
                'question': question[:60],
                'market_price': current_price,
                'ai_probability': None,
                'confidence': None,
                'edge': 0,
                'side': None,
                'decision': 'NO_TRADE',
                'reason': 'No resolution rules available',
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._analyses.insert(0, analysis)
            self._analyses = self._analyses[:50]
            print(f"[AI] SKIP: No resolution rules for {question[:40]}...")
            await self._broadcast_update()
            return
        
        # Call real Claude AI for analysis
        self._ai_calls += 1
        print(f"[AI] Analyzing: {question[:50]}...")
        
        result: AISignalResult = await self._ai_generator.generate_signal(
            market_question=question,
            current_price=current_price,
            spread=market.get('spread', 0.02),
            resolution_rules=resolution_rules,
            resolution_date=None,  # TODO: parse from market
            volume_24h=market.get('volume_24h', 0),
            liquidity=market.get('liquidity', 0),
            recent_price_path=market.get('price_history', []),
            category=market.get('category'),
        )
        
        # Handle AI failure gracefully
        if not result.success:
            analysis = {
                'market_id': market_id,
                'question': question[:60],
                'market_price': current_price,
                'ai_probability': None,
                'confidence': None,
                'edge': 0,
                'side': None,
                'decision': 'NO_TRADE',
                'reason': f"AI failed: {result.error}",
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._analyses.insert(0, analysis)
            self._analyses = self._analyses[:50]
            print(f"[AI] FAILED: {result.error}")
            await self._broadcast_update()
            return
        
        # AI succeeded
        self._ai_successes += 1
        signal = result.signal
        raw_prob = signal.raw_prob
        confidence = signal.confidence
        
        # Apply calibration layer
        calibrated_prob = raw_prob
        adjusted_prob = raw_prob
        calibration_method = "none"
        
        if self._calibration:
            try:
                calib_result: CalibrationResult = await self._calibration.calibrate(
                    raw_prob=raw_prob,
                    market_price=current_price,
                    confidence=confidence,
                    category=market.get('category'),
                )
                calibrated_prob = calib_result.calibrated_prob
                adjusted_prob = calib_result.adjusted_prob
                calibration_method = calib_result.method
                print(f"[Calibration] Raw: {raw_prob:.2%} → Calibrated: {calibrated_prob:.2%} → Adjusted: {adjusted_prob:.2%} ({calibration_method})")
            except Exception as e:
                print(f"[Calibration] Error: {e} - using raw probability")
        
        # Calculate edge using adjusted (calibrated) probability
        edge_yes = adjusted_prob - current_price
        edge_no = (1 - adjusted_prob) - (1 - current_price)
        
        best_side = 'YES' if edge_yes > edge_no else 'NO'
        best_edge = max(edge_yes, edge_no)
        best_prob = adjusted_prob if best_side == 'YES' else (1 - adjusted_prob)
        
        stats = self._get_stats()
        should_trade = (
            best_edge >= self.min_edge and
            confidence >= self.min_confidence and
            len(self._positions) < 5 and
            stats['available'] >= 10
        )
        
        # Build analysis record with AI reasoning
        analysis = {
            'market_id': market_id,
            'question': question[:60],
            'market_price': current_price,
            'ai_probability': raw_prob,
            'calibrated_probability': calibrated_prob,
            'adjusted_probability': adjusted_prob,
            'confidence': confidence,
            'edge': best_edge,
            'side': best_side,
            'decision': 'TRADE' if should_trade else 'NO_TRADE',
            'timestamp': datetime.utcnow().isoformat(),
            'key_reasons': signal.key_reasons,
            'disconfirming': signal.disconfirming_evidence,
            'info_quality': signal.information_quality,
            'latency_ms': result.latency_ms,
            'calibration_method': calibration_method,
        }
        
        # Log decision to database
        if self._db_connected:
            try:
                reason_codes = []
                if best_edge < self.min_edge:
                    reason_codes.append("LOW_EDGE")
                if confidence < self.min_confidence:
                    reason_codes.append("LOW_CONFIDENCE")
                if len(self._positions) >= 5:
                    reason_codes.append("MAX_POSITIONS")
                if stats['available'] < 10:
                    reason_codes.append("LOW_FUNDS")
                if should_trade:
                    reason_codes.append("CRITERIA_MET")
                
                await self._db.log_decision(
                    market_id=market_id,
                    token_id=market.get('token_id', market_id),
                    price=current_price,
                    decision='TRADE' if should_trade else 'NO_TRADE',
                    reason_codes=reason_codes,
                    spread=market.get('spread'),
                    volume_24h=market.get('volume_24h'),
                    raw_prob=raw_prob,
                    confidence=confidence,
                    calibrated_prob=calibrated_prob,
                    adjusted_prob=adjusted_prob,
                    edge=best_edge,
                    side=best_side if should_trade else None,
                    ai_latency_ms=result.latency_ms,
                )
                
                # Also log calibration sample for future calibration
                await self._calibration.record_prediction(
                    market_id=market_id,
                    raw_prob=raw_prob,
                    market_price=current_price,
                    category=market.get('category'),
                    calibrated_prob=calibrated_prob,
                )
            except Exception as e:
                print(f"[DB] Failed to log decision: {e}")
        
        if should_trade:
            # Use RiskEngine for sophisticated position sizing
            # For Kelly, pass the probability and price for the side we're actually trading
            trade_prob = best_prob  # Probability our chosen side wins
            trade_price = current_price if best_side == 'YES' else (1 - current_price)
            
            size = await self._risk_engine.calculate_position_size(
                adjusted_prob=trade_prob,
                market_price=trade_price,
                edge=best_edge,
                confidence=confidence,
                market_id=market_id,
            )
            
            # Validate trade against risk limits
            is_valid, risk_reasons = await self._risk_engine.validate_trade(
                size=size,
                price=current_price,
                token_id=market.get('token_id', market_id),
            )
            
            if not is_valid or size < 5:
                should_trade = False
                analysis['decision'] = 'NO_TRADE'
                analysis['reason'] = f"Risk rejected: {', '.join(risk_reasons) if risk_reasons else 'size too small'}"
            else:
                analysis['reason'] = f"Edge {best_edge*100:.1f}% | {signal.key_reasons[0] if signal.key_reasons else 'AI approved'}"
                await self._enter_position(market, best_side, size, best_prob, best_edge, confidence)
        else:
            reasons = []
            if best_edge < self.min_edge:
                reasons.append(f"edge {best_edge*100:.1f}%")
            if confidence < self.min_confidence:
                reasons.append(f"conf {confidence*100:.0f}%")
            if len(self._positions) >= 5:
                reasons.append("max positions")
            if stats['available'] < 10:
                reasons.append("low funds")
            analysis['reason'] = ', '.join(reasons) if reasons else "criteria not met"
        
        self._analyses.insert(0, analysis)
        self._analyses = self._analyses[:50]
        
        decision_str = "✓ TRADE" if should_trade else "✗ SKIP"
        print(f"[AI] {question[:40]}... | {decision_str} | Edge: {best_edge*100:+.1f}% | Conf: {confidence*100:.0f}%")
        
        await self._broadcast_update()
    
    async def _enter_position(self, market: dict, side: str, size: float, prob: float, edge: float, confidence: float = 0.5):
        pos_id = f"pos_{int(datetime.utcnow().timestamp()*1000)}"
        
        position = {
            'id': pos_id,
            'market_id': market['id'],
            'question': market['question'],
            'side': side,
            'size': size,
            'entry_price': market['price'],
            'current_price': market['price'],
            'ai_probability': prob,
            'edge': edge,
            'confidence': confidence,
            'entry_time': datetime.utcnow().isoformat(),
            'unrealized_pnl': 0,
            'db_trade_id': None,  # Will be set if DB logging succeeds
        }
        
        # Log trade entry to database
        if self._db_connected:
            try:
                db_trade_id = await self._db.log_trade_entry(
                    decision_id=0,  # We don't track this linking yet
                    market_id=market['id'],
                    token_id=market.get('token_id', market['id']),
                    entry_price=market['price'],
                    entry_side=side,
                    size=size,
                    raw_prob=prob,
                    adjusted_prob=prob,
                    edge=edge,
                    confidence=confidence,
                )
                position['db_trade_id'] = db_trade_id
            except Exception as e:
                print(f"[DB] Failed to log trade entry: {e}")
        
        self._positions[pos_id] = position
        
        trade = {
            'id': pos_id,
            'market_id': market['id'],
            'question': market['question'],
            'action': 'ENTRY',
            'side': side,
            'price': market['price'],
            'size': size,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.insert(0, trade)  # Newest first
        
        mode = "[DRY RUN] " if self.dry_run else ""
        print(f"{mode}ENTERED: {side} ${size:.2f} @ {market['price']*100:.0f}¢ | {market['question'][:40]}...")
        
        self._save_state()  # Persist positions to disk
        await self._broadcast_update()
    
    async def _exit_position(self, pos_id: str, exit_price: float, pnl: float, reason: str):
        position = self._positions.pop(pos_id, None)
        if not position:
            return
        
        # Record trade result in risk engine (updates bankroll, drawdown, etc.)
        await self._risk_engine.record_trade_result(pnl)
        
        # Log trade exit to database
        if self._db_connected and position.get('db_trade_id'):
            try:
                await self._db.log_trade_exit(
                    trade_id=position['db_trade_id'],
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    fees_estimate=0.0,  # TODO: estimate fees
                )
            except Exception as e:
                print(f"[DB] Failed to log trade exit: {e}")
        
        trade = {
            'id': pos_id,
            'market_id': position['market_id'],
            'question': position['question'],
            'action': 'EXIT',
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': round(pnl, 2),
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.insert(0, trade)  # Newest first
        
        mode = "[DRY RUN] " if self.dry_run else ""
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        print(f"{mode}EXITED: {pnl_str} ({reason}) | {position['question'][:40]}...")
        
        self._save_state()  # Persist positions to disk
        await self._broadcast_update()
    
    def _parse_market(self, m: dict) -> dict:
        price = 0.5
        if 'outcomePrices' in m:
            prices = m['outcomePrices']
            if isinstance(prices, list) and len(prices) > 0:
                try:
                    price = float(prices[0])
                except:
                    pass
        
        volume = float(m.get('volume', 0) or m.get('volume24hr', 0) or 0)
        
        # Extract spread from bid/ask if available
        spread = 0.02  # Default 2 cents
        best_bid = m.get('bestBid')
        best_ask = m.get('bestAsk')
        if best_bid and best_ask:
            try:
                spread = float(best_ask) - float(best_bid)
            except:
                pass
        
        # Extract liquidity
        liquidity = float(m.get('liquidity', 0) or m.get('liquidityClob', 0) or 0)
        
        end_date = None
        for field in ['endDate', 'end_date', 'resolutionTime']:
            if field in m and m[field]:
                try:
                    end_date = m[field]
                except:
                    pass
        
        question = m.get('question', '').lower()
        if any(w in question for w in ['bitcoin', 'btc', 'eth', 'crypto']):
            category = 'crypto'
        elif any(w in question for w in ['trump', 'biden', 'election', 'president', 'congress']):
            category = 'politics'
        elif any(w in question for w in ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football']):
            category = 'sports'
        else:
            category = 'other'
        
        slug = m.get('slug', '')
        
        # Get token ID for WebSocket subscriptions
        token_id = m.get('conditionId', '')
        if 'clobTokenIds' in m:
            token_ids = m['clobTokenIds']
            if isinstance(token_ids, list) and len(token_ids) > 0:
                token_id = token_ids[0]
            elif isinstance(token_ids, str):
                try:
                    parsed = json.loads(token_ids)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        token_id = parsed[0]
                except:
                    pass
        
        return {
            'id': m.get('conditionId', ''),
            'token_id': token_id,
            'question': m.get('question', 'Unknown'),
            'description': m.get('description') or m.get('resolutionDetails') or m.get('rules') or '',
            'price': price,
            'price_pct': int(price * 100),
            'spread': spread,
            'spread_display': f"{spread*100:.1f}¢",
            'liquidity': liquidity,
            'volume_24h': volume,
            'volume_display': f"${volume/1000:.1f}k" if volume < 1000000 else f"${volume/1000000:.1f}M",
            'end_date': end_date,
            'category': category,
            'url': f"https://polymarket.com/event/{slug}" if slug else "",
            'image': m.get('image', ''),
        }
    
    async def _broadcast_update(self):
        if not self._websockets:
            return
        
        data = {
            'type': 'update',
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        }
        
        msg = json.dumps(data, default=str)
        dead = set()
        for ws in self._websockets:
            try:
                await ws.send_str(msg)
            except:
                dead.add(ws)
        self._websockets -= dead
    
    async def _handle_index(self, request):
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_state(self, request):
        return web.json_response({
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        })
    
    async def _handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)
        
        await ws.send_str(json.dumps({
            'type': 'init',
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        }, default=str))
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self._websockets.discard(ws)
        return ws


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battle-Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }
        
        .header { background: #161b22; padding: 12px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
        .logo { font-size: 18px; font-weight: 700; }
        .logo .b1 { color: #58a6ff; }
        .logo .b2 { color: #f0883e; }
        .header-info { display: flex; align-items: center; gap: 16px; font-size: 12px; }
        .status { display: flex; align-items: center; gap: 6px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot.live { background: #3fb950; animation: pulse 2s infinite; }
        .dot.off { background: #f85149; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        .badge { padding: 4px 10px; border-radius: 10px; font-size: 10px; font-weight: 600; }
        .badge.dry { background: #9e6a03; }
        .badge.live { background: #238636; }
        .runtime { color: #8b949e; }
        
        .tabs { display: flex; gap: 2px; padding: 8px 20px; background: #161b22; border-bottom: 1px solid #30363d; }
        .tab { padding: 8px 16px; border: none; background: transparent; color: #8b949e; cursor: pointer; font-size: 13px; border-radius: 6px; transition: all 0.2s; }
        .tab:hover { background: #21262d; color: #e6edf3; }
        .tab.active { background: #58a6ff; color: #fff; }
        
        .content { padding: 16px 20px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Stats Grid */
        .stats-section { margin-bottom: 20px; }
        .stats-title { font-size: 11px; color: #8b949e; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; }
        .stat { background: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #30363d; }
        .stat-label { font-size: 10px; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
        .stat-value { font-size: 18px; font-weight: 700; }
        .stat-value.pos { color: #3fb950; }
        .stat-value.neg { color: #f85149; }
        .stat-value.neu { color: #58a6ff; }
        .stat-sub { font-size: 10px; color: #8b949e; margin-top: 2px; }
        
        /* Sections */
        .section { margin-bottom: 24px; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .section-title { font-size: 14px; font-weight: 600; }
        .section-count { background: #30363d; padding: 2px 8px; border-radius: 10px; font-size: 11px; color: #8b949e; }
        
        /* Cards Grid */
        .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; transition: border-color 0.2s; }
        .card:hover { border-color: #484f58; }
        
        /* Position Card */
        .pos-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px; }
        .pos-question { font-size: 12px; line-height: 1.4; flex: 1; margin-right: 8px; }
        .pos-side { padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
        .pos-side.yes { background: #238636; }
        .pos-side.no { background: #da3633; }
        .pos-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
        .pos-stat { text-align: center; }
        .pos-stat-label { font-size: 9px; color: #8b949e; text-transform: uppercase; }
        .pos-stat-value { font-size: 13px; font-weight: 600; margin-top: 2px; }
        
        /* Trade Item */
        .trade-list { display: flex; flex-direction: column; gap: 6px; }
        .trade-item { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; display: flex; justify-content: space-between; align-items: center; }
        .trade-item.entry { border-left: 3px solid #238636; }
        .trade-item.win { border-left: 3px solid #3fb950; }
        .trade-item.loss { border-left: 3px solid #f85149; }
        .trade-info { flex: 1; }
        .trade-question { font-size: 12px; margin-bottom: 3px; }
        .trade-meta { font-size: 10px; color: #8b949e; }
        .trade-result { text-align: right; }
        .trade-pnl { font-size: 14px; font-weight: 600; }
        .trade-pnl.pos { color: #3fb950; }
        .trade-pnl.neg { color: #f85149; }
        .trade-type { font-size: 10px; color: #8b949e; }
        
        /* Analysis Item */
        .analysis-item { background: #21262d; border-radius: 6px; padding: 10px 12px; margin-bottom: 6px; }
        .analysis-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .analysis-question { font-size: 12px; flex: 1; margin-right: 8px; }
        .analysis-decision { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
        .analysis-decision.trade { background: #238636; }
        .analysis-decision.skip { background: #30363d; color: #8b949e; }
        .analysis-stats { display: flex; gap: 16px; font-size: 11px; color: #8b949e; }
        .analysis-stat { display: flex; align-items: center; gap: 4px; }
        .analysis-stat strong { color: #e6edf3; }
        .analysis-reason { font-size: 10px; color: #8b949e; margin-top: 6px; font-style: italic; }
        
        /* Market Card */
        .market-card { cursor: pointer; }
        .market-card:hover { border-color: #58a6ff; }
        .market-header { display: flex; gap: 10px; margin-bottom: 8px; }
        .market-img { width: 40px; height: 40px; border-radius: 6px; background: #30363d; object-fit: cover; flex-shrink: 0; }
        .market-question { font-size: 12px; line-height: 1.3; }
        .market-footer { display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid #30363d; }
        .price-bar { flex: 1; margin-right: 12px; }
        .price-labels { display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 4px; }
        .price-yes { color: #3fb950; font-weight: 600; }
        .price-no { color: #f85149; }
        .price-track { height: 6px; background: #f8514920; border-radius: 3px; overflow: hidden; }
        .price-fill { height: 100%; background: linear-gradient(90deg, #3fb950, #2ea043); border-radius: 3px; transition: width 0.3s; }
        .market-vol { font-size: 10px; color: #8b949e; }
        
        .empty { text-align: center; padding: 40px 20px; color: #8b949e; }
        .empty-icon { font-size: 36px; margin-bottom: 10px; }
        .empty-text { font-size: 13px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo"><span class="b1">Battle</span><span class="b2">Bot</span></div>
        <div class="header-info">
            <div class="status">
                <div id="dot" class="dot live"></div>
                <span id="connStatus">Connected</span>
            </div>
            <span id="runtime" class="runtime">0h 0m 0s</span>
            <div id="modeBadge" class="badge dry">DRY RUN</div>
            <div id="wsBadge" class="badge" style="background:#666;margin-left:8px">WS: --</div>
            <div id="aiBadge" class="badge" style="background:#666;margin-left:8px">AI: --</div>
        </div>
    </div>
    
    <div class="tabs">
        <button class="tab active" data-tab="portfolio">Portfolio</button>
        <button class="tab" data-tab="activity">Activity</button>
        <button class="tab" data-tab="markets">Markets</button>
    </div>
    
    <div class="content">
        <!-- PORTFOLIO TAB -->
        <div id="portfolio" class="tab-content active">
            <div class="stats-section">
                <div class="stats-title">Account</div>
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-label">Available</div>
                        <div id="available" class="stat-value neu">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">At Risk</div>
                        <div id="atRisk" class="stat-value neu">$0.00</div>
                        <div id="positionsOpen" class="stat-sub">0 positions</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Total Value</div>
                        <div id="totalValue" class="stat-value neu">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Return</div>
                        <div id="returnPct" class="stat-value neu">0.00%</div>
                    </div>
                </div>
            </div>
            
            <div class="stats-section">
                <div class="stats-title">Performance</div>
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-label">Realized P&L</div>
                        <div id="realizedPnl" class="stat-value neu">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Unrealized P&L</div>
                        <div id="unrealizedPnl" class="stat-value neu">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Win Rate</div>
                        <div id="winRate" class="stat-value neu">0%</div>
                        <div id="winLoss" class="stat-sub">0W / 0L</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Best Trade</div>
                        <div id="bestTrade" class="stat-value pos">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Worst Trade</div>
                        <div id="worstTrade" class="stat-value neg">$0.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Total Trades</div>
                        <div id="totalTrades" class="stat-value neu">0</div>
                        <div id="entryExit" class="stat-sub">0 entries / 0 exits</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <div class="section-title">Active Positions</div>
                    <span id="posCount" class="section-count">0</span>
                </div>
                <div id="positions" class="cards"></div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <div class="section-title">Recent Trades</div>
                    <span id="recentTradesCount" class="section-count">0</span>
                </div>
                <div id="recentTrades" class="trade-list"></div>
            </div>
        </div>
        
        <!-- ACTIVITY TAB -->
        <div id="activity" class="tab-content">
            <div class="section">
                <div class="section-header">
                    <div class="section-title">AI Analyses</div>
                    <span id="analysesCount" class="section-count">0</span>
                </div>
                <div id="analyses"></div>
            </div>
        </div>
        
        <!-- MARKETS TAB -->
        <div id="markets" class="tab-content">
            <div class="section">
                <div class="section-header">
                    <div class="section-title">Monitored Markets</div>
                    <span id="marketsCount" class="section-count">0</span>
                </div>
                <div id="marketsList" class="cards"></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let data = { stats: {}, positions: [], trades: [], analyses: [], monitored: [] };
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('dot').className = 'dot live';
                document.getElementById('connStatus').textContent = 'Connected';
            };
            
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.stats) data.stats = msg.stats;
                if (msg.positions) data.positions = msg.positions;
                if (msg.trades) data.trades = msg.trades;
                if (msg.analyses) data.analyses = msg.analyses;
                if (msg.monitored) data.monitored = msg.monitored;
                render();
            };
            
            ws.onclose = () => {
                document.getElementById('dot').className = 'dot off';
                document.getElementById('connStatus').textContent = 'Reconnecting...';
                setTimeout(connect, 2000);
            };
        }
        
        function render() {
            const s = data.stats;
            
            // Header
            document.getElementById('runtime').textContent = s.runtime || '0h 0m 0s';
            document.getElementById('modeBadge').textContent = s.dry_run ? 'DRY RUN' : 'LIVE';
            document.getElementById('modeBadge').className = `badge ${s.dry_run ? 'dry' : 'live'}`;
            
            // WebSocket status
            const wsBadge = document.getElementById('wsBadge');
            if (s.ws_connected) {
                wsBadge.textContent = `WS: ${s.price_updates || 0}`;
                wsBadge.style.background = '#22c55e';
            } else {
                wsBadge.textContent = 'WS: OFF';
                wsBadge.style.background = '#666';
            }
            
            // AI status
            const aiBadge = document.getElementById('aiBadge');
            if (s.ai_available) {
                aiBadge.textContent = `AI: ${s.ai_successes || 0}/${s.ai_calls || 0}`;
                aiBadge.style.background = '#8b5cf6';
            } else {
                aiBadge.textContent = 'AI: OFF';
                aiBadge.style.background = '#666';
            }
            
            // Account stats
            document.getElementById('available').textContent = `$${(s.available || 0).toFixed(2)}`;
            document.getElementById('atRisk').textContent = `$${(s.at_risk || 0).toFixed(2)}`;
            document.getElementById('positionsOpen').textContent = `${s.open_positions || 0} positions`;
            
            const tv = s.total_value || s.initial_bankroll || 1000;
            document.getElementById('totalValue').textContent = `$${tv.toFixed(2)}`;
            
            const rp = s.return_pct || 0;
            const rpEl = document.getElementById('returnPct');
            rpEl.textContent = `${rp >= 0 ? '+' : ''}${rp.toFixed(2)}%`;
            rpEl.className = `stat-value ${rp > 0 ? 'pos' : rp < 0 ? 'neg' : 'neu'}`;
            
            // Performance stats
            const realized = s.realized_pnl || 0;
            const realizedEl = document.getElementById('realizedPnl');
            realizedEl.textContent = `${realized >= 0 ? '+' : ''}$${realized.toFixed(2)}`;
            realizedEl.className = `stat-value ${realized > 0 ? 'pos' : realized < 0 ? 'neg' : 'neu'}`;
            
            const unrealized = s.unrealized_pnl || 0;
            const unrealizedEl = document.getElementById('unrealizedPnl');
            unrealizedEl.textContent = `${unrealized >= 0 ? '+' : ''}$${unrealized.toFixed(2)}`;
            unrealizedEl.className = `stat-value ${unrealized > 0 ? 'pos' : unrealized < 0 ? 'neg' : 'neu'}`;
            
            document.getElementById('winRate').textContent = `${((s.win_rate || 0) * 100).toFixed(0)}%`;
            document.getElementById('winLoss').textContent = `${s.winning_trades || 0}W / ${s.losing_trades || 0}L`;
            
            document.getElementById('bestTrade').textContent = `+$${(s.best_trade || 0).toFixed(2)}`;
            document.getElementById('worstTrade').textContent = `$${(s.worst_trade || 0).toFixed(2)}`;
            
            document.getElementById('totalTrades').textContent = (s.total_entries || 0) + (s.total_exits || 0);
            document.getElementById('entryExit').textContent = `${s.total_entries || 0} entries / ${s.total_exits || 0} exits`;
            
            renderPositions();
            renderTrades();
            renderAnalyses();
            renderMarkets();
        }
        
        function renderPositions() {
            const el = document.getElementById('positions');
            const pos = data.positions || [];
            document.getElementById('posCount').textContent = pos.length;
            
            if (!pos.length) {
                el.innerHTML = '<div class="empty"><div class="empty-icon">📭</div><div class="empty-text">No active positions</div></div>';
                return;
            }
            
            el.innerHTML = pos.map(p => {
                const pnl = p.unrealized_pnl || 0;
                const pnlClass = pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : 'neu';
                return `
                    <div class="card">
                        <div class="pos-header">
                            <div class="pos-question">${esc(p.question)}</div>
                            <span class="pos-side ${p.side?.toLowerCase()}">${p.side}</span>
                        </div>
                        <div class="pos-grid">
                            <div class="pos-stat">
                                <div class="pos-stat-label">Entry</div>
                                <div class="pos-stat-value">${((p.entry_price || 0) * 100).toFixed(0)}¢</div>
                            </div>
                            <div class="pos-stat">
                                <div class="pos-stat-label">Current</div>
                                <div class="pos-stat-value">${((p.current_price || p.entry_price || 0) * 100).toFixed(0)}¢</div>
                            </div>
                            <div class="pos-stat">
                                <div class="pos-stat-label">Size</div>
                                <div class="pos-stat-value">$${(p.size || 0).toFixed(2)}</div>
                            </div>
                            <div class="pos-stat">
                                <div class="pos-stat-label">P&L</div>
                                <div class="pos-stat-value ${pnlClass}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function renderTrades() {
            const el = document.getElementById('recentTrades');
            const trades = (data.trades || []).slice(0, 10);
            document.getElementById('recentTradesCount').textContent = trades.length;
            
            if (!trades.length) {
                el.innerHTML = '<div class="empty"><div class="empty-icon">📜</div><div class="empty-text">No trades yet</div></div>';
                return;
            }
            
            el.innerHTML = trades.map(t => {
                const isEntry = t.action === 'ENTRY';
                const pnl = t.pnl || 0;
                const itemClass = isEntry ? 'entry' : (pnl >= 0 ? 'win' : 'loss');
                const pnlClass = pnl >= 0 ? 'pos' : 'neg';
                
                return `
                    <div class="trade-item ${itemClass}">
                        <div class="trade-info">
                            <div class="trade-question">${esc((t.question || '').substring(0, 50))}...</div>
                            <div class="trade-meta">
                                ${t.side} @ ${((t.price || t.entry_price || 0) * 100).toFixed(0)}¢ · 
                                $${(t.size || 0).toFixed(2)} · 
                                ${new Date(t.timestamp).toLocaleTimeString()}
                                ${t.reason ? ` · ${t.reason}` : ''}
                            </div>
                        </div>
                        <div class="trade-result">
                            ${isEntry 
                                ? '<div class="trade-pnl neu">—</div><div class="trade-type">ENTRY</div>'
                                : `<div class="trade-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</div><div class="trade-type">EXIT</div>`
                            }
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function renderAnalyses() {
            const el = document.getElementById('analyses');
            const analyses = (data.analyses || []).slice(0, 20);
            document.getElementById('analysesCount').textContent = analyses.length;
            
            if (!analyses.length) {
                el.innerHTML = '<div class="empty"><div class="empty-icon">🧠</div><div class="empty-text">No analyses yet</div></div>';
                return;
            }
            
            el.innerHTML = analyses.map(a => {
                const hasAiData = a.ai_probability !== null && a.ai_probability !== undefined;
                const keyReasons = (a.key_reasons || []).slice(0, 2).map(r => `• ${esc(r)}`).join('<br>');
                const infoQuality = a.info_quality ? `<span style="color:${a.info_quality === 'high' ? '#22c55e' : a.info_quality === 'medium' ? '#f59e0b' : '#ef4444'}">${a.info_quality}</span>` : '';
                const latency = a.latency_ms ? `<span style="color:#8b949e;font-size:10px">${a.latency_ms}ms</span>` : '';
                
                return `
                <div class="analysis-item">
                    <div class="analysis-header">
                        <div class="analysis-question">${esc(a.question)}</div>
                        <span class="analysis-decision ${a.decision === 'TRADE' ? 'trade' : 'skip'}">${a.decision}</span>
                    </div>
                    <div class="analysis-stats">
                        ${hasAiData ? `
                        <div class="analysis-stat">AI: <strong>${(a.ai_probability * 100).toFixed(0)}%</strong></div>
                        <div class="analysis-stat">Market: <strong>${((a.market_price || 0) * 100).toFixed(0)}%</strong></div>
                        <div class="analysis-stat">Edge: <strong style="color:${a.edge >= 0.03 ? '#22c55e' : '#8b949e'}">${((a.edge || 0) * 100).toFixed(1)}%</strong></div>
                        <div class="analysis-stat">Conf: <strong>${((a.confidence || 0) * 100).toFixed(0)}%</strong></div>
                        ${infoQuality ? `<div class="analysis-stat">Quality: ${infoQuality}</div>` : ''}
                        ${latency}
                        ` : `<div class="analysis-stat" style="color:#ef4444">AI unavailable</div>`}
                    </div>
                    ${a.reason ? `<div class="analysis-reason">${esc(a.reason)}</div>` : ''}
                    ${keyReasons ? `<div class="analysis-reason" style="font-style:normal;color:#8b949e;margin-top:4px">${keyReasons}</div>` : ''}
                </div>
            `}).join('');
        }
        
        function renderMarkets() {
            const el = document.getElementById('marketsList');
            const markets = data.monitored || [];
            document.getElementById('marketsCount').textContent = markets.length;
            
            if (!markets.length) {
                el.innerHTML = '<div class="empty"><div class="empty-icon">🔍</div><div class="empty-text">Loading markets...</div></div>';
                return;
            }
            
            el.innerHTML = markets.map(m => `
                <div class="card market-card" onclick="window.open('${m.url}', '_blank')">
                    <div class="market-header">
                        ${m.image ? `<img class="market-img" src="${m.image}" alt="">` : '<div class="market-img"></div>'}
                        <div class="market-question">${esc(m.question)}</div>
                    </div>
                    <div class="market-footer">
                        <div class="price-bar">
                            <div class="price-labels">
                                <span class="price-yes">${m.price_pct || 50}% Yes</span>
                                <span class="price-no">${100 - (m.price_pct || 50)}% No</span>
                            </div>
                            <div class="price-track">
                                <div class="price-fill" style="width: ${m.price_pct || 50}%"></div>
                            </div>
                        </div>
                        <div class="market-vol">${m.volume_display || '$0'}</div>
                    </div>
                </div>
            `).join('');
        }
        
        function esc(str) {
            const div = document.createElement('div');
            div.textContent = str || '';
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
        
        connect();
    </script>
</body>
</html>
'''


async def main():
    port = int(os.getenv('PORT', 8080))
    bot = BattleBot(port=port)
    await bot.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
