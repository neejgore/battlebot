#!/usr/bin/env python3
"""Battle-Bot for Kalshi - CFTC-regulated prediction market.

Legal for US residents including California.
Uses same AI strategy as Polymarket version.
"""

import asyncio
import json
import os
from datetime import datetime
from aiohttp import web
import httpx
from dotenv import load_dotenv

from logic.ai_signal import AISignalGenerator, AISignalResult
from logic.calibration import CalibrationEngine, CalibrationResult
from logic.risk_engine import RiskEngine, RiskLimits
from data.database import TelemetryDB
from services.kalshi_client import KalshiClient, parse_kalshi_market

load_dotenv()


class KalshiBattleBot:
    """BattleBot for Kalshi prediction markets."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._app = None
        self._runner = None
        self._websockets: set[web.WebSocketResponse] = set()
        
        # Storage directory for persistence
        self._storage_dir = os.getenv('STORAGE_DIR', 'storage')
        os.makedirs(self._storage_dir, exist_ok=True)
        
        # Config from env
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.initial_bankroll = float(os.getenv('INITIAL_BANKROLL', 1000))
        self.min_edge = float(os.getenv('MIN_EDGE', 0.05))  # 5% min edge (lower for dry run testing)
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.5))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 50))
        self.kelly_fraction = float(os.getenv('FRACTIONAL_KELLY', 0.1))
        self.simulate_prices = os.getenv('SIMULATE_PRICES', 'false').lower() == 'true'
        
        # Kalshi client
        use_demo = os.getenv('KALSHI_USE_DEMO', 'false').lower() == 'true'
        self._kalshi = KalshiClient(use_demo=use_demo)
        
        # Market data
        self._markets: dict[str, dict] = {}
        self._monitored: dict[str, dict] = {}
        self._target_series: list[str] = []  # Cached target series tickers (discovered once)
        
        # Trading state (persisted)
        self._state_file = f"{self._storage_dir}/kalshi_state.json"
        self._positions: dict[str, dict] = {}
        self._pending_orders: dict[str, dict] = {}  # Orders placed but not yet filled
        self._trades: list[dict] = []
        self._analyses: list[dict] = []
        self._load_state()
        
        self._running = False
        self._last_analysis: dict[str, datetime] = {}
        self._last_analysis_price: dict[str, float] = {}
        self._analysis_cooldown = 1800  # 30 minutes
        self._price_change_threshold = 0.02  # Re-analyze on 2% move
        self._start_time = None
        
        # Price updates counter (no WebSocket for Kalshi, use polling)
        self._price_update_count = 0
        
        # AI Signal Generator
        self._ai_generator = AISignalGenerator()
        self._ai_calls = 0
        self._ai_successes = 0
        
        # Database
        self._db = TelemetryDB(f"{self._storage_dir}/kalshi.db")
        self._db_connected = False
        
        # Calibration
        self._calibration: CalibrationEngine = None
        
        # Risk Engine
        self._risk_limits = RiskLimits(
            max_daily_drawdown=0.15,
            max_position_size=self.max_position_size,
            max_percent_bankroll_per_market=0.10,
            max_total_open_risk=0.30,
            max_positions=20,  # Allow more concurrent positions for tracking
            profit_take_pct=0.03,
            stop_loss_pct=0.03,
            time_stop_hours=720,
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
        # Check if state should be cleared (for recovering from phantom positions)
        if os.getenv('CLEAR_STATE', '').lower() == 'true':
            print("[State] CLEAR_STATE=true - clearing all positions, orders, and trades")
            self._positions = {}
            self._pending_orders = {}
            self._trades = []
            self._save_state()
            return
            
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._pending_orders = state.get('pending_orders', {})
                    self._trades = state.get('trades', [])
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._pending_orders)} pending orders, {len(self._trades)} trades")
            elif os.path.exists(self._state_file + '.backup'):
                print(f"[State] Main file missing, loading from backup...")
                with open(self._state_file + '.backup', 'r') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._pending_orders = state.get('pending_orders', {})
                    self._trades = state.get('trades', [])
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._pending_orders)} pending orders, {len(self._trades)} trades from backup")
        except json.JSONDecodeError as e:
            print(f"[State] WARNING: Corrupted state file, starting fresh: {e}")
            self._positions = {}
            self._pending_orders = {}
            self._trades = []
        except Exception as e:
            print(f"[State] Failed to load: {e}")
            self._positions = {}
            self._pending_orders = {}
            self._trades = []
    
    def _save_state(self):
        """Save positions and trades to disk with atomic write."""
        try:
            os.makedirs(os.path.dirname(self._state_file) or '.', exist_ok=True)
            
            state_data = {
                'positions': self._positions,
                'pending_orders': self._pending_orders,
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
    
    async def _sync_pending_orders_on_startup(self):
        """Check status of pending orders from previous session."""
        filled = 0
        canceled = 0
        still_pending = 0
        
        for order_id, order in list(self._pending_orders.items()):
            try:
                result = await self._kalshi.get_order(order_id)
                order_data = result.get('order', {})
                status = order_data.get('status', '').lower()
                
                if status == 'executed':
                    # Order filled while bot was down - create position
                    fill_price = order_data.get('average_fill_price', order['entry_price'] * 100) / 100
                    fill_count = order_data.get('filled_count', order['contracts'])
                    
                    pos_id = order.get('id', f"pos_{order_id}")
                    pos = {
                        'id': pos_id,
                        'order_id': order_id,
                        'market_id': order['market_id'],
                        'question': order.get('question', ''),
                        'side': order['side'],
                        'size': order['size'],
                        'entry_price': fill_price,
                        'current_price': fill_price,
                        'contracts': fill_count,
                        'ai_probability': order.get('ai_probability', 0.5),
                        'edge': order.get('edge', 0),
                        'confidence': order.get('confidence', 0),
                        'entry_time': order.get('placed_time', datetime.utcnow().isoformat()),
                        'unrealized_pnl': 0.0,
                    }
                    self._positions[pos_id] = pos
                    self._pending_orders.pop(order_id)
                    filled += 1
                    print(f"[Startup] Order {order_id[:8]}... was FILLED @ {fill_price*100:.0f}¢")
                    
                elif status in ('canceled', 'cancelled'):
                    self._pending_orders.pop(order_id)
                    canceled += 1
                    print(f"[Startup] Order {order_id[:8]}... was CANCELED")
                    
                else:  # still resting
                    # Cancel stale limit orders (older than 1 hour)
                    placed_time_str = order.get('placed_time', '')
                    is_stale = False
                    if placed_time_str:
                        try:
                            placed_time = datetime.fromisoformat(placed_time_str.replace('Z', '+00:00'))
                            age_hours = (datetime.now(placed_time.tzinfo) - placed_time).total_seconds() / 3600
                            is_stale = age_hours > 1
                        except:
                            is_stale = True  # If we can't parse time, assume stale
                    
                    if is_stale:
                        try:
                            await self._kalshi.cancel_order(order_id)
                            self._pending_orders.pop(order_id)
                            canceled += 1
                            print(f"[Startup] Order {order_id[:8]}... CANCELED (stale limit order)")
                        except Exception as cancel_err:
                            still_pending += 1
                            print(f"[Startup] Order {order_id[:8]}... still PENDING (cancel failed: {cancel_err})")
                    else:
                        still_pending += 1
                        print(f"[Startup] Order {order_id[:8]}... still PENDING")
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                if '404' in str(e).lower():
                    # Order doesn't exist - remove it
                    self._pending_orders.pop(order_id, None)
                    print(f"[Startup] Order {order_id[:8]}... NOT FOUND (removed)")
                else:
                    print(f"[Startup] Error checking order {order_id[:8]}...: {e}")
        
        self._save_state()
        print(f"[Startup] Sync complete: {filled} filled, {canceled} canceled, {still_pending} still pending")
    
    def _get_stats(self) -> dict:
        """Calculate all stats from current state."""
        at_risk = sum(p['size'] for p in self._positions.values())
        realized_pnl = sum(t.get('pnl', 0) for t in self._trades if t.get('action') == 'EXIT')
        available = self.initial_bankroll - at_risk + realized_pnl
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in self._positions.values())
        total_value = available + at_risk + unrealized_pnl
        
        exits = [t for t in self._trades if t.get('action') == 'EXIT']
        winning = len([t for t in exits if t.get('pnl', 0) > 0])
        losing = len([t for t in exits if t.get('pnl', 0) < 0])
        
        pnls = [t.get('pnl', 0) for t in exits]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        total_closed = winning + losing
        win_rate = winning / total_closed if total_closed > 0 else 0
        return_pct = ((total_value - self.initial_bankroll) / self.initial_bankroll) * 100

        runtime = "0h 0m 0s"
        if self._start_time:
            delta = datetime.utcnow() - self._start_time
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            seconds = int(delta.total_seconds() % 60)
            runtime = f"{hours}h {minutes}m {seconds}s"

        return {
            'platform': 'kalshi',
            'dry_run': self.dry_run,
            'running': self._running,
            'runtime': runtime,
            'initial_bankroll': self.initial_bankroll,
            'available': available,
            'at_risk': at_risk,
            'total_value': total_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'return_pct': return_pct,
            'total_entries': len([t for t in self._trades if t.get('action') == 'ENTRY']),
            'total_exits': len(exits),
            'winning_trades': winning,
            'losing_trades': losing,
            'breakeven_trades': len([t for t in exits if t.get('pnl', 0) == 0]),
            'win_rate': win_rate,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'open_positions': len(self._positions),
            'markets_monitored': len(self._monitored),
            'analyses_count': len(self._analyses),
            'ws_connected': False,  # Kalshi uses polling, not WebSocket
            'price_updates': self._price_update_count,
            'ai_available': self._ai_generator._client is not None,
            'ai_calls': self._ai_calls,
            'ai_successes': self._ai_successes,
            'min_edge': self.min_edge,
            'min_confidence': self.min_confidence,
            'max_position_size': self.max_position_size,
            'kelly_fraction': self.kelly_fraction,
            'kill_switch': self._risk_engine.daily_stats.kill_switch_triggered,
            'daily_drawdown': self._risk_engine.daily_stats.current_drawdown_pct,
            'exposure_ratio': at_risk / self.initial_bankroll,
            'trading_allowed': self._risk_engine.is_trading_allowed,
        }
    
    async def start(self):
        """Start the Kalshi bot and dashboard."""
        # Setup web app
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
        
        # Connect database
        await self._db.connect()
        self._db_connected = True
        print("[DB] Telemetry database connected")
        
        # Initialize calibration
        self._calibration = CalibrationEngine(self._db)
        
        # Print startup banner
        mode = "DRY RUN" if self.dry_run else "LIVE"
        print(f"\n{'='*50}")
        print(f"KALSHI BATTLE-BOT [{mode}]")
        print(f"{'='*50}")
        print(f"Platform: Kalshi (CFTC-regulated, US legal)")
        print(f"Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Min Edge: {self.min_edge*100:.1f}%")
        print(f"Max Position: ${self.max_position_size:,.2f}")
        if self.simulate_prices:
            print(f"\n⚠️  SIMULATE_PRICES=true (testing mode)")
        print(f"\nDashboard: http://localhost:{self.port}")
        print("Press Ctrl+C to stop\n")
        
        # Sync pending orders from previous session (LIVE mode only)
        if not self.dry_run and self._pending_orders:
            print(f"[Startup] Checking {len(self._pending_orders)} pending orders from previous session...")
            await self._sync_pending_orders_on_startup()
        
        # Start background tasks
        asyncio.create_task(self._market_loop())
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._price_refresh_loop())
        if not self.dry_run:
            asyncio.create_task(self._position_sync_loop())
        
        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()
    
    async def _shutdown(self):
        """Clean shutdown."""
        self._running = False
        self._save_state()
        if self._db_connected:
            await self._db.close()
        if self._runner:
            await self._runner.cleanup()
        print("\n[Shutdown] Kalshi BattleBot stopped")
    
    async def _market_loop(self):
        """Fetch markets from Kalshi periodically."""
        while self._running:
            try:
                await self._fetch_markets()
                await self._select_markets()
            except Exception as e:
                print(f"[Market Error] {e}")
            await asyncio.sleep(60)  # Refresh every minute
    
    async def _fetch_markets(self):
        """Fetch ALL useful markets from Kalshi - no artificial limits.
        
        Strategy:
        1. Discover all series ONCE and cache target categories
        2. Fetch markets from cached target series (politics, economics, etc.)
        3. Also fetch general non-MVE markets with full pagination
        4. Deduplicate and sort by liquidity
        """
        try:
            all_markets = []
            
            # Categories we want (excludes Sports, Entertainment, Mentions, Social)
            target_categories = [
                'Politics', 'Economics', 'Financials', 'Elections', 
                'Climate and Weather', 'Science and Technology', 
                'Health', 'World', 'Companies', 'Crypto', 'Transportation', 'Education'
            ]
            
            # STEP 1: Discover series (only on first run, then use cache)
            if not self._target_series:
                try:
                    series_response = await self._kalshi.get_series_list()
                    all_series = series_response.get('series', [])
                    print(f"[Discovery] Found {len(all_series)} total series on Kalshi")
                    
                    # Count by category
                    by_category = {}
                    for s in all_series:
                        cat = s.get('category', 'Unknown')
                        by_category[cat] = by_category.get(cat, 0) + 1
                        
                        # Collect target series tickers
                        if cat in target_categories:
                            ticker = s.get('ticker')
                            if ticker:
                                self._target_series.append(ticker)
                    
                    print(f"[Categories]")
                    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                        marker = " <-- TARGET" if cat in target_categories else ""
                        print(f"  {cat}: {count}{marker}")
                    
                    print(f"[Target Series] {len(self._target_series)} series in target categories (cached for future refreshes)")
                    
                except Exception as e:
                    print(f"[Discovery Error] {e}")
            else:
                print(f"[Refresh] Using cached {len(self._target_series)} target series")
            
            # STEP 2: Fetch markets from target series
            if self._target_series:
                print(f"[Fetching] Markets from {len(self._target_series)} target series...")
                series_with_markets = 0
                total_from_series = 0
                
                for i, series_ticker in enumerate(self._target_series):
                    try:
                        # Rate limiting - brief pause between calls
                        if i > 0 and i % 10 == 0:
                            await asyncio.sleep(0.5)
                        else:
                            await asyncio.sleep(0.1)
                        
                        result = await self._kalshi.get_markets(
                            status='open',
                            series_ticker=series_ticker,
                            limit=200
                        )
                        markets = result.get('markets', [])
                        if markets:
                            series_with_markets += 1
                            total_from_series += len(markets)
                            all_markets.extend(markets)
                            
                    except Exception as e:
                        if '429' in str(e):
                            print(f"[Rate Limited] Pausing for 3 seconds...")
                            await asyncio.sleep(3)
                        continue
                    
                    # Progress update every 100 series
                    if (i + 1) % 100 == 0:
                        print(f"[Progress] {i + 1}/{len(self._target_series)} series checked, {total_from_series} markets found")
                
                print(f"[Series Fetch Complete] {series_with_markets} series had open markets, {total_from_series} total")
            
            # STEP 3: Paginated general fetch - get ALL non-MVE markets
            print(f"[General Fetch] Fetching all non-MVE markets with pagination...")
            cursor = None
            pages_fetched = 0
            total_general = 0
            
            while True:
                try:
                    await asyncio.sleep(0.2)
                    result = await self._kalshi.get_markets(
                        status='open',
                        limit=1000,  # Max per page
                        cursor=cursor,
                        exclude_mve=True
                    )
                    general = result.get('markets', [])
                    if not general:
                        break
                        
                    all_markets.extend(general)
                    total_general += len(general)
                    pages_fetched += 1
                    
                    cursor = result.get('cursor')
                    if not cursor:
                        break
                        
                    # Safety limit - don't fetch forever
                    if pages_fetched >= 20:
                        print(f"[General Fetch] Reached page limit")
                        break
                        
                except Exception as e:
                    if '429' in str(e):
                        print(f"[Rate Limited] Pausing for 3 seconds...")
                        await asyncio.sleep(3)
                        continue
                    print(f"[General fetch error] {e}")
                    break
            
            print(f"[General Fetch Complete] {total_general} markets from {pages_fetched} pages")
            
            # STEP 4: Deduplicate by ticker
            seen = set()
            unique_markets = []
            for m in all_markets:
                ticker = m.get('ticker')
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    unique_markets.append(m)
            all_markets = unique_markets
            
            print(f"[Total] {len(all_markets)} unique markets after deduplication")
            
            # Sort by open interest (better for holding positions to resolution)
            all_markets.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
            
            # Log top markets by open interest
            top_10 = all_markets[:10]
            if top_10:
                print(f"[Top 10 by Open Interest]")
                for m in top_10:
                    vol = m.get('volume', 0)
                    oi = m.get('open_interest', 0)
                    ticker = m.get('ticker', '')[:50]
                    cat = m.get('category', '')
                    print(f"  oi=${oi:,} vol=${vol:,} [{cat}] {ticker}")
            
            # Group by event and show event-level volume
            event_volumes = {}
            for m in all_markets:
                event = m.get('event_ticker', 'unknown')
                vol = m.get('volume', 0) or 0
                event_volumes[event] = event_volumes.get(event, 0) + vol
            
            top_events = sorted(event_volumes.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"[Top Events by Volume]")
            for event, vol in top_events:
                print(f"  ${vol:,} | {event}")
            
            # Process markets - NO category filtering, let volume/spread filters decide
            category_counts = {}
            
            for m in all_markets:
                try:
                    market = parse_kalshi_market(m)
                    if market['id'] and market['id'] not in self._markets:
                        self._markets[market['id']] = market
                        category = m.get('category', 'unknown')
                        category_counts[category] = category_counts.get(category, 0) + 1
                except Exception as e:
                    continue
            
            added = len(self._markets)
            categories_str = ', '.join(f"{k}:{v}" for k, v in sorted(category_counts.items()))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {added} markets total")
            if categories_str:
                print(f"[Categories] {categories_str}")
        except Exception as e:
            print(f"[Kalshi API Error] {e}")
    
    def _is_combo_market(self, market: dict) -> bool:
        """Detect multi-leg combo markets that have poor liquidity.
        
        NOTE: Disabled for now - relying on volume filter instead.
        Kalshi's API returns many MULTIGAME markets, but some may have
        decent liquidity. Let volume filtering handle market quality.
        """
        # Disabled - let volume filter handle this
        # The volume filter (MIN_VOLUME_24H) is a better proxy for liquidity
        return False
    
    async def _select_markets(self):
        """Filter markets using eligibility criteria."""
        eligible = []
        rejection_counts = {
            'no_end_date': 0, 'too_far_out': 0, 'low_oi': 0,
            'wide_spread': 0, 'extreme_price': 0, 'low_liquidity': 0,
            'combo_market': 0,
        }
        
        # Minimum volume threshold - only trade markets with real activity
        # Set to 0 by default to allow markets through, rely on spread filter
        min_volume = float(os.getenv('MIN_VOLUME_24H', '0'))
        
        # Max days to resolution - political/economic markets often have longer timeframes
        # Default 180 days (6 months) to capture most active markets
        max_days = int(os.getenv('MAX_DAYS_TO_RESOLUTION', '365'))
        
        for m in self._markets.values():
            # Must have end date
            if not m.get('end_date'):
                rejection_counts['no_end_date'] += 1
                continue
            
            # Time to resolution check
            try:
                end_date = datetime.fromisoformat(m['end_date'].replace('Z', '+00:00'))
                days_to_resolution = (end_date.replace(tzinfo=None) - datetime.utcnow()).days
                if days_to_resolution > max_days:
                    rejection_counts['too_far_out'] += 1
                    continue
                if days_to_resolution < 0:
                    continue
            except:
                pass
            
            # Minimum open interest filter - require real liquidity
            # OI = committed capital, better than volume for holding positions
            oi = m.get('open_interest', 0) or 0
            min_oi = int(os.getenv('MIN_OPEN_INTEREST', '10'))  # Default 10 contracts
            if oi < min_oi:
                rejection_counts['low_oi'] += 1
                # Debug: log first few rejections
                if rejection_counts['low_oi'] <= 3:
                    print(f"[Debug] Rejected for low_oi: {m.get('ticker', '')[:30]} oi={oi}")
                continue
            
            # Spread check - configurable, default 6 cents
            max_spread = float(os.getenv('MAX_SPREAD', '0.06'))
            if m.get('spread', max_spread) > max_spread:
                rejection_counts['wide_spread'] += 1
                continue
            
            # Price range (not too extreme)
            price = m.get('price', 0.5)
            if price < 0.05 or price > 0.95:
                rejection_counts['extreme_price'] += 1
                continue
            
            eligible.append(m)
        
        # Take top 15 by open interest (most liquid markets)
        eligible.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
        self._monitored = {m['id']: m for m in eligible[:15]}
        
        # Log top monitored markets
        if self._monitored:
            top_3 = list(self._monitored.values())[:3]
            print(f"[Monitoring] Top 3: {[m.get('question', '')[:30] + '...' for m in top_3]}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market eligibility: {len(eligible)} eligible / {len(self._markets)} total")
        
        if rejection_counts:
            reasons = [f"{k}={v}" for k, v in rejection_counts.items() if v > 0]
            if reasons:
                print(f"[Filter] Rejected: {', '.join(reasons)}")
    
    async def _trading_loop(self):
        """Main trading loop - analyze markets and enter positions."""
        await asyncio.sleep(5)  # Wait for initial market fetch
        
        while self._running:
            try:
                if not self._risk_engine.is_trading_allowed:
                    await asyncio.sleep(60)
                    continue
                
                for market_id, market in list(self._monitored.items()):
                    if len(self._positions) >= self._risk_limits.max_positions:
                        break
                    
                    # Check cooldown and price movement
                    last = self._last_analysis.get(market_id)
                    last_price = self._last_analysis_price.get(market_id, 0)
                    current_price = market.get('price', 0.5)
                    
                    price_moved = abs(current_price - last_price) >= self._price_change_threshold if last_price else False
                    
                    if last and not price_moved:
                        if (datetime.utcnow() - last).total_seconds() < self._analysis_cooldown:
                            continue
                    
                    await self._analyze_market(market)
                    self._last_analysis[market_id] = datetime.utcnow()
                    self._last_analysis_price[market_id] = current_price
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"[Trading Error] {e}")
            await asyncio.sleep(30)
    
    async def _analyze_market(self, market: dict):
        """Analyze market with AI and decide whether to trade."""
        market_id = market['id']
        question = market.get('question', '')[:50]
        current_price = market.get('price', 0.5)
        
        print(f"[AI] Analyzing: {question}...")
        self._ai_calls += 1
        
        # Get AI signal
        try:
            result = await self._ai_generator.generate_signal(
                market_question=market.get('question', ''),
                current_price=current_price,
                spread=market.get('spread', 0.02),
                resolution_rules=market.get('rules', '') or market.get('description', ''),
                volume_24h=market.get('volume_24h', 0),
            )
            
            # Check if AI call succeeded (result is AISignalResult with success flag)
            if not result or not result.success or not result.signal:
                error_msg = result.error if result else "No response"
                print(f"[AI] FAILED: {error_msg}")
                # Store failed analysis
                self._analyses.insert(0, {
                    'market_id': market_id,
                    'question': market.get('question', ''),
                    'market_price': current_price,
                    'ai_probability': None,
                    'confidence': None,
                    'edge': 0,
                    'side': None,
                    'decision': 'NO_TRADE',
                    'reason': f"AI failed: {error_msg}",
                    'timestamp': datetime.utcnow().isoformat(),
                })
                self._analyses = self._analyses[:50]
                await self._broadcast_update()
                return
            
            # Extract signal data from result
            signal = result.signal
            self._ai_successes += 1
            
        except Exception as e:
            print(f"[AI] FAILED: {e}")
            return
        
        # Calibrate probability (async call)
        try:
            cal_result = await self._calibration.calibrate(
                raw_prob=signal.raw_prob,
                market_price=current_price,
                confidence=signal.confidence,
                category=market.get('category'),
            )
            calibrated_prob = cal_result.calibrated_prob
            adjusted_prob = cal_result.adjusted_prob
            calibration_method = cal_result.method
        except Exception as e:
            print(f"[Calibration] Error: {e} - using raw probability")
            calibrated_prob = signal.raw_prob
            adjusted_prob = signal.raw_prob
            calibration_method = "error"
        
        # Determine trade side and edge using ACTUAL prices from Kalshi
        yes_price = market.get('yes_price', current_price)
        no_price = market.get('no_price', 1 - current_price)  # Fallback if not available
        
        if adjusted_prob > yes_price:
            # Bet YES: we think YES is more likely than the market price implies
            side = 'YES'
            edge = adjusted_prob - yes_price
            trade_prob = adjusted_prob  # Our belief in YES winning
            trade_price = yes_price     # Actual cost to buy YES
        else:
            # Bet NO: we think NO is more likely than the market implies
            side = 'NO'
            no_prob = 1 - adjusted_prob  # Our belief in NO winning
            edge = no_prob - no_price    # Edge = belief - cost
            trade_prob = no_prob         # Our belief in NO winning
            trade_price = no_price       # Actual cost to buy NO
        
        # Store analysis
        self._analyses.insert(0, {
            'market_id': market_id,
            'question': market.get('question', ''),
            'market_price': current_price,
            'ai_probability': signal.raw_prob,
            'calibrated_probability': calibrated_prob,
            'adjusted_probability': adjusted_prob,
            'confidence': signal.confidence,
            'edge': edge,
            'side': side,
            'decision': 'PENDING',
            'timestamp': datetime.utcnow().isoformat(),
            'key_reasons': signal.key_reasons[:3] if signal.key_reasons else [],
            'info_quality': signal.information_quality,
            'latency_ms': result.latency_ms,
            'calibration_method': calibration_method,
        })
        self._analyses = self._analyses[:50]
        
        # Check if we should trade
        should_trade = True
        reasons = []
        
        if edge < self.min_edge:
            should_trade = False
            reasons.append('LOW_EDGE')
        
        if signal.confidence < self.min_confidence:
            should_trade = False
            reasons.append('LOW_CONFIDENCE')
        
        if market_id in [p.get('market_id') for p in self._positions.values()]:
            should_trade = False
            reasons.append('ALREADY_IN_POSITION')
        
        if len(self._positions) >= self._risk_limits.max_positions:
            should_trade = False
            reasons.append('MAX_POSITIONS')
        
        # Log decision
        decision = 'TRADE' if should_trade else 'NO_TRADE'
        self._analyses[0]['decision'] = decision
        self._analyses[0]['reason'] = ', '.join(reasons) if reasons else 'CRITERIA_MET'
        
        if should_trade:
            # Calculate position size (async with proper params)
            print(f"[Debug] Calculating size: prob={trade_prob:.2f}, price={trade_price:.2f}, edge={edge:.2f}, conf={signal.confidence:.2f}")
            position_size = await self._risk_engine.calculate_position_size(
                adjusted_prob=trade_prob,
                market_price=trade_price,
                edge=edge,
                confidence=signal.confidence,
                market_id=market_id,
            )
            print(f"[Debug] Position size: ${position_size:.2f}")
            
            if position_size > 0:
                await self._enter_position(market, side, position_size, adjusted_prob, edge, signal.confidence)
                print(f"[AI] {question}... | ✓ TRADE | Edge: +{edge*100:.1f}% | Conf: {signal.confidence*100:.0f}%")
            else:
                print(f"[AI] {question}... | ✗ SKIP | Size: $0 (edge={edge:.2%}, conf={signal.confidence:.2%})")
        else:
            print(f"[AI] {question}... | ✗ SKIP | {', '.join(reasons)}")
        
        await self._broadcast_update()
    
    async def _enter_position(self, market: dict, side: str, size: float, prob: float, edge: float, confidence: float):
        """Enter a new position."""
        market_id = market['id']
        pos_id = f"pos_{int(datetime.utcnow().timestamp()*1000)}"
        
        # Use the correct price for the side we're trading
        if side.upper() == 'YES':
            entry_price = market.get('yes_price', market['price'])
        else:
            entry_price = market.get('no_price', 1 - market['price'])
        
        contracts = int(size / entry_price) if entry_price > 0 else 0
        
        # In LIVE mode, actually place the order on Kalshi
        order_id = None
        if not self.dry_run:
            try:
                if contracts < 1:
                    print(f"[Order] Size ${size:.2f} too small for 1 contract at {int(entry_price*100)}¢")
                    return
                
                # Use very aggressive limit orders (effectively market orders with price protection)
                # Cross the spread by 10¢ to sweep the order book and fill immediately
                base_price_cents = int(entry_price * 100)
                aggressive_price_cents = min(base_price_cents + 10, 95)  # Pay up to 10¢ more, cap at 95¢
                
                result = await self._kalshi.place_order(
                    ticker=market_id,
                    side=side.lower(),  # 'yes' or 'no'
                    count=contracts,
                    price=aggressive_price_cents,
                    order_type='limit',
                )
                order_id = result.get('order', {}).get('order_id')
                print(f"[LIVE ORDER] Placed: {contracts} {side} @ {aggressive_price_cents}¢ (mid: {base_price_cents}¢) | Order ID: {order_id}")
            except Exception as e:
                print(f"[Order Error] Failed to place order on Kalshi: {e}")
                return  # Don't record if order failed
        
        order_data = {
            'id': pos_id,
            'order_id': order_id,
            'market_id': market_id,
            'question': market.get('question', ''),
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'contracts': contracts,
            'ai_probability': prob,
            'edge': edge,
            'confidence': confidence,
            'placed_time': datetime.utcnow().isoformat(),
        }
        
        # In dry run mode, immediately treat as filled position
        if self.dry_run:
            pos = {
                **order_data,
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'entry_time': datetime.utcnow().isoformat(),
            }
            # Log to database
            if self._db_connected:
                try:
                    db_trade_id = await self._db.log_trade_entry(
                        decision_id=0,
                        market_id=market_id,
                        token_id=market.get('token_id', market_id),
                        entry_price=entry_price,
                        entry_side=side,
                        size=size,
                        raw_prob=prob,
                        adjusted_prob=prob,
                        edge=edge,
                        confidence=confidence,
                    )
                    pos['db_trade_id'] = db_trade_id
                except Exception as e:
                    print(f"[DB] Failed to log trade entry: {e}")
            self._positions[pos_id] = pos
            # Record trade for dry run
            trade = {
                'id': pos_id,
                'market_id': market_id,
                'question': market.get('question', ''),
                'action': 'ENTRY',
                'side': side,
                'price': entry_price,
                'size': size,
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._trades.insert(0, trade)
            print(f"[DRY RUN] ENTERED: {side} ${size:.2f} @ {int(entry_price*100)}¢ | {market.get('question', '')[:50]}...")
        else:
            # In LIVE mode, track as pending until we confirm fill from Kalshi
            self._pending_orders[order_id] = order_data
            # Record as pending order (not entry yet)
            trade = {
                'id': pos_id,
                'market_id': market_id,
                'question': market.get('question', ''),
                'action': 'ORDER_PLACED',
                'side': side,
                'price': entry_price,
                'size': size,
                'timestamp': datetime.utcnow().isoformat(),
                'order_id': order_id,
                'status': 'pending',
            }
            self._trades.insert(0, trade)
            print(f"[PENDING] Order {order_id} placed, waiting to fill...")
        
        self._save_state()
        await self._broadcast_update()
    
    async def _position_monitor_loop(self):
        """Monitor positions for exit conditions."""
        import random
        while self._running:
            try:
                for pos_id, pos in list(self._positions.items()):
                    market = self._markets.get(pos.get('market_id'))
                    if not market:
                        continue
                    
                    entry_price = pos['entry_price']
                    side = pos['side']
                    
                    # Get current price for the side we hold
                    if side.upper() == 'YES':
                        current_price = market.get('yes_price', market['price'])
                    else:
                        current_price = market.get('no_price', 1 - market['price'])
                    
                    # Simulate price movement if enabled
                    if self.simulate_prices:
                        drift = random.gauss(0, 0.02)
                        simulated_price = max(0.05, min(0.95, entry_price + drift))
                        current_price = simulated_price
                    
                    # Calculate P&L: (current_price - entry_price) × contracts
                    # Same formula for both YES and NO since we're using the actual price for each side
                    contracts = pos.get('contracts', int(pos['size'] / entry_price) if entry_price > 0 else 0)
                    price_change = current_price - entry_price
                    
                    unrealized_pnl = price_change * contracts
                    pos['current_price'] = current_price
                    pos['unrealized_pnl'] = unrealized_pnl
                    
                    # Exit conditions based on actual position value
                    # Max gain per contract = 1 - entry_price (for YES) or entry_price (for NO)
                    # Max loss per contract = entry_price (for YES) or 1 - entry_price (for NO)
                    profit_take_pct = self._risk_limits.profit_take_pct
                    stop_loss_pct = self._risk_limits.stop_loss_pct
                    
                    # Calculate profit target and stop loss based on contract cost
                    cost_basis = contracts * entry_price  # What we paid
                    profit_target = cost_basis * profit_take_pct
                    stop_loss = -cost_basis * stop_loss_pct
                    
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
                
            except Exception as e:
                print(f"[Monitor Error] {e}")
            await asyncio.sleep(10)
    
    async def _exit_position(self, pos_id: str, exit_price: float, pnl: float, reason: str):
        """Exit a position."""
        position = self._positions.get(pos_id)
        if not position:
            return
        
        # In LIVE mode, actually sell the position on Kalshi
        if not self.dry_run:
            try:
                contracts = position.get('contracts', 0)
                if contracts > 0:
                    # Aggressive exit: accept 10¢ less than mid for immediate fill
                    base_price_cents = int(exit_price * 100)
                    aggressive_price_cents = max(base_price_cents - 10, 5)  # Accept 10¢ less, floor at 5¢
                    
                    result = await self._kalshi.sell_position(
                        ticker=position['market_id'],
                        side=position['side'].lower(),
                        count=contracts,
                        price=aggressive_price_cents,
                        order_type='limit',
                    )
                    exit_order_id = result.get('order', {}).get('order_id')
                    print(f"[LIVE SELL] Placed: {contracts} {position['side']} @ {aggressive_price_cents}¢ (mid: {base_price_cents}¢) | Order ID: {exit_order_id}")
                    
                    # Track as pending exit - don't remove position until sell fills
                    position['pending_exit'] = {
                        'order_id': exit_order_id,
                        'exit_price': exit_price,
                        'reason': reason,
                        'placed_time': datetime.utcnow().isoformat(),
                    }
                    self._save_state()
                    return  # Don't record exit yet - wait for fill confirmation
                    
            except Exception as e:
                print(f"[Order Error] Failed to place sell order on Kalshi: {e}")
                return  # Keep position, don't exit
        
        # For dry run mode, or after sell is confirmed (called from sync loop)
        self._positions.pop(pos_id, None)
        
        await self._risk_engine.record_trade_result(pnl)
        
        # Log to database
        if self._db_connected and position.get('db_trade_id'):
            try:
                await self._db.log_trade_exit(
                    trade_id=position['db_trade_id'],
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    fees_estimate=0.0,
                )
            except Exception as e:
                print(f"[DB] Failed to log trade exit: {e}")
        
        contracts = position.get('contracts', 0)
        trade = {
            'id': pos_id,
            'market_id': position['market_id'],
            'question': position.get('question', ''),
            'action': 'EXIT',
            'side': position['side'],
            'entry_price': position.get('entry_price', 0),
            'exit_price': exit_price,
            'contracts': contracts,
            'size': position['size'],
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.insert(0, trade)
        
        mode = "[DRY RUN]" if self.dry_run else "[LIVE]"
        print(f"{mode} EXITED: ${pnl:+.2f} ({reason}) | {contracts} contracts @ {int(exit_price*100)}¢ | {position.get('question', '')[:50]}...")
        
        self._save_state()
        await self._broadcast_update()
    
    async def _price_refresh_loop(self):
        """Periodically refresh prices from Kalshi API."""
        while self._running:
            try:
                for market_id in list(self._monitored.keys()):
                    try:
                        result = await self._kalshi.get_market(market_id)
                        if result and 'market' in result:
                            updated = parse_kalshi_market(result['market'])
                            self._markets[market_id] = updated
                            self._monitored[market_id] = updated
                            self._price_update_count += 1
                    except:
                        pass
                    await asyncio.sleep(1)  # Rate limit
            except Exception as e:
                print(f"[Price Refresh Error] {e}")
            await asyncio.sleep(30)  # Refresh every 30 seconds
    
    async def _check_pending_exits(self):
        """Check status of pending sell/exit orders."""
        positions_with_pending_exit = [
            (pos_id, pos) for pos_id, pos in self._positions.items() 
            if pos.get('pending_exit')
        ]
        
        if not positions_with_pending_exit:
            return
        
        for pos_id, position in positions_with_pending_exit:
            pending_exit = position.get('pending_exit', {})
            exit_order_id = pending_exit.get('order_id')
            if not exit_order_id:
                continue
            
            try:
                result = await self._kalshi.get_order(exit_order_id)
                order_data = result.get('order', {})
                status = order_data.get('status', '').lower()
                
                if status == 'executed':
                    # Sell order filled - finalize the exit
                    fill_price = order_data.get('average_fill_price', pending_exit.get('exit_price', 0.5) * 100) / 100
                    reason = pending_exit.get('reason', 'UNKNOWN')
                    
                    # Calculate final PnL using ACTUAL fill prices and contract count
                    entry_price = position['entry_price']  # Actual buy fill price
                    side = position['side']
                    contracts = position.get('contracts', 0)
                    
                    if side.upper() == 'YES':
                        price_change = fill_price - entry_price
                    else:
                        price_change = entry_price - fill_price
                    
                    pnl = price_change * contracts  # PnL per contract × number of contracts
                    
                    # Now actually remove the position and record exit
                    self._positions.pop(pos_id, None)
                    
                    await self._risk_engine.record_trade_result(pnl)
                    
                    # Log to database
                    if self._db_connected and position.get('db_trade_id'):
                        try:
                            await self._db.log_trade_exit(
                                trade_id=position['db_trade_id'],
                                exit_price=fill_price,
                                exit_reason=reason,
                                pnl=pnl,
                                fees_estimate=0.0,
                            )
                        except Exception as e:
                            print(f"[DB] Failed to log trade exit: {e}")
                    
                    # Record trade with actual fill data
                    trade = {
                        'id': pos_id,
                        'market_id': position['market_id'],
                        'question': position.get('question', ''),
                        'action': 'EXIT',
                        'side': position['side'],
                        'entry_price': entry_price,  # Actual buy fill price
                        'exit_price': fill_price,    # Actual sell fill price
                        'contracts': contracts,
                        'size': position.get('size', 0),
                        'pnl': pnl,
                        'reason': reason,
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    self._trades.insert(0, trade)
                    
                    print(f"[EXIT CONFIRMED] ${pnl:+.2f} ({reason}) | {contracts} @ {entry_price*100:.0f}¢→{fill_price*100:.0f}¢ | {position.get('question', '')[:50]}...")
                    self._save_state()
                    await self._broadcast_update()
                    
                elif status in ('canceled', 'cancelled'):
                    # Sell order was canceled - remove pending exit flag, position still open
                    position.pop('pending_exit', None)
                    print(f"[EXIT CANCELED] Position {pos_id} still open - sell order was canceled")
                    self._save_state()
                
                # If still 'resting', do nothing - wait for fill
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                if '404' in str(e).lower():
                    # Order not found - might have been filled, check positions
                    position.pop('pending_exit', None)
                    print(f"[EXIT ORDER] {exit_order_id} not found - checking actual positions...")
                else:
                    print(f"[Exit Check Error] {pos_id}: {e}")
    
    async def _position_sync_loop(self):
        """Sync positions with Kalshi API to detect filled orders."""
        await asyncio.sleep(5)  # Wait for startup
        
        while self._running:
            try:
                # First, check pending EXIT orders (sells on existing positions)
                await self._check_pending_exits()
                
                # Then check pending BUY orders
                if not self._pending_orders:
                    await asyncio.sleep(10)
                    continue
                
                # Check each pending order's status directly
                filled_orders = []
                canceled_orders = []
                
                for order_id, order in list(self._pending_orders.items()):
                    try:
                        # Get order status from Kalshi API
                        result = await self._kalshi.get_order(order_id)
                        order_data = result.get('order', {})
                        status = order_data.get('status', '').lower()
                        
                        if status == 'executed':
                            # Order fully filled
                            fill_count = order_data.get('filled_count', order['contracts'])
                            fill_price = order_data.get('average_fill_price', order['entry_price'] * 100) / 100
                            filled_orders.append({
                                'order_id': order_id,
                                'order': order,
                                'fill_count': fill_count,
                                'fill_price': fill_price,
                            })
                            print(f"[ORDER FILLED] {order_id} | {fill_count} contracts @ {fill_price*100:.0f}¢")
                        
                        elif status == 'canceled' or status == 'cancelled':
                            canceled_orders.append(order_id)
                            print(f"[ORDER CANCELED] {order_id}")
                        
                        elif status == 'resting':
                            # Still waiting to fill - check for partial fills
                            filled_count = order_data.get('filled_count', 0)
                            if filled_count > 0:
                                print(f"[PARTIAL FILL] {order_id} | {filled_count}/{order['contracts']} contracts filled")
                        
                        await asyncio.sleep(0.5)  # Rate limit between API calls
                        
                    except Exception as e:
                        # Order might not exist anymore (already executed/canceled)
                        error_str = str(e).lower()
                        if '404' in error_str or 'not found' in error_str:
                            # Order not found - likely already executed, check fills
                            print(f"[Order {order_id}] Not found, checking fills...")
                            try:
                                fills = await self._kalshi.get_fills(ticker=order['market_id'])
                                for fill in fills.get('fills', []):
                                    if fill.get('order_id') == order_id:
                                        filled_orders.append({
                                            'order_id': order_id,
                                            'order': order,
                                            'fill_count': fill.get('count', order['contracts']),
                                            'fill_price': fill.get('price', order['entry_price'] * 100) / 100,
                                        })
                                        print(f"[FILL FOUND] {order_id} from fills API")
                                        break
                            except:
                                pass
                        else:
                            print(f"[Order Status Error] {order_id}: {e}")
                
                # Process filled orders - move to positions
                for filled in filled_orders:
                    order_id = filled['order_id']
                    order = self._pending_orders.pop(order_id, None)
                    if not order:
                        continue
                    
                    pos_id = order.get('id', f"pos_{order_id}")
                    fill_price = filled.get('fill_price', order['entry_price'])
                    
                    pos = {
                        'id': pos_id,
                        'order_id': order_id,
                        'market_id': order['market_id'],
                        'question': order.get('question', ''),
                        'side': order['side'],
                        'size': order['size'],
                        'entry_price': fill_price,
                        'current_price': fill_price,
                        'contracts': filled.get('fill_count', order['contracts']),
                        'ai_probability': order.get('ai_probability', 0.5),
                        'edge': order.get('edge', 0),
                        'confidence': order.get('confidence', 0),
                        'entry_time': datetime.utcnow().isoformat(),
                        'unrealized_pnl': 0.0,
                    }
                    self._positions[pos_id] = pos
                    
                    # Log to database
                    if self._db_connected:
                        try:
                            db_trade_id = await self._db.log_trade_entry(
                                decision_id=0,
                                market_id=order['market_id'],
                                token_id=order['market_id'],
                                entry_price=fill_price,
                                entry_side=order['side'],
                                size=order['size'],
                                raw_prob=order.get('ai_probability', 0.5),
                                adjusted_prob=order.get('ai_probability', 0.5),
                                edge=order.get('edge', 0),
                                confidence=order.get('confidence', 0),
                            )
                            pos['db_trade_id'] = db_trade_id
                        except Exception as e:
                            print(f"[DB] Failed to log filled trade: {e}")
                    
                    print(f"[POSITION OPENED] {order['side']} ${order['size']:.2f} @ {fill_price*100:.0f}¢ | {order.get('question', '')[:50]}...")
                    
                    # Update trade record
                    for trade in self._trades:
                        if trade.get('order_id') == order_id:
                            trade['action'] = 'ENTRY'
                            trade['status'] = 'filled'
                            trade['fill_price'] = fill_price
                            break
                
                # Remove canceled orders
                for order_id in canceled_orders:
                    order = self._pending_orders.pop(order_id, None)
                    if order:
                        print(f"[REMOVED] Canceled order {order_id}")
                        # Update trade record
                        for trade in self._trades:
                            if trade.get('order_id') == order_id:
                                trade['status'] = 'canceled'
                                break
                
                # Save state if anything changed
                if filled_orders or canceled_orders:
                    self._save_state()
                    await self._broadcast_update()
                
                # Log sync status
                if self._pending_orders:
                    print(f"[Sync] {len(self._pending_orders)} orders pending, {len(self._positions)} positions active")
                
            except Exception as e:
                print(f"[Position Sync Error] {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _broadcast_update(self):
        """Send update to all connected WebSocket clients."""
        if not self._websockets:
            return
        
        data = json.dumps({
            'type': 'update',
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'pending_orders': list(self._pending_orders.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        })
        
        for ws in list(self._websockets):
            try:
                await ws.send_str(data)
            except:
                self._websockets.discard(ws)
    
    async def _handle_index(self, request):
        """Serve the dashboard HTML."""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_state(self, request):
        """API endpoint for current state."""
        return web.json_response({
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        })
    
    async def _handle_websocket(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)
        
        # Send initial state
        await ws.send_str(json.dumps({
            'type': 'init',
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
        }))
        
        try:
            async for msg in ws:
                pass
        finally:
            self._websockets.discard(ws)
        
        return ws


# Dashboard HTML (same as Polymarket version with minor text changes)
DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalshi BattleBot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; }
        .header { background: #161b22; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; }
        .logo { font-size: 18px; font-weight: 600; color: #58a6ff; }
        .status { display: flex; align-items: center; gap: 12px; font-size: 13px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #f85149; }
        .dot.live { background: #3fb950; }
        .badge { padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }
        .badge.dry { background: #f0883e; color: #000; }
        .badge.live { background: #3fb950; color: #000; }
        .tabs { display: flex; gap: 8px; padding: 12px 20px; background: #161b22; }
        .tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; color: #8b949e; }
        .tab:hover { color: #e6edf3; background: #21262d; }
        .tab.active { background: #1f6feb; color: #fff; }
        .content { padding: 20px; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card-label { font-size: 11px; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
        .card-value { font-size: 24px; font-weight: 600; }
        .card-sub { font-size: 12px; color: #8b949e; margin-top: 2px; }
        .green { color: #3fb950; }
        .red { color: #f85149; }
        .yellow { color: #f0883e; }
        .section-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #8b949e; }
        .position { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
        .position-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
        .position-title { font-size: 13px; font-weight: 500; flex: 1; }
        .position-side { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px; }
        .position-side.yes { background: #238636; }
        .position-side.no { background: #f85149; }
        .position-details { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 12px; }
        .position-detail-label { color: #8b949e; }
        .position-detail-value { font-weight: 500; }
        .trade { display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #161b22; border-left: 3px solid #30363d; margin-bottom: 4px; font-size: 13px; }
        .trade.entry { border-left-color: #1f6feb; }
        .trade.exit { border-left-color: #3fb950; }
        .trade.exit.loss { border-left-color: #f85149; }
        .trade-info { flex: 1; }
        .trade-action { padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-left: 8px; }
        .trade-action.entry { background: #1f6feb33; color: #58a6ff; }
        .trade-action.exit { background: #3fb95033; color: #3fb950; }
        .market { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
        .market-question { font-size: 13px; margin-bottom: 8px; }
        .market-stats { display: flex; gap: 16px; font-size: 12px; color: #8b949e; }
        .hidden { display: none; }
        .analysis { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
        .analysis-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .analysis-decision { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .analysis-decision.trade { background: #238636; }
        .analysis-decision.skip { background: #6e7681; }
        .badge-group { display: flex; gap: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">Kalshi BattleBot</div>
        <div class="status">
            <span><span class="dot" id="dot"></span> <span id="connStatus">Connecting...</span></span>
            <span id="runtime">0h 0m 0s</span>
            <span class="badge dry" id="modeBadge">DRY RUN</span>
            <div class="badge-group">
                <span class="badge" id="wsBadge" style="background:#6e7681">Poll</span>
                <span class="badge" id="aiBadge" style="background:#6e7681">AI: --</span>
            </div>
        </div>
    </div>
    <div class="tabs">
        <div class="tab active" data-tab="portfolio">Portfolio</div>
        <div class="tab" data-tab="activity">Activity</div>
        <div class="tab" data-tab="markets">Markets</div>
    </div>
    <div class="content">
        <div id="portfolio" class="tab-content">
            <div class="section-title">ACCOUNT</div>
            <div class="grid">
                <div class="card"><div class="card-label">Available</div><div class="card-value green" id="available">$0.00</div></div>
                <div class="card"><div class="card-label">At Risk</div><div class="card-value yellow" id="atRisk">$0.00</div><div class="card-sub" id="posCount">0 positions</div></div>
                <div class="card"><div class="card-label">Total Value</div><div class="card-value" id="totalValue">$0.00</div></div>
                <div class="card"><div class="card-label">Return</div><div class="card-value green" id="returnPct">0.00%</div></div>
            </div>
            <div class="section-title">PERFORMANCE</div>
            <div class="grid">
                <div class="card"><div class="card-label">Realized P&L</div><div class="card-value green" id="realizedPnl">$0.00</div></div>
                <div class="card"><div class="card-label">Unrealized P&L</div><div class="card-value" id="unrealizedPnl">$0.00</div></div>
                <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="winRate">0%</div><div class="card-sub" id="winLoss">0W / 0L</div></div>
                <div class="card"><div class="card-label">Best Trade</div><div class="card-value green" id="bestTrade">$0.00</div></div>
                <div class="card"><div class="card-label">Worst Trade</div><div class="card-value red" id="worstTrade">$0.00</div></div>
                <div class="card"><div class="card-label">Total Trades</div><div class="card-value" id="totalTrades">0</div><div class="card-sub" id="tradeBreakdown">0 entries / 0 exits</div></div>
            </div>
            <div class="section-title">Active Positions <span id="positionCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="positions"></div>
            <div class="section-title" style="margin-top:20px">Recent Trades <span id="tradeCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="trades"></div>
        </div>
        <div id="activity" class="tab-content hidden">
            <div class="section-title">AI Analyses</div>
            <div id="analyses"></div>
        </div>
        <div id="markets" class="tab-content hidden">
            <div class="section-title">Monitored Markets <span id="marketCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="marketList"></div>
        </div>
    </div>
    <script>
        let ws;
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
                document.getElementById(tab.dataset.tab).classList.remove('hidden');
            });
        });
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('dot').className = 'dot live';
                document.getElementById('connStatus').textContent = 'Connected';
            };
            
            ws.onclose = () => {
                document.getElementById('dot').className = 'dot';
                document.getElementById('connStatus').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.stats) updateStats(data.stats);
                if (data.positions) renderPositions(data.positions);
                if (data.trades) renderTrades(data.trades);
                if (data.analyses) renderAnalyses(data.analyses);
                if (data.monitored) renderMarkets(data.monitored);
            };
        }
        
        function updateStats(s) {
            document.getElementById('available').textContent = '$' + s.available.toFixed(2);
            document.getElementById('atRisk').textContent = '$' + s.at_risk.toFixed(2);
            document.getElementById('posCount').textContent = s.open_positions + ' positions';
            document.getElementById('totalValue').textContent = '$' + s.total_value.toFixed(2);
            document.getElementById('returnPct').textContent = (s.return_pct >= 0 ? '+' : '') + s.return_pct.toFixed(2) + '%';
            document.getElementById('returnPct').className = 'card-value ' + (s.return_pct >= 0 ? 'green' : 'red');
            document.getElementById('realizedPnl').textContent = (s.realized_pnl >= 0 ? '+' : '') + '$' + s.realized_pnl.toFixed(2);
            document.getElementById('realizedPnl').className = 'card-value ' + (s.realized_pnl >= 0 ? 'green' : 'red');
            document.getElementById('unrealizedPnl').textContent = (s.unrealized_pnl >= 0 ? '+' : '') + '$' + s.unrealized_pnl.toFixed(2);
            document.getElementById('unrealizedPnl').className = 'card-value ' + (s.unrealized_pnl >= 0 ? 'green' : 'red');
            document.getElementById('winRate').textContent = (s.win_rate * 100).toFixed(0) + '%';
            document.getElementById('winLoss').textContent = s.winning_trades + 'W / ' + s.losing_trades + 'L';
            document.getElementById('bestTrade').textContent = '+$' + s.best_trade.toFixed(2);
            document.getElementById('worstTrade').textContent = '$' + s.worst_trade.toFixed(2);
            document.getElementById('totalTrades').textContent = s.total_entries + s.total_exits;
            document.getElementById('tradeBreakdown').textContent = s.total_entries + ' entries / ' + s.total_exits + ' exits';
            document.getElementById('runtime').textContent = s.runtime;
            document.getElementById('modeBadge').textContent = s.dry_run ? 'DRY RUN' : 'LIVE';
            document.getElementById('modeBadge').className = 'badge ' + (s.dry_run ? 'dry' : 'live');
            
            const wsBadge = document.getElementById('wsBadge');
            wsBadge.textContent = 'Poll: ' + (s.price_updates || 0);
            wsBadge.style.background = '#6e7681';
            
            const aiBadge = document.getElementById('aiBadge');
            if (s.ai_available) {
                aiBadge.textContent = 'AI: ' + s.ai_successes + '/' + s.ai_calls;
                aiBadge.style.background = '#238636';
            } else {
                aiBadge.textContent = 'AI: OFF';
                aiBadge.style.background = '#f85149';
            }
        }
        
        function renderPositions(positions) {
            document.getElementById('positionCount').textContent = positions.length;
            const html = positions.map(p => `
                <div class="position">
                    <div class="position-header">
                        <div class="position-title">${p.question}</div>
                        <span class="position-side ${p.side.toLowerCase()}">${p.side}</span>
                    </div>
                    <div class="position-details">
                        <div><div class="position-detail-label">ENTRY</div><div class="position-detail-value">${(p.entry_price*100).toFixed(0)}¢</div></div>
                        <div><div class="position-detail-label">CURRENT</div><div class="position-detail-value">${(p.current_price*100).toFixed(0)}¢</div></div>
                        <div><div class="position-detail-label">SIZE</div><div class="position-detail-value">$${p.size.toFixed(2)}</div></div>
                        <div><div class="position-detail-label">P&L</div><div class="position-detail-value ${p.unrealized_pnl >= 0 ? 'green' : 'red'}">${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)}</div></div>
                    </div>
                </div>
            `).join('');
            document.getElementById('positions').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No open positions</div>';
        }
        
        function renderTrades(trades) {
            document.getElementById('tradeCount').textContent = trades.length;
            const html = trades.slice(0, 10).map(t => {
                const isExit = t.action === 'EXIT';
                const isLoss = isExit && t.pnl < 0;
                return `
                    <div class="trade ${t.action.toLowerCase()} ${isLoss ? 'loss' : ''}">
                        <div class="trade-info">
                            <div>${t.question}</div>
                            <div style="color:#8b949e;font-size:11px;">${t.side} @ ${(t.price*100).toFixed(0)}¢ · $${t.size.toFixed(2)} · ${new Date(t.timestamp).toLocaleTimeString()}${t.reason ? ' · ' + t.reason : ''}</div>
                        </div>
                        ${isExit ? `<div class="${t.pnl >= 0 ? 'green' : 'red'}">${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}</div>` : ''}
                        <span class="trade-action ${t.action.toLowerCase()}">${t.action}</span>
                    </div>
                `;
            }).join('');
            document.getElementById('trades').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No trades yet</div>';
        }
        
        function renderAnalyses(analyses) {
            const html = analyses.map(a => `
                <div class="analysis">
                    <div class="analysis-header">
                        <div style="flex:1">${a.question}</div>
                        <span class="analysis-decision ${a.decision === 'TRADE' ? 'trade' : 'skip'}">${a.decision}</span>
                    </div>
                    <div style="font-size:12px;color:#8b949e;margin-bottom:8px;">
                        AI: ${(a.ai_probability*100).toFixed(0)}% → Adj: ${(a.adjusted_probability*100).toFixed(0)}% | 
                        Market: ${(a.market_price*100).toFixed(0)}¢ | 
                        Edge: ${a.edge >= 0 ? '+' : ''}${(a.edge*100).toFixed(1)}% |
                        ${a.side} |
                        ${a.latency_ms}ms
                    </div>
                    ${a.key_reasons ? `<div style="font-size:12px;color:#58a6ff;">• ${a.key_reasons.slice(0,2).join('<br>• ')}</div>` : ''}
                </div>
            `).join('');
            document.getElementById('analyses').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No analyses yet</div>';
        }
        
        function renderMarkets(markets) {
            document.getElementById('marketCount').textContent = markets.length;
            const html = markets.map(m => `
                <div class="market">
                    <div class="market-question">${m.question}</div>
                    <div class="market-stats">
                        <span>Price: ${m.price_pct}¢</span>
                        <span>Spread: ${m.spread_display}</span>
                        <span>Volume: ${m.volume_display}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById('marketList').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No markets monitored</div>';
        }
        
        connect();
    </script>
</body>
</html>
'''


async def main():
    port = int(os.getenv('PORT', 8080))
    bot = KalshiBattleBot(port=port)
    await bot.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
