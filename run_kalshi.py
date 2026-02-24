#!/usr/bin/env python3
"""Battle-Bot for Kalshi - CFTC-regulated prediction market.

Legal for US residents including California.
Uses same AI strategy as Polymarket version.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*utcnow.*')

import asyncio
import json
import os
from datetime import datetime, timedelta
from aiohttp import web
import httpx
from dotenv import load_dotenv

from logic.ai_signal import AISignalGenerator, AISignalResult
from logic.calibration import CalibrationEngine, CalibrationResult
from logic.risk_engine import RiskEngine, RiskLimits
from data.database import TelemetryDB
from services.kalshi_client import KalshiClient, parse_kalshi_market
from services.market_intelligence import get_intelligence_service, MarketIntelligence

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
        
        # Config from env - STRICT defaults for profitability
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.initial_bankroll = float(os.getenv('INITIAL_BANKROLL', 100))
        self.min_edge = max(0.07, float(os.getenv('MIN_EDGE', 0.08)))  # 8% min edge — filters noise without blocking moderate edges
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.40))  # 40% confidence — filters low-quality calls; 50% was too strict, blocked all trades
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 25))  # $25 max - bigger bets on better opportunities
        self.max_days_to_resolution = float(os.getenv('MAX_DAYS_TO_RESOLUTION', 45))  # Skip markets resolving > 45 days out
        self.kelly_fraction = float(os.getenv('FRACTIONAL_KELLY', 0.20))  # 20% Kelly - higher conviction on filtered bets
        self.max_oi_pct = float(os.getenv('MAX_OI_PCT', 0.10))  # Max 10% of open interest
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
        # Signal log for backtesting - tracks ALL AI signals and their outcomes
        self._signal_log: list[dict] = []  # Persisted, checked for outcomes
        # Actual Kalshi values from API (synced every 5 min)
        self._kalshi_cash = None       # Available cash
        self._kalshi_portfolio = None  # Current positions value
        self._kalshi_total = None      # Total portfolio value
        self._load_state()
        
        self._running = False
        self._last_analysis: dict[str, datetime] = {}
        self._last_analysis_price: dict[str, float] = {}
        self._recently_exited: dict[str, datetime] = {}  # Track exits to prevent re-entry
        self._recently_exited_reason: dict[str, str] = {}  # Exit reason per market
        self._analysis_cooldown = 1800  # 30 minutes (overridden per time-horizon in trading loop)
        self._price_change_threshold = 0.02  # Re-analyze on 2% move
        # Probability cache: {market_id: (timestamp, price_when_cached, ai_prob, confidence)}
        # Reused when price hasn't moved significantly — avoids redundant Claude calls
        self._prob_cache: dict[str, tuple[datetime, float, float, float]] = {}
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
        
        # Risk Engine - selective but confident
        self._risk_limits = RiskLimits(
            max_daily_drawdown=0.15,
            max_position_size=self.max_position_size,  # $30 max
            max_percent_bankroll_per_market=0.25,  # 25% per market - fewer bets = bigger size
            max_total_open_risk=0.90,  # 90% max exposure - 10% reserve
            max_positions=25,  # Max 25 positions - deploy more capital
            profit_take_pct=999.0,  # DISABLED - let bets settle naturally
            stop_loss_pct=999.0,  # DISABLED - let bets settle naturally
            time_stop_hours=720,  # 30 days max - but prefer settlement
            edge_scale=0.10,
            min_edge=self.min_edge,
        )
        self._risk_engine = RiskEngine(
            initial_bankroll=self.initial_bankroll,
            fractional_kelly=self.kelly_fraction,
            limits=self._risk_limits,
        )
        
        # Market Intelligence Service (news, domain data, inefficiency detection)
        self._intelligence = get_intelligence_service()
        self._use_intelligence = os.getenv('USE_INTELLIGENCE', 'true').lower() == 'true'
        self._prefer_inefficient = os.getenv('PREFER_INEFFICIENT_MARKETS', 'true').lower() == 'true'
        self._use_contrarian = os.getenv('USE_CONTRARIAN_TIMING', 'true').lower() == 'true'
        
        # Inefficiency threshold: prefer markets with score > this
        self._min_inefficiency_score = float(os.getenv('MIN_INEFFICIENCY_SCORE', '0.1'))
    
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
        
        # CLEAR_BUGGY_TRADES: opt-in only via env var (no longer the default)
        clear_buggy = os.getenv('CLEAR_BUGGY_TRADES', 'false').lower() == 'true'
        if clear_buggy:
            print("[State] Clearing reconciled trades (CLEAR_BUGGY_TRADES=true)...")
            
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._pending_orders = state.get('pending_orders', {})
                    self._trades = state.get('trades', [])
                    self._signal_log = state.get('signal_log', [])
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._pending_orders)} pending orders, {len(self._trades)} trades, {len(self._signal_log)} signals")
                    
                    # Clear ALL reconciled trades - they have corrupted P&L data
                    if clear_buggy:
                        before = len(self._trades)
                        # Remove ALL reconciled trades (they have wrong P&L calculations)
                        self._trades = [t for t in self._trades if not t.get('reconciled')]
                        removed = before - len(self._trades)
                        if removed > 0:
                            print(f"[State] Removed {removed} buggy reconciled trades with corrupted P&L")
                            self._save_state()
                            
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
            
            daily_snapshots = getattr(self, '_daily_snapshots', {})
            portfolio_hourly = getattr(self, '_portfolio_hourly', {})
            
            state_data = {
                'positions': self._positions,
                'pending_orders': self._pending_orders,
                'trades': self._trades[-100:],
                'signal_log': self._signal_log[-500:],
                'daily_snapshots': daily_snapshots,
                'portfolio_hourly': portfolio_hourly,
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
                        'end_date': order.get('end_date'),  # Preserve for time horizon
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
    
    async def _cancel_all_resting_orders(self):
        """Cancel ALL resting orders on Kalshi at startup.
        
        This prevents accumulation of unfilled orders from previous sessions.
        Fresh start every time.
        """
        print("[Startup] Checking for stale resting orders...")
        try:
            result = await self._kalshi.get_orders(status="resting")
            orders = result.get('orders', [])
            
            if not orders:
                print("[Startup] No resting orders found.")
                return
            
            print(f"[Startup] Found {len(orders)} resting orders - CANCELING ALL")
            
            canceled = 0
            failed = 0
            for order in orders:
                order_id = order.get('order_id')
                if not order_id:
                    continue
                    
                try:
                    await self._kalshi.cancel_order(order_id)
                    canceled += 1
                    ticker = order.get('ticker', 'unknown')
                    side = order.get('side', '?')
                    count = order.get('remaining_count', order.get('count', '?'))
                    print(f"  [CANCELED] {order_id[:8]}... | {count} {side.upper()} on {ticker[:30]}")
                    await asyncio.sleep(0.3)  # Rate limit
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] {order_id[:8]}...: {e}")
            
            print(f"[Startup] Order cleanup complete: {canceled} canceled, {failed} failed")
            
            # Also clear bot's internal pending orders since they're all canceled
            if self._pending_orders:
                print(f"[Startup] Clearing {len(self._pending_orders)} internal pending order records")
                self._pending_orders.clear()
                self._save_state()
                
        except Exception as e:
            print(f"[Startup] Error fetching resting orders: {e}")
    
    async def _reconcile_settlements_from_fills(self):
        """Reconcile settlements by checking fills against trade history.
        
        This is critical for Railway deployments where state may be lost.
        Queries Kalshi fills and backfills any missed settlement P&L.
        """
        print("[Reconcile] Checking for missed settlements from Kalshi fills...")
        
        try:
            # Get recent fills from Kalshi (our actual trade history)
            result = await self._kalshi.get_fills(limit=200)
            fills = result.get('fills', [])
            
            if not fills:
                print("[Reconcile] No fills found in Kalshi history")
                return
            
            print(f"[Reconcile] Found {len(fills)} fills in Kalshi history")
            
            # Group fills by ticker to find net positions
            fills_by_ticker = {}
            for fill in fills:
                ticker = fill.get('ticker', '')
                if not ticker:
                    continue
                if ticker not in fills_by_ticker:
                    fills_by_ticker[ticker] = []
                fills_by_ticker[ticker].append(fill)
            
            # Get tickers that have EXIT trades in our history
            exit_tickers = set()
            entry_by_ticker = {}
            for trade in self._trades:
                ticker = trade.get('market_id', '')
                if trade.get('action') == 'EXIT':
                    exit_tickers.add(ticker)
                elif trade.get('action') == 'ENTRY':
                    entry_by_ticker[ticker] = trade
            
            reconciled = 0
            checked = 0
            for ticker, ticker_fills in fills_by_ticker.items():
                # Skip if we already have an EXIT for this ticker
                if ticker in exit_tickers:
                    continue
                
                # Calculate net position from fills
                net_contracts = 0
                avg_price = 0.0
                total_cost = 0.0
                side = None
                
                for fill in ticker_fills:
                    action = fill.get('action', '')
                    count = fill.get('count', 0)
                    # Kalshi fills API 'price' field is in DOLLARS (e.g., 0.18 = 18 cents)
                    price = fill.get('price', 0.50)
                    fill_side = fill.get('side', '')
                    
                    if action == 'buy':
                        net_contracts += count
                        total_cost += price * count
                        side = fill_side
                    elif action == 'sell':
                        net_contracts -= count
                
                # Skip if no net position (fully closed via sells)
                if net_contracts <= 0:
                    continue
                
                avg_price = total_cost / net_contracts if net_contracts > 0 else 0.5
                
                # Check if this market has settled
                checked += 1
                try:
                    market_result = await self._kalshi.get_market(ticker)
                    market_data = market_result.get('market', {}) if market_result else {}
                    status = market_data.get('status', '')
                    result = market_data.get('result', '')
                    title = market_data.get('title', ticker)[:50]
                    
                    print(f"[Reconcile] Checking {ticker[:30]}... | net={net_contracts} | status={status} | result={result}")
                    
                    # Kalshi uses 'finalized' or 'settled' for completed markets
                    if status not in ('settled', 'finalized') or not result:
                        # Market not settled yet - skip
                        continue
                    
                    # Market settled! Calculate P&L
                    our_side = (side or '').lower()
                    if our_side == result:
                        exit_price = 1.0
                        pnl = (exit_price - avg_price) * net_contracts
                        outcome = 'WIN'
                    else:
                        exit_price = 0.0
                        pnl = -avg_price * net_contracts
                        outcome = 'LOSS'
                    
                    print(f"[Reconcile] FOUND MISSED SETTLEMENT: {outcome}! ${pnl:+.2f} | {our_side.upper()} on '{result.upper()}' result | {title}")
                    
                    # Record EXIT trade
                    trade = {
                        'id': f"reconciled_{ticker}_{datetime.utcnow().timestamp()}",
                        'market_id': ticker,
                        'question': title,
                        'action': 'EXIT',
                        'side': side,
                        'entry_price': avg_price,
                        'exit_price': exit_price,
                        'contracts': net_contracts,
                        'size': avg_price * net_contracts,
                        'pnl': pnl,
                        'reason': f'SETTLED_{outcome}',
                        'timestamp': datetime.utcnow().isoformat(),
                        'reconciled': True,  # Flag that this was backfilled
                    }
                    self._trades.insert(0, trade)
                    
                    # DON'T record historical settlements in risk engine's daily stats
                    # These losses already happened - they shouldn't trigger today's kill switch
                    # The P&L is already reflected in Kalshi account balance
                    # await self._risk_engine.record_trade_result(pnl)  # DISABLED
                    
                    reconciled += 1
                    await asyncio.sleep(0.2)  # Rate limit API calls
                    
                except Exception as e:
                    print(f"[Reconcile] Error checking market {ticker}: {e}")
                    continue
            
            print(f"[Reconcile] Checked {checked} positions with net buys")
            if reconciled > 0:
                self._save_state()
                print(f"[Reconcile] Backfilled {reconciled} missed settlements!")
            else:
                print("[Reconcile] No missed settlements found - all trades accounted for")
                
        except Exception as e:
            print(f"[Reconcile] Error during reconciliation: {e}")
            import traceback
            traceback.print_exc()
    
    async def _sync_positions_with_kalshi(self):
        """Sync internal positions with Kalshi's actual positions and balance.
        
        This ensures the bot's state matches reality by:
        1. Fetching actual balance from Kalshi API
        2. Fetching actual positions from Kalshi API
        3. Removing phantom positions that don't exist on Kalshi
        4. Adding any positions that exist on Kalshi but not in bot state
        5. Updating entry prices to match Kalshi's records
        """
        print("[Sync] Fetching actual data from Kalshi...")
        
        try:
            # Fetch actual balance and portfolio value from Kalshi
            try:
                balance_result = await self._kalshi.get_balance()
                
                # Debug: Log raw response to see what fields are available
                print(f"[Sync] Raw balance response: {balance_result}")
                
                # Kalshi returns all values in cents
                balance_cents = balance_result.get('balance', 0)
                portfolio_cents = balance_result.get('portfolio_value', 0)
                
                self._kalshi_cash = balance_cents / 100  # Convert to dollars
                self._kalshi_portfolio = portfolio_cents / 100  # Convert to dollars
                self._kalshi_total = self._kalshi_cash + self._kalshi_portfolio
                
                print(f"[Sync] Kalshi: Cash=${self._kalshi_cash:.2f}, Positions=${self._kalshi_portfolio:.2f}, Total=${self._kalshi_total:.2f}")
                
                # Keep intraday peak in sync with the same number the user sees on the dashboard
                await self._save_daily_snapshot(self._kalshi_total, self._kalshi_cash, self._kalshi_portfolio)
            except Exception as e:
                print(f"[Sync] Could not fetch balance: {e}")
                import traceback
                traceback.print_exc()
                self._kalshi_cash = None
                self._kalshi_portfolio = None
                self._kalshi_total = None
            
            result = await self._kalshi.get_positions()
            kalshi_positions = result.get('market_positions', []) or result.get('positions', [])
            
            if not kalshi_positions:
                print("[Sync] No positions found on Kalshi")
                if self._positions:
                    print(f"[Sync] WARNING: Bot has {len(self._positions)} phantom positions - clearing them")
                    self._positions.clear()
                    self._save_state()
                return
            
            print(f"[Sync] Found {len(kalshi_positions)} positions on Kalshi")
            
            # Debug: Log first position's structure
            if kalshi_positions:
                sample = kalshi_positions[0]
                print(f"[Sync] Sample position fields: {list(sample.keys())}")
            
            # Build a map of Kalshi positions by ticker
            kalshi_by_ticker = {}
            for kp in kalshi_positions:
                ticker = kp.get('ticker', kp.get('market_ticker', ''))
                if ticker:
                    kalshi_by_ticker[ticker] = kp
                    # Debug log each position
                    pos = kp.get('position', 0)
                    exposure = kp.get('market_exposure', 0)
                    print(f"[Sync] Kalshi: {ticker[:30]}... | pos={pos} | exposure={exposure}¢")
            
            # Check for phantom positions (in bot but not on Kalshi)
            # Positions gone from Kalshi: remove only. Do NOT infer P&L or record EXIT.
            # Settled P&L is only from Kalshi fills (Settlements tab). This avoids phantom losses.
            removed = 0
            for pos_id in list(self._positions.keys()):
                pos = self._positions[pos_id]
                market_id = pos.get('market_id', '')
                if market_id not in kalshi_by_ticker:
                    print(f"[Sync] Position no longer on Kalshi, removing (use Settlements for P&L): {pos.get('question', '')[:40]}...")
                    self._positions.pop(pos_id)
                    removed += 1
            if removed:
                self._save_state()
                await self._broadcast_update()
            
            # Check for missing positions (on Kalshi but not in bot)
            # and update entry prices for existing positions
            existing_tickers = {p.get('market_id') for p in self._positions.values()}
            added_count = 0
            updated_count = 0
            
            for ticker, kp in kalshi_by_ticker.items():
                position_count = kp.get('position', 0)
                if position_count == 0:
                    continue  # No actual position
                
                # Get the actual entry price from Kalshi
                # Kalshi API returns market_exposure (cost in cents) and position (contracts)
                # Entry price = market_exposure / abs(position)
                market_exposure_cents = kp.get('market_exposure', 0)
                contracts = abs(position_count)
                
                if contracts > 0 and market_exposure_cents > 0:
                    entry_price = (market_exposure_cents / contracts) / 100  # Convert cents to dollars
                else:
                    # Try dollars field as fallback
                    market_exposure_dollars = kp.get('market_exposure_dollars', '0')
                    try:
                        exposure_float = float(market_exposure_dollars)
                        entry_price = exposure_float / contracts if contracts > 0 else 0.5
                    except:
                        entry_price = 0.5  # Fallback
                
                # Determine side from position count
                side = 'YES' if position_count > 0 else 'NO'
                
                # Calculate actual size (cost) from market_exposure
                actual_size = market_exposure_cents / 100 if market_exposure_cents > 0 else contracts * entry_price
                
                if ticker not in existing_tickers:
                    # Add missing position
                    pos_id = f"pos_kalshi_{ticker[:20]}_{int(datetime.utcnow().timestamp())}"
                    question = kp.get('title', kp.get('market_title', ticker))
                    
                    self._positions[pos_id] = {
                        'id': pos_id,
                        'market_id': ticker,
                        'question': question,
                        'side': side,
                        'size': actual_size,
                        'entry_price': entry_price,
                        'current_price': entry_price,
                        'contracts': contracts,
                        'ai_probability': 0.5,
                        'edge': 0,
                        'confidence': 0.5,
                        'entry_time': datetime.utcnow().isoformat(),
                        'unrealized_pnl': 0.0,
                        'synced_from_kalshi': True,
                    }
                    print(f"[Sync] Added position from Kalshi: {side} {contracts} @ {entry_price*100:.0f}¢ (${actual_size:.2f}) | {question[:40]}...")
                    added_count += 1
                else:
                    # Update existing position with correct entry price and size
                    for pos_id, pos in self._positions.items():
                        if pos.get('market_id') == ticker:
                            old_price = pos.get('entry_price', 0)
                            old_size = pos.get('size', 0)
                            
                            # Check if anything needs updating
                            price_diff = abs(old_price - entry_price) > 0.005
                            size_diff = abs(old_size - actual_size) > 0.01
                            
                            if price_diff or size_diff:
                                print(f"[Sync] Updating: {old_price*100:.0f}¢→{entry_price*100:.0f}¢, ${old_size:.2f}→${actual_size:.2f} | {pos.get('question', '')[:40]}...")
                                pos['entry_price'] = entry_price
                                pos['contracts'] = contracts
                                pos['size'] = actual_size
                                pos['side'] = side  # Also update side in case it was wrong
                                updated_count += 1
                            break
            
            if added_count or updated_count or removed:
                self._save_state()
                print(f"[Sync] Complete: {added_count} added, {updated_count} updated, {removed} removed")
            else:
                print("[Sync] Positions already in sync with Kalshi")
                
        except Exception as e:
            print(f"[Sync] Error syncing with Kalshi: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_historical_performance(self) -> dict:
        """Calculate win/loss performance by category from trade history.
        
        Returns dict with category stats that can be fed to AI for learning.
        """
        # Get all completed exits with P&L
        exits = [t for t in self._trades if t.get('action') == 'EXIT' and 'pnl' in t]
        
        if not exits:
            return {'summary': 'No completed trades yet.', 'by_category': {}}
        
        # Group by category
        by_category = {}
        for trade in exits:
            # Try to get category from trade, or infer from question
            category = trade.get('category', 'unknown')
            if category == 'unknown':
                question = trade.get('question', '').lower()
                if any(word in question for word in ['basketball', 'nba', 'nfl', 'mlb', 'nhl', 'hockey', 'football', 'baseball', 'soccer', 'tennis', 'golf', 'ufc', 'boxing', 'double', 'points', 'rebounds', 'assists', 'wins']):
                    category = 'sports'
                elif any(word in question for word in ['bitcoin', 'btc', 'eth', 'crypto', 'coin']):
                    category = 'crypto'
                elif any(word in question for word in ['trump', 'biden', 'congress', 'senate', 'election', 'president', 'governor', 'fed', 'chair', 'nominee', 'bill', 'law']):
                    category = 'politics'
                elif any(word in question for word in ['s&p', 'stock', 'nasdaq', 'dow', 'market', 'gdp', 'inflation', 'rate']):
                    category = 'economics'
                else:
                    category = 'other'
            
            if category not in by_category:
                by_category[category] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
            
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                by_category[category]['wins'] += 1
            else:
                by_category[category]['losses'] += 1
            by_category[category]['total_pnl'] += pnl
        
        # Build summary string for AI
        lines = []
        for cat, stats in sorted(by_category.items(), key=lambda x: x[1]['wins'] + x[1]['losses'], reverse=True):
            total = stats['wins'] + stats['losses']
            win_rate = stats['wins'] / total * 100 if total > 0 else 0
            pnl = stats['total_pnl']
            lines.append(f"- {cat.upper()}: {stats['wins']}W/{stats['losses']}L ({win_rate:.0f}% win rate), P&L: ${pnl:+.2f}")
        
        # Add recommendations based on performance
        recommendations = []
        for cat, stats in by_category.items():
            total = stats['wins'] + stats['losses']
            if total >= 2:  # Need at least 2 trades to judge
                win_rate = stats['wins'] / total
                if win_rate < 0.3:
                    recommendations.append(f"AVOID {cat.upper()} - historically poor ({stats['wins']}W/{stats['losses']}L)")
                elif win_rate > 0.6:
                    recommendations.append(f"FAVOR {cat.upper()} - historically strong ({stats['wins']}W/{stats['losses']}L)")
        
        summary = "HISTORICAL PERFORMANCE:\n" + "\n".join(lines)
        if recommendations:
            summary += "\n\nRECOMMENDATIONS:\n" + "\n".join(recommendations)
        
        return {
            'summary': summary,
            'by_category': by_category,
            'total_trades': len(exits),
            'recommendations': recommendations,
        }
    
    def _get_intel_effectiveness(self) -> dict:
        """Analyze win rates for trades with vs without news intelligence."""
        exits = [t for t in self._trades if t.get('action') == 'EXIT']
        
        if not exits:
            return {
                'summary': 'No completed trades to analyze',
                'with_news': {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0},
                'without_news': {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0},
            }
        
        with_news = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
        without_news = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
        
        for trade in exits:
            pnl = trade.get('pnl', 0)
            has_intel = trade.get('has_intel', False)
            news_count = trade.get('news_count', 0)
            
            bucket = with_news if (has_intel or news_count > 0) else without_news
            bucket['trades'] += 1
            bucket['pnl'] += pnl
            if pnl > 0:
                bucket['wins'] += 1
            elif pnl < 0:
                bucket['losses'] += 1
        
        # Calculate win rates
        with_news['win_rate'] = with_news['wins'] / with_news['trades'] if with_news['trades'] > 0 else 0
        without_news['win_rate'] = without_news['wins'] / without_news['trades'] if without_news['trades'] > 0 else 0
        
        # Build summary
        wn_str = (
            f"WITH NEWS    ({with_news['trades']:>3} trades): "
            f"{with_news['win_rate']*100:.0f}% win rate "
            f"({with_news['wins']}W/{with_news['losses']}L) "
            f"P&L ${with_news['pnl']:+.2f}"
        ) if with_news['trades'] > 0 else "WITH NEWS: No trades yet"

        non_str = (
            f"WITHOUT NEWS ({without_news['trades']:>3} trades): "
            f"{without_news['win_rate']*100:.0f}% win rate "
            f"({without_news['wins']}W/{without_news['losses']}L) "
            f"P&L ${without_news['pnl']:+.2f}"
        ) if without_news['trades'] > 0 else "WITHOUT NEWS: No trades yet"

        lines = [
            "=== NEWS INTELLIGENCE EFFECTIVENESS ===",
            wn_str,
            non_str,
        ]
        
        # Conclusion
        if with_news['trades'] >= 3 and without_news['trades'] >= 3:
            diff = with_news['win_rate'] - without_news['win_rate']
            if diff > 0.1:
                lines.append(f"\n→ NEWS IS HELPING: +{diff*100:.0f}pp win rate improvement")
            elif diff < -0.1:
                lines.append(f"\n→ NEWS NOT HELPING: {diff*100:.0f}pp win rate difference")
            else:
                lines.append(f"\n→ INCONCLUSIVE: Only {diff*100:+.0f}pp difference")
        else:
            lines.append("\n→ Need more trades to determine effectiveness")
        
        return {
            'summary': '\n'.join(lines),
            'with_news': with_news,
            'without_news': without_news,
        }
    
    def _get_stats(self) -> dict:
        """Calculate all stats from current state."""
        # Pending/resting orders (always from internal state)
        pending_at_risk = sum(o.get('size', 0) for o in self._pending_orders.values())
        
        # Realized P&L from closed trades
        realized_pnl = sum(t.get('pnl', 0) for t in self._trades if t.get('action') == 'EXIT')
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in self._positions.values())
        
        # Internal calculation for reference
        internal_positions = sum(p['size'] for p in self._positions.values())
        
        # Use ACTUAL Kalshi values if available (these are ground truth)
        if self._kalshi_total is not None and self._kalshi_portfolio is not None:
            # Kalshi API provides authoritative values
            available = self._kalshi_cash
            positions_at_risk = self._kalshi_portfolio  # Current value of filled positions
            at_risk = positions_at_risk + pending_at_risk  # Total at risk
            total_value = self._kalshi_total  # Cash + positions
            return_pct_actual = ((self._kalshi_total - self.initial_bankroll) / self.initial_bankroll) * 100
            using_kalshi = True
        else:
            # Fallback to internal calculations (less accurate)
            positions_at_risk = internal_positions
            at_risk = positions_at_risk + pending_at_risk
            available = self.initial_bankroll - at_risk + realized_pnl
            total_value = available + positions_at_risk + unrealized_pnl
            return_pct_actual = None
            using_kalshi = False
        
        # Debug: Log which values are being used (every 10th call to avoid spam)
        if not hasattr(self, '_stats_debug_counter'):
            self._stats_debug_counter = 0
        self._stats_debug_counter += 1
        if self._stats_debug_counter % 10 == 1:
            print(f"[Stats] Source: {'KALSHI API' if using_kalshi else 'INTERNAL'}")
            print(f"[Stats] Cash=${available:.2f}, Positions=${positions_at_risk:.2f}, Total=${total_value:.2f}")
            if using_kalshi:
                print(f"[Stats] Kalshi raw: cash={self._kalshi_cash}, portfolio={self._kalshi_portfolio}, total={self._kalshi_total}")
            print(f"[Stats] Internal positions sum: ${internal_positions:.2f}")
        
        # Calculate at-risk by time horizon (positions only)
        at_risk_ultra_short = 0  # ≤24 hours
        at_risk_short = 0        # 1-7 days
        at_risk_medium = 0       # 8+ days
        now = datetime.utcnow()
        
        for pos in self._positions.values():
            size = pos.get('size', 0)
            market = self._markets.get(pos.get('market_id', ''))
            
            # Calculate time to resolution from end_date if available
            hours_to_res = 9999
            days_to_res = 9999
            
            # Try to get end_date from market or position itself
            end_date_str = None
            if market:
                # Try cached values first
                hours_to_res = market.get('hours_to_resolution', 9999)
                days_to_res = market.get('days_to_resolution', 9999)
                end_date_str = market.get('end_date')
            
            # Fallback: check position's own end_date (stored at entry time)
            if hours_to_res == 9999 and not end_date_str:
                end_date_str = pos.get('end_date')
            
            # Calculate from end_date if we have it
            if hours_to_res == 9999 and end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    time_to_res = end_date.replace(tzinfo=None) - now
                    hours_to_res = time_to_res.total_seconds() / 3600
                    days_to_res = time_to_res.days
                except:
                    pass
            
            # Categorize by time horizon
            if hours_to_res <= 24:
                at_risk_ultra_short += size
            elif days_to_res <= 7:
                at_risk_short += size
            else:
                at_risk_medium += size
        
        exits = [t for t in self._trades if t.get('action') == 'EXIT']
        winning = len([t for t in exits if t.get('pnl', 0) > 0])
        losing = len([t for t in exits if t.get('pnl', 0) < 0])
        
        pnls = [t.get('pnl', 0) for t in exits]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        total_closed = winning + losing
        win_rate = winning / total_closed if total_closed > 0 else 0
        
        # TODAY'S PERFORMANCE - filter by today's date
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        today_exits = [t for t in exits if t.get('timestamp', '').startswith(today_str)]
        today_pnl = sum(t.get('pnl', 0) for t in today_exits)
        today_wins = len([t for t in today_exits if t.get('pnl', 0) > 0])
        today_losses = len([t for t in today_exits if t.get('pnl', 0) < 0])
        today_trades = today_wins + today_losses
        
        # Use actual Kalshi return if available, otherwise calculated
        if return_pct_actual is not None:
            return_pct = return_pct_actual
        else:
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
            'positions_at_risk': positions_at_risk,  # Uses Kalshi value when available
            'pending_at_risk': pending_at_risk,
            'pending_orders': len(self._pending_orders),
            'at_risk_ultra_short': at_risk_ultra_short,
            'at_risk_short': at_risk_short,
            'at_risk_medium': at_risk_medium,
            'total_value': total_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'return_pct': return_pct,
            'kalshi_synced': self._kalshi_total is not None,  # Shows if using real Kalshi data
            # Today's performance
            'today_pnl': today_pnl,
            'today_wins': today_wins,
            'today_losses': today_losses,
            'today_trades': today_trades,
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
            'max_oi_pct': self.max_oi_pct,
            'max_days_to_resolution': self.max_days_to_resolution,
            'max_cluster_positions': int(os.getenv('MAX_CLUSTER_POSITIONS', '3')),
            'profit_lock_pct': float(os.getenv('PROFIT_LOCK_PCT', '0.50')),
            'kill_switch': self._risk_engine.daily_stats.kill_switch_triggered,
            'daily_drawdown': self._risk_engine.daily_stats.current_drawdown_pct,
            'exposure_ratio': at_risk / self.initial_bankroll,
            'trading_allowed': self._risk_engine.is_trading_allowed,
            # Intelligence features
            'use_intelligence': self._use_intelligence,
            'prefer_inefficient': self._prefer_inefficient,
            'use_contrarian': self._use_contrarian,
            'min_inefficiency_score': self._min_inefficiency_score,
            # Intel effectiveness metrics
            'intel_effectiveness': self._get_intel_effectiveness(),
        }
    
    async def start(self):
        """Start the Kalshi bot and dashboard."""
        # Setup web app
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/state', self._handle_state)
        self._app.router.add_get('/api/signals', self._handle_signals)
        self._app.router.add_get('/api/fills', self._handle_fills)
        self._app.router.add_get('/api/performance', self._handle_performance)
        self._app.router.add_get('/api/settlements', self._handle_settlements)
        self._app.router.add_get('/api/debug-reconcile', self._handle_debug_reconcile)
        
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
        print(f"\n{'='*60}")
        print(f"KALSHI BATTLE-BOT [{mode}]")
        print(f"{'='*60}")
        print(f"Platform: Kalshi (CFTC-regulated, US legal)")
        print(f"Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Min Edge: {self.min_edge*100:.1f}%")
        print(f"Max Position: ${self.max_position_size:,.2f}")
        print(f"\n[Intelligence Features]")
        print(f"  News Integration: {'ON' if self._use_intelligence else 'OFF'}")
        # Check if Brave is configured
        try:
            brave_key = self._intelligence.news_service._brave_api_key
            if brave_key:
                print(f"  News Source: Brave Search API + Reddit RSS (configured)")
            else:
                print(f"  News Source: Google News RSS + Reddit RSS (no Brave key)")
        except Exception:
            print(f"  News Source: Unknown")
        print(f"  Prefer Inefficient Markets: {'ON' if self._prefer_inefficient else 'OFF'}")
        print(f"  Contrarian Timing: {'ON' if self._use_contrarian else 'OFF'}")
        print(f"  Min Inefficiency Score: {self._min_inefficiency_score:.2f}")
        if self.simulate_prices:
            print(f"\n⚠️  SIMULATE_PRICES=true (testing mode)")
        print(f"\nDashboard: http://localhost:{self.port}")
        print("Press Ctrl+C to stop\n")
        
        # CRITICAL: Cancel ALL stale resting orders on startup (LIVE mode only)
        if not self.dry_run:
            await self._cancel_all_resting_orders()
        
        # Sync pending orders from previous session (LIVE mode only)
        if not self.dry_run and self._pending_orders:
            print(f"[Startup] Checking {len(self._pending_orders)} pending orders from previous session...")
            await self._sync_pending_orders_on_startup()
        
        # Sync positions with Kalshi to ensure accuracy (LIVE mode only)
        if not self.dry_run:
            await self._sync_positions_with_kalshi()
        
        # DISABLED: Reconciliation was creating buggy P&L data that corrupted dashboard
        # The bot will track trades going forward; historical trades are in Kalshi account
        # if not self.dry_run:
        #     await self._reconcile_settlements_from_fills()
        
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
            await asyncio.sleep(900)  # Refresh every 15 minutes (markets don't change rapidly)
    
    async def _fetch_markets(self):
        """Fetch ALL markets from Kalshi using fast paginated fetch.
        
        Strategy: Single paginated fetch gets ALL open markets (~20 API calls, ~2 min)
        - Uses mve_filter='exclude' to skip combo/parlay markets at API level
        - No need to iterate through 4000+ series individually
        """
        try:
            all_markets = []
            
            # Fetch all open markets - exclude combo markets at API level for efficiency
            print(f"[Fetching] All open markets (excluding combos) with pagination...")
            cursor = None
            pages_fetched = 0
            
            while True:
                try:
                    await asyncio.sleep(0.2)  # Rate limiting
                    result = await self._kalshi.get_markets(
                        status='open',
                        limit=1000,  # Max per page
                        cursor=cursor,
                        exclude_mve=True,  # Skip combo/parlay markets at API level
                    )
                    markets = result.get('markets', [])
                    if not markets:
                        break
                        
                    all_markets.extend(markets)
                    pages_fetched += 1
                    print(f"[Progress] Page {pages_fetched}: {len(all_markets)} markets total")
                    
                    cursor = result.get('cursor')
                    if not cursor:
                        break
                        
                    # Safety limit
                    if pages_fetched >= 50:
                        print(f"[Fetch] Reached page limit (50)")
                        break
                        
                except Exception as e:
                    if '429' in str(e):
                        print(f"[Rate Limited] Pausing for 3 seconds...")
                        await asyncio.sleep(3)
                        continue
                    print(f"[Fetch error] {e}")
                    break
            
            print(f"[Fetch Complete] {len(all_markets)} markets from {pages_fetched} pages")
            
            # Deduplicate by ticker (pagination may have overlap)
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
            
            # Debug: Sample market to verify API fields
            if all_markets:
                sample = all_markets[0]
                print(f"[Debug] Sample market fields (ticker={sample.get('ticker', 'N/A')[:30]}):")
                print(f"  Time: close={sample.get('close_time')}, expected_exp={sample.get('expected_expiration_time')}")
                print(f"  Price: last_price_dollars={sample.get('last_price_dollars')}, last_price={sample.get('last_price')}")
                print(f"  Bid/Ask: yes_bid_dollars={sample.get('yes_bid_dollars')}, yes_ask_dollars={sample.get('yes_ask_dollars')}")
                print(f"  Volume: volume={sample.get('volume')}, volume_24h={sample.get('volume_24h')}, oi={sample.get('open_interest')}")
            
            # Count markets by time field availability
            time_field_stats = {'close_time': 0, 'expected_exp': 0, 'expiration': 0, 'none': 0}
            for m in all_markets:
                if m.get('expected_expiration_time'):
                    time_field_stats['expected_exp'] += 1
                elif m.get('expiration_time'):
                    time_field_stats['expiration'] += 1
                elif m.get('close_time'):
                    time_field_stats['close_time'] += 1
                else:
                    time_field_stats['none'] += 1
            print(f"[Time Fields] expected_exp={time_field_stats['expected_exp']}, expiration={time_field_stats['expiration']}, close_time={time_field_stats['close_time']}, none={time_field_stats['none']}")
            
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
        
        MULTIGAME, PARLAY, COMBO, and MVE markets are multi-leg bets with no liquidity.
        Also checks for multiple player props in the question (e.g., "yes X, yes Y").
        """
        market_id_upper = (market.get('id', '') or market.get('ticker', '')).upper()
        question = market.get('question', '') or market.get('title', '')
        
        # Check ticker patterns
        if any(pattern in market_id_upper for pattern in ['MULTIGAME', 'PARLAY', 'COMBO', 'MVE']):
            return True
        
        # Check for multi-player combo pattern in question: "yes X, yes Y" or multiple "15+", "25+"
        question_lower = question.lower()
        if question_lower.count('yes ') >= 2:  # Multiple "yes" statements = combo
            return True
        if question_lower.count('+,') >= 2:  # Multiple "+," patterns like "15+,yes" = combo
            return True
        
        return False
    
    async def _select_markets(self):
        """Filter markets using eligibility criteria.
        
        PRIORITIZES by time horizon for fastest feedback:
        1. Ultra-short (≤24 hours) - immediate validation
        2. Short-term (1-7 days) - quick feedback
        3. Medium-term (8-365 days) - diversity
        """
        ultra_short = []  # ≤24 hours to resolution
        short_term = []   # 1-7 days to resolution
        medium_term = []  # 8-365 days to resolution
        rejection_counts = {
            'no_end_date': 0, 'too_far_out': 0, 'low_oi': 0,
            'wide_spread': 0, 'extreme_price': 0, 'low_liquidity': 0,
            'combo_market': 0, 'no_volume': 0,
        }
        ultra_short_rejected = {'low_oi': 0, 'wide_spread': 0, 'extreme_price': 0, 'too_close': 0, 'combo': 0, 'no_volume': 0}
        
        max_days = self.max_days_to_resolution  # shared with trading-loop filter (default 45)
        now = datetime.utcnow()
        
        # Pre-scan to understand what's available
        pre_scan = {'ultra': 0, 'short': 0, 'medium': 0, 'no_end': 0, 'past': 0}
        ultra_examples = []
        for m in self._markets.values():
            end_date_str = m.get('end_date')
            if not end_date_str:
                pre_scan['no_end'] += 1
                continue
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                hours = (end_date.replace(tzinfo=None) - now).total_seconds() / 3600
                if hours < 0:
                    pre_scan['past'] += 1
                elif hours <= 24:
                    pre_scan['ultra'] += 1
                    if len(ultra_examples) < 5:
                        ultra_examples.append({
                            'ticker': m.get('id', '')[:30],
                            'hours': hours,
                            'question': m.get('question', '')[:40],
                            'oi': m.get('open_interest', 0),
                            'spread': m.get('spread', 0),
                            'price': m.get('price', 0.5),
                        })
                elif hours <= 168:  # 7 days
                    pre_scan['short'] += 1
                else:
                    pre_scan['medium'] += 1
            except Exception as e:
                if pre_scan['no_end'] < 3:
                    print(f"[Debug] Failed to parse end_date '{end_date_str}': {e}")
        print(f"[Pre-Filter] Total markets by time: ultra(≤24h)={pre_scan['ultra']}, short(1-7d)={pre_scan['short']}, medium(8+d)={pre_scan['medium']}, no_end={pre_scan['no_end']}, past={pre_scan['past']}")
        
        # Show examples of ultra-short markets BEFORE filtering
        if ultra_examples:
            print(f"[Ultra-Short Examples (pre-filter)]")
            for ex in ultra_examples:
                print(f"  {ex['hours']:.1f}h | oi={ex['oi']} | spread={ex['spread']:.2%} | price={ex['price']:.0%} | {ex['question']}...")
        
        for m in self._markets.values():
            if not m.get('end_date'):
                rejection_counts['no_end_date'] += 1
                continue
            
            # Calculate time to resolution in hours AND days
            try:
                end_date = datetime.fromisoformat(m['end_date'].replace('Z', '+00:00'))
                time_to_resolution = end_date.replace(tzinfo=None) - now
                hours_to_resolution = time_to_resolution.total_seconds() / 3600
                days_to_resolution = time_to_resolution.days
                
                if days_to_resolution > max_days:
                    rejection_counts['too_far_out'] += 1
                    continue
                if hours_to_resolution < 1:  # Less than 1 hour - too close
                    if hours_to_resolution > 0:
                        ultra_short_rejected['too_close'] += 1
                    continue
            except:
                continue
            
            # Filter out MULTIGAME combo/parlay markets - they have no liquidity
            if self._is_combo_market(m):
                rejection_counts['combo_market'] += 1
                if hours_to_resolution <= 24:
                    ultra_short_rejected['combo'] += 1
                continue
            
            # Require some trading activity (volume) - markets with 0 volume have no liquidity
            volume = m.get('volume', 0) or m.get('volume_24h', 0) or 0
            min_volume = int(os.getenv('MIN_VOLUME', '1'))  # At least 1 contract traded
            if volume < min_volume:
                rejection_counts['no_volume'] += 1
                if hours_to_resolution <= 24:
                    ultra_short_rejected['no_volume'] += 1
                continue
            
            # Minimum open interest - lower bar for ultra-short (more forgiving)
            oi = m.get('open_interest', 0) or 0
            min_oi = int(os.getenv('MIN_OPEN_INTEREST', '10'))
            # Lower OI requirement for ultra-short markets (1/4 of normal)
            effective_min_oi = max(1, min_oi // 4) if hours_to_resolution <= 24 else min_oi
            if oi < effective_min_oi:
                rejection_counts['low_oi'] += 1
                if hours_to_resolution <= 24:
                    ultra_short_rejected['low_oi'] += 1
                if rejection_counts['low_oi'] <= 3:
                    print(f"[Debug] Rejected for low_oi: {m.get('ticker', '')[:30]} oi={oi} (need {effective_min_oi})")
                continue
            
            # Spread check - more lenient for ultra-short (daily markets have wider spreads)
            max_spread = float(os.getenv('MAX_SPREAD', '0.06'))
            # Allow 2x wider spread for ultra-short markets (daily events)
            effective_max_spread = max_spread * 2 if hours_to_resolution <= 24 else max_spread
            # Default to 1.0 (100%) if missing - require valid spread data
            actual_spread = m.get('spread', 1.0)
            if actual_spread > effective_max_spread:
                rejection_counts['wide_spread'] += 1
                if hours_to_resolution <= 24:
                    ultra_short_rejected['wide_spread'] += 1
                    if ultra_short_rejected['wide_spread'] <= 3:
                        print(f"[Debug] Ultra-short rejected for spread: {m.get('ticker', '')[:30]} spread={actual_spread:.2%} (max {effective_max_spread:.2%})")
                continue
            
            # Price range - more lenient for ultra-short (daily events can trade near extremes)
            price = m.get('price', 0.5)
            if hours_to_resolution <= 24:
                # Ultra-short: allow 3-97¢ range
                if price < 0.03 or price > 0.97:
                    rejection_counts['extreme_price'] += 1
                    ultra_short_rejected['extreme_price'] += 1
                    if ultra_short_rejected['extreme_price'] <= 3:
                        print(f"[Debug] Ultra-short rejected for price: {m.get('ticker', '')[:30]} price={price:.0%}")
                    continue
            else:
                # Standard: 5-95¢ range
                if price < 0.05 or price > 0.95:
                    rejection_counts['extreme_price'] += 1
                    continue
            
            # Store time info
            m['hours_to_resolution'] = hours_to_resolution
            m['days_to_resolution'] = days_to_resolution
            
            # Calculate inefficiency score for prioritization
            if self._prefer_inefficient:
                ineff_score, ineff_reasons = self._intelligence.inefficiency_detector.calculate_inefficiency_score(
                    volume=volume,
                    spread=m.get('spread', 0.02),
                    open_interest=oi,
                    price=m.get('price', 0.5),
                    recent_price_change=0.0,  # Will be filled by intelligence service later
                )
                m['inefficiency_score'] = ineff_score
                m['inefficiency_reasons'] = ineff_reasons
            else:
                m['inefficiency_score'] = 0.0
                m['inefficiency_reasons'] = []
            
            # Categorize by time horizon
            if hours_to_resolution <= 24:
                ultra_short.append(m)
            elif days_to_resolution <= 7:
                short_term.append(m)
            else:
                medium_term.append(m)
        
        # Sort each category by composite score:
        # - Inefficiency score (higher = better opportunity)
        # - Open interest (higher = more liquid for execution)
        # - Category bonus (weather=100% win, sports_winner=28% win)
        # Balance: inefficiency × 0.5 + normalized_oi × 0.3 + category_bonus × 0.2
        def market_score(m):
            ineff = m.get('inefficiency_score', 0) * 0.5
            oi = min(m.get('open_interest', 0) / 1000, 1.0) * 0.3
            q = m.get('question', '').lower()
            ticker = m.get('id', '').upper()

            # Category bonus: reward markets where Claude has real signal
            # Penalize markets where Claude has <30% confidence historically
            category_bonus = 0.0
            if any(x in q for x in ['doge', 'trump', 'fed ', 'rate cut', 'deportat', 'congress',
                                      'senate', 'executive order', 'tariff', 'election', 'fomc']):
                category_bonus = 1.0   # Politics/policy: highest edge opportunity
            elif any(x in q for x in ['gdp', 'cpi', 'inflation', 'unemployment', 'jobs report']):
                category_bonus = 0.8   # Economics: good signal from FRED data
            elif any(x in q for x in ['bitcoin price', 'ethereum price', 'btc price', 'eth price',
                                       'bitcoin range', 'btc range']) \
                 or any(x in ticker for x in ['KXBTC', 'KXETH', 'KXBTCD', 'KXETHD']):
                category_bonus = -1.0  # Intraday crypto price: <30% confidence, always at cluster cap
            elif any(x in q for x in ['snow', 'temperature', 'high temp', 'high of', 'low of', '°', 'degrees']):
                category_bonus = -1.0  # Weather: currently losing, overly concentrated
            elif any(sport in q for sport in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'soccer', 'golf', 'tennis']):
                category_bonus = -2.0  # Sports: blocked anyway, lowest priority

            return ineff + oi + (category_bonus * 0.2)
        
        if self._prefer_inefficient:
            ultra_short.sort(key=market_score, reverse=True)
            short_term.sort(key=market_score, reverse=True)
            medium_term.sort(key=market_score, reverse=True)
        else:
            # Original behavior: sort by OI only
            ultra_short.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
            short_term.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
            medium_term.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
        
        # PRIORITIZE: Short-term political/policy markets first, then ultra-short, then medium
        # Ultra-short cap at 10: intraday crypto/weather hog slots with low-signal analysis
        selected = short_term[:40] + ultra_short[:10] + medium_term[:20]
        
        # Log what we found
        print(f"[Time Horizon] Ultra-short (≤24h): {len(ultra_short)} | Short (1-7d): {len(short_term)} | Medium (8-365d): {len(medium_term)}")
        
        if ultra_short:
            print(f"[Ultra-short Markets - resolving within 24h]")
            for m in ultra_short[:5]:
                hours = m.get('hours_to_resolution', 0)
                oi = m.get('open_interest', 0)
                vol = m.get('volume', 0) or m.get('volume_24h', 0) or 0
                ticker = m.get('id', '')[:25]
                q = m.get('question', '')[:35]
                print(f"  {hours:.1f}h | oi={oi} vol={vol} | {ticker} | {q}...")
        
        if short_term and not ultra_short:
            print(f"[Short-term Top 3]")
            for m in short_term[:3]:
                days = m.get('days_to_resolution', '?')
                oi = m.get('open_interest', 0)
                q = m.get('question', '')[:45]
                print(f"  {days}d | oi=${oi:,} | {q}...")
        
        self._monitored = {m['id']: m for m in selected}
        
        total_eligible = len(ultra_short) + len(short_term) + len(medium_term)
        ultra_count = len([m for m in selected if m.get('hours_to_resolution', 999) <= 24])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market eligibility: {total_eligible} eligible / {len(self._markets)} total")
        print(f"[Monitoring] {len(selected)} markets selected ({ultra_count} ultra-short, resolving in hours)")
        
        if rejection_counts:
            reasons = [f"{k}={v}" for k, v in rejection_counts.items() if v > 0]
            if reasons:
                print(f"[Filter] Rejected: {', '.join(reasons)}")
        
        # Log ultra-short specific rejections to help debug
        ultra_rejected = [f"{k}={v}" for k, v in ultra_short_rejected.items() if v > 0]
        if ultra_rejected:
            print(f"[Ultra-Short Debug] Rejected: {', '.join(ultra_rejected)}")
    
    async def _trading_loop(self):
        """Main trading loop - analyze markets and enter positions.
        
        PRIORITIZES ultra-short markets (≤24h) for immediate feedback.
        """
        await asyncio.sleep(5)  # Wait for initial market fetch
        
        loop_counter = 0
        INTEL_LOG_INTERVAL = 50  # Log intel stats every ~50 loops
        
        while self._running:
            try:
                if not self._risk_engine.is_trading_allowed:
                    await asyncio.sleep(60)
                    continue
                
                # Sort monitored markets by:
                # 1. Time to resolution (shorter = more urgent)
                # 2. News relevance score (higher = news matters more)
                # Combined score: lower time + higher news relevance = higher priority
                from services.market_intelligence import NewsService
                
                def market_priority(m):
                    hours = m.get('hours_to_resolution', 9999)
                    news_score = NewsService.score_news_relevance(
                        m.get('question', ''),
                        m.get('category')
                    )
                    q = m.get('question', '').lower()
                    ticker = m.get('id', '').upper()

                    # Heavily deprioritize intraday crypto price and weather markets:
                    # Claude has <30% confidence on these, and we're usually at cluster cap.
                    # Push them to the back of the queue so political/policy markets
                    # get analyzed first every cycle.
                    is_crypto_price = any(x in q for x in ['bitcoin price', 'ethereum price', 'btc price', 'eth price', 'bitcoin range', 'btc range']) \
                        or any(x in ticker for x in ['KXBTC', 'KXETH', 'KXBTCD', 'KXETHD'])
                    is_weather = any(x in q for x in ['snow', 'temperature', 'high temp', 'high of', 'low of', '°', 'degrees'])
                    depriority = 10.0 if (is_crypto_price or is_weather) else 0.0

                    time_score = min(hours, 168) / 168  # Normalize to 0-1 (cap at 1 week)
                    return depriority + time_score - (news_score * 0.3)
                
                markets_by_urgency = sorted(
                    self._monitored.values(),
                    key=market_priority
                )
                
                for market in markets_by_urgency:
                    market_id = market.get('id')
                    if not market_id:
                        continue
                    
                    # Skip markets where news has very low value or outcomes are predictable
                    question_lower = market.get('question', '').lower()
                    question_raw = market.get('question', '')
                    
                    # FILTER 1: Skip entertainment/noise markets
                    if any(term in question_lower for term in ['mention', 'announcer', 'say during', 'tweet', 'post about']):
                        continue
                    
                    # FILTER 2: Skip ALL player prop markets with low thresholds
                    # These are usually very predictable (star players hit basic stats)
                    player_prop_patterns = [
                        ': 1+', ':1+', ': 2+', ':2+', ': 3+', ':3+',
                        ': 4+', ':4+', ': 5+', ':5+',
                        'three pointers', 'three-pointers',
                        '+ points', '+ rebounds', '+ assists', '+ steals', '+ blocks',
                        'points or more', 'rebounds or more', 'assists or more',
                    ]
                    if any(pattern in question_lower or pattern in question_raw for pattern in player_prop_patterns):
                        continue
                    
                    # FILTER 2b: Skip point spreads and totals - too volatile, no edge
                    point_spread_patterns = [
                        'total points', 'over 1', 'over 2', 'over 3', 'over 4', 'over 5',
                        'over 6', 'over 7', 'over 8', 'over 9',
                        'wins by over', 'wins by under', 'point spread',
                        '.5 points', 'points scored',
                    ]
                    if any(pattern in question_lower for pattern in point_spread_patterns):
                        continue
                    
                    # FILTER 2c: HARD BLOCK ALL SPORTS - historically poor (28% win rate, -$32 losses)
                    # Sports outcomes are too unpredictable, no sustainable edge
                    sports_patterns = [
                        # Basketball
                        'nba', 'ncaa', 'basketball', 'wnba', 'nbl',
                        # Football
                        'nfl', 'football', 'touchdown', 'quarterback',
                        # Baseball
                        'mlb', 'baseball',
                        # Hockey
                        'nhl', 'hockey',
                        # Soccer
                        'soccer', 'premier league', 'champions league', 'mls',
                        # Golf
                        'golf', 'pga', 'lpga', 'genesis invitational',
                        # Tennis
                        'tennis', 'atp', 'wta',
                        # Fighting
                        'ufc', 'boxing', 'mma', 'fight', 'bout', 'knockout', 'k.o.',
                        'heavyweight', 'lightweight', 'middleweight', 'welterweight',
                        # Generic vs. matchup (two fighters/teams)
                        ' vs ', ' vs. ',
                        # Other sports
                        'f1', 'nascar', 'olympics', 'world cup', 'world series',
                        'super bowl', 'championship', 'tournament', 'playoffs',
                        # Generic sports terms
                        'wins the match', 'win the game', 'beat ', 'defeats ',
                        'rebounds', 'assists', 'three-pointers', '3-pointers',
                        'pitcher', 'batter', 'innings', 'overtime', 'penalty kick',
                    ]
                    if any(pattern in question_lower for pattern in sports_patterns):
                        continue  # SKIP ALL SPORTS - no edge, high losses
                    
                    # Also check ticker patterns for sports
                    ticker_upper = market_id.upper()
                    sports_ticker_patterns = [
                        'KXNBA', 'KXNFL', 'KXMLB', 'KXNHL', 'KXNCAA', 'KXPGA', 'KXLPGA',
                        'KXUFC', 'KXBOX', 'KXTEN', 'KXSOC', 'KXWNH', 'KXWOH',
                        'KXDPWORLD', 'KXNBL', 'KXBRASIL', 'KXWCC', 'KXFIGHT',
                        'KXMATCH', 'KXBOUT', 'KXCHAMP',
                    ]
                    if any(pattern in ticker_upper for pattern in sports_ticker_patterns):
                        continue  # SKIP sports by ticker pattern

                    # FILTER: Skip foreign elections with no reliable intelligence
                    # These consistently produce LOW_CONFIDENCE signals and waste API credits
                    # Filter markets by explicit country/region name — avoids false positives
                    # from generic terms like "house of rep" matching foreign parliaments
                    no_intel_patterns = [
                        # Foreign countries/elections
                        'nepal', 'kenya', 'nigeria', 'pakistan', 'bangladesh',
                        'ethiopia', 'myanmar', 'cambodia', 'laos', 'mozambique',
                        'zimbabwe', 'zambia', 'malawi', 'botswana', 'namibia',
                        'colombian chamber', 'colombia election',
                        # Foreign central banks — consistently LOW_CONFIDENCE, no news edge
                        "people's bank of china", 'pboc', 'bank of china cut',
                        'ecb cut', 'bank of england cut', 'bank of japan cut',
                    ]
                    if any(p in question_lower for p in no_intel_patterns):
                        continue  # Skip: no reliable news intelligence for this market

                    # FILTER 3: Skip markets where probability is extreme (< 10% or > 90%)
                    # These have low expected value and high variance
                    market_price = market.get('price', 0.5)
                    yes_price = market.get('yes_price', market_price)
                    no_price = market.get('no_price', 1 - market_price)
                    
                    if yes_price < 0.10 or yes_price > 0.90:
                        continue  # Skip extreme probability markets
                    
                    # FILTER 4: Skip markets with low volume (need real liquidity)
                    volume = market.get('volume', 0) or market.get('open_interest', 0) or 0
                    if volume < 500:
                        continue  # Skip illiquid markets - need 500+ for reliable exits
                    
                    # FILTER 5: Skip narrow-range weather/temperature markets (coin flips)
                    weather_patterns = ['temperature', '° to ', '°-', 'degrees', 'high of', 'low of']
                    if any(p in question_lower for p in weather_patterns):
                        continue  # Weather ranges are unpredictable coin flips

                    # FILTER 6: Skip markets resolving too far out (high uncertainty, hard to lock gains)
                    hours_to_res_check = market.get('hours_to_resolution', 0)
                    max_hours = self.max_days_to_resolution * 24
                    if hours_to_res_check > max_hours:
                        continue  # Too far out — too much can change before resolution
                        
                    if len(self._positions) >= self._risk_limits.max_positions:
                        break

                    # PRE-FILTER: Skip Claude call if we know the market will be rejected anyway.
                    # These checks mirror _analyze_market but cost nothing vs an API call.

                    # Already holding this market
                    if market_id in [p.get('market_id') for p in self._positions.values()]:
                        continue

                    # Recently exited — still in cooldown
                    recent_exit = self._recently_exited.get(market_id)
                    if recent_exit:
                        exit_reason_stored = self._recently_exited_reason.get(market_id, '')
                        cooldown_hours = 6.0 if 'PROFIT_LOCK' in exit_reason_stored or 'NEAR_SETTLEMENT' in exit_reason_stored else 2.0
                        if (datetime.utcnow() - recent_exit).total_seconds() / 3600 < cooldown_hours:
                            continue

                    # Cluster cap reached — no point asking Claude
                    _pre_cluster_key = self._get_cluster_key(market.get('question', ''))
                    _pre_cluster_count = sum(
                        1 for p in self._positions.values()
                        if self._get_cluster_key(p.get('question', '')) == _pre_cluster_key
                    )
                    MAX_POSITIONS_PER_CLUSTER = int(os.getenv('MAX_CLUSTER_POSITIONS', '3'))
                    if _pre_cluster_count >= MAX_POSITIONS_PER_CLUSTER:
                        continue

                    # Check cooldown and price movement
                    last = self._last_analysis.get(market_id)
                    last_price = self._last_analysis_price.get(market_id, 0)
                    current_price = market.get('price', 0.5)

                    price_moved = abs(current_price - last_price) >= self._price_change_threshold if last_price else False

                    # Tiered cooldown based on time horizon — medium-term markets need less frequent re-analysis
                    hours_to_res = market.get('hours_to_resolution', 9999)
                    if hours_to_res <= 24:
                        cooldown = 300          # 5 min — ultra-short, price moves fast
                    elif hours_to_res <= 168:
                        cooldown = 3600         # 60 min — short-term (1-7 days)
                    else:
                        cooldown = 7200         # 2 hours — medium-term (7-45 days)

                    if last and not price_moved:
                        if (datetime.utcnow() - last).total_seconds() < cooldown:
                            continue

                    # Log time horizon and news relevance for visibility
                    news_relevance = NewsService.score_news_relevance(
                        market.get('question', ''),
                        market.get('category')
                    )
                    time_label = "ULTRA-SHORT" if hours_to_res <= 24 else "SHORT" if hours_to_res <= 168 else "MEDIUM"
                    if hours_to_res <= 168 or news_relevance > 0.6:
                        print(f"[{time_label}] {hours_to_res:.1f}h | News relevance: {news_relevance:.2f}")

                    await self._analyze_market(market)
                    self._last_analysis[market_id] = datetime.utcnow()
                    self._last_analysis_price[market_id] = current_price
                    await asyncio.sleep(2)
                    
                # Periodic intel status logging
                loop_counter += 1
                if loop_counter >= INTEL_LOG_INTERVAL:
                    loop_counter = 0
                    try:
                        news_stats = self._intelligence.news_service.get_usage_stats()
                        intel_eff = self._get_intel_effectiveness()
                        print(f"\n[INTEL STATUS] Brave searches: {news_stats.get('brave_searches', 0)}, Cache size: {news_stats.get('cache_size', 0)}")
                        if intel_eff.get('with_news', {}).get('trades', 0) > 0 or intel_eff.get('without_news', {}).get('trades', 0) > 0:
                            print(intel_eff['summary'])
                    except Exception as e:
                        print(f"[INTEL STATUS] Error: {e}")
                    
            except Exception as e:
                print(f"[Trading Error] {e}")
            await asyncio.sleep(30)
    
    def _get_cluster_key(self, question: str) -> str:
        """Map a market question to a correlation cluster key.
        
        Markets in the same cluster move together (same underlying news driver).
        We cap how many positions we hold per cluster to avoid over-concentration.
        """
        q = question.lower()
        # DOGE / federal spending cuts
        if any(x in q for x in ['doge', 'elon', 'federal spending', 'budget cut', 'cut the budget',
                                  'cut between', 'cut less than', 'cut more than']):
            return 'doge_spending'
        # Deportations / immigration
        if any(x in q for x in ['deport', 'immigration', 'migrant', 'border']):
            return 'deportation'
        # State of the Union / congressional events — check BEFORE trump_policy so
        # "Will X attend the State of the Union" doesn't get swallowed by trump_policy
        if any(x in q for x in ['state of the union', 'sotu', 'inaugur', 'joint session', 'attend the']):
            return 'political_events'
        # Trump nickname / mention markets (separate cluster from policy actions)
        if any(x in q for x in ['trump say', 'trump mention', 'trump tweet', 'trump post',
                                  'trump nickname', 'will trump use', 'trump refer']):
            return 'trump_speech'
        # Trump meetings / travel
        if any(x in q for x in ['trump meet', 'trump trip', 'mar-a-lago', 'trump travel',
                                  'trump visit', 'days at']):
            return 'trump_travel'
        # Trump executive actions (broad)
        if any(x in q for x in ['trump', 'executive order', 'tariff', 'maga']):
            return 'trump_policy'
        # Fed / US interest rates
        if any(x in q for x in ['federal reserve', 'fomc', 'rate cut', 'rate hike',
                                  'fed funds', 'basis point']):
            return 'fed_rates'
        # China / international central banks (separate cluster from US Fed)
        if any(x in q for x in ["people's bank", "pboc", "china rate", "ecb", "bank of england",
                                  "bank of japan", "boj"]):
            return 'intl_central_banks'
        # Crypto price buckets (BTC/ETH)
        if any(x in q for x in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana']):
            return 'crypto_price'
        # GDP / economic indicators (same release)
        if any(x in q for x in ['gdp', 'recession', 'inflation', 'cpi', 'unemployment',
                                  'jobs report', 'payroll', 'nonfarm']):
            return 'macro_economics'
        # Weather — per city to avoid over-concentration
        for city in ['new york', 'chicago', 'los angeles', 'houston', 'miami',
                     'philadelphia', 'boston', 'seattle', 'dallas', 'atlanta']:
            if city in q:
                return f'weather_{city.replace(" ", "_")}'
        if any(x in q for x in ['snow', 'temperature', 'high temp', 'precipitation', 'rainfall']):
            return 'weather_other'
        # Sports same league/team
        if any(x in q for x in ['nba', 'lakers', 'celtics', 'warriors', 'nets']):
            return 'nba'
        if any(x in q for x in ['nfl', 'chiefs', 'eagles', 'cowboys', '49ers']):
            return 'nfl'
        if any(x in q for x in ['mlb', 'yankees', 'dodgers', 'astros']):
            return 'mlb'
        # Middle East / geopolitics
        if any(x in q for x in ['israel', 'gaza', 'hamas', 'iran', 'ukraine', 'russia',
                                  'taiwan', 'china war', 'ceasefire']):
            return 'geopolitics'
        return 'other'

    async def _analyze_market(self, market: dict):
        """Analyze market with AI and decide whether to trade.
        
        Enhanced with:
        1. Market intelligence (news, domain data)
        2. Inefficiency detection
        3. Contrarian timing signals
        """
        market_id = market['id']
        question = market.get('question', '')[:50]
        current_price = market.get('price', 0.5)
        
        print(f"[AI] Analyzing: {question}...")
        self._ai_calls += 1
        
        # Step 1: Gather market intelligence (news, domain data, overreaction detection)
        intel: MarketIntelligence = None
        if self._use_intelligence:
            try:
                intel = await self._intelligence.gather_intelligence(
                    market_id=market_id,
                    market_question=market.get('question', ''),
                    current_price=current_price,
                    spread=market.get('spread', 0.02),
                    volume=market.get('volume_24h', 0),
                    open_interest=market.get('open_interest', 0),
                    category=market.get('category'),
                )
                if intel.news_items:
                    print(f"[Intel] Found {len(intel.news_items)} news items, inefficiency={intel.inefficiency_score:.2f}")
                if intel.overreaction_detected:
                    print(f"[Intel] OVERREACTION detected: {intel.overreaction_direction} {intel.overreaction_magnitude:.1%}")
            except Exception as e:
                print(f"[Intel] Failed to gather: {e}")
                intel = None
        
        # Build overreaction info string for AI
        overreaction_info = None
        if intel and intel.overreaction_detected:
            overreaction_info = (
                f"ALERT: Market moved {intel.overreaction_magnitude:.1%} {intel.overreaction_direction} recently. "
                f"Price change 24h: {intel.recent_price_change:+.1%}. "
                f"This could be an overreaction - consider if the move is justified."
            )
        
        # Step 2: Get historical performance for learning
        historical = self._get_historical_performance()

        # Step 3: Get AI signal — use probability cache if price is stable (saves API credits)
        # Cache is valid for up to 2h if price moved <1.5¢ since last call
        cached = self._prob_cache.get(market_id)
        use_cache = False
        if cached:
            cache_time, cache_price, cache_prob, cache_conf = cached
            price_drift = abs(current_price - cache_price)
            cache_age = (datetime.utcnow() - cache_time).total_seconds()
            hours_to_res = market.get('hours_to_resolution', 9999)
            max_cache_age = 300 if hours_to_res <= 24 else 3600  # 5 min for ultra-short, 1 hr otherwise
            if price_drift < 0.015 and cache_age < max_cache_age:
                use_cache = True

        if use_cache:
            # Reconstruct a minimal signal result from cache — no API call needed
            from logic.ai_signal import TradeSignal
            cached_signal = TradeSignal(
                raw_prob=cache_prob,
                confidence=cache_conf,
                key_reasons=["(cached — price unchanged)"],
            )
            class _CachedResult:
                success = True
                error = None
                latency_ms = 0
                signal = cached_signal
            result = _CachedResult()
            signal = cached_signal
            self._ai_successes += 1
            print(f"[AI] {question[:45]}... | ♻ CACHED (drift {abs(current_price - cache_price):.3f})")
        else:
            try:
                result = await self._ai_generator.generate_signal(
                    market_question=market.get('question', ''),
                    current_price=current_price,
                    spread=market.get('spread', 0.02),
                    resolution_rules=market.get('rules', '') or market.get('description', ''),
                    volume_24h=market.get('volume_24h', 0),
                    category=market.get('category'),
                    # Intelligence data
                    news_summary=intel.news_summary if intel else None,
                    domain_summary=intel.domain_summary if intel else None,
                    recent_price_change=intel.recent_price_change if intel else 0.0,
                    overreaction_info=overreaction_info,
                    # Historical performance for learning
                    historical_performance=historical.get('summary') if historical.get('total_trades', 0) > 0 else None,
                )
            
                # Check if AI call succeeded
                if not result or not result.success or not result.signal:
                    error_msg = result.error if result else "No response"
                    print(f"[AI] FAILED: {error_msg}")
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
                # Store in probability cache for reuse when price is stable
                self._prob_cache[market_id] = (datetime.utcnow(), current_price, signal.raw_prob, signal.confidence)

            except Exception as e:
                print(f"[AI] FAILED: {e}")
                return

        if not signal:
            return
        
        # Step 3: Calibrate probability (async call)
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
        
        # Step 4: Determine trade side and edge using ACTUAL prices from Kalshi
        yes_price = market.get('yes_price', current_price)
        no_price = market.get('no_price', 1 - current_price)  # Fallback if not available
        
        if adjusted_prob > yes_price:
            # Bet YES: AI thinks YES is more likely than the market price implies
            side = 'YES'
            edge = adjusted_prob - yes_price
            trade_prob = adjusted_prob
            trade_price = yes_price
        else:
            # Bet NO: AI thinks NO is more likely than the market price implies
            side = 'NO'
            no_prob = 1 - adjusted_prob  # Our belief in NO winning
            edge = no_prob - no_price    # Edge = belief - cost
            trade_prob = no_prob         # Our belief in NO winning
            trade_price = no_price       # Actual cost to buy NO
        
        # Step 5: Apply contrarian edge adjustment
        contrarian_multiplier = 1.0
        if self._use_contrarian and intel and intel.overreaction_detected:
            ai_predicted_direction = "yes" if side == "YES" else "no"
            contrarian_multiplier = self._intelligence.contrarian_detector.get_contrarian_edge_boost(
                is_overreaction=intel.overreaction_detected,
                overreaction_direction=intel.overreaction_direction,
                ai_predicted_direction=ai_predicted_direction,
                magnitude=intel.overreaction_magnitude,
            )
            if contrarian_multiplier != 1.0:
                print(f"[Contrarian] Edge multiplier: {contrarian_multiplier:.2f}x ({'BOOST' if contrarian_multiplier > 1 else 'REDUCE'})")
            edge = edge * contrarian_multiplier
        
        # Store analysis with intelligence data
        analysis_record = {
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
        }
        
        # Add intelligence data if available
        if intel:
            analysis_record['has_intel'] = True
            analysis_record['news_count'] = len(intel.news_items)
            analysis_record['inefficiency_score'] = intel.inefficiency_score
            analysis_record['overreaction'] = intel.overreaction_detected
            analysis_record['contrarian_multiplier'] = contrarian_multiplier
            if intel.inefficiency_reasons:
                analysis_record['inefficiency_reasons'] = intel.inefficiency_reasons[:3]
        
        self._analyses.insert(0, analysis_record)
        self._analyses = self._analyses[:50]
        
        # Log signal for backtesting (track ALL signals regardless of trade decision)
        signal_entry = {
            'market_id': market_id,
            'question': question[:80],
            'edge': edge,
            'confidence': signal.confidence,
            'side': side,
            'market_price': current_price,
            'ai_probability': signal.raw_prob,
            'close_time': market.get('close_time'),
            'timestamp': datetime.utcnow().isoformat(),
            'outcome': None,  # To be filled when market settles
            'outcome_checked': False,
        }
        self._signal_log.append(signal_entry)
        
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
        
        # Don't re-enter markets we recently exited
        # Cooldown varies by exit reason:
        #  - PROFIT_LOCK exit: 6h cooldown (position won, don't re-buy same thing)
        #  - Other exits: 2h cooldown (prevent chasing)
        recent_exit = self._recently_exited.get(market_id)
        if recent_exit:
            exit_reason_stored = self._recently_exited_reason.get(market_id, '')
            cooldown_hours = 6.0 if 'PROFIT_LOCK' in exit_reason_stored or 'NEAR_SETTLEMENT' in exit_reason_stored else 2.0
            hours_since_exit = (datetime.utcnow() - recent_exit).total_seconds() / 3600
            if hours_since_exit < cooldown_hours:
                should_trade = False
                reasons.append(f'RECENTLY_EXITED_{hours_since_exit:.1f}h_AGO_{exit_reason_stored[:20]}')
        
        if len(self._positions) >= self._risk_limits.max_positions:
            should_trade = False
            reasons.append('MAX_POSITIONS')
        
        # CORRELATION CLUSTER CAP: Limit exposure to same news theme
        # Prevents piling into 10 DOGE-cut variants when 2-3 are sufficient.
        MAX_POSITIONS_PER_CLUSTER = int(os.getenv('MAX_CLUSTER_POSITIONS', '3'))
        cluster_key = self._get_cluster_key(market.get('question', ''))
        cluster_count = sum(
            1 for p in self._positions.values()
            if self._get_cluster_key(p.get('question', '')) == cluster_key
        )
        if cluster_count >= MAX_POSITIONS_PER_CLUSTER:
            should_trade = False
            reasons.append(f'CLUSTER_CAP_{cluster_key[:20]}_{cluster_count}')
        
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
            
            # Liquidity cap: don't exceed X% of open interest
            open_interest = market.get('open_interest', 0) or 0
            if open_interest > 0 and self.max_oi_pct > 0:
                # Each contract costs ~trade_price, so max_contracts = OI * max_oi_pct
                max_contracts_by_liquidity = int(open_interest * self.max_oi_pct)
                max_size_by_liquidity = max_contracts_by_liquidity * trade_price
                if position_size > max_size_by_liquidity:
                    print(f"[Liquidity Cap] ${position_size:.2f} → ${max_size_by_liquidity:.2f} (OI={open_interest}, max {self.max_oi_pct*100:.0f}%)")
                    position_size = max_size_by_liquidity
            
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
        
        # STRICT LIMITS - prevent runaway orders
        MAX_CONTRACTS_PER_ORDER = 25  # Max 25 contracts - bigger bets on quality opportunities
        MIN_PRICE_CENTS = 15  # Don't trade below 15¢ - cheap contracts have 6% win rate
        MAX_PRICE_CENTS = 95  # Allow up to 95¢ - data shows expensive NO still profitable (+4% ROI)
        
        # Use the correct price for the side we're trading
        if side.upper() == 'YES':
            entry_price = market.get('yes_price', market['price'])
        else:
            entry_price = market.get('no_price', 1 - market['price'])
        
        # Convert to cents for validation
        price_cents = int(entry_price * 100)
        
        # VALIDATION: Skip extreme prices
        if price_cents < MIN_PRICE_CENTS:
            print(f"[Order] SKIPPED: Price {price_cents}¢ too low (min {MIN_PRICE_CENTS}¢) - too risky")
            return
        if price_cents > MAX_PRICE_CENTS:
            print(f"[Order] SKIPPED: Price {price_cents}¢ too high (max {MAX_PRICE_CENTS}¢) - not enough upside")
            return
        
        # Calculate contracts with STRICT LIMIT
        contracts = int(size / entry_price) if entry_price > 0 else 0
        if contracts > MAX_CONTRACTS_PER_ORDER:
            print(f"[Order] Capping contracts: {contracts} → {MAX_CONTRACTS_PER_ORDER} (max per order)")
            contracts = MAX_CONTRACTS_PER_ORDER
        
        # Recalculate actual size based on capped contracts
        actual_size = contracts * entry_price
        
        # In LIVE mode, actually place the order on Kalshi
        order_id = None
        if not self.dry_run:
            try:
                if contracts < 1:
                    print(f"[Order] Size ${size:.2f} too small for 1 contract at {price_cents}¢")
                    return
                
                # Use AGGRESSIVE slippage to ensure orders fill quickly
                # Slippage: 10¢ or 15%, whichever is larger - this ensures fills
                slippage_cents = max(10, int(price_cents * 0.15))
                order_price_cents = min(price_cents + slippage_cents, 95)
                
                print(f"[Order] Placing: {contracts} {side} @ {order_price_cents}¢ (market: {price_cents}¢, slippage: {slippage_cents}¢)")
                
                result = await self._kalshi.place_order(
                    ticker=market_id,
                    side=side.lower(),  # 'yes' or 'no'
                    count=contracts,
                    price=order_price_cents,
                    order_type='limit',
                )
                order_id = result.get('order', {}).get('order_id')
                print(f"[LIVE ORDER] Placed: {contracts} {side} @ {order_price_cents}¢ | Order ID: {order_id}")
            except Exception as e:
                print(f"[Order Error] Failed to place order on Kalshi: {e}")
                return  # Don't record if order failed
        
        # Update size to actual
        size = actual_size
        
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
            'end_date': market.get('end_date'),  # Store for time horizon calc
            'category': market.get('category', 'unknown'),  # Store for learning
        }
        
        # In dry run mode, immediately treat as filled position
        if self.dry_run:
            # Get intel data from most recent analysis for this market
            recent_analysis = next((a for a in self._analyses if a.get('market_id') == market_id), None)
            has_intel = recent_analysis.get('has_intel', False) if recent_analysis else False
            news_count = recent_analysis.get('news_count', 0) if recent_analysis else 0
            
            pos = {
                **order_data,
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'entry_time': datetime.utcnow().isoformat(),
                'edge': edge,  # Store for exit tracking
                'confidence': confidence,  # Store for exit tracking
                'has_intel': has_intel,  # Store for exit tracking
                'news_count': news_count,  # Store for exit tracking
                'category': market.get('category', 'unknown'),  # Store for learning
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
            
            trade = {
                'id': pos_id,
                'market_id': market_id,
                'question': market.get('question', ''),
                'action': 'ENTRY',
                'side': side,
                'price': entry_price,
                'size': size,
                'edge': edge,  # Track for performance analysis
                'confidence': confidence,  # Track for performance analysis
                'category': market.get('category', 'unknown'),  # Track for learning
                'has_intel': has_intel,  # Track if news was used
                'news_count': news_count,  # Track how many news items
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._trades.insert(0, trade)
            news_indicator = f" [NEWS:{news_count}]" if news_count > 0 else ""
            print(f"[DRY RUN] ENTERED: {side} ${size:.2f} @ {int(entry_price*100)}¢ E:{edge*100:.0f}% C:{confidence*100:.0f}%{news_indicator} | {market.get('question', '')[:50]}...")
        else:
            # In LIVE mode, track as pending until we confirm fill from Kalshi
            # Get intel data from most recent analysis for this market
            recent_analysis = next((a for a in self._analyses if a.get('market_id') == market_id), None)
            has_intel = recent_analysis.get('has_intel', False) if recent_analysis else False
            news_count = recent_analysis.get('news_count', 0) if recent_analysis else 0
            
            # Store intel info in order_data so it persists through fill
            order_data['edge'] = edge
            order_data['confidence'] = confidence
            order_data['has_intel'] = has_intel
            order_data['news_count'] = news_count
            
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
                'edge': edge,  # Track for performance analysis
                'confidence': confidence,  # Track for performance analysis
                'has_intel': has_intel,
                'news_count': news_count,
                'timestamp': datetime.utcnow().isoformat(),
                'order_id': order_id,
                'status': 'pending',
            }
            self._trades.insert(0, trade)
            news_indicator = f" [NEWS:{news_count}]" if news_count > 0 else ""
            print(f"[PENDING] Order {order_id} placed E:{edge*100:.0f}% C:{confidence*100:.0f}%{news_indicator}, waiting to fill...")
        
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
                    
                    # EXIT STRATEGY:
                    # - NO stop losses (they cut winners - historical data confirms)
                    # - YES selective profit-lock for large gains near resolution
                    #   Rationale: "was up yesterday, gave it back today" pattern
                    #   caused by unrealized gains reversing with no lock mechanism.
                    
                    cost_basis = contracts * entry_price
                    gain_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0
                    
                    # Calculate days to resolution
                    days_to_res = None
                    end_date_str = pos.get('end_date')
                    if end_date_str:
                        try:
                            end_dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                            days_to_res = (end_dt - datetime.now(end_dt.tzinfo)).total_seconds() / 86400
                        except Exception:
                            pass
                    
                    should_exit = False
                    exit_reason_mon = ""
                    
                    # PROFIT-LOCK: Exit when position has gained significantly
                    # Logic: if we're up 50%+, the market has already priced in our view —
                    # locking in is better than waiting for a potential reversal.
                    PROFIT_LOCK_PCT = float(os.getenv('PROFIT_LOCK_PCT', '0.50'))  # 50% gain threshold
                    
                    if gain_pct >= PROFIT_LOCK_PCT and cost_basis > 0:
                        should_exit = True
                        exit_reason_mon = f"PROFIT_LOCK_{gain_pct*100:.0f}pct"
                    
                    # NEAR-SETTLEMENT LOCK: Within 4h of resolution, lock any gain > 25%
                    # This captures "approaching certainty" price moves before final settlement
                    elif days_to_res is not None and days_to_res <= 0.17 and gain_pct >= 0.25:
                        should_exit = True
                        exit_reason_mon = f"NEAR_SETTLEMENT_LOCK_{gain_pct*100:.0f}pct"
                    
                    # Logging only (no exit) for monitoring — throttle to once per hour per position
                    elif unrealized_pnl < -0.70 * cost_basis:
                        if not hasattr(self, '_monitor_log_times'):
                            self._monitor_log_times = {}
                        last_log = self._monitor_log_times.get(pos_id, 0)
                        if (datetime.utcnow().timestamp() - last_log) > 3600:
                            print(f"[Position Monitor] {pos_id[:8]} down 70%+ - holding for settlement | {pos.get('question','')[:40]}")
                            self._monitor_log_times[pos_id] = datetime.utcnow().timestamp()
                    
                    if should_exit:
                        print(f"[Position Monitor] {pos_id[:8]} {exit_reason_mon} | "
                              f"entry={entry_price*100:.0f}¢ cur={current_price*100:.0f}¢ "
                              f"uPnL=${unrealized_pnl:+.2f} | {pos.get('question','')[:40]}")
                        await self._exit_position(pos_id, current_price, unrealized_pnl, exit_reason_mon)
                
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
                    # Skip if position already has a pending exit order
                    if position.get('pending_exit'):
                        return  # Silently skip - don't spam logs
                    
                    # Use a limit sell at current bid minus 2¢ for quick fills without
                    # giving away value. Floor at 1¢ to always be marketable.
                    current_side_price = exit_price  # exit_price is the current market price for this side
                    sell_price_cents = max(1, int(current_side_price * 100) - 2)
                    
                    print(f"[LIVE SELL] Placing limit order: {contracts} {position['side']} @ {sell_price_cents}¢")
                    
                    result = await self._kalshi.sell_position(
                        ticker=position['market_id'],
                        side=position['side'].lower(),
                        count=contracts,
                        price=sell_price_cents,
                        order_type='limit',
                    )
                    exit_order_id = result.get('order', {}).get('order_id')
                    print(f"[LIVE SELL] Order placed @ {sell_price_cents}¢ | Order ID: {exit_order_id}")
                    
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
        
        # Track exit to prevent re-entry
        self._recently_exited[position['market_id']] = datetime.utcnow()
        self._recently_exited_reason[position['market_id']] = reason
        
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
            'edge': position.get('edge'),  # Carry forward for performance analysis
            'confidence': position.get('confidence'),  # Carry forward for performance analysis
            'category': position.get('category', 'unknown'),  # Track for learning
            'has_intel': position.get('has_intel', False),  # Carry forward for effectiveness tracking
            'news_count': position.get('news_count', 0),  # Carry forward for effectiveness tracking
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
                    # Profit = sell_price - buy_price (same formula for BOTH YES and NO)
                    entry_price = position['entry_price']  # Actual buy fill price
                    contracts = position.get('contracts', 0)
                    
                    price_change = fill_price - entry_price  # Same for YES and NO
                    pnl = price_change * contracts  # PnL per contract × number of contracts
                    
                    # Now actually remove the position and record exit
                    self._positions.pop(pos_id, None)
                    
                    # Track exit to prevent re-entry
                    self._recently_exited[position['market_id']] = datetime.utcnow()
                    self._recently_exited_reason[position['market_id']] = reason
                    
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
                        'edge': position.get('edge'),  # Track for performance analysis
                        'confidence': position.get('confidence'),  # Track for performance analysis
                        'category': position.get('category', 'unknown'),  # Track for learning
                        'has_intel': position.get('has_intel', False),  # Track for effectiveness
                        'news_count': position.get('news_count', 0),  # Track for effectiveness
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
    
    async def _check_signal_outcomes(self):
        """Check outcomes of logged signals for backtesting analysis."""
        if not self._signal_log:
            return
        
        # Find signals that haven't been checked yet
        unchecked = [s for s in self._signal_log if not s.get('outcome_checked')]
        if not unchecked:
            return
            
        checked_count = 0
        for signal in unchecked[:20]:  # Check up to 20 at a time
            market_id = signal.get('market_id')
            if not market_id:
                continue
                
            try:
                # Fetch market status from Kalshi
                raw = await self._kalshi.get_market(market_id)
                if not raw:
                    continue
                market = raw.get('market', raw)  # unwrap {'market': {...}} envelope

                status = market.get('status', '')

                if status == 'settled':
                    # Market has settled - record outcome
                    result = market.get('result', '')  # 'yes' or 'no'
                    signal['outcome_checked'] = True
                    
                    if result:
                        predicted_side = signal.get('side', '').lower()
                        actual_result = result.lower()
                        
                        # Did our prediction win?
                        signal['actual_result'] = actual_result
                        signal['predicted_correct'] = (predicted_side == actual_result)
                        
                        # Calculate what P&L would have been
                        entry_price = signal.get('market_price', 0)
                        if signal['predicted_correct']:
                            # Won: paid entry_price, received $1
                            signal['theoretical_pnl'] = 1.0 - entry_price
                        else:
                            # Lost: paid entry_price, received $0
                            signal['theoretical_pnl'] = -entry_price
                        
                        signal['outcome'] = 'WIN' if signal['predicted_correct'] else 'LOSS'
                        checked_count += 1
                    
                elif status == 'closed':
                    # Market closed but not yet settled
                    pass
                    
                await asyncio.sleep(0.3)  # Rate limit
                
            except Exception as e:
                if '404' not in str(e).lower():
                    print(f"[Signal Check] Error for {market_id}: {e}")
        
        if checked_count > 0:
            print(f"[Signals] Checked {checked_count} signal outcomes")
            self._save_state()
    
    def _get_signal_performance(self) -> dict:
        """Analyze signal log to determine optimal thresholds."""
        if not self._signal_log:
            return {'error': 'No signals logged yet'}
        
        # Separate by outcome
        settled = [s for s in self._signal_log if s.get('outcome')]
        pending = [s for s in self._signal_log if not s.get('outcome_checked')]
        
        if not settled:
            return {
                'total_signals': len(self._signal_log),
                'pending': len(pending),
                'settled': 0,
                'message': 'No settled signals yet - need to wait for markets to resolve'
            }
        
        wins = [s for s in settled if s.get('outcome') == 'WIN']
        losses = [s for s in settled if s.get('outcome') == 'LOSS']
        
        # Analyze by edge threshold
        edge_analysis = []
        for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            above = [s for s in settled if s.get('edge', 0) >= threshold]
            if above:
                above_wins = len([s for s in above if s.get('outcome') == 'WIN'])
                above_pnl = sum(s.get('theoretical_pnl', 0) for s in above)
                edge_analysis.append({
                    'threshold': f"{threshold*100:.0f}%",
                    'signals': len(above),
                    'wins': above_wins,
                    'win_rate': f"{above_wins/len(above)*100:.1f}%",
                    'theoretical_pnl': f"${above_pnl:.2f}",
                })
        
        # Analyze by confidence threshold
        conf_analysis = []
        for threshold in [0.25, 0.35, 0.50, 0.60, 0.70, 0.80]:
            above = [s for s in settled if s.get('confidence', 0) >= threshold]
            if above:
                above_wins = len([s for s in above if s.get('outcome') == 'WIN'])
                above_pnl = sum(s.get('theoretical_pnl', 0) for s in above)
                conf_analysis.append({
                    'threshold': f"{threshold*100:.0f}%",
                    'signals': len(above),
                    'wins': above_wins,
                    'win_rate': f"{above_wins/len(above)*100:.1f}%",
                    'theoretical_pnl': f"${above_pnl:.2f}",
                })
        
        return {
            'total_signals': len(self._signal_log),
            'settled': len(settled),
            'pending': len(pending),
            'wins': len(wins),
            'losses': len(losses),
            'overall_win_rate': f"{len(wins)/len(settled)*100:.1f}%" if settled else "N/A",
            'overall_pnl': f"${sum(s.get('theoretical_pnl', 0) for s in settled):.2f}",
            'by_edge_threshold': edge_analysis,
            'by_confidence_threshold': conf_analysis,
            'recent_signals': [
                {
                    'question': s.get('question', '')[:50],
                    'edge': f"{s.get('edge', 0)*100:.1f}%",
                    'confidence': f"{s.get('confidence', 0)*100:.0f}%",
                    'outcome': s.get('outcome', 'pending'),
                    'pnl': f"${s.get('theoretical_pnl', 0):.2f}" if s.get('theoretical_pnl') else 'N/A',
                }
                for s in sorted(settled, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
            ]
        }
    
    async def _position_sync_loop(self):
        """Sync positions with Kalshi API to detect filled orders and resolved markets."""
        await asyncio.sleep(5)  # Wait for startup
        
        sync_counter = 0
        FULL_SYNC_INTERVAL = 30  # Full sync every 30 iterations (~5 minutes)
        
        while self._running:
            try:
                sync_counter += 1
                
                # Periodically do a full position sync to detect resolved markets
                if sync_counter >= FULL_SYNC_INTERVAL:
                    sync_counter = 0
                    await self._sync_positions_with_kalshi()
                    # Also cancel any stale resting orders that accumulated
                    await self._cancel_all_resting_orders()
                    # Check signal outcomes for backtesting
                    await self._check_signal_outcomes()
                
                # First, check pending EXIT orders (sells on existing positions)
                await self._check_pending_exits()
                
                # Then check pending BUY orders
                if not self._pending_orders:
                    await asyncio.sleep(10)
                    continue  # Still loop to trigger full sync periodically
                
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
                            
                            # CANCEL STALE ORDERS: If order has been resting > 10 minutes, cancel it
                            STALE_ORDER_MINUTES = 10
                            placed_time_str = order.get('placed_time', order.get('entry_time', ''))
                            if placed_time_str:
                                try:
                                    placed_time = datetime.fromisoformat(placed_time_str)
                                    age_minutes = (datetime.utcnow() - placed_time).total_seconds() / 60
                                    if age_minutes > STALE_ORDER_MINUTES:
                                        print(f"[STALE ORDER] {order_id} | {age_minutes:.0f}min old - canceling...")
                                        try:
                                            await self._kalshi.cancel_order(order_id)
                                            canceled_orders.append(order_id)
                                            print(f"[ORDER CANCELED] {order_id} (stale)")
                                        except Exception as cancel_err:
                                            print(f"[Cancel Error] {order_id}: {cancel_err}")
                                except Exception as time_err:
                                    pass  # Can't parse time, skip
                        
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
                        'end_date': order.get('end_date'),  # Preserve for time horizon
                        'has_intel': order.get('has_intel', False),  # Preserve for effectiveness tracking
                        'news_count': order.get('news_count', 0),  # Preserve for effectiveness tracking
                        'category': order.get('category', 'unknown'),  # Preserve for learning
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
    
    async def _handle_signals(self, request):
        """API endpoint for signal backtesting analysis."""
        return web.json_response(self._get_signal_performance())
    
    async def _handle_fills(self, request):
        """Debug endpoint to see Kalshi fills."""
        try:
            result = await self._kalshi.get_fills(limit=100)
            fills = result.get('fills', [])
            
            # Show RAW first fill to see actual structure
            raw_sample = fills[0] if fills else {}
            
            # Simplify for debug output - include all relevant fields
            simplified = []
            for f in fills[:50]:
                simplified.append({
                    'ticker': f.get('ticker', ''),
                    'action': f.get('action', ''),
                    'side': f.get('side', ''),
                    'count': f.get('count', 0),
                    'yes_price': f.get('yes_price'),
                    'no_price': f.get('no_price'),
                    'price': f.get('price'),
                    'created': f.get('created_time', '')[:19],
                })
            
            return web.json_response({
                'total_fills': len(fills),
                'raw_sample': raw_sample,
                'fills': simplified
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_performance(self, request):
        """API endpoint for real performance data from Kalshi.
        
        This calculates performance directly from Kalshi API data,
        not from internal trade tracking which can be buggy.
        """
        try:
            # 1. Get actual account balance from Kalshi (ground truth)
            balance_result = await self._kalshi.get_balance()
            cash = balance_result.get('balance', 0) / 100.0  # cents to dollars
            
            # 2. Get positions value from Kalshi
            positions_result = await self._kalshi.get_positions()
            kalshi_positions = positions_result.get('market_positions', [])
            
            # Calculate position values
            total_position_cost = 0
            total_position_value = 0
            positions_detail = []
            
            for pos in kalshi_positions:
                contracts = pos.get('position', 0)
                if contracts == 0:
                    continue
                    
                ticker = pos.get('ticker', '')
                side = 'yes' if contracts > 0 else 'no'
                contracts = abs(contracts)
                
                # Get current market price
                market_result = await self._kalshi.get_market(ticker)
                market = market_result.get('market', {}) if market_result else {}
                
                yes_price = market.get('yes_bid', 50) / 100.0
                no_price = market.get('no_bid', 50) / 100.0
                current_price = yes_price if side == 'yes' else no_price
                
                # Get entry price from our fills
                entry_price = 0.50  # default
                for p in self._positions.values():
                    if p.get('market_id') == ticker:
                        entry_price = p.get('entry_price', 0.50)
                        break
                
                cost = contracts * entry_price
                value = contracts * current_price
                unrealized = value - cost  # same sign for both YES and NO: higher price = gain
                
                total_position_cost += cost
                total_position_value += value
                
                positions_detail.append({
                    'ticker': ticker,
                    'question': market.get('title', ticker)[:50],
                    'side': side.upper(),
                    'contracts': contracts,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'cost': cost,
                    'unrealized_pnl': unrealized,
                })
            
            # 3. Calculate account metrics
            total_deposits = float(os.getenv('TOTAL_DEPOSITS', '150'))  # User's total deposits
            # Prefer Kalshi's own portfolio valuation (same number shown in the ACCOUNT section)
            # so that Today's Peak tracks the same figure the user sees.
            account_value = self._kalshi_total if self._kalshi_total is not None else (cash + total_position_value)
            total_return = account_value - total_deposits
            return_pct = (total_return / total_deposits * 100) if total_deposits > 0 else 0
            
            # 4. Calculate unrealized P&L
            unrealized_pnl = sum(p['unrealized_pnl'] for p in positions_detail)
            
            # 5. Count winning vs losing positions
            winning_positions = len([p for p in positions_detail if p['unrealized_pnl'] > 0])
            losing_positions = len([p for p in positions_detail if p['unrealized_pnl'] < 0])
            
            # 6. Get today's activity
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            today_entries = [t for t in self._trades 
                           if t.get('action') == 'ENTRY' and 
                           t.get('timestamp', '').startswith(today_str)]
            
            # 7. Calculate exposure breakdown
            exposure_by_category = {}
            for p in positions_detail:
                ticker = p['ticker']
                cost = p['cost']
                
                # Categorize by ticker prefix
                if 'SNOW' in ticker or 'RAIN' in ticker:
                    cat = 'Weather'
                elif 'BTC' in ticker or 'ETH' in ticker or 'SOL' in ticker:
                    cat = 'Crypto'
                elif 'NBA' in ticker or 'NFL' in ticker or 'MLB' in ticker:
                    cat = 'Sports'
                else:
                    cat = 'Other'
                
                exposure_by_category[cat] = exposure_by_category.get(cat, 0) + cost
            
            # 8. Save daily snapshot (and start-of-day value for today's P&L)
            await self._save_daily_snapshot(account_value, cash, total_position_value)
            
            # 9. Today's P&L = current account value - start of day value (single source of truth)
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            today_snapshot = self._daily_snapshots.get(today_str, {})
            start_of_day_value = today_snapshot.get('start_of_day_value')
            if start_of_day_value is not None:
                today_pnl = round(account_value - start_of_day_value, 2)
                today_pnl_pct = round((today_pnl / start_of_day_value * 100), 1) if start_of_day_value else 0
            else:
                today_pnl = None
                today_pnl_pct = None
            
            # 10. Portfolio history: daily (14 days) + today's hourly for chart
            history_daily, history_today_hourly = self._get_performance_history()
            
            return web.json_response({
                # SINGLE SOURCE OF TRUTH: Kalshi account value only
                'account': {
                    'cash': round(cash, 2),
                    'positions_value': round(total_position_value, 2),
                    'total_value': round(account_value, 2),
                    'total_deposits': total_deposits,
                },
                # All P&L derived from account value (no fill-based totals)
                'performance': {
                    'total_return': round(total_return, 2),       # total_value - total_deposits
                    'return_pct': round(return_pct, 1),
                    'today_pnl': today_pnl,                       # total_value - start_of_day
                    'today_pnl_pct': today_pnl_pct,
                    'start_of_day_value': start_of_day_value,
                    'unrealized_pnl': round(unrealized_pnl, 2),
                    # Intraday high/low for "was up, gave it back" analysis
                    'intraday_high': today_snapshot.get('intraday_high'),
                    'intraday_high_time': today_snapshot.get('intraday_high_time'),
                    'intraday_low': today_snapshot.get('intraday_low'),
                    'peak_giveback': round(today_snapshot.get('intraday_high', account_value) - account_value, 2)
                                     if today_snapshot.get('intraday_high') else None,
                },
                # Positions Summary
                'positions': {
                    'count': len(positions_detail),
                    'winning': winning_positions,
                    'losing': losing_positions,
                    'total_cost': round(total_position_cost, 2),
                    'details': positions_detail[:10],  # Top 10
                },
                # Today's Activity
                'today': {
                    'new_bets': len(today_entries),
                    'total_deployed': sum(t.get('size', 0) for t in today_entries),
                },
                # Exposure by Category
                'exposure_by_category': {k: round(v, 2) for k, v in exposure_by_category.items()},
                # Portfolio value history (Kalshi): 14 days + hourly for today
                'history': history_daily,
                'today_hourly': history_today_hourly,
                # Data source confirmation
                'data_source': 'kalshi_api',
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            import traceback
            return web.json_response({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)
    
    async def _save_daily_snapshot(self, account_value: float, cash: float, positions_value: float):
        """Save portfolio value for history. First value of the day = start-of-day. Hourly points for today."""
        now = datetime.utcnow()
        today_str = now.strftime('%Y-%m-%d')
        hour_str = now.strftime('%H')
        val = round(account_value, 2)
        
        if not hasattr(self, '_daily_snapshots'):
            self._daily_snapshots = {}
            self._portfolio_hourly = {}
            try:
                if os.path.exists(self._state_file):
                    with open(self._state_file, 'r') as f:
                        state = json.load(f)
                        self._daily_snapshots = state.get('daily_snapshots', {})
                        self._portfolio_hourly = state.get('portfolio_hourly', {})
            except Exception:
                pass
        
        # Start-of-day value (first we see today)
        snapshot = self._daily_snapshots.get(today_str, {})
        if snapshot.get('start_of_day_value') is None:
            snapshot['start_of_day_value'] = val
        snapshot['account_value'] = val
        snapshot['cash'] = round(cash, 2)
        snapshot['positions_value'] = round(positions_value, 2)
        snapshot['timestamp'] = now.isoformat()
        
        # Track intraday high watermark — shows "was up X, now at Y" pattern
        current_high = snapshot.get('intraday_high', val)
        if val > current_high:
            snapshot['intraday_high'] = val
            snapshot['intraday_high_time'] = now.isoformat()
        else:
            snapshot['intraday_high'] = current_high
        
        # Track intraday low watermark
        current_low = snapshot.get('intraday_low', val)
        if val < current_low:
            snapshot['intraday_low'] = val
            snapshot['intraday_low_time'] = now.isoformat()
        else:
            snapshot['intraday_low'] = current_low
        
        self._daily_snapshots[today_str] = snapshot
        
        # Hourly series for today (Kalshi portfolio value)
        if today_str not in self._portfolio_hourly:
            self._portfolio_hourly[today_str] = {}
        self._portfolio_hourly[today_str][hour_str] = val
        # Keep only last 14 days of hourly
        sorted_dates = sorted(self._portfolio_hourly.keys(), reverse=True)
        for old in sorted_dates[14:]:
            del self._portfolio_hourly[old]
        
        # Prune daily to 30 days
        sorted_dates = sorted(self._daily_snapshots.keys(), reverse=True)
        for old in sorted_dates[30:]:
            del self._daily_snapshots[old]
        
        if not hasattr(self, '_last_snapshot_file_save') or (now - self._last_snapshot_file_save).total_seconds() > 300:
            self._save_state()
            self._last_snapshot_file_save = now
    
    def _get_performance_history(self) -> tuple:
        """Return (daily history for last 14 days, today's hourly points for chart)."""
        if not hasattr(self, '_daily_snapshots'):
            self._daily_snapshots = {}
        if not hasattr(self, '_portfolio_hourly'):
            self._portfolio_hourly = {}
        
        daily = []
        today = datetime.utcnow().date()
        for i in range(14):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self._daily_snapshots:
                s = self._daily_snapshots[date_str]
                daily.append({
                    'date': date_str,
                    'account_value': s.get('account_value'),
                    'cash': s.get('cash'),
                    'positions_value': s.get('positions_value'),
                    'start_of_day_value': s.get('start_of_day_value'),
                    'intraday_high': s.get('intraday_high'),
                    'intraday_low': s.get('intraday_low'),
                    'intraday_high_time': s.get('intraday_high_time'),
                })
            else:
                daily.append({'date': date_str, 'account_value': None, 'cash': None,
                              'positions_value': None, 'intraday_high': None, 'intraday_low': None})
        
        today_str = today.strftime('%Y-%m-%d')
        today_hourly = []
        if today_str in self._portfolio_hourly:
            for h in sorted(self._portfolio_hourly[today_str].keys()):
                today_hourly.append({'hour': int(h), 'value': self._portfolio_hourly[today_str][h]})
        
        return daily, today_hourly
    
    async def _handle_settlements(self, request):
        """API endpoint for accurate settlement log from Kalshi fills.
        
        This calculates P&L for each settled bet directly from Kalshi data:
        - Gets all fills (buy/sell history)
        - Groups by ticker
        - For settled markets, calculates exact P&L based on result
        
        P&L Calculation:
        - WIN (our side == result): P&L = contracts * (1.0 - avg_entry_price)
        - LOSS (our side != result): P&L = contracts * (-avg_entry_price)
        """
        try:
            # Get all fills from Kalshi (our trade history)
            result = await self._kalshi.get_fills(limit=500)
            fills = result.get('fills', [])
            
            if not fills:
                return web.json_response({
                    'settlements': [],
                    'summary': {'total': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0},
                    'today': {'wins': 0, 'losses': 0, 'pnl': 0},
                })
            
            # Group fills by ticker
            fills_by_ticker = {}
            for fill in fills:
                ticker = fill.get('ticker', '')
                if not ticker:
                    continue
                if ticker not in fills_by_ticker:
                    fills_by_ticker[ticker] = []
                fills_by_ticker[ticker].append(fill)
            
            settlements = []
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            
            # Process each ticker: include every ticker we BOUGHT that has since settled
            # (After settlement, Kalshi may show sells that zero the position — we still want the settlement.)
            for ticker, ticker_fills in fills_by_ticker.items():
                total_bought = 0
                total_cost = 0.0
                total_sold = 0
                side = None
                earliest_fill_time = None
                
                for fill in ticker_fills:
                    action = fill.get('action', '')
                    count = fill.get('count', 0)
                    price = fill.get('price', 0.50)  # Kalshi fills: price in DOLLARS
                    fill_side = fill.get('side', '')
                    fill_time = fill.get('created_time', '')
                    
                    if action == 'buy':
                        total_bought += count
                        total_cost += price * count
                        side = fill_side
                        if not earliest_fill_time or fill_time < earliest_fill_time:
                            earliest_fill_time = fill_time
                    elif action == 'sell':
                        total_sold += count
                
                # Skip if we never bought
                if total_bought <= 0:
                    continue
                
                avg_entry_price = total_cost / total_bought
                our_side = (side or '').lower()
                
                try:
                    market_result = await self._kalshi.get_market(ticker)
                    market = market_result.get('market', {}) if market_result else {}
                    status = market.get('status', '')
                    result = market.get('result', '')
                    title = market.get('title', ticker)
                    close_time = market.get('close_time', '')
                    
                    # Only include settled markets
                    if status not in ('settled', 'finalized') or not result:
                        continue
                    
                    # Contracts that settled = what we bought (sells may be settlement payout)
                    contracts_settled = total_bought
                    market_result_lower = result.lower()
                    
                    if our_side == market_result_lower:
                        pnl = contracts_settled * (1.0 - avg_entry_price)
                        outcome = 'WIN'
                    else:
                        pnl = contracts_settled * (-avg_entry_price)
                        outcome = 'LOSS'
                    
                    settlement_date = close_time[:10] if close_time else today_str
                    
                    settlements.append({
                        'ticker': ticker,
                        'question': title,
                        'side': our_side.upper(),
                        'contracts': contracts_settled,
                        'entry_price': round(avg_entry_price, 4),
                        'entry_price_cents': int(round(avg_entry_price * 100)),
                        'cost': round(contracts_settled * avg_entry_price, 2),
                        'result': market_result_lower.upper(),
                        'outcome': outcome,
                        'pnl': round(pnl, 2),
                        'settlement_date': settlement_date,
                        'fill_time': earliest_fill_time,
                    })
                    
                except Exception as e:
                    print(f"[Settlements] Error checking market {ticker}: {e}")
                    continue
                
                await asyncio.sleep(0.1)
            
            # Sort by settlement date (newest first)
            settlements.sort(key=lambda x: x.get('settlement_date', ''), reverse=True)
            
            # Calculate summary stats
            wins = [s for s in settlements if s['outcome'] == 'WIN']
            losses = [s for s in settlements if s['outcome'] == 'LOSS']
            total_pnl = sum(s['pnl'] for s in settlements)
            
            # Today's stats
            today_settlements = [s for s in settlements if s.get('settlement_date', '').startswith(today_str)]
            today_wins = len([s for s in today_settlements if s['outcome'] == 'WIN'])
            today_losses = len([s for s in today_settlements if s['outcome'] == 'LOSS'])
            today_pnl = sum(s['pnl'] for s in today_settlements)
            
            # Calculate win rate
            total_settled = len(wins) + len(losses)
            win_rate = (len(wins) / total_settled * 100) if total_settled > 0 else 0
            
            # Best and worst trades
            pnls = [s['pnl'] for s in settlements]
            best_pnl = max(pnls) if pnls else 0
            worst_pnl = min(pnls) if pnls else 0
            
            total_deposits = float(os.getenv('TOTAL_DEPOSITS', '150'))
            
            return web.json_response({
                'settlements': settlements,
                'total_deposits': total_deposits,
                'summary': {
                    'total': len(settlements),
                    'wins': len(wins),
                    'losses': len(losses),
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(total_pnl, 2),
                    'best_trade': round(best_pnl, 2),
                    'worst_trade': round(worst_pnl, 2),
                    'avg_win': round(sum(s['pnl'] for s in wins) / len(wins), 2) if wins else 0,
                    'avg_loss': round(sum(s['pnl'] for s in losses) / len(losses), 2) if losses else 0,
                },
                'today': {
                    'settlements': len(today_settlements),
                    'wins': today_wins,
                    'losses': today_losses,
                    'pnl': round(today_pnl, 2),
                },
                'data_source': 'kalshi_fills_api',
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            import traceback
            return web.json_response({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)
    
    async def _handle_debug_reconcile(self, request):
        """Debug endpoint to trace reconciliation logic."""
        debug_output = []
        
        try:
            # Get fills
            result = await self._kalshi.get_fills(limit=200)
            fills = result.get('fills', [])
            debug_output.append(f"Total fills: {len(fills)}")
            
            # Group by ticker
            fills_by_ticker = {}
            for fill in fills:
                ticker = fill.get('ticker', '')
                if not ticker:
                    continue
                if ticker not in fills_by_ticker:
                    fills_by_ticker[ticker] = []
                fills_by_ticker[ticker].append(fill)
            
            debug_output.append(f"Unique tickers: {len(fills_by_ticker)}")
            
            # Get exit tickers from trades
            exit_tickers = set()
            for trade in self._trades:
                ticker = trade.get('market_id', '')
                if trade.get('action') == 'EXIT':
                    exit_tickers.add(ticker)
            
            debug_output.append(f"Exit tickers in history: {len(exit_tickers)}")
            
            # Check Feb 20 tickers specifically
            feb20_checks = []
            for ticker in ['KXBTCD-26FEB2017-T65749.99', 'KXBTCD-26FEB2017-T66249.99', 
                          'KXBTC-26FEB2017-B66000', 'KXBTC-26FEB2017-B66500']:
                check = {'ticker': ticker}
                
                # Is it in fills?
                check['in_fills'] = ticker in fills_by_ticker
                
                # Is it in exit_tickers?
                check['has_exit'] = ticker in exit_tickers
                
                if ticker in fills_by_ticker:
                    # Calculate net position
                    net = 0
                    side = None
                    for f in fills_by_ticker[ticker]:
                        if f.get('action') == 'buy':
                            net += f.get('count', 0)
                            side = f.get('side')
                        elif f.get('action') == 'sell':
                            net -= f.get('count', 0)
                    check['net_contracts'] = net
                    check['side'] = side
                    
                    # Check market status
                    try:
                        market_result = await self._kalshi.get_market(ticker)
                        market_data = market_result.get('market', {}) if market_result else {}
                        check['market_status'] = market_data.get('status', 'unknown')
                        check['market_result'] = market_data.get('result', 'none')
                        check['market_title'] = market_data.get('title', '')[:40]
                    except Exception as e:
                        check['market_error'] = str(e)
                
                feb20_checks.append(check)
            
            return web.json_response({
                'debug': debug_output,
                'feb20_checks': feb20_checks
            })
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
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
        <div class="tab" data-tab="txlog">Settlements</div>
        <div class="tab" data-tab="activity">Activity</div>
        <div class="tab" data-tab="markets">Markets</div>
    </div>
    <div class="content">
        <div id="portfolio" class="tab-content">
            <div class="section-title">ACCOUNT</div>
            <div class="grid">
                <div class="card"><div class="card-label">Available</div><div class="card-value green" id="available">$0.00</div></div>
                <div class="card"><div class="card-label">At Risk (Total)</div><div class="card-value yellow" id="atRisk">$0.00</div><div class="card-sub" id="posCount">0 positions</div></div>
                <div class="card"><div class="card-label">Total Value</div><div class="card-value" id="totalValue">$0.00</div></div>
                <div class="card"><div class="card-label">Return</div><div class="card-value green" id="returnPct">0.00%</div></div>
            </div>
            <div class="section-title">AT RISK BREAKDOWN</div>
            <div class="grid">
                <div class="card"><div class="card-label">Filled Positions</div><div class="card-value" id="positionsAtRisk">$0.00</div><div class="card-sub" id="filledCount">0 filled</div></div>
                <div class="card"><div class="card-label">Resting Orders</div><div class="card-value" id="pendingAtRisk">$0.00</div><div class="card-sub" id="restingCount">0 resting</div></div>
            </div>
            <div class="section-title">AT RISK BY TIME HORIZON (Filled Only)</div>
            <div class="grid">
                <div class="card"><div class="card-label">Ultra-Short (≤24h)</div><div class="card-value" id="atRiskUltra">$0.00</div><div class="card-sub">resolves in hours</div></div>
                <div class="card"><div class="card-label">Short (1-7d)</div><div class="card-value" id="atRiskShort">$0.00</div><div class="card-sub">resolves in days</div></div>
                <div class="card"><div class="card-label">Medium (8+d)</div><div class="card-value" id="atRiskMedium">$0.00</div><div class="card-sub">resolves in weeks+</div></div>
            </div>
            <div class="section-title">YOUR ACCOUNT <span style="font-size:10px;opacity:0.6;">(live from Kalshi)</span></div>
            <div class="grid">
                <div class="card"><div class="card-label">Account value</div><div class="card-value" id="accountValue">$0.00</div><div class="card-sub">cash + positions</div></div>
                <div class="card"><div class="card-label">Cash</div><div class="card-value" id="cashAvailable">$0.00</div></div>
                <div class="card"><div class="card-label">In positions</div><div class="card-value" id="positionsValue">$0.00</div><div class="card-sub" id="positionsSummary">0 bets</div></div>
            </div>
            <div class="section-title">REAL P&L <span style="font-size:10px;opacity:0.6;">(value minus what you deposited)</span></div>
            <div class="grid">
                <div class="card"><div class="card-label">Total vs <span id="depositsLabel">$150</span> deposited</div><div class="card-value" id="totalReturn">$0.00</div><div class="card-sub" id="returnPct">0%</div></div>
                <div class="card"><div class="card-label">Today</div><div class="card-value" id="todayPnl">—</div><div class="card-sub" id="todayPnlSub">vs start of day</div></div>
                <div class="card"><div class="card-label">Today's Peak</div><div class="card-value" id="intradayHigh">—</div><div class="card-sub" id="peakGiveback">intraday high watermark</div></div>
                <div class="card"><div class="card-label">Positions</div><div class="card-value" id="positionStatus">0 winning / 0 losing</div><div class="card-sub" id="positionStatusSub">vs entry price</div></div>
            </div>
            <div class="section-title">TODAY'S ACTIVITY</div>
            <div class="grid">
                <div class="card"><div class="card-label">New Bets</div><div class="card-value" id="todayBets">0</div><div class="card-sub" id="todayDeployed">$0 deployed</div></div>
                <div class="card"><div class="card-label">Open Positions</div><div class="card-value" id="openPositions">0</div><div class="card-sub" id="positionLimit">of 25 max</div></div>
                <div class="card"><div class="card-label">Bot Status</div><div class="card-value" id="botStatus">Active</div><div class="card-sub" id="botRuntime">0h 0m</div></div>
            </div>
            <div class="section-title">STRATEGY READOUT</div>
            <div id="strategyReadout" class="card" style="padding:16px;line-height:1.6;font-size:14px;">
                <p id="readoutMoney">—</p>
                <p id="readoutToday">—</p>
                <p id="readoutSignal">—</p>
            </div>
            <div class="section-title">EXPOSURE BY CATEGORY</div>
            <div class="grid" id="exposureGrid">
                <div class="card"><div class="card-label">Weather</div><div class="card-value" id="expWeather">$0.00</div></div>
                <div class="card"><div class="card-label">Crypto</div><div class="card-value" id="expCrypto">$0.00</div></div>
                <div class="card"><div class="card-label">Other</div><div class="card-value" id="expOther">$0.00</div></div>
            </div>
            <div class="section-title">TRADE ENTRY THRESHOLDS <span style="font-size:10px;opacity:0.6;">a market must clear every threshold to get a bet — see Activity tab for per-market pass/fail</span></div>
            <div id="filtersGrid" class="grid" style="grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-bottom:16px;">
                <div class="card"><div class="card-label">Min Edge</div><div class="card-value" id="cfgMinEdge">—</div><div class="card-sub">AI edge must exceed</div></div>
                <div class="card"><div class="card-label">Min Confidence</div><div class="card-value" id="cfgMinConf">—</div><div class="card-sub">AI certainty must exceed</div></div>
                <div class="card"><div class="card-label">Max Horizon</div><div class="card-value" id="cfgMaxDays">—</div><div class="card-sub">market must resolve within</div></div>
                <div class="card"><div class="card-label">Profit-Lock Exit</div><div class="card-value" id="cfgProfitLock">—</div><div class="card-sub">auto-sell at this gain</div></div>
                <div class="card"><div class="card-label">Cluster Cap</div><div class="card-value" id="cfgCluster">—</div><div class="card-sub">max bets per news theme</div></div>
                <div class="card"><div class="card-label">Max Bet Size</div><div class="card-value" id="cfgMaxBet">—</div><div class="card-sub">per position ceiling</div></div>
                <div class="card"><div class="card-label">Kelly Fraction</div><div class="card-value" id="cfgKelly">—</div><div class="card-sub">sizing conservatism</div></div>
                <div class="card"><div class="card-label">Trading</div><div class="card-value" id="cfgTrading">—</div><div class="card-sub" id="cfgTradingSub">kill-switch status</div></div>
            </div>
            <div class="section-title" style="display:flex;align-items:center;justify-content:space-between;">
                <span>PORTFOLIO VALUE (Kalshi) <span style="font-size:10px;opacity:0.6;"><span style="color:#3fb950;">green</span> = above deposits · <span style="color:#f85149;">red</span> = below</span></span>
                <span id="chartToggle" style="display:flex;gap:4px;">
                    <button onclick="setChartMode('hourly')" id="btnHourly" style="font-size:11px;padding:3px 10px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;">Hourly</button>
                    <button onclick="setChartMode('daily')" id="btnDaily" style="font-size:11px;padding:3px 10px;border-radius:6px;border:1px solid #58a6ff;background:#1f3a5f;color:#58a6ff;cursor:pointer;">Daily</button>
                    <button onclick="setChartMode('monthly')" id="btnMonthly" style="font-size:11px;padding:3px 10px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;">Monthly</button>
                </span>
            </div>
            <div id="historyChart" style="padding:12px 0 4px 0;min-height:230px;"></div>
            <div class="section-title">Active Positions <span id="positionCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="positions"></div>
            <div class="section-title" style="margin-top:20px">Recent Trades <span id="tradeCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="trades"></div>
        </div>
        <div id="txlog" class="tab-content hidden">
            <div class="section-title">YOUR ACCOUNT <span style="font-size:10px;opacity:0.6;">(same as Portfolio — from Kalshi)</span></div>
            <div class="grid">
                <div class="card"><div class="card-label">Account value</div><div class="card-value" id="truthValue">$0.00</div></div>
                <div class="card"><div class="card-label">Real P&L vs <span id="settlDepositsLabel">$150</span></div><div class="card-value" id="truthPnl">$0.00</div><div class="card-sub" id="truthPct">0%</div></div>
                <div class="card"><div class="card-label">Today</div><div class="card-value" id="truthToday">—</div><div class="card-sub">vs start of day</div></div>
            </div>
            <div class="section-title">SETTLED BETS <span style="font-size:10px;opacity:0.6;">(list only — true P&L is above)</span> <span id="settlementCount" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">0</span></div>
            <div id="settlementList"></div>
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
                if (tab.dataset.tab === 'txlog') fetchSettlements();
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
                
                // Fetch real performance and settlements from Kalshi API
                fetchPerformance();
            };
        }
        
        function updateStats(s) {
            // Legacy stats (At Risk section)
            document.getElementById('available').textContent = '$' + s.available.toFixed(2);
            document.getElementById('atRisk').textContent = '$' + s.at_risk.toFixed(2);
            document.getElementById('posCount').textContent = s.open_positions + ' positions + ' + (s.pending_orders || 0) + ' resting';
            document.getElementById('positionsAtRisk').textContent = '$' + (s.positions_at_risk || 0).toFixed(2);
            document.getElementById('filledCount').textContent = s.open_positions + ' filled';
            document.getElementById('pendingAtRisk').textContent = '$' + (s.pending_at_risk || 0).toFixed(2);
            document.getElementById('restingCount').textContent = (s.pending_orders || 0) + ' resting';
            document.getElementById('atRiskUltra').textContent = '$' + (s.at_risk_ultra_short || 0).toFixed(2);
            document.getElementById('atRiskShort').textContent = '$' + (s.at_risk_short || 0).toFixed(2);
            document.getElementById('atRiskMedium').textContent = '$' + (s.at_risk_medium || 0).toFixed(2);
            document.getElementById('totalValue').textContent = '$' + s.total_value.toFixed(2);
            
            // Header badges
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
            
            // Bot Status in Today's Activity
            document.getElementById('botStatus').textContent = s.trading_allowed ? 'Active' : 'Paused';
            document.getElementById('botStatus').className = 'card-value ' + (s.trading_allowed ? 'green' : 'red');
            document.getElementById('botRuntime').textContent = s.runtime;

            // BOT FILTERS ACTIVE section — surface all live thresholds
            const fEdge = document.getElementById('cfgMinEdge');
            const fConf = document.getElementById('cfgMinConf');
            const fDays = document.getElementById('cfgMaxDays');
            const fLock = document.getElementById('cfgProfitLock');
            const fClus = document.getElementById('cfgCluster');
            const fBet  = document.getElementById('cfgMaxBet');
            const fKelly= document.getElementById('cfgKelly');
            const fTrad = document.getElementById('cfgTrading');
            const fTradSub = document.getElementById('cfgTradingSub');
            if (fEdge) fEdge.textContent = (s.min_edge * 100).toFixed(0) + '%';
            if (fConf) fConf.textContent = (s.min_confidence * 100).toFixed(0) + '%';
            if (fDays) fDays.textContent = s.max_days_to_resolution != null ? s.max_days_to_resolution + 'd' : '—';
            if (fLock) fLock.textContent = s.profit_lock_pct != null ? '+' + (s.profit_lock_pct * 100).toFixed(0) + '%' : '—';
            if (fClus) fClus.textContent = s.max_cluster_positions != null ? s.max_cluster_positions + ' pos' : '—';
            if (fBet)  fBet.textContent  = s.max_position_size != null ? '$' + s.max_position_size : '—';
            if (fKelly)fKelly.textContent = s.kelly_fraction != null ? (s.kelly_fraction * 100).toFixed(0) + '%' : '—';
            if (fTrad) {
                const ks = s.kill_switch;
                fTrad.textContent = ks ? 'HALTED' : (s.trading_allowed ? 'ON' : 'PAUSED');
                fTrad.className = 'card-value ' + (ks ? 'red' : s.trading_allowed ? 'green' : 'yellow');
            }
            if (fTradSub) fTradSub.textContent = s.kill_switch ? 'kill-switch triggered' :
                ('drawdown ' + (s.daily_drawdown != null ? (s.daily_drawdown * 100).toFixed(1) + '%' : '0%'));
        }
        
        // Fetch performance first so dashboard always shows Kalshi numbers; then state + signals for readout
        async function fetchPerformance() {
            try {
                const perfRes = await fetch('/api/performance');
                const p = await perfRes.json();
                if (p.error) {
                    console.error('Performance API error:', p.error);
                    return;
                }
                if (!p.account || !p.performance) {
                    console.error('Performance API: missing account or performance');
                    return;
                }
                
                // Account Value section
                document.getElementById('accountValue').textContent = '$' + p.account.total_value.toFixed(2);
                document.getElementById('accountValue').className = 'card-value';
                document.getElementById('cashAvailable').textContent = '$' + p.account.cash.toFixed(2);
                document.getElementById('positionsValue').textContent = '$' + p.account.positions_value.toFixed(2);
                document.getElementById('positionsSummary').textContent = p.positions.count + ' active bets';
                
                // Real P&L: only from account value (negative = red, positive = green)
                const ret = p.performance.total_return;
                const returnPctVal = p.performance.return_pct;
                const depositsLbl = document.getElementById('depositsLabel');
                if (depositsLbl) depositsLbl.textContent = '$' + (p.account.total_deposits || 150);
                document.getElementById('totalReturn').textContent = (ret >= 0 ? '+' : '') + '$' + ret.toFixed(2);
                document.getElementById('totalReturn').className = 'card-value ' + (ret >= 0 ? 'green' : 'red');
                document.getElementById('returnPct').textContent = (returnPctVal >= 0 ? '+' : '') + returnPctVal.toFixed(1) + '%';
                document.getElementById('returnPct').className = 'card-sub ' + (returnPctVal >= 0 ? 'green' : 'red');
                
                // Today's change = account value now - start of day (from Kalshi)
                const todayPn = p.performance.today_pnl;
                const todayPct = p.performance.today_pnl_pct;
                const todayEl = document.getElementById('todayPnl');
                const todaySubEl = document.getElementById('todayPnlSub');
                if (todayPn != null && todayPct != null) {
                    todayEl.textContent = (todayPn >= 0 ? '+' : '') + '$' + todayPn.toFixed(2);
                    todayEl.className = 'card-value ' + (todayPn >= 0 ? 'green' : 'red');
                    todaySubEl.textContent = (todayPct >= 0 ? '+' : '') + todayPct.toFixed(1) + '% vs start of day';
                } else {
                    todayEl.textContent = '—';
                    todayEl.className = 'card-value';
                    todaySubEl.textContent = 'vs start of day (set tomorrow)';
                }
                
                document.getElementById('positionStatus').textContent = p.positions.winning + ' winning / ' + p.positions.losing + ' losing';
                document.getElementById('positionStatusSub').textContent = 'vs entry price';
                // Keep Settlements tab deposit label in sync
                const sdl = document.getElementById('settlDepositsLabel');
                if (sdl) sdl.textContent = '$' + (p.account.total_deposits || 150);
                
                // Intraday high watermark — "was up X, gave back Y today"
                const highEl = document.getElementById('intradayHigh');
                const givebackEl = document.getElementById('peakGiveback');
                if (highEl && p.performance.intraday_high != null) {
                    const high = p.performance.intraday_high;
                    const giveback = p.performance.peak_giveback;
                    const startVal = p.performance.start_of_day_value || high;
                    const highGain = high - startVal;
                    highEl.textContent = '$' + high.toFixed(2) + (highGain >= 0 ? ' (+$' + highGain.toFixed(2) + ')' : '');
                    highEl.className = 'card-value ' + (highGain >= 0 ? 'green' : 'red');
                    if (givebackEl && giveback != null && giveback > 0.01) {
                        givebackEl.textContent = '-$' + giveback.toFixed(2) + ' given back from peak';
                        givebackEl.className = 'card-sub red';
                    } else if (givebackEl) {
                        givebackEl.textContent = 'at or near peak';
                        givebackEl.className = 'card-sub green';
                    }
                }
                
                // Today's Activity
                document.getElementById('todayBets').textContent = p.today.new_bets;
                document.getElementById('todayDeployed').textContent = '$' + (p.today.total_deployed || 0).toFixed(2) + ' deployed';
                document.getElementById('openPositions').textContent = p.positions.count;
                document.getElementById('positionLimit').textContent = 'of 25 max';
                
                // Exposure by Category
                document.getElementById('expWeather').textContent = '$' + (p.exposure_by_category.Weather || 0).toFixed(2);
                document.getElementById('expCrypto').textContent = '$' + (p.exposure_by_category.Crypto || 0).toFixed(2);
                document.getElementById('expOther').textContent = '$' + (p.exposure_by_category.Other || 0).toFixed(2);
                
                // Settlements tab: same truth (optional elements)
                const tv = document.getElementById('truthValue');
                const tp = document.getElementById('truthPnl');
                const tpc = document.getElementById('truthPct');
                const tt = document.getElementById('truthToday');
                if (tv) tv.textContent = '$' + p.account.total_value.toFixed(2);
                if (tp) { tp.textContent = (ret >= 0 ? '+' : '') + '$' + ret.toFixed(2); tp.className = 'card-value ' + (ret >= 0 ? 'green' : 'red'); }
                if (tpc) { tpc.textContent = (returnPctVal >= 0 ? '+' : '') + returnPctVal.toFixed(1) + '%'; tpc.className = 'card-sub ' + (returnPctVal >= 0 ? 'green' : 'red'); }
                if (tt) { tt.textContent = (todayPn != null ? (todayPn >= 0 ? '+' : '') + '$' + todayPn.toFixed(2) : '—'); tt.className = 'card-value' + (todayPn != null ? ' ' + (todayPn >= 0 ? 'green' : 'red') : ''); }
                
                // Portfolio value chart — store data globally so toggle buttons can re-render
                if (p.history) {
                    window._chartHistory   = p.history;
                    window._chartHourly    = p.today_hourly || [];
                    window._chartDeposits  = p.account.total_deposits || 150;
                    renderHistoryChart(window._chartHistory, window._chartHourly, window._chartDeposits, window._chartMode || 'daily');
                }
                
                // Strategy readout (state + signals); don't block main UI if these fail
                try {
                    const [stateRes, signalsRes] = await Promise.all([fetch('/api/state'), fetch('/api/signals')]);
                    const state = stateRes.ok ? await stateRes.json() : {};
                    const sig = signalsRes.ok ? await signalsRes.json() : {};
                    updateStrategyReadout(p, state, sig);
                } catch (e) { if (document.getElementById('readoutMoney')) document.getElementById('readoutMoney').textContent = 'Account loaded; readout skipped.'; }
                
            } catch (e) {
                console.error('Failed to fetch performance:', e);
            }
        }
        
        function updateStrategyReadout(p, state, sig) {
            const moneyEl = document.getElementById('readoutMoney');
            const todayEl = document.getElementById('readoutToday');
            const signalEl = document.getElementById('readoutSignal');
            if (!moneyEl || !todayEl || !signalEl) return;
            const val = p.account.total_value;
            const deposits = p.account.total_deposits || 150;
            const ret = p.performance.total_return;
            const pct = p.performance.return_pct;
            const todayPn = p.performance.today_pnl;
            const todayPct = p.performance.today_pnl_pct;
            moneyEl.innerHTML = 'Account value is <strong>$' + val.toFixed(2) + '</strong> (deposits $' + deposits + '). Total return ' + (ret >= 0 ? '<span class="green">+' : '<span class="red">') + '$' + ret.toFixed(2) + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%)</span>.';
            const todayStr = todayPn != null ? 'Today: ' + (todayPn >= 0 ? '<span class="green">+' : '<span class="red">') + '$' + todayPn.toFixed(2) + ' (' + (todayPct >= 0 ? '+' : '') + todayPct.toFixed(1) + '%)</span> vs start of day.' : 'Today: — (set after first snapshot).';
            todayEl.innerHTML = 'New bets today: <strong>' + (p.today.new_bets || 0) + '</strong>, $' + (p.today.total_deployed || 0).toFixed(2) + ' deployed. Closed today: ' + (state.stats ? (state.stats.today_wins || 0) + ' wins, ' + (state.stats.today_losses || 0) + ' losses (' + (state.stats.today_trades || 0) + ' exits).' : '—') + ' ' + todayStr;
            if (sig.error || !sig.settled) {
                signalEl.textContent = 'Signals: ' + (sig.message || sig.error || 'No settled signals yet — need resolved markets to judge AI.');
            } else {
                const wr = sig.overall_win_rate || 'N/A';
                const pnl = sig.overall_pnl || '$0';
                const bestEdge = (sig.by_edge_threshold && sig.by_edge_threshold.length) ? sig.by_edge_threshold[sig.by_edge_threshold.length - 1] : null;
                const bestConf = (sig.by_confidence_threshold && sig.by_confidence_threshold.length) ? sig.by_confidence_threshold[sig.by_confidence_threshold.length - 1] : null;
                let verdict = 'Settled signals: ' + sig.settled + ', win rate ' + wr + ', theoretical P&L ' + pnl + '.';
                if (bestEdge) verdict += ' Best edge band ' + bestEdge.threshold + '+: ' + bestEdge.win_rate + ' win rate, ' + bestEdge.theoretical_pnl + '.';
                if (bestConf) verdict += ' Best confidence ' + bestConf.threshold + '+: ' + bestConf.win_rate + ', ' + bestConf.theoretical_pnl + '.';
                signalEl.innerHTML = verdict;
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
            // Pull live thresholds from the filter cards so comparison is always current
            const threshEdge = parseFloat((document.getElementById('cfgMinEdge') || {}).textContent) / 100 || 0.12;
            const threshConf = parseFloat((document.getElementById('cfgMinConf') || {}).textContent) / 100 || 0.50;

            const html = analyses.map(a => {
                const isTrade   = a.decision === 'TRADE';
                const edge      = a.edge != null ? a.edge : 0;
                const conf      = a.confidence != null ? a.confidence : null;
                const edgePass  = edge >= threshEdge;
                const confPass  = conf == null || conf >= threshConf;
                const edgeCol   = edgePass ? '#3fb950' : '#f85149';
                const confCol   = confPass ? '#3fb950' : '#f85149';

                // Rejection reason — humanised
                const rawReason = a.reason || '';
                let rejLabel = '';
                if (!isTrade && rawReason) {
                    const parts = [];
                    if (rawReason.includes('LOW_EDGE'))        parts.push(`Edge ${(edge*100).toFixed(1)}% < ${(threshEdge*100).toFixed(0)}% min`);
                    if (rawReason.includes('LOW_CONFIDENCE'))  parts.push(`Conf ${conf != null ? (conf*100).toFixed(0)+'%' : '?'} < ${(threshConf*100).toFixed(0)}% min`);
                    if (rawReason.includes('YES_BETS_DISABLED')) parts.push('YES side disabled');
                    if (rawReason.includes('MAX_POSITIONS'))   parts.push('Position limit reached');
                    if (rawReason.includes('CLUSTER_CAP'))     parts.push('Cluster cap hit');
                    if (rawReason.includes('RECENTLY_EXITED')) parts.push('Cooldown after exit');
                    if (rawReason.includes('ALREADY_IN'))      parts.push('Already in this market');
                    if (parts.length === 0)                    parts.push(rawReason.slice(0, 60));
                    rejLabel = parts.join(' · ');
                }

                return `
                <div class="analysis">
                    <div class="analysis-header">
                        <div style="flex:1;font-size:13px;">${a.question}</div>
                        <span class="analysis-decision ${isTrade ? 'trade' : 'skip'}">${isTrade ? '✓ TRADE' : '✗ SKIP'}</span>
                    </div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:6px 0 4px 0;">
                        <span>AI: <strong>${a.ai_probability != null ? (a.ai_probability*100).toFixed(0)+'%' : '—'}</strong></span>
                        <span>Market: <strong>${(a.market_price*100).toFixed(0)}¢</strong></span>
                        <span style="color:${edgeCol};">Edge: <strong>${edge >= 0 ? '+' : ''}${(edge*100).toFixed(1)}%</strong> ${edgePass ? '✓' : '✗ need '+((threshEdge)*100).toFixed(0)+'%'}</span>
                        ${conf != null ? `<span style="color:${confCol};">Conf: <strong>${(conf*100).toFixed(0)}%</strong> ${confPass ? '✓' : '✗ need '+(threshConf*100).toFixed(0)+'%'}</span>` : ''}
                        <span style="color:#8b949e;">${a.side || ''} · ${a.latency_ms != null ? a.latency_ms+'ms' : ''}</span>
                    </div>
                    ${rejLabel ? `<div style="font-size:11px;color:#f85149;margin-bottom:4px;">↳ Skipped: ${rejLabel}</div>` : ''}
                    ${a.key_reasons && a.key_reasons.length ? `<div style="font-size:11px;color:#58a6ff;line-height:1.5;">• ${a.key_reasons.slice(0,2).join('<br>• ')}</div>` : ''}
                </div>`;
            }).join('');
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
        
        // Fetch and render settlements from Kalshi API
        async function fetchSettlements() {
            try {
                const res = await fetch('/api/settlements');
                const data = await res.json();
                if (data.error) {
                    console.error('Settlements API error:', data.error);
                    return;
                }
                
                const settlements = data.settlements || [];
                document.getElementById('settlementCount').textContent = settlements.length;
                
                // Render settlement list (flag any trade where cost exceeded stated deposits)
                const totalDeposits = data.total_deposits || 150;
                const html = settlements.map(s => {
                    const isWin = s.outcome === 'WIN';
                    const pnlClass = s.pnl > 0 ? 'green' : (s.pnl < 0 ? 'red' : '');
                    const pnlSign = s.pnl >= 0 ? '+' : '';
                    const outcomeIcon = isWin ? '✓' : '✗';
                    const outcomeClass = isWin ? 'green' : 'red';
                    const costExceedsDeposits = s.cost > totalDeposits;
                    const warningLine = costExceedsDeposits ? '<div style="font-size:10px;color:#f85149;margin-top:2px;">Cost $' + s.cost.toFixed(2) + ' &gt; $' + totalDeposits + ' deposited — verify on Kalshi</div>' : '';
                    return `
                        <div style="display:flex;justify-content:space-between;align-items:center;padding:12px;border-bottom:1px solid #30363d;">
                            <div style="flex:1;min-width:0;">
                                <div style="font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${s.question}</div>
                                <div style="font-size:11px;color:#8b949e;margin-top:2px;">${s.settlement_date} · Result: ${s.result}</div>
                                ${warningLine}
                            </div>
                            <div style="text-align:center;padding:0 16px;min-width:70px;">
                                <div style="font-size:13px;font-weight:600;color:#58a6ff;">${s.side}</div>
                                <div style="font-size:11px;color:#8b949e;">${s.contracts} @ ${s.entry_price_cents}¢</div>
                            </div>
                            <div style="text-align:right;min-width:90px;">
                                <div style="font-size:15px;font-weight:600;" class="${pnlClass}">${pnlSign}$${Math.abs(s.pnl).toFixed(2)}</div>
                                <div style="font-size:11px;" class="${outcomeClass}">${outcomeIcon} ${s.outcome}</div>
                            </div>
                        </div>
                    `;
                }).join('');
                document.getElementById('settlementList').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No settled bets yet - positions will appear here when markets resolve</div>';
                
            } catch (e) {
                console.error('Failed to fetch settlements:', e);
            }
        }
        
        // Chart mode toggle
        window._chartMode = 'daily';
        function setChartMode(mode) {
            window._chartMode = mode;
            ['hourly','daily','monthly'].forEach(m => {
                const btn = document.getElementById('btn' + m.charAt(0).toUpperCase() + m.slice(1));
                if (!btn) return;
                if (m === mode) {
                    btn.style.borderColor = '#58a6ff'; btn.style.background = '#1f3a5f'; btn.style.color = '#58a6ff';
                } else {
                    btn.style.borderColor = '#30363d'; btn.style.background = '#21262d'; btn.style.color = '#8b949e';
                }
            });
            if (window._chartHistory) {
                renderHistoryChart(window._chartHistory, window._chartHourly || [], window._chartDeposits || 150, mode);
            }
        }

        // Portfolio value (Kalshi) day-over-day: line + bar chart, updates hourly
        function renderHistoryChart(history, todayHourly, depositsArg, mode) {
            mode = mode || 'daily';
            const container = document.getElementById('historyChart');
            if (!history || history.length === 0) {
                container.innerHTML = '<div style="color:#8b949e;font-size:13px;padding:24px 0;text-align:center;">No history yet — data updates hourly once the bot syncs with Kalshi.</div>';
                return;
            }

            const DEPOSITS = parseFloat(depositsArg) || 150;
            const ordered = history.slice().reverse(); // oldest → newest
            const todayStr = new Date().toISOString().slice(0, 10);
            const todayDay  = ordered.find(h => h.date === todayStr);

            let points = [];
            let todaySepIdx = 0;

            if (mode === 'hourly') {
                // Hourly mode: show only today's hourly data
                if (!todayHourly || todayHourly.length === 0) {
                    container.innerHTML = '<div style="color:#8b949e;font-size:13px;padding:24px 0;text-align:center;">No hourly data yet for today.</div>';
                    return;
                }
                todayHourly.forEach((pt, i) => {
                    const hh = String(pt.hour).padStart(2,'0');
                    points.push({ label: hh+':00', dayLabel: 'Today '+hh+':00', value: pt.value,
                                  isToday: true, isHourly: true, isLatest: i === todayHourly.length - 1 });
                });
                todaySepIdx = 0;
            } else {
                // Daily / Monthly: show past days + today
                const daysBack = mode === 'monthly' ? 30 : 14;
                const cutoff = new Date(); cutoff.setDate(cutoff.getDate() - daysBack);
                const cutoffStr = cutoff.toISOString().slice(0, 10);
                const pastDays = ordered.filter(h => h.date < todayStr && h.date >= cutoffStr && h.account_value != null);
                pastDays.forEach(h => {
                    const d = new Date(h.date + 'T12:00:00');
                    points.push({
                        label:    d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        dayLabel: d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }),
                        value: h.account_value, isToday: false, isHourly: false,
                        high: h.intraday_high, low: h.intraday_low,
                    });
                });
                todaySepIdx = points.length;
                if (todayHourly && todayHourly.length > 0) {
                    todayHourly.forEach((pt, i) => {
                        const hh = String(pt.hour).padStart(2,'0');
                        points.push({ label: hh+':00', dayLabel: 'Today '+hh+':00', value: pt.value,
                                      isToday: true, isHourly: true, isLatest: i === todayHourly.length - 1 });
                    });
                } else if (todayDay && todayDay.account_value != null) {
                    points.push({ label: 'Today', dayLabel: 'Today', value: todayDay.account_value,
                                  isToday: true, isHourly: false,
                                  high: todayDay.intraday_high, low: todayDay.intraday_low, isLatest: true });
                }
            }

            if (points.length === 0) {
                container.innerHTML = '<div style="color:#8b949e;font-size:13px;padding:24px 0;text-align:center;">No portfolio values recorded yet. Check back soon.</div>';
                return;
            }
            // Ensure at least 2 points for a visible line
            if (points.length === 1) {
                points.unshift({ label: '', dayLabel: '', value: DEPOSITS, isToday: false, isHourly: false, ghost: true });
            }

            // ── Dimensions ────────────────────────────────────────────────────
            const W = 780, H = 220;
            const pad = { top: 28, right: 78, bottom: 38, left: 54 };
            const cW = W - pad.left - pad.right;
            const cH = H - pad.top - pad.bottom;

            const allVals = points.map(p => p.value).filter(v => v != null);
            const highVals = points.filter(p => p.high).map(p => p.high);
            // Include today's intraday high so the watermark line never clips above the chart
            const todayHighVal = (todayDay && todayDay.intraday_high) ? todayDay.intraday_high : 0;
            const spread = Math.max(...[...allVals, ...highVals, DEPOSITS, todayHighVal]) - Math.min(...allVals, DEPOSITS);
            const vPad = Math.max(spread * 0.18, 3);
            const minY = Math.max(0, Math.min(...allVals, DEPOSITS) - vPad);
            const maxY = Math.max(...allVals, ...highVals, DEPOSITS, todayHighVal) + vPad;
            const yRange = maxY - minY || 1;

            const xS = i  => pad.left + (i / Math.max(points.length - 1, 1)) * cW;
            const yS = v  => pad.top  + (1 - (v - minY) / yRange) * cH;
            const depY    = yS(DEPOSITS);
            const botY    = pad.top + cH;

            // ── Y-axis ticks ──────────────────────────────────────────────────
            const rawStep = (maxY - minY) / 4;
            const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
            const niceStep = [1,2,5,10].map(f => f * mag).find(s => s >= rawStep) || mag;
            const ticks = [];
            for (let t = Math.ceil(minY / niceStep) * niceStep; t <= maxY + 0.001; t = Math.round((t + niceStep) * 1e6) / 1e6) {
                ticks.push(Math.round(t * 100) / 100);
            }
            if (!ticks.some(t => Math.abs(t - DEPOSITS) < 0.01)) ticks.push(DEPOSITS);

            // ── Smooth Catmull-Rom path through valid points ──────────────────
            const vpts = points
                .map((p, i) => ({ ...p, x: xS(i), y: p.value != null ? yS(p.value) : null }))
                .filter(p => p.y != null);

            function catmull(pts) {
                if (pts.length === 1) return `M${pts[0].x},${pts[0].y}`;
                let d = `M${pts[0].x.toFixed(2)},${pts[0].y.toFixed(2)}`;
                for (let i = 0; i < pts.length - 1; i++) {
                    const p0 = pts[Math.max(0, i-1)], p1 = pts[i],
                          p2 = pts[i+1],               p3 = pts[Math.min(pts.length-1, i+2)];
                    const t = 0.35;
                    const cx1 = (p1.x + (p2.x - p0.x) * t).toFixed(2);
                    const cy1 = (p1.y + (p2.y - p0.y) * t).toFixed(2);
                    const cx2 = (p2.x - (p3.x - p1.x) * t).toFixed(2);
                    const cy2 = (p2.y - (p3.y - p1.y) * t).toFixed(2);
                    d += ` C${cx1},${cy1} ${cx2},${cy2} ${p2.x.toFixed(2)},${p2.y.toFixed(2)}`;
                }
                return d;
            }

            const linePath = catmull(vpts);
            const fp = vpts[0], lp = vpts[vpts.length - 1];
            const areaPath = linePath + ` L${lp.x.toFixed(2)},${botY} L${fp.x.toFixed(2)},${botY} Z`;

            const curVal  = lp.value;
            const curY    = lp.y;
            const curCol  = curVal >= DEPOSITS ? '#3fb950' : '#f85149';
            const uid     = 'c' + Date.now();

            // ── Today high watermark ──────────────────────────────────────────
            const todayHigh    = todayDay ? todayDay.intraday_high  : null;
            const todayLow     = todayDay ? todayDay.intraday_low   : null;
            const todayStart   = todayDay ? todayDay.start_of_day_value : null;
            const peakGiveback = todayHigh && curVal ? (todayHigh - curVal) : 0;
            const todayHighY   = todayHigh ? yS(todayHigh) : null;
            const todaySepX    = (todaySepIdx > 0 && todaySepIdx < points.length) ? xS(todaySepIdx) : null;

            // ── X labels: pick at most 12, always include first & last ────────
            const maxLbls = 12;
            const step    = Math.max(1, Math.floor(points.length / maxLbls));
            const lblIdxs = new Set();
            for (let i = 0; i < points.length; i += step) lblIdxs.add(i);
            lblIdxs.add(points.length - 1);

            // ── Build SVG ─────────────────────────────────────────────────────
            const svg = `
<svg viewBox="0 0 ${W} ${H}" style="max-width:100%;height:auto;display:block;overflow:visible;" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="${uid}gA" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3fb950" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#3fb950" stop-opacity="0.03"/>
    </linearGradient>
    <linearGradient id="${uid}gB" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f85149" stop-opacity="0.03"/>
      <stop offset="100%" stop-color="#f85149" stop-opacity="0.55"/>
    </linearGradient>
    <clipPath id="${uid}ca"><rect x="${pad.left}" y="${pad.top}" width="${cW}" height="${Math.max(0, depY-pad.top).toFixed(2)}"/></clipPath>
    <clipPath id="${uid}cb"><rect x="${pad.left}" y="${depY.toFixed(2)}" width="${cW}" height="${Math.max(0, botY-depY).toFixed(2)}"/></clipPath>
    <filter id="${uid}glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>

  <!-- Background chart area -->
  <rect x="${pad.left}" y="${pad.top}" width="${cW}" height="${cH}" fill="#0d1117" rx="2" opacity="0.4"/>

  <!-- Y grid lines + labels -->
  ${ticks.map(t => {
    const ty = yS(t).toFixed(2);
    const isDep = Math.abs(t - DEPOSITS) < 0.01;
    return `<line x1="${pad.left}" y1="${ty}" x2="${pad.left+cW}" y2="${ty}"
              stroke="${isDep ? '#58a6ff' : '#21262d'}" stroke-width="${isDep ? 1.2 : 0.5}"
              stroke-dasharray="${isDep ? '5,3' : 'none'}" opacity="${isDep ? 0.9 : 1}"/>
    <text x="${(pad.left-7).toFixed(1)}" y="${(parseFloat(ty)+4).toFixed(1)}"
          font-size="10" fill="${isDep ? '#58a6ff' : '#484f58'}" text-anchor="end"
          font-weight="${isDep ? 'bold' : 'normal'}">$${t%1===0?t:t.toFixed(0)}</text>`;
  }).join('')}

  <!-- Deposit label on right -->
  <text x="${(pad.left+cW+5).toFixed(1)}" y="${(depY+4).toFixed(1)}"
        font-size="9" fill="#58a6ff" opacity="0.8">deposited</text>

  <!-- Area fills (split green above / red below deposit line) -->
  <path d="${areaPath}" clip-path="url(#${uid}ca)" fill="url(#${uid}gA)" stroke="none"/>
  <path d="${areaPath}" clip-path="url(#${uid}cb)" fill="url(#${uid}gB)" stroke="none"/>

  <!-- Today separator -->
  ${todaySepX != null ? `
  <line x1="${todaySepX.toFixed(2)}" y1="${pad.top}" x2="${todaySepX.toFixed(2)}" y2="${botY}"
        stroke="#58a6ff" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.5"/>
  <rect x="${(todaySepX-16).toFixed(1)}" y="${(pad.top-20).toFixed(1)}" width="32" height="14" rx="3"
        fill="#1f2937" stroke="#58a6ff" stroke-width="0.5" opacity="0.8"/>
  <text x="${todaySepX.toFixed(1)}" y="${(pad.top-9).toFixed(1)}"
        font-size="9" fill="#58a6ff" text-anchor="middle" font-weight="bold">TODAY</text>` : ''}

  <!-- Today intraday high watermark (peak giveback indicator) -->
  ${todayHighY != null && peakGiveback > 0.5 && todaySepX != null ? `
  <line x1="${todaySepX.toFixed(1)}" y1="${todayHighY.toFixed(1)}" x2="${(lp.x+4).toFixed(1)}" y2="${todayHighY.toFixed(1)}"
        stroke="#e3b341" stroke-width="1" stroke-dasharray="3,2" opacity="0.75"/>
  <circle cx="${todaySepX.toFixed(1)}" cy="${todayHighY.toFixed(1)}" r="2.5" fill="#e3b341" opacity="0.8"/>
  <text x="${((todaySepX + lp.x)/2).toFixed(1)}" y="${(todayHighY-5).toFixed(1)}"
        font-size="9" fill="#e3b341" text-anchor="middle" opacity="0.9">↑ peak $${todayHigh.toFixed(2)}</text>
  <line x1="${lp.x.toFixed(1)}" y1="${todayHighY.toFixed(1)}" x2="${lp.x.toFixed(1)}" y2="${curY.toFixed(1)}"
        stroke="#e3b341" stroke-width="0.8" stroke-dasharray="2,2" opacity="0.6"/>
  <text x="${(lp.x+6).toFixed(1)}" y="${((todayHighY+curY)/2+4).toFixed(1)}"
        font-size="9" fill="#e3b341" opacity="0.85">-$${peakGiveback.toFixed(2)}</text>` : ''}

  <!-- Coloured line (above deposit = green, below = red) -->
  <path d="${linePath}" clip-path="url(#${uid}ca)" fill="none" stroke="#3fb950" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="${linePath}" clip-path="url(#${uid}cb)" fill="none" stroke="#f85149" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Past day data points -->
  ${vpts.filter(p => !p.isToday && !p.ghost).map(p => {
    const col = p.value >= DEPOSITS ? '#3fb950' : '#f85149';
    return `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="4"
              fill="${col}" stroke="#0d1117" stroke-width="1.5">
              <title>${p.dayLabel}: $${p.value.toFixed(2)}</title></circle>`;
  }).join('')}

  <!-- Today hourly dots (smaller) -->
  ${vpts.filter(p => p.isToday && !p.isLatest).map(p => {
    const col = p.value >= DEPOSITS ? '#3fb950' : '#f85149';
    return `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="2.5"
              fill="${col}" stroke="#0d1117" stroke-width="1" opacity="0.65">
              <title>${p.dayLabel}: $${p.value.toFixed(2)}</title></circle>`;
  }).join('')}

  <!-- Current value dot (highlighted) -->
  <circle cx="${lp.x.toFixed(2)}" cy="${curY.toFixed(2)}" r="9" fill="${curCol}" opacity="0.12"/>
  <circle cx="${lp.x.toFixed(2)}" cy="${curY.toFixed(2)}" r="5" fill="${curCol}"
          stroke="#0d1117" stroke-width="2" filter="url(#${uid}glow)"/>

  <!-- Current value badge -->
  <rect x="${(pad.left+cW+8).toFixed(1)}" y="${(curY-11).toFixed(1)}" width="56" height="20" rx="5"
        fill="${curVal >= DEPOSITS ? '#238636' : '#da3633'}"/>
  <text x="${(pad.left+cW+36).toFixed(1)}" y="${(curY+3.5).toFixed(1)}"
        font-size="11.5" fill="white" text-anchor="middle" font-weight="bold">$${curVal.toFixed(2)}</text>

  <!-- X-axis labels -->
  ${points.map((p, i) => {
    if (!lblIdxs.has(i) || p.ghost) return '';
    const col = p.isToday ? '#58a6ff' : '#484f58';
    return `<text x="${xS(i).toFixed(1)}" y="${(H-6).toFixed(1)}"
              font-size="9.5" fill="${col}" text-anchor="middle">${p.label}</text>`;
  }).join('')}

  <!-- Chart border -->
  <rect x="${pad.left}" y="${pad.top}" width="${cW}" height="${cH}"
        fill="none" stroke="#21262d" stroke-width="0.5" rx="2"/>
</svg>
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;padding:0 ${pad.left}px;">
  <span style="font-size:10px;color:#484f58;">Portfolio value from Kalshi API · <span style="color:#58a6ff;">── $${DEPOSITS} deposited</span> · Updates every ~5 min</span>
  ${peakGiveback > 0.5 ? `<span style="font-size:10px;color:#e3b341;">⚠ Today's peak was $${(todayHigh||0).toFixed(2)} · gave back $${peakGiveback.toFixed(2)}</span>` : ''}
</div>`;

            container.innerHTML = svg;
        }
        
        connect();
        
        // Initial performance fetch and regular updates
        fetchPerformance();
        fetchSettlements();
        setInterval(fetchPerformance, 15000); // Update every 15 seconds
        setInterval(fetchSettlements, 30000); // Update settlements every 30 seconds
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
