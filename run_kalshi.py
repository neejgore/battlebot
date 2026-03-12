#!/usr/bin/env python3
"""Battle-Bot for Kalshi - CFTC-regulated prediction market.

Legal for US residents including California.
Uses same AI strategy as Polymarket version.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*utcnow.*')

import asyncio
import collections
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from aiohttp import web
import httpx
from dotenv import load_dotenv

from logic.ai_signal import AISignalGenerator, AISignalResult
from logic.calibration import CalibrationEngine, CalibrationResult
from logic.risk_engine import RiskEngine, RiskLimits
from data.database import TelemetryDB
from services.kalshi_client import KalshiClient, parse_kalshi_market
from services.kalshi_websocket import KalshiWebSocketClient
from services.market_intelligence import get_intelligence_service, MarketIntelligence
from services.crypto_edge_service import CryptoEdgeService, CryptoEdgeResult

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
        self.min_edge = max(0.05, float(os.getenv('MIN_EDGE', 0.12)))  # 12% edge floor — data shows 8-11% range is net-negative
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.72))  # 72% confidence floor
        # Historical note: "0.65-0.79 loses money, only >=0.80 profitable" — that was true before
        # the consensus guard, temporal guard, econ sanity, and intel filters were added.
        # Those guards now filter out the low-quality signals that dragged down 0.65-0.79 winrate.
        # 0.72 allows high-quality economic signals (0.72-0.79 band) while blocking true noise.
        # max_position_size scales with bankroll: default 15% of INITIAL_BANKROLL.
        # At $1000 bankroll this is $150/bet; set MAX_POSITION_SIZE env var to override.
        _default_max_pos = float(os.getenv('INITIAL_BANKROLL', 100)) * 0.15
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', _default_max_pos))
        self.max_days_to_resolution = float(os.getenv('MAX_DAYS_TO_RESOLUTION', 30))  # 30 days max — tighter than 45
        self.min_days_to_resolution = float(os.getenv('MIN_DAYS_TO_RESOLUTION', 0))  # No minimum — sports/weather filters handle bad short-horizon bets; BTC range needs access
        self.kelly_fraction = float(os.getenv('FRACTIONAL_KELLY', 0.25))  # 25% (Quarter-Kelly) — standard conservative sizing
        self.max_oi_pct = float(os.getenv('MAX_OI_PCT', 0.10))  # Max 10% of open interest
        self.simulate_prices = os.getenv('SIMULATE_PRICES', 'false').lower() == 'true'
        
        # Kalshi client
        use_demo = os.getenv('KALSHI_USE_DEMO', 'false').lower() == 'true'
        self._kalshi = KalshiClient(use_demo=use_demo)

        # Kalshi WebSocket client — provides real-time ticker and fill updates.
        # Shares the same credentials as the REST client.
        self._ws_client = KalshiWebSocketClient(
            api_key_id=self._kalshi.api_key_id,
            private_key=self._kalshi._private_key,
            use_demo=use_demo,
        )

        # Quantitative edge engine for crypto range markets
        self._crypto_edge = CryptoEdgeService()
        
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
        # Filter diagnostic log — records every market skipped before Claude, with reason
        self._filter_log: collections.deque = collections.deque(maxlen=500)
        # Nightly strategy report — populated by _nightly_strategy_loop at midnight UTC
        self._last_nightly_report: dict = {}
        # Actual Kalshi values from API (synced every 5 min)
        # ALL three are set together in _sync_positions_with_kalshi so they stay consistent.
        self._kalshi_cash = None            # Available cash (from balance.balance)
        self._kalshi_positions_mv = None    # Market value of open positions (from balance.portfolio_value)
        self._kalshi_portfolio = None       # Cost basis of open positions (sum of market_exposure) — for "capital at risk" display only
        self._kalshi_total = None           # cash + positions_mv — single source of truth for ALL account value displays
        self._kalshi_positions_raw: list = []   # Raw positions from last sync (used by performance endpoint)
        self._settlements_cache: dict | None = None      # Cached settlements response
        self._settlements_cache_time: datetime | None = None  # When cache was populated
        
        # All attributes that _load_state() will populate must be initialised BEFORE
        # _load_state() is called so that _save_state() never writes {} for them.
        self._daily_snapshots: dict = {}
        self._portfolio_hourly: dict = {}
        self._recently_exited: dict = {}        # exit-cooldown timestamps (restored by _load_state)
        self._recently_exited_reason: dict = {} # exit-cooldown reasons (restored by _load_state)
        # Stats debug counter — initialised here to avoid hasattr in _get_stats hot path
        self._stats_debug_counter: int = 0
        # Snapshot file-save throttle — initialised here to avoid hasattr in _save_daily_snapshot
        self._last_snapshot_file_save: datetime | None = None

        # Risk Engine must be created BEFORE _load_state() so the kill-switch restore
        # logic inside _load_state() can write to self._risk_engine.daily_stats.
        # All env vars it needs (initial_bankroll, kelly_fraction, min_edge,
        # max_position_size) are already set above.
        self._risk_limits = RiskLimits(
            max_daily_drawdown=0.10,  # 10% daily loss circuit breaker (was 15%)
            max_position_size=self.max_position_size,
            max_percent_bankroll_per_market=0.25,
            max_total_open_risk=0.90,
            max_positions=25,
            profit_take_pct=999.0,    # DISABLED - let bets settle naturally
            stop_loss_pct=999.0,      # DISABLED - let bets settle naturally
            time_stop_hours=720,
            edge_scale=0.10,
            min_edge=self.min_edge,
        )
        self._risk_engine = RiskEngine(
            initial_bankroll=self.initial_bankroll,
            fractional_kelly=self.kelly_fraction,
            limits=self._risk_limits,
        )

        self._load_state()
        # Only wipe intraday_high if the saved timestamp is from a *previous* session that
        # ran on a different UTC day — i.e. the snapshot's intraday_high_time date doesn't
        # match today.  Wiping unconditionally was destroying a legitimate same-day peak on
        # every Railway restart, leaving "Today's Peak" blank until the next sync cycle.
        _today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if _today_str in self._daily_snapshots:
            _snap = self._daily_snapshots[_today_str]
            _high_time = _snap.get('intraday_high_time', '')
            # Only clear if the high was recorded on a different date (genuinely stale).
            if _high_time and not _high_time.startswith(_today_str):
                _snap.pop('intraday_high', None)
                _snap.pop('intraday_high_time', None)
        
        self._running = False
        self._last_analysis: dict[str, datetime] = {}
        self._last_analysis_price: dict[str, float] = {}
        self._analysis_cooldown = 1800  # 30 minutes (overridden per time-horizon in trading loop)
        self._price_change_threshold = 0.02  # Re-analyze on 2% move
        # Probability cache: {market_id: (timestamp, price_when_cached, ai_prob, confidence)}
        # Reused when price hasn't moved significantly — avoids redundant Claude calls
        self._prob_cache: dict[str, tuple[datetime, float, float, float]] = {}
        self._start_time = None
        # Position monitor helpers — initialised here to avoid hasattr checks in hot loops
        self._news_check_times: dict[str, float] = {}   # pos_id -> last news fetch epoch
        self._btc_price_cache: dict = {'price': None, 'fetched_at': 0}
        self._monitor_log_times: dict[str, float] = {}  # pos_id -> last 70%-down log epoch
        # Profit-lock threshold — read once from env, not per-position per-cycle
        self._profit_lock_pct: float = float(os.getenv('PROFIT_LOCK_PCT', '0.50'))
        
        # Kill-switch fire date: set once (False→True transition), never overwritten.
        # Persisted to state so a restart after midnight doesn't keep the switch active.
        self._kill_switch_fire_date: str = ''

        # 500ms order rate-limit gate (workspace rule: min 500ms between placements)
        self._last_order_time: datetime = datetime.utcnow() - timedelta(seconds=1)

        # Entry lock: prevents concurrent _analyze_market coroutines from entering
        # the same market simultaneously. Without this, two tasks can both pass the
        # ALREADY_IN_POSITION check before either has updated _pending_orders.
        self._entry_lock: asyncio.Lock = asyncio.Lock()

        # Price updates counter (incremented by WS ticker callback and REST refresh)
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
        
        # (_risk_limits and _risk_engine are initialised earlier, before _load_state())

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
                with open(self._state_file, 'r', encoding='utf-8', errors='replace') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._pending_orders = state.get('pending_orders', {})
                    self._trades = state.get('trades', [])
                    self._signal_log = state.get('signal_log', [])
                    self._daily_snapshots = state.get('daily_snapshots', {})
                    self._portfolio_hourly = state.get('portfolio_hourly', {})
                    # Restore exit cooldowns (ISO strings → datetime)
                    for k, v in state.get('recently_exited', {}).items():
                        try:
                            self._recently_exited[k] = datetime.fromisoformat(v)
                        except Exception:
                            pass
                    self._recently_exited_reason = state.get('recently_exited_reason', {})
                    # _today must be defined before any snapshot lookup that uses it.
                    _today = datetime.utcnow().strftime('%Y-%m-%d')
                    _today_snap = self._daily_snapshots.get(_today, {})
                    # If the kill switch was manually reset today, use the reset baseline
                    # (stored as _ks_reset_baseline) instead of the original start_of_day.
                    # This survives redeploys so the reset isn't wiped every time Railway restarts.
                    _ks_reset_baseline = _today_snap.get('_ks_reset_baseline')
                    _today_start = _ks_reset_baseline or _today_snap.get('start_of_day_value')
                    if _today_start and _today_start > 0:
                        self._risk_engine.daily_stats.starting_bankroll = _today_start
                        _label = 'reset baseline' if _ks_reset_baseline else 'start_of_day'
                        print(f"[State] Restored today's starting bankroll ({_label}): ${_today_start:.2f}")
                    # Restore kill switch ONLY if it fired today AND user has NOT manually reset it.
                    # Without the reset-baseline guard, a bad Kalshi API reading re-triggers the
                    # kill switch after a manual reset, saves triggered=True, and every subsequent
                    # redeploy (e.g. from a code push) starts halted again.
                    # Rule: if _ks_reset_baseline exists, the user overrode the kill switch today —
                    # respect that across redeploys. The kill switch can still fire during the run
                    # from genuine losses; it just won't auto-start the bot in a halted state.
                    _ks_date = state.get('kill_switch_date', '')
                    _user_reset_today = bool(_ks_reset_baseline)
                    if state.get('kill_switch_triggered') and _ks_date == _today and not _user_reset_today:
                        self._risk_engine.daily_stats.kill_switch_triggered = True
                        self._kill_switch_fire_date = _ks_date
                        print(f"[State] Kill switch restored from state (triggered today {_ks_date})")
                    elif state.get('kill_switch_triggered') and _ks_date == _today and _user_reset_today:
                        print(f"[State] Kill switch NOT restored — user manually reset today (baseline ${_ks_reset_baseline:.2f})")
                    # Restore analysis cooldowns to prevent burst re-analysis on restart
                    for k, v in state.get('last_analysis', {}).items():
                        try:
                            self._last_analysis[k] = datetime.fromisoformat(v)
                        except Exception:
                            pass
                    print(f"[State] Loaded {len(self._positions)} positions, {len(self._pending_orders)} pending orders, {len(self._trades)} trades, {len(self._signal_log)} signals, {len(self._recently_exited)} exit cooldowns")
                    
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
                with open(self._state_file + '.backup', 'r', encoding='utf-8', errors='replace') as f:
                    state = json.load(f)
                    self._positions = state.get('positions', {})
                    self._pending_orders = state.get('pending_orders', {})
                    self._trades = state.get('trades', [])
                    self._signal_log = state.get('signal_log', [])
                    self._daily_snapshots = state.get('daily_snapshots', {})
                    self._portfolio_hourly = state.get('portfolio_hourly', {})
                    for k, v in state.get('recently_exited', {}).items():
                        try:
                            self._recently_exited[k] = datetime.fromisoformat(v)
                        except Exception:
                            pass
                    self._recently_exited_reason = state.get('recently_exited_reason', {})
                    # Restore kill-switch state (same date-gate as primary path)
                    _ks_date_bak = state.get('kill_switch_date', '')
                    _today_bak = datetime.utcnow().strftime('%Y-%m-%d')
                    _today_snap_bak = self._daily_snapshots.get(_today_bak, {})
                    _ks_reset_baseline_bak = _today_snap_bak.get('_ks_reset_baseline')
                    _today_start_bak = _ks_reset_baseline_bak or _today_snap_bak.get('start_of_day_value')
                    if _today_start_bak and _today_start_bak > 0:
                        self._risk_engine.daily_stats.starting_bankroll = _today_start_bak
                        print(f"[State] Restored today's starting bankroll from backup: ${_today_start_bak:.2f}")
                    _user_reset_today_bak = bool(_ks_reset_baseline_bak)
                    if state.get('kill_switch_triggered') and _ks_date_bak == _today_bak and not _user_reset_today_bak:
                        self._risk_engine.daily_stats.kill_switch_triggered = True
                        print(f"[State] Kill-switch restored from backup (triggered today)")
                    elif state.get('kill_switch_triggered') and _ks_date_bak == _today_bak and _user_reset_today_bak:
                        print(f"[State] Kill-switch NOT restored from backup — user manually reset today")
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
            
            # Serialize recently_exited (datetime values → ISO strings)
            recently_exited_serial = {
                k: v.isoformat() for k, v in self._recently_exited.items()
            }
            state_data = {
                'positions': self._positions,
                'pending_orders': self._pending_orders,
                'trades': self._trades[:500],   # insert(0,...) means newest is at front
                'signal_log': self._signal_log[-500:],
                'daily_snapshots': daily_snapshots,
                'portfolio_hourly': portfolio_hourly,
                'recently_exited': recently_exited_serial,
                'recently_exited_reason': self._recently_exited_reason,
                'kill_switch_triggered': self._risk_engine.daily_stats.kill_switch_triggered,
                # Persist the exact date the kill-switch fired (set once on False→True
                # transition; never overwritten here).
                'kill_switch_date': self._kill_switch_fire_date,
                # Persist per-market analysis timestamps so cooldowns survive restarts
                # and prevent burst re-analysis + potential duplicate orders on startup.
                'last_analysis': {k: v.isoformat() for k, v in self._last_analysis.items()},
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
                    # Order filled while bot was down - create position.
                    # Kalshi API may use 'fill_count' OR 'filled_count' depending on SDK version.
                    # average_fill_price may not exist; derive from taker+maker cost if available.
                    fill_count = (order_data.get('fill_count')
                                  or order_data.get('filled_count')
                                  or order['contracts'])
                    _taker = order_data.get('taker_fill_cost', 0)
                    _maker = order_data.get('maker_fill_cost', 0)
                    if (_taker + _maker) > 0 and fill_count > 0:
                        fill_price = (_taker + _maker) / fill_count / 100  # costs in cents
                    else:
                        fill_price = order_data.get('average_fill_price', order['entry_price'] * 100) / 100

                    # Scale size to actual fill (partial fills cost less than planned)
                    total_contracts = order.get('contracts', 1) or 1
                    fill_ratio = min(1.0, fill_count / total_contracts)
                    actual_size = fill_count * fill_price  # true dollar cost at fill price
                    
                    pos_id = order.get('id', f"pos_{order_id}")
                    pos = {
                        'id': pos_id,
                        'order_id': order_id,
                        'market_id': order['market_id'],
                        'question': order.get('question', ''),
                        'side': order['side'],
                        'size': actual_size,
                        'entry_price': fill_price,
                        'current_price': fill_price,
                        'contracts': fill_count,
                        'ai_probability': order.get('ai_probability', 0.5),
                        'edge': order.get('edge', 0),
                        'confidence': order.get('confidence', 0),
                        'entry_time': order.get('placed_time', datetime.utcnow().isoformat()),
                        'unrealized_pnl': 0.0,
                        'end_date': order.get('end_date'),
                        'has_intel': order.get('has_intel', False),
                        'news_count': order.get('news_count', 0),
                        'category': order.get('category', 'unknown'),
                    }
                    self._positions[pos_id] = pos
                    self._pending_orders.pop(order_id)
                    filled += 1
                    partial_flag = f" (partial: {fill_count}/{total_contracts})" if fill_ratio < 1.0 else ""
                    print(f"[Startup] Order {order_id[:8]}... was FILLED{partial_flag} @ {fill_price*100:.0f}¢ (${actual_size:.2f})")
                    
                elif status in ('canceled', 'cancelled'):
                    # B49 fix: check for partial fill before treating as a clean cancel
                    startup_partial = order_data.get('filled_count', 0)
                    if startup_partial and startup_partial > 0:
                        fill_price = order_data.get('average_fill_price', order['entry_price'] * 100) / 100
                        actual_size = startup_partial * fill_price
                        pos_id = order.get('id', f"pos_{order_id}")
                        pos = {
                            'id': pos_id,
                            'order_id': order_id,
                            'market_id': order['market_id'],
                            'question': order.get('question', ''),
                            'side': order['side'],
                            'size': actual_size,
                            'entry_price': fill_price,
                            'current_price': fill_price,
                            'contracts': startup_partial,
                            'ai_probability': order.get('ai_probability', 0.5),
                            'edge': order.get('edge', 0),
                            'confidence': order.get('confidence', 0),
                            'entry_time': order.get('placed_time', datetime.utcnow().isoformat()),
                            'unrealized_pnl': 0.0,
                            'end_date': order.get('end_date'),
                            'has_intel': order.get('has_intel', False),
                            'news_count': order.get('news_count', 0),
                            'category': order.get('category', 'unknown'),
                        }
                        self._positions[pos_id] = pos
                        filled += 1
                        print(f"[Startup] Order {order_id[:8]}... PARTIAL+CANCELED — {startup_partial}/{order.get('contracts','?')} contracts @ {fill_price*100:.0f}¢ (${actual_size:.2f}) — position created")
                    else:
                        canceled += 1
                        print(f"[Startup] Order {order_id[:8]}... was CANCELED")
                    self._pending_orders.pop(order_id)
                    
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
                total_buy_contracts = 0  # Track buys separately for avg_price
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
                        total_buy_contracts += count
                        total_cost += price * count
                        if fill_side:  # Only update side if the field is present
                            side = fill_side
                    elif action == 'sell':
                        net_contracts -= count

                # Skip if no net position (fully closed via normal sells or already-posted settlement fills).
                # Use total_buy_contracts as fallback so markets that settled before this loop ran
                # (where Kalshi has already posted the settlement sell fill) are still reconciled.
                effective_contracts = net_contracts if net_contracts > 0 else (
                    total_buy_contracts if total_buy_contracts > 0 else 0
                )
                if effective_contracts <= 0:
                    continue
                # If net_contracts was <=0 but we have buys, use total_buy_contracts for P&L
                if net_contracts <= 0:
                    net_contracts = total_buy_contracts
                
                # avg_price is per-contract cost (use total buys, not net, to get true cost basis)
                avg_price = total_cost / total_buy_contracts if total_buy_contracts > 0 else 0.5
                
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
                    # Lowercase both sides — Kalshi may return 'Yes', 'YES', or 'yes'.
                    # If the fill API did not return a 'side' field, fall back to the
                    # ENTRY record in self._trades (or the open position if still tracked).
                    if not side:
                        _entry = entry_by_ticker.get(ticker, {})
                        side = _entry.get('side', '')
                        if not side:
                            _pos_match = next(
                                (p for p in self._positions.values()
                                 if p.get('market_id') == ticker), None
                            )
                            if _pos_match:
                                side = _pos_match.get('side', '')
                    our_side = (side or '').lower()
                    # If side is empty after all three fallbacks (fills API, _trades,
                    # _positions), we cannot determine WIN vs LOSS. Skip rather than
                    # recording a guaranteed-wrong LOSS that corrupts calibration data.
                    if not our_side:
                        print(f"[Reconcile] Skipping {ticker[:30]} — cannot determine side (fills API omitted it and no trade/position record found)")
                        continue
                    if our_side == (result or '').lower():
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

                    # B16 fix: update calibration DB with ground-truth outcome so
                    # the calibration engine can learn from settled markets.
                    if self._calibration and self._db_connected:
                        try:
                            yes_won = (result or '').lower() == 'yes'
                            updated = await self._calibration.record_outcome_by_market(
                                market_id=ticker,
                                yes_won=yes_won,
                            )
                            if updated:
                                print(f"[Calibration] Updated {updated} sample(s) for {ticker[:30]} → {'YES' if yes_won else 'NO'}")
                        except Exception as _cal_err:
                            print(f"[Calibration] record_outcome_by_market error: {_cal_err}")
                    
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

    async def _settlement_reconcile_loop(self):
        """Run settlement reconciliation every 6 hours to capture settled markets."""
        await asyncio.sleep(3600)  # First run 1 hour after startup (startup already ran it)
        while self._running:
            try:
                if not self.dry_run:
                    await self._reconcile_settlements_from_fills()
            except Exception as e:
                print(f"[SettleLoop] Error: {e}")
            await asyncio.sleep(6 * 3600)  # Every 6 hours

    async def _run_nightly_report(self) -> None:
        """Calculate and log the nightly strategy performance report."""
        now = datetime.now(timezone.utc)
        # Truncate to 19 chars (YYYY-MM-DDTHH:MM:SS) so that string comparison
        # against naive trade/analysis timestamps is consistent.
        cutoff_30d = (now - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')
        cutoff_7d  = (now - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
        cutoff_24h = (now - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S')

        exits = [t for t in self._trades if t.get('action') == 'EXIT' and t.get('pnl') is not None]
        recent = [t for t in exits if t.get('timestamp', '') >= cutoff_30d]
        week   = [t for t in exits if t.get('timestamp', '') >= cutoff_7d]

        # P&L by cluster/category
        cat: dict = collections.defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
        for t in recent:
            pnl = t.get('pnl', 0) or 0
            key = self._get_cluster_key(t.get('question', ''))
            cat[key]['pnl'] += pnl
            if pnl > 0: cat[key]['wins'] += 1
            else:       cat[key]['losses'] += 1

        # Filter activity last 24h
        filter_counts: dict = collections.defaultdict(int)
        for entry in self._filter_log:
            if entry['ts'] >= cutoff_24h:
                # Group LOW_VOLUME_NNN under LOW_VOLUME
                reason = entry['reason'] if not entry['reason'].startswith('LOW_VOLUME_') else 'LOW_VOLUME'
                filter_counts[reason] += 1

        week_pnl   = sum(t.get('pnl', 0) or 0 for t in week)
        new_today  = sum(1 for t in self._trades
                         if t.get('action') in ('ENTRY', 'ORDER_PLACED')
                         and t.get('timestamp', '')[:10] == now.strftime('%Y-%m-%d'))

        # Run market gap discovery — find liquid series we haven't analysed today
        market_gaps = await self._discover_market_gaps()

        self._last_nightly_report = {
            'generated_at': now.isoformat(),
            'period_days': 30,
            'total_settled_30d': len(recent),
            'settled_7d': len(week),
            'pnl_7d': round(week_pnl, 2),
            'new_bets_today': new_today,
            'open_positions': len(self._positions),
            'category_stats': {
                k: {'wins': v['wins'], 'losses': v['losses'],
                    'win_rate': round(v['wins'] / (v['wins'] + v['losses']), 3) if (v['wins'] + v['losses']) > 0 else 0,
                    'pnl': round(v['pnl'], 2)}
                for k, v in cat.items()
            },
            'filter_counts_24h': dict(sorted(filter_counts.items(), key=lambda x: -x[1])),
            'market_gaps': market_gaps,   # series with liquid markets not analysed today
        }

        # Log to Railway
        print(f"\n{'='*58}")
        print(f"[NIGHTLY REPORT]  {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*58}")
        print(f"  7-day P&L: ${week_pnl:+.2f}  |  30d settled: {len(recent)}  |  Open: {len(self._positions)}")
        print(f"  New bets today: {new_today}")
        print(f"  Category breakdown (30d):")
        for k, v in sorted(self._last_nightly_report['category_stats'].items(), key=lambda x: -x[1]['pnl']):
            print(f"    {k:<22} {v['wins']}W/{v['losses']}L  WR={v['win_rate']:.0%}  P&L=${v['pnl']:+.2f}")
        print(f"  Filter activity (24h): {self._last_nightly_report['filter_counts_24h']}")
        if market_gaps:
            print(f"  ⚠️  MARKET GAPS — liquid series NOT analysed today ({len(market_gaps)}):")
            for g in market_gaps[:10]:
                print(f"    [{g['series']}] bid={g['yes_bid']:.0%} oi={g['open_interest']:,} — {g['example_question'][:55]}")
        else:
            print(f"  ✅  No market gaps — all liquid series covered.")
        print(f"{'='*58}\n")

    async def _discover_market_gaps(self) -> list[dict]:
        """Scan ALL Kalshi series for liquid markets (10-90c YES, OI ≥ 200) that the bot
        has NOT analyzed in the last 24 hours.  Returns a list of gap records so the
        nightly report can flag them and operators can decide whether to act.

        This prevents the KXETH / KXBTCR→KXBTC class of blind spots:
        even if a new series appears with a different naming convention, this job
        will surface it automatically at midnight.
        """
        gaps: list[dict] = []
        # Regex for Kalshi ticker format: PREFIX-YYMONDD... e.g. KXBTC-26MAR06...
        _series_re = re.compile(r'^([A-Z0-9]+)-\d{2}[A-Z]{3}')
        try:
            # Markets analysed in the last 24 hours — series we already cover
            # Use strftime to produce a naive-compatible ISO string (no timezone suffix)
            # so string comparison against naive analysis timestamps works correctly.
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S')
            covered_series: set[str] = set()
            for a in self._analyses:
                if a.get('timestamp', '') >= cutoff:
                    mm = _series_re.match(a.get('market_id', '').upper())
                    if mm:
                        covered_series.add(mm.group(1))

            # Also include series we know about from the filter log (even if filtered out)
            for entry in self._filter_log:
                if entry.get('ts', '') >= cutoff:
                    mm = _series_re.match(entry.get('market_id', '').upper())
                    if mm:
                        covered_series.add(mm.group(1))

            # Walk through all currently-fetched PARSED markets and find liquid ones
            # whose series we haven't touched.  self._markets is a dict[id -> market_dict].
            seen_gap_series: set[str] = set()
            for mkt in self._markets.values():
                mid = mkt.get('id', '')
                mm = _series_re.match(mid.upper())
                if not mm:
                    continue
                series = mm.group(1)
                if series in covered_series or series in seen_gap_series:
                    continue

                yes_price = mkt.get('yes_price', 0) or 0
                oi        = mkt.get('open_interest', 0) or 0
                vol       = mkt.get('volume', 0) or oi

                # Liquid + eligible price range (yes_price is the midpoint from parse_kalshi_market)
                if 0.10 <= yes_price <= 0.90 and vol >= 200:
                    seen_gap_series.add(series)
                    gaps.append({
                        'series': series,
                        'example_ticker': mid,
                        'example_question': mkt.get('question', '')[:80],
                        'yes_bid': round(yes_price, 2),
                        'open_interest': oi,
                    })

        except Exception as e:
            print(f"[Market Gap Discovery] Error: {e}")

        return sorted(gaps, key=lambda x: -x['open_interest'])

    async def _nightly_strategy_loop(self) -> None:
        """Fire _run_nightly_report every day at midnight UTC."""
        # On startup, check whether the nightly reset was missed (e.g. the bot
        # crashed after midnight but before the reset ran).  If the daily_stats
        # are still keyed to a prior day, force an immediate reset so today's
        # kill-switch and drawdown counter start from today's baseline.
        try:
            _now_utc = datetime.now(timezone.utc)
            _today_str = _now_utc.strftime('%Y-%m-%d')
            _stats_date = getattr(self._risk_engine.daily_stats, 'date', None)
            if _stats_date is not None:
                _stats_date_str = (
                    _stats_date.strftime('%Y-%m-%d')
                    if hasattr(_stats_date, 'strftime')
                    else str(_stats_date)
                )
                if _stats_date_str < _today_str:
                    print(f"[Nightly] Missed reset detected (stats from {_stats_date_str}, today={_today_str}) — resetting now")
                    await self._risk_engine.reset_daily_stats()
                    self._save_state()
        except Exception as _e:
            print(f"[Nightly] Startup reset check failed (non-fatal): {_e}")

        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                await asyncio.sleep((next_midnight - now).total_seconds())
                # Reset daily risk stats so the kill-switch and drawdown counter
                # start fresh each day.  Must happen BEFORE the nightly report so
                # the report sees a clean slate for the new day.
                await self._risk_engine.reset_daily_stats()
                print("[Nightly] Daily risk stats reset for new day")
                # Persist the reset immediately so a crash between now and the next
                # organic _save_state() call doesn't restore a stale kill-switch from
                # yesterday's date into today's session.
                self._save_state()
                # Run the nightly report in its own try/except so a report failure
                # does not count as a full-loop error and defer the next midnight
                # sleep by 23 hours (which would skip tomorrow's daily reset).
                try:
                    await self._run_nightly_report()
                except Exception as _rpt_err:
                    print(f"[Nightly] Report failed (non-fatal): {_rpt_err}")
            except Exception as e:
                print(f"[NightlyLoop Error] {e}")
                await asyncio.sleep(3600)  # retry in 1h on error

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
            # Step 1: Fetch cash balance from Kalshi.
            # Kalshi's /portfolio/balance returns BOTH cash (balance) AND
            # portfolio_value (current market value of open positions at live prices).
            # Using portfolio_value directly is more accurate than recomputing from
            # cached prices, which may be a few minutes stale.
            _balance_ok = False
            _kalshi_portfolio_value = None  # market value direct from Kalshi API
            try:
                balance_result = await self._kalshi.get_balance()
                print(f"[Sync] Raw balance response: {balance_result}")
                balance_cents = balance_result.get('balance', 0)
                self._kalshi_cash = balance_cents / 100
                pv_cents = balance_result.get('portfolio_value', 0) or 0
                if pv_cents > 0:
                    _kalshi_portfolio_value = pv_cents / 100
                _balance_ok = True
            except Exception as e:
                print(f"[Sync] Could not fetch balance: {e}")
                import traceback
                traceback.print_exc()
                # Preserve last-known cash value on transient failure.

            # Step 2: Fetch open positions from Kalshi.
            result = await self._kalshi.get_positions()
            # Try all known field names (Kalshi uses market_positions; fall back to alternatives)
            kalshi_positions = (result.get('market_positions') or result.get('positions') or
                                result.get('portfolio_positions') or result.get('data') or [])
            print(f"[Sync] get_positions: {len(kalshi_positions)} market_positions returned")
            # Cache for performance endpoint so it doesn't need its own live API calls.
            # Always update (including to []) so the performance tab doesn't show stale
            # positions after they all close. The "no positions" early-return below only
            # fires when our own _positions dict is also empty — i.e. genuinely nothing open.
            # Normalize field names: Kalshi changed position → position_fp (string),
            # market_exposure → market_exposure_dollars (string, already in dollars not cents).
            # We normalize once here so all downstream code continues to use the old names.
            def _normalize_pos(kp: dict) -> dict:
                # Contracts: positive = long YES, negative = long NO
                if 'position_fp' in kp and 'position' not in kp:
                    kp = dict(kp)
                    kp['position'] = int(round(float(kp.get('position_fp') or 0)))
                # Market exposure: convert dollars → cents to match old field convention
                if 'market_exposure_dollars' in kp and 'market_exposure' not in kp:
                    kp = dict(kp) if isinstance(kp, dict) else kp
                    kp['market_exposure'] = round(float(kp.get('market_exposure_dollars') or 0) * 100)
                # Realized PnL: convert dollars → cents for consistency
                if 'realized_pnl_dollars' in kp and 'realized_pnl' not in kp:
                    kp['realized_pnl'] = round(float(kp.get('realized_pnl_dollars') or 0) * 100)
                return kp
            kalshi_positions = [_normalize_pos(kp) for kp in kalshi_positions]
            self._kalshi_positions_raw = kalshi_positions
            
            # Step 3: Compute portfolio value from positions data.
            # _kalshi_portfolio     = cost basis (sum of |market_exposure|) — capital at risk
            # _kalshi_positions_mv  = Kalshi's own live MtM (balance.portfolio_value)
            #                         This is exactly what Kalshi's UI shows as "Positions: $X"
            #                         and matches their internal pricing — preferred over our
            #                         cached prices which can lag by up to 20 min.
            # Fallback to contracts × cached_price when portfolio_value is unavailable
            # (e.g. a position was just filled and hasn't synced to balance yet).
            if _balance_ok and self._kalshi_cash is not None:
                _portfolio_exposure_cents = sum(
                    abs(kp.get('market_exposure', 0))
                    for kp in kalshi_positions
                    if kp.get('position', 0) != 0
                )
                self._kalshi_portfolio = _portfolio_exposure_cents / 100

                # Compute our own cached-price MtM as a fallback and for per-position P&L.
                _cache_market_value = 0.0
                _seen_tickers: set = set()
                for kp in kalshi_positions:
                    _contracts = kp.get('position', 0)
                    if _contracts == 0:
                        continue
                    _ticker = kp.get('ticker', '')
                    _seen_tickers.add(_ticker)
                    _side = 'yes' if _contracts > 0 else 'no'
                    _contracts = abs(_contracts)
                    _cached = self._markets.get(_ticker, {})
                    _price = _cached.get('yes_price') if _side == 'yes' else _cached.get('no_price')
                    if _price is None:
                        _exposure = abs(kp.get('market_exposure', 0)) / 100
                        _price = (_exposure / _contracts) if _contracts > 0 else 0.5
                    _cache_market_value += _contracts * _price

                # Include internal positions not yet in Kalshi sync (e.g. just filled)
                for _ipos in self._positions.values():
                    _ticker = _ipos.get('market_id', '')
                    if not _ticker or _ticker in _seen_tickers:
                        continue
                    _side = _ipos.get('side', 'YES').lower()
                    _contracts = _ipos.get('contracts', 0)
                    _cached = self._markets.get(_ticker, {})
                    _price = _cached.get('yes_price') if _side == 'yes' else _cached.get('no_price')
                    if _price is None:
                        _price = _ipos.get('current_price', _ipos.get('entry_price', 0.5))
                    _cache_market_value += _contracts * _price

                # Sanity-check Kalshi's portfolio_value before trusting it.
                # Kalshi's API is inconsistent: it sometimes returns face value (max payout
                # if all positions win), sometimes cost basis, sometimes real MtM. This causes
                # the reported total to swing wildly (e.g. $726 → $411 intraday) even when
                # positions haven't moved significantly, which falsely triggers the kill switch.
                #
                # Strategy: use Kalshi's value when it's plausible (within 80%-200% of our own
                # mid-price calculation). Outside that band, fall back to our own calculation.
                _pv_source = 'cache_fallback'
                if _kalshi_portfolio_value is not None and _cache_market_value > 1:
                    _ratio = _kalshi_portfolio_value / _cache_market_value
                    if 0.5 <= _ratio <= 2.5:
                        # Plausible — use Kalshi's value (it often has more positions than we track)
                        _positions_mv = _kalshi_portfolio_value
                        _pv_source = 'kalshi_api'
                    else:
                        # Out of band — Kalshi returned garbage (face value or near-zero)
                        _positions_mv = _cache_market_value
                        _pv_source = 'cache_sanity_override'
                        print(f"[Sync] WARNING: Kalshi portfolio_value=${_kalshi_portfolio_value:.2f} "
                              f"ratio={_ratio:.2f}x vs cache=${_cache_market_value:.2f} "
                              f"— using cache to prevent bad kill-switch trigger")
                elif _kalshi_portfolio_value is not None:
                    _positions_mv = _kalshi_portfolio_value
                    _pv_source = 'kalshi_api'
                else:
                    _positions_mv = _cache_market_value

                # Single source of truth for ALL dashboard value displays:
                #   _kalshi_positions_mv = live MtM (Kalshi API or cache fallback)
                #   _kalshi_total        = cash + MtM (every dashboard metric derives from this)
                self._kalshi_positions_mv = _positions_mv
                self._kalshi_total = self._kalshi_cash + self._kalshi_positions_mv
                _account_for_snapshot = self._kalshi_total
                print(f"[Sync] Kalshi: Cash=${self._kalshi_cash:.2f}, "
                      f"Positions(cost)=${self._kalshi_portfolio:.2f}, "
                      f"Positions(MtM/{_pv_source})=${self._kalshi_positions_mv:.2f}, "
                      f"CachedMtM=${_cache_market_value:.2f}, "
                      f"Total=${self._kalshi_total:.2f}")
                await self._save_daily_snapshot(_account_for_snapshot, self._kalshi_cash, _positions_mv)

                # Sync the risk engine with current market value (not cost basis) so the
                # kill-switch sees real unrealized losses, not just settled cash changes.
                #
                # Spike guard: same logic as _save_daily_snapshot — if the new total is
                # >35% below the previous sync value in a single cycle, it's almost certainly
                # a bad Kalshi API reading. Don't pass it to sync_bankroll or the kill switch
                # fires on garbage data. Log a warning and skip this sync's bankroll update.
                _prev_bankroll = self._risk_engine.daily_stats.current_bankroll or _account_for_snapshot
                _bankroll_ratio = _account_for_snapshot / _prev_bankroll if _prev_bankroll > 0 else 1.0
                if _bankroll_ratio < 0.65 and _prev_bankroll > 50:
                    print(f"[SyncGuard] SPIKE: skipping sync_bankroll — "
                          f"total=${_account_for_snapshot:.2f} vs prev=${_prev_bankroll:.2f} "
                          f"({_bankroll_ratio:.2f}x). Kalshi API bad reading, kill switch protected.")
                else:
                    _ks_was_triggered = self._risk_engine.daily_stats.kill_switch_triggered
                    self._risk_engine.sync_bankroll(_account_for_snapshot)
                    # Stamp kill-switch fire date exactly once on False→True transition.
                    if (not _ks_was_triggered
                            and self._risk_engine.daily_stats.kill_switch_triggered
                            and not self._kill_switch_fire_date):
                        self._kill_switch_fire_date = datetime.utcnow().strftime('%Y-%m-%d')
                        print(f"[KILL SWITCH] Triggered — fire date stamped: {self._kill_switch_fire_date}")

            if not kalshi_positions:
                # An empty response means either we genuinely have no open positions,
                # OR the API is returning a partial/bad response (token expiry, 200-with-null,
                # pagination issue). Clearing local positions on a bad response would be
                # catastrophic — we'd lose all tracking and charge phantom losses.
                # Safe policy: if we have local positions, treat an empty Kalshi response
                # as a data fetch failure and skip this sync cycle entirely.
                if self._positions:
                    print(f"[Sync] WARNING: Kalshi returned 0 positions but bot tracks "
                          f"{len(self._positions)} — skipping sync (likely API glitch)")
                    return
                print("[Sync] No positions on Kalshi or in bot — nothing to reconcile")
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
            
            # Check for phantom positions (in bot but not on Kalshi).
            # Guard: if Kalshi returned fewer positions than we track locally, do NOT charge
            # P&L — this almost certainly means a paginated/partial API response, not a
            # genuine settlement. Charging losses on a partial response would falsely trigger
            # the daily drawdown kill-switch. Only remove + charge when Kalshi plausibly
            # returned a complete picture (at least as many positions as we know about).
            _kalshi_count = len(kalshi_by_ticker)
            _local_count = len(self._positions)
            _safe_to_charge_pnl = _kalshi_count >= _local_count or _local_count == 0
            removed = 0
            if not _safe_to_charge_pnl:
                # Kalshi returned fewer positions than we track — almost certainly a
                # partial/paginated API response, not genuine settlement.  Removing
                # positions here would orphan real open positions (blacklisting them
                # for 48 h) and could falsely fire the kill-switch.  Skip the removal
                # pass; the next full sync will reconcile when it gets a complete response.
                print(f"[Sync] WARNING: Kalshi returned {_kalshi_count} positions vs "
                      f"{_local_count} local — skipping phantom-removal (partial response)")
            else:
                for pos_id in list(self._positions.keys()):
                    pos = self._positions[pos_id]
                    market_id = pos.get('market_id', '')
                    if market_id not in kalshi_by_ticker:
                        # Don't assume total loss. The position is gone from Kalshi because
                        # it settled (WIN or LOSS) or was externally closed. We don't know
                        # the outcome here, and sync_bankroll() (called just above) already
                        # reflects the real account value. Charging a fake negative
                        # pnl_estimate would push risk engine bankroll below Kalshi actuals,
                        # potentially triggering a false kill-switch for ~5 minutes.
                        # Rely solely on sync_bankroll for accuracy; skip record_trade_result.
                        print(f"[Sync] Position no longer on Kalshi — removing "
                              f"(charge skipped, sync_bankroll handles real P&L): "
                              f"{pos.get('question', '')[:40]}...")
                        self._positions.pop(pos_id)
                        self._recently_exited[market_id] = datetime.utcnow()
                        self._recently_exited_reason[market_id] = 'SYNC_REMOVED'
                        # Save state after EACH removal so a crash mid-loop doesn't
                        # cause double-removal on the next restart.
                        removed += 1
                        self._save_state()
            if removed:
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
                    # Don't re-add positions we intentionally wrote off or exited.
                    # TERMINAL_WRITEOFF removes the position locally but leaves it on
                    # Kalshi (no sell order placed). Without this guard, the sync loop
                    # would re-add it every 5 minutes, double-counting losses.
                    if ticker in self._recently_exited:
                        hours_since = (datetime.utcnow() - self._recently_exited[ticker]).total_seconds() / 3600
                        if hours_since < 48:  # 48h window covers all realistic settlement delays
                            print(f"[Sync] Skipping re-add of recently exited position: {ticker[:30]} ({hours_since:.1f}h ago)")
                            continue

                    # Race condition guard: the trading loop may have entered this market
                    # between the last sync and now (startup race). Don't create a duplicate.
                    _already_held_mids = {p.get('market_id') for p in self._positions.values()}
                    if ticker in _already_held_mids:
                        print(f"[Sync] Skipping duplicate sync — already holding {ticker[:30]} via analysis")
                        continue

                    # Add missing position
                    pos_id = f"pos_kalshi_{ticker[:20]}_{int(datetime.utcnow().timestamp())}"
                    cached_market = self._markets.get(ticker, {})
                    question = (
                        kp.get('title') or kp.get('market_title')
                        or cached_market.get('question') or cached_market.get('title')
                        or ticker
                    )
                    
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
                        'end_date': cached_market.get('end_date'),
                        'has_intel': False,
                        'news_count': 0,
                        'category': cached_market.get('category', 'unknown'),
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
                            
                            # Back-fill question if stored as empty OR as the raw ticker.
                            # (Old code wrote `or ticker` as a fallback, so ticker string
                            # ended up stored as a truthy "question" — treat it as missing.)
                            _stored_q = pos.get('question', '')
                            if not _stored_q or _stored_q == ticker:
                                _cache_m = self._markets.get(ticker, {})
                                _bfq = (
                                    kp.get('title') or kp.get('market_title')
                                    or _cache_m.get('question') or _cache_m.get('title')
                                )
                                # Only store if it's a real improvement (not also the ticker)
                                if _bfq and _bfq != ticker:
                                    pos['question'] = _bfq

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

            # Enrich any positions whose question text is still the raw ticker
            await self._enrich_position_questions()
                
        except Exception as e:
            print(f"[Sync] Error syncing with Kalshi: {e}")
            import traceback
            traceback.print_exc()

    async def _enrich_position_questions(self) -> None:
        """Back-fill human-readable question text for positions that were stored
        with question == market_id (a previous fallback wrote the raw ticker string).

        Calls the Kalshi /markets/{ticker} endpoint for each affected position so
        we get the actual market title.  Falls back to the ticker decoder if the API
        also returns an empty title.
        """
        from services.kalshi_client import parse_kalshi_market as _parse
        needs = [
            (pos_id, pos, pos['market_id'])
            for pos_id, pos in self._positions.items()
            if (q := pos.get('question', '')) == '' or q == pos.get('market_id', '')
        ]
        if not needs:
            return
        print(f"[Enrich] {len(needs)} position(s) need readable question text")
        changed = False
        for pos_id, pos, ticker in needs:
            # 1. Try the in-memory markets cache first
            cached = self._markets.get(ticker, {})
            cache_q = cached.get('question', '') or cached.get('title', '')
            if cache_q and cache_q != ticker:
                pos['question'] = cache_q
                print(f"[Enrich] (cache) {ticker[:30]} → {cache_q[:50]}")
                changed = True
                continue
            # 2. Call the single-market endpoint
            try:
                await asyncio.sleep(0.25)  # gentle rate-limiting
                raw = await self._kalshi.get_market(ticker)
                mkt = raw.get('market', raw)
                enriched = _parse(mkt)
                api_q = enriched.get('question', '')
                if api_q and api_q != ticker:
                    pos['question'] = api_q
                    print(f"[Enrich] (api)   {ticker[:30]} → {api_q[:50]}")
                    changed = True
                    continue
            except Exception as _e:
                print(f"[Enrich] API error for {ticker}: {_e}")
            # 3. Decoder fallback — always produces something readable
            decoded = self._decode_kalshi_ticker(ticker)
            pos['question'] = decoded
            print(f"[Enrich] (decode) {ticker[:30]} → {decoded[:50]}")
            changed = True
        if changed:
            self._save_state()

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
    
    def _log_filter(self, market_id: str, question: str, reason: str, price: float = 0.0) -> None:
        """Record a market that was filtered before reaching Claude analysis."""
        self._filter_log.append({
            'ts': datetime.utcnow().isoformat(),
            'market_id': market_id,
            'question': question[:80],
            'reason': reason,
            'price': round(price, 3),
        })

    def _get_stats(self) -> dict:
        """Calculate all stats from current state."""
        # Pending/resting orders (always from internal state)
        pending_at_risk = sum(o.get('size', 0) for o in self._pending_orders.values())
        
        # Realized P&L from closed trades
        realized_pnl = sum(t.get('pnl', 0) for t in self._trades if t.get('action') == 'EXIT')
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in self._positions.values())
        
        # Internal calculation for reference
        internal_positions = sum(p['size'] for p in self._positions.values())
        
        # Use ACTUAL Kalshi values if available (these are ground truth).
        # Compute positions value at CURRENT MARKET PRICES (not cost basis / market_exposure)
        # so that Total Value, Filled Positions, Account Value, and the chart all show
        # the same number: cash + what positions are actually worth right now.
        if self._kalshi_cash is not None and self._kalshi_total is not None:
            available = self._kalshi_cash
            # Use _kalshi_positions_mv (set by sync loop from Kalshi API) so that
            # Total Value, Account Value, and the graph all derive from the same number.
            positions_at_risk = round(self._kalshi_positions_mv or 0.0, 2)
            at_risk = positions_at_risk + pending_at_risk
            total_value = round(self._kalshi_total, 2)
            return_pct_actual = ((total_value - self.initial_bankroll) / self.initial_bankroll) * 100 if self.initial_bankroll else 0.0
            using_kalshi = True
        else:
            # Kalshi hasn't synced yet (first startup, or transient API error).
            # DO NOT use realized_pnl from the trade log — it has been historically
            # corrupted by wrong fill-price bugs and would produce values like $824
            # when the real account is $578.  Show $0 for monetary fields until
            # Kalshi returns a real answer; the dashboard shows "--" for unknown fields.
            positions_at_risk = internal_positions
            at_risk = positions_at_risk + pending_at_risk
            available = 0.0
            total_value = 0.0
            return_pct_actual = None
            using_kalshi = False
        
        # Log which values are being used (every 200th call — ~once per 10 min at dashboard poll rate)
        self._stats_debug_counter += 1
        if self._stats_debug_counter % 200 == 1:
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
            return_pct = ((total_value - self.initial_bankroll) / self.initial_bankroll) * 100 if self.initial_bankroll else 0.0

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
            'kalshi_synced': self._kalshi_cash is not None,  # Shows if using real Kalshi data
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
            'ws_connected': self._ws_client.is_connected,
            'ws_stats': self._ws_client.get_stats(),
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
            'min_days_to_resolution': self.min_days_to_resolution,
            'max_cluster_positions': int(os.getenv('MAX_CLUSTER_POSITIONS', '3')),
            'profit_lock_pct': float(os.getenv('PROFIT_LOCK_PCT', '0.50')),
            'kill_switch': self._risk_engine.daily_stats.kill_switch_triggered,
            'daily_drawdown': self._risk_engine.daily_stats.current_drawdown_pct,
            'exposure_ratio': (at_risk / self.initial_bankroll) if self.initial_bankroll else 0.0,
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
        # Fail fast if API credentials are missing in LIVE mode — without them every
        # Kalshi API call silently returns 401 and the bot runs indefinitely doing nothing.
        if not self.dry_run:
            if not self._kalshi.api_key_id or not self._kalshi._private_key:
                raise ValueError(
                    "LIVE mode requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY "
                    "(or KALSHI_PRIVATE_KEY_PATH) to be set. Bot cannot start."
                )

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
        self._app.router.add_get('/api/filters', self._handle_filters)
        self._app.router.add_get('/api/nightly', self._handle_nightly)
        self._app.router.add_get('/api/reset-killswitch', self._handle_reset_killswitch)
        self._app.router.add_get('/api/force-exit', self._handle_force_exit)
        self._app.router.add_get('/api/positions', self._handle_positions)
        self._app.router.add_get('/api/debug-positions', self._handle_debug_positions)
        self._app.router.add_get('/healthz', self._handle_healthz)
        
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
        print(f"Min Edge: {self.min_edge*100:.1f}%  |  Min Confidence: {self.min_confidence*100:.0f}%")
        print(f"Max Position: ${self.max_position_size:,.2f}  |  Kelly: {self.kelly_fraction*100:.0f}%")
        print(f"Horizon: 12h min (hard floor) — {self.max_days_to_resolution:.0f}d max  [BTC range bets: allowed]")
        print(f"Daily Loss Limit: 10%  |  Cluster Cap: {os.getenv('MAX_CLUSTER_POSITIONS','3')} (trump_speech=1)")
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
        
        # Sync pending orders FIRST — before canceling resting orders — so fills
        # that completed during downtime are detected and promoted to positions.
        # _cancel_all_resting_orders() clears _pending_orders afterward, making the
        # guard at the original location always False and _sync_pending_orders_on_startup
        # unreachable. Swapping the order fixes this.
        if not self.dry_run and self._pending_orders:
            print(f"[Startup] Checking {len(self._pending_orders)} pending orders from previous session...")
            await self._sync_pending_orders_on_startup()

        # CRITICAL: Cancel ALL stale resting orders on startup (LIVE mode only)
        if not self.dry_run:
            await self._cancel_all_resting_orders()
        
        # Sync positions with Kalshi to ensure accuracy (LIVE mode only)
        if not self.dry_run:
            await self._sync_positions_with_kalshi()
        
        # Run once on startup to backfill any settlements that happened during downtime
        if not self.dry_run:
            await self._reconcile_settlements_from_fills()

        # Register WebSocket callbacks before starting the run loop so we
        # never miss a fill or ticker that arrives right after connect.
        self._ws_client.add_ticker_callback(self._on_ws_ticker)
        self._ws_client.add_fill_callback(self._on_ws_fill)

        # Start background tasks
        asyncio.create_task(self._ws_client.run())           # Real-time WS feed
        asyncio.create_task(self._market_loop())
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._price_refresh_loop())
        asyncio.create_task(self._settlement_reconcile_loop())  # Periodic settlement P&L backfill
        asyncio.create_task(self._nightly_strategy_loop())     # Nightly performance report at midnight UTC
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
        await self._ws_client.stop()
        self._save_state()
        await self._crypto_edge.close()
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
                # Immediately sync WS subscriptions after markets are updated so the bot
                # starts receiving real-time ticker prices without waiting for the 5-minute
                # price_refresh_loop sleep cycle.  Include open-position markets too so they
                # keep getting WS price updates even if dropped from _monitored.
                _ws_tickers = set(self._monitored.keys())
                for _pos in self._positions.values():
                    _mid = _pos.get('market_id')
                    if _mid:
                        _ws_tickers.add(_mid)
                if _ws_tickers:
                    await self._ws_client.sync_subscriptions(list(_ws_tickers))
                    print(f"[WS] Subscribed to {len(_ws_tickers)} market tickers after market scan")
            except Exception as e:
                import traceback as _tb; print(f"[Market Error] {e}"); _tb.print_exc()
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
                        pages_fetched += 1  # Count against safety limit to prevent livelock
                        print(f"[Rate Limited] Pausing for 3 seconds (attempt {pages_fetched})...")
                        await asyncio.sleep(3)
                        continue
                    print(f"[Fetch error] {e}")
                    break
            
            print(f"[Fetch Complete] {len(all_markets)} markets from {pages_fetched} pages")
            # If the fetch returned zero markets (e.g., 50 consecutive 429s burned the safety
            # limit before any page was loaded), abort and keep the existing _markets intact.
            # Wiping _markets here would leave the trading loop with nothing to analyze for
            # up to 15 minutes and give no indication in the logs that something went wrong.
            if not all_markets:
                print(f"[Fetch] WARNING: 0 markets fetched — preserving existing {len(self._markets)} cached markets")
                return
            
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
                    if market['id']:
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
            elif any(x in ticker for x in ['KXBTCD', 'KXETHD']):
                category_bonus = -1.0  # Exact crypto above/below price: blocked by CRYPTO_EXACT_PRICE filter
            elif any(ticker.startswith(p) for p in ['KXBTC-', 'KXETH-', 'KXNASDAQ100-', 'KXDOGE-', 'KXSOL-']):
                category_bonus = 0.5   # Range bucket markets: our profitable strategy, prioritize
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
        
        # PRIORITIZE: Force-include top range series markets first (KXBTC-, KXETH-, KXNASDAQ100-
        # etc.) so they are never crowded out by high-OI macro markets.  Then fill remaining
        # slots with the best non-range short-term and ultra-short markets.
        _RANGE_PREFIXES = ('KXBTC-', 'KXETH-', 'KXNASDAQ100-', 'KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXBCH-')
        # Cap per-series (top 2 buckets each) so one liquid series cannot crowd out others.
        # A flat [:10] cap could let KXBTC- fill every slot, blocking DOGE/SOL/XRP/BCH/NASDAQ.
        _range_by_series: dict[str, list] = {p: [] for p in _RANGE_PREFIXES}
        # Include medium_term range markets too — monthly KXBTC-/KXETH-/etc. buckets
        # would otherwise be excluded from the guaranteed priority slots and silently
        # crowded out if 20+ high-OI non-range medium-term markets exist.
        for m in short_term + ultra_short + medium_term:
            mid = m.get('id', '').upper()
            for p in _RANGE_PREFIXES:
                if mid.startswith(p):
                    _range_by_series[p].append(m)
                    break
        _range_markets = []
        for p in _RANGE_PREFIXES:
            # Top 5 per series (was 3): gives the quant model more price levels to find edge.
            # BTC at $67,500 might have best edge in the 4th or 5th bucket — not just the
            # top-OI ones. Quality unchanged: same quant edge + confidence filters apply.
            _top5 = sorted(_range_by_series[p],
                           key=lambda x: x.get('open_interest', 0) or 0, reverse=True)[:5]
            _range_markets.extend(_top5)
        _range_ids = {m['id'] for m in _range_markets}
        # Exclude range markets from the other buckets to avoid analysing the same market twice
        _short_non_range = [m for m in short_term if m['id'] not in _range_ids]
        _ultra_non_range  = [m for m in ultra_short  if m['id'] not in _range_ids]
        # Ultra-short markets resolve within 24h — widest coverage here for maximum trade frequency.
        # Short-term raised 100 → 150: more political/economic 1-7d markets in the analysis pool.
        # Medium-term raised 50 → 75: more 8-45d markets; same strict quality filters apply.
        selected = _range_markets + _short_non_range[:300] + _ultra_non_range[:150] + medium_term[:150]
        
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

                    # Deprioritize exact-price crypto ladders and weather (no edge).
                    # Range series (KXBTC-, KXETH-, etc.) are intentionally NOT deprioritized
                    # — they have a quant edge and should be analyzed promptly each cycle.
                    _EXACT_PRICE_TICKERS = ('KXBTCD', 'KXETHD', 'KXBTC15M', 'KXETH15M',
                                             'KXDOGED', 'KXSOL15M', 'KXXRP15M', 'KXBCH15M')
                    _RANGE_SERIES = ('KXBTC-', 'KXETH-', 'KXNASDAQ100-', 'KXDOGE-',
                                     'KXSOL-', 'KXXRP-', 'KXBCH-')
                    _is_range_series = any(ticker.startswith(p) for p in _RANGE_SERIES)
                    _is_exact_price_ticker = any(ticker.startswith(p) for p in _EXACT_PRICE_TICKERS)
                    is_crypto_price = not _is_range_series and (
                        _is_exact_price_ticker
                        or any(x in q for x in ['bitcoin price', 'ethereum price',
                                                 'btc price', 'eth price'])
                    )
                    is_weather = any(x in q for x in ['snow', 'temperature', 'high temp',
                                                       'high of', 'low of', '°', 'degrees'])
                    depriority = 10.0 if (is_crypto_price or is_weather) else 0.0

                    time_score = min(hours, 168) / 168  # Normalize to 0-1 (cap at 1 week)
                    return depriority + time_score - (news_score * 0.3)
                
                markets_by_urgency = sorted(
                    self._monitored.values(),
                    key=market_priority
                )

                # Constants for the crypto filter — defined once per scan cycle, not per market
                _CRYPTO_EXACT_SUFFIXES = (
                    'KXBTCD', 'KXETHD', 'KXDOGED', 'KXXRPD', 'KXBCHD',
                    'KXBTC15M', 'KXETH15M', 'KXSOL15M', 'KXDOGE15M', 'KXXRP15M', 'KXBCH15M',
                )
                _CRYPTO_RANGE_PREFIXES = (
                    'KXBTC-', 'KXETH-', 'KXBCH-', 'KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXNASDAQ100-',
                )
                _EXACT_PRICE_RE = re.compile(
                    r'(above|below|hit|reach|exceed|surpass|touch)\s+\$[\d,.]+',
                )

                for market in markets_by_urgency:
                    market_id = market.get('id')
                    if not market_id:
                        continue
                    
                    # Skip markets where news has very low value or outcomes are predictable
                    question_lower = market.get('question', '').lower()
                    question_raw = market.get('question', '')
                    
                    # FILTER 1: Skip entertainment/noise/pop-culture markets.
                    # These have no reliable public information edge — AI has no predictive
                    # ability on movie scores, award winners, celebrity events, or social
                    # media activity.  Block both by question text and by ticker prefix.
                    _ent_question_patterns = [
                        # Social-media / speech noise
                        'mention', 'announcer', 'say during', 'tweet', 'post about',
                        # Movies / TV / streaming
                        'rotten tomatoes', 'box office', 'opening weekend', 'domestic gross',
                        'metacritic', 'imdb', 'certified fresh', 'audience score',
                        'streaming service', 'streaming platform', 'streaming subscriber',
                        'netflix', 'hulu', 'disney+', 'max series', 'peacock',
                        'season finale', 'series finale', 'pilot episode',
                        'tv show', 'tv series', 'movie rating', 'film rating',
                        # Awards ceremonies
                        'oscar', 'academy award', 'golden globe', 'emmy award', 'grammy',
                        'bafta', 'sag award', 'tony award', 'peoples choice', 'mtv award',
                        'billboard music', 'american music award', 'bet award',
                        # Celebrity / pop culture
                        'celebrity', 'kardashian', 'taylor swift', 'beyonce',
                        'will smith', 'kanye', 'elon musk tweet',  # specific celeb noise
                        # Game shows / reality TV
                        'survivor', 'bachelor', 'bachelorette', 'big brother',
                        'american idol', 'the voice', 'dancing with the stars',
                        'drag race', 'love island',
                        # Video game awards
                        'game of the year', 'the game awards',
                    ]
                    _ent_ticker_prefixes = [
                        'KXRT-',       # Rotten Tomatoes scores
                        'KXOSCARS', 'KXGRAMMYS', 'KXEMMYS', 'KXGLOBES',
                        'KXBOX-',      # Box office
                        'KXMOVIE', 'KXFILM', 'KXCELEB', 'KXPOP',
                        'KXREALITY', 'KXAWARD',
                    ]
                    _ent_question_hit = any(p in question_lower for p in _ent_question_patterns)
                    _ent_ticker_hit   = any(market_id.upper().startswith(p) for p in _ent_ticker_prefixes)
                    if _ent_question_hit or _ent_ticker_hit:
                        self._log_filter(market_id, question_raw, 'ENTERTAINMENT_NOISE', market.get('price', 0))
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
                        self._log_filter(market_id, question_raw, 'PLAYER_PROP', market.get('price', 0))
                        continue
                    
                    # FILTER 2b: Skip point spreads and totals - too volatile, no edge
                    # NOTE: 'over 1'-'over 9' removed — they false-positively matched
                    # macro questions like "over 10 million deportations" or "over 1% GDP growth".
                    # Half-point lines (.5) are purely sports and safe to block as substrings.
                    point_spread_patterns = [
                        'total points', 'wins by over', 'wins by under', 'point spread',
                        '.5 points', 'points scored',
                        # Half-point over/under lines — exclusively sports betting terminology
                        'over 0.5', 'over 1.5', 'over 2.5', 'over 3.5', 'over 4.5',
                        'over 5.5', 'over 6.5', 'over 7.5', 'over 8.5', 'over 9.5',
                        'under 0.5', 'under 1.5', 'under 2.5', 'under 3.5', 'under 4.5',
                        'under 5.5', 'under 6.5', 'under 7.5', 'under 8.5', 'under 9.5',
                    ]
                    if any(pattern in question_lower for pattern in point_spread_patterns):
                        self._log_filter(market_id, question_raw, 'POINT_SPREAD', market.get('price', 0))
                        continue
                    
                    # FILTER 2c: HARD BLOCK ALL SPORTS - historically poor (28% win rate, -$32 losses)
                    # Sports outcomes are too unpredictable, no sustainable edge
                    #
                    # Guard: economic and political markets are NEVER sports.
                    # Without this, 'nfl' matches "i**nfl**ation" and 'mma' matches "em**mma**nuel".
                    _econ_politics_guard = [
                        'inflation', 'cpi', 'gdp', 'unemployment', 'federal reserve',
                        'interest rate', 'treasury', 'yield', 'tariff', 'trade deficit',
                        'election', 'mayoral', 'parliamentary', 'congress', 'senate',
                        'president', 'prime minister', 'chancellor', 'minister',
                    ]
                    _is_econ_politics = any(k in question_lower for k in _econ_politics_guard)

                    # Short abbreviations (3-4 chars) require word-boundary matching to avoid
                    # false positives like 'nfl' in "inflation", 'mma' in "emmanuel", etc.
                    _sports_abbrevs = [
                        'nba', 'ncaa', 'wnba', 'nbl', 'nfl', 'mlb', 'nhl',
                        'mls', 'pga', 'lpga', 'atp', 'wta', 'ufc', 'mma',
                        'cfb', 'cfl', 'ipl', 'bbl',  # college football, CFL, cricket, Aus baseball
                    ]
                    # Longer patterns are safe as plain substrings
                    sports_patterns = [
                        'basketball', 'football', 'touchdown', 'quarterback',
                        'baseball', 'hockey',
                        'soccer', 'premier league', 'champions league',
                        'golf', 'genesis invitational', 'the masters', 'ryder cup',
                        'us open', 'wimbledon', 'french open', 'australian open',
                        'tennis',
                        # Rugby — was completely unblocked (Six Nations, World Cup, etc.)
                        'rugby', 'six nations', 'six-nations', 'rugby league', 'rugby union',
                        'rugby world cup', 'super rugby', 'premiership rugby',
                        'pro14', 'pro12', 'top 14', 'top14',
                        'six nations championship', 'tri nations', 'the rugby championship',
                        'bledisloe', 'calcutta cup', 'grand slam',
                        'boxing', 'fight night', 'knockout', 'k.o.',
                        'heavyweight', 'lightweight', 'middleweight', 'welterweight',
                        'nascar', 'olympics', 'world cup', 'world series',
                        'super bowl', 'championship game', 'playoff game',
                        'march madness', 'ncaa tournament', 'final four',
                        'stanley cup', 'nba finals', 'nba championship', 'nba title',
                        'wins the series', 'take the series', 'win the series',
                        'wins the match', 'win the game', 'beat the spread', 'defeats ',
                        'wins the game', 'wins the championship', 'wins the title',
                        'rebounds', 'assists', 'three-pointers', '3-pointers',
                        'pitcher', 'batter', 'innings', 'home run', 'strikeout',
                        'overtime period', 'penalty kick', 'penalty shootout',
                        'field goal', 'slam dunk', 'hat trick',
                        # Season win-total / over-under markets
                        'win at least', 'win more than', 'win fewer than',
                        'regular season wins', 'season record',
                        # Draft / combine markets
                        'nfl draft', 'nba draft', 'mlb draft', 'nhl draft',
                        'first overall pick', 'top pick',
                        # Motorsport / F1
                        'grand prix', 'formula 1', 'formula one', 'fastest lap',
                        'pole position', 'qualifying lap', 'f1 race', 'pitstop',
                    ]
                    _abbrev_hit = (
                        not _is_econ_politics
                        and any(
                            bool(re.search(r'\b' + abbrev + r'\b', question_lower))
                            for abbrev in _sports_abbrevs
                        )
                    )
                    _pattern_hit = (
                        not _is_econ_politics
                        and any(pattern in question_lower for pattern in sports_patterns)
                    )
                    if _abbrev_hit or _pattern_hit:
                        self._log_filter(market_id, question_raw, 'SPORTS_QUESTION', market.get('price', 0))
                        continue  # SKIP ALL SPORTS - no edge, high losses
                    
                    # Also check ticker patterns for sports
                    ticker_upper = market_id.upper()
                    sports_ticker_patterns = [
                        'KXNBA', 'KXNFL', 'KXMLB', 'KXNHL', 'KXNCAA', 'KXPGA', 'KXLPGA',
                        'KXUFC', 'KXBOX', 'KXTEN', 'KXSOC', 'KXWNH', 'KXWOH',
                        'KXDPWORLD', 'KXNBL', 'KXBRASIL', 'KXWCC', 'KXFIGHT',
                        'KXMATCH', 'KXBOUT', 'KXCHAMP',
                        # MLS soccer (were leaking — no edge without live sports stats)
                        'KXMLS',
                        # European/international soccer leagues (caught by question " vs " but add tickers too)
                        'KXLALIGA', 'KXSERIEA', 'KXBUNDES', 'KXLIGUE1', 'KXPREMIER',
                        'KXLIGAMX', 'KXCOPPA', 'KXFACUP', 'KXEUROPA', 'KXCHAMPIONSLEAGUE',
                        # Race sports / motorsport
                        'KXNASCARRACE', 'KXF1RACE', 'KXF1', 'KXGRANDPRIX', 'KXGP',
                        # Generic game/match tickers not caught above
                        'KXNBLGAME', 'KXWOMHOCKEY', 'KXLALIGA2GAME', 'KXSERIEAGAME',
                        'KXLIGAMXGAME', 'KXBUNDESGAME', 'KXLIGUE1GAME',
                        # Esports (were leaking through — no edge, waste AI credits)
                        'KXCOD',        # Call of Duty esports (KXCODGAME etc.)
                        'KXVALORANT', 'KXDOTA', 'KXCSGO', 'KXLOLGAME', 'KXROCKETLEAGUE',
                        'KXOVERWATCH', 'KXAPEXLEGENDS', 'KXFORTNITEGAME', 'KXESPORT',
                        # International basketball (FIBA, EuroLeague — were leaking)
                        'KXFIBA', 'KXEUROLEAGUE', 'KXEUROCUP', 'KXACBGAME', 'KXLNBGAME',
                        'KXBBLGAME', 'KXEKOGAME', 'KXBSGAME', 'KXLBAGAME',
                        # Season win totals / standings (NBA, NFL, MLB, NHL)
                        'KXNBAWIN', 'KXNFLWIN', 'KXMLBWIN', 'KXNHLWIN',
                        'KXNBAEAST', 'KXNBAWEST', 'KXNFCEAST', 'KXNFCWEST',
                        'KXNFCNORTH', 'KXNFCSOUTH', 'KXAFCEAST', 'KXAFCWEST',
                        'KXAFCNORTH', 'KXAFCSOUTH',
                        # Draft markets
                        'KXNFLDRAFT', 'KXNBADRAFT', 'KXMLBDRAFT', 'KXNHLDRAFT',
                        # Awards / MVP / Cy Young etc.
                        'KXNBAMVP', 'KXNFLMVP', 'KXMLBMVP', 'KXNHLMVP',
                        'KXNBAAWARDS', 'KXNFLAWARDS', 'KXMLBAWARDS', 'KXNHLAWARDS',
                        # Golf tournaments
                        'KXMASTERS', 'KXPGACHAMP', 'KXUSOPEN', 'KXTHEOPENGOLF',
                        'KXRYDERCUP', 'KXPRESIDENTSCUP',
                        # Tennis grand slams
                        'KXWIMBLEDON', 'KXUSOPENTEN', 'KXFRENCHOPEN', 'KXAUSTOPEN',
                        # College football / playoffs
                        'KXCFP', 'KXCFBGAME', 'KXNCAAFB',
                        # Combat sports catch-alls
                        'KXWWE', 'KXBELLATOR', 'KXPFL',
                        # Rugby (was completely unblocked)
                        'KXRUGBY', 'KXSIXNATIONS', 'KXRWC', 'KXRUGBYWC',
                        'KXSUPERRUGBY', 'KXPRUGBY', 'KXTOP14',
                    ]
                    if any(pattern in ticker_upper for pattern in sports_ticker_patterns):
                        self._log_filter(market_id, question_raw, 'SPORTS_TICKER', market.get('price', 0))
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
                        # Foreign macro/GDP — no edge, repeatedly rejected by AI (wastes API credits)
                        'brazilian gdp', 'brazil gdp', 'brazil q4', 'brazil q3',
                        'chinese gdp', 'china gdp', 'eurozone gdp', 'uk gdp',
                        'german gdp', 'japan gdp', 'india gdp',
                        'gdp (yoy)', 'gdp growth', 'gdp reading',
                        # Dogecoin exact-price/sentiment — AI has no edge here
                        # NOTE: 'doge price range' / 'dogecoin price range' are
                        # intentionally EXCLUDED so KXDOGE- range-series markets
                        # reach the quant gate in _analyze_market.
                        'doge trimmed mean', 'dogecoin trimmed mean',
                        'doge price', 'dogecoin price',
                        # Press secretary speech markets — priced efficiently, AI has no edge (<8%)
                        'will the white house press secretary say',
                        'will the press secretary say',
                        # Fed/Powell speech prediction markets — 2% edge every time, no predictive value
                        'will powell say', 'powell say trump', 'powell say "',
                        # Unemployment rate — analyzed 8x today at 25-30% confidence every time
                        # AI cannot predict monthly BLS data releases with any reliability
                        'unemployment rate in ',
                        # Foreign central banks — consistently LOW_CONFIDENCE, no news edge
                        "people's bank of china", 'pboc', 'bank of china cut',
                        'ecb cut', 'bank of england cut', 'bank of japan cut',
                        # Weather — bot has zero meteorological data; all snow/rain bets have lost money
                        'snow', 'rainfall', 'precipitation', 'blizzard', 'snowfall',
                        'rain in ', 'will it rain', 'inches of rain', 'inches of snow',
                        'storm in ', 'hurricane', 'tornado',
                        # ETH/crypto exact-time price bucket markets ("price at 5pm") —
                        # Claude gives 0.001-0.014 edge and 0.15 confidence every single time.
                        # These are NOT range markets (which win); they're time-locked price
                        # snapshots with no AI advantage. Blocking frees up Claude cycles for
                        # political/economics markets where real edge exists.
                        'ethereum price at ', 'eth price at ',
                        'bitcoin price at ', 'btc price at ',
                        'solana price at ', 'sol price at ',
                        'xrp price at ', 'ripple price at ',
                    ]
                    # Known range-series markets (KXBTC-, KXDOGE-, etc.) bypass the
                    # no_intel_patterns gate entirely — they go straight to the quant
                    # gate in _analyze_market which is the real signal for them.
                    _ticker_for_range_check = market_id.upper()
                    _is_range_series_skip = any(
                        _ticker_for_range_check.startswith(p)
                        for p in ('KXBTC-', 'KXETH-', 'KXBCH-', 'KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXNASDAQ100-')
                    )
                    if not _is_range_series_skip and any(p in question_lower for p in no_intel_patterns):
                        self._log_filter(market_id, question_raw, 'NO_INTEL_PATTERN', market.get('price', 0))
                        continue  # Skip: no reliable news intelligence for this market

                    # FILTER: Block crypto EXACT-PRICE series — these have no edge (29% WR, -$67).
                    # Strategy: INVERTED BLOCKLIST — only block tickers we know are exact-price ladders.
                    # This means any NEW range/bucket series Kalshi adds is automatically eligible
                    # (it passes EXTREME_PRICE + VOLUME gates below, then Claude decides).
                    _ticker_upper = market_id.upper()
                    _is_exact_price_series = any(_ticker_upper.startswith(p) for p in _CRYPTO_EXACT_SUFFIXES)
                    if _is_exact_price_series:
                        self._log_filter(market_id, question_raw, 'CRYPTO_EXACT_PRICE', market.get('price', 0))
                        continue  # Known exact-price ladder — no edge
                    # Also block one-off "will X hit $Y" phrasing that is NOT a known range series
                    _is_known_range_series = any(_ticker_upper.startswith(p) for p in _CRYPTO_RANGE_PREFIXES)
                    _is_crypto = any(x in question_lower for x in
                                     ['bitcoin', 'btc ', 'ethereum', ' eth ', 'solana', 'sol price',
                                      'xrp', 'ripple', 'crypto'])
                    _has_exact_price_phrasing = bool(_EXACT_PRICE_RE.search(question_lower))
                    if _is_crypto and _has_exact_price_phrasing and not _is_known_range_series:
                        self._log_filter(market_id, question_raw, 'CRYPTO_EXACT_PRICE', market.get('price', 0))
                        continue  # Crypto one-off threshold bet — no edge

                    # FILTER 3: Skip markets where probability is extreme (< 10% or > 90%)
                    # These have low expected value and high variance
                    market_price = market.get('price', 0.5)
                    yes_price = market.get('yes_price', market_price)
                    no_price = market.get('no_price', 1 - market_price)
                    
                    if yes_price < 0.10 or yes_price > 0.90:
                        self._log_filter(market_id, question_raw, 'EXTREME_PRICE', yes_price)
                        continue  # Skip extreme probability markets
                    
                    # FILTER 4: Skip markets with low volume (need real liquidity)
                    # Range series (BTC, ETH, NASDAQ etc) get a lower 200-volume floor since
                    # new daily/weekly contracts open with less liquidity initially.
                    volume = market.get('volume', 0) or market.get('open_interest', 0) or 0
                    _is_btc_range_market = ('bitcoin' in question_lower and 'range' in question_lower) or _ticker_upper.startswith('KXBTC-')
                    _is_range_market = _is_known_range_series or _is_btc_range_market
                    _min_vol = 200 if _is_range_market else 500
                    if volume < _min_vol:
                        self._log_filter(market_id, question_raw, f'LOW_VOLUME_{int(volume)}', market.get('price', 0))
                        continue  # Skip illiquid markets
                    
                    # FILTER 5: Skip all temperature/weather range markets (coin flips, no edge)
                    weather_patterns = [
                        'temperature', '° to ', '°-', 'degrees', 'high of', 'low of',
                        'high temp', 'low temp', 'max temp', 'min temp',
                        'weather in', 'climate', 'sunny', 'cloudy', 'foggy',
                    ]
                    if any(p in question_lower for p in weather_patterns):
                        self._log_filter(market_id, question_raw, 'WEATHER_RANGE', market.get('price', 0))
                        continue  # Weather ranges are unpredictable coin flips

                    # FILTER 6: Skip markets resolving too far out (high uncertainty, hard to lock gains)
                    hours_to_res_check = market.get('hours_to_resolution', 0)
                    max_hours = self.max_days_to_resolution * 24
                    if hours_to_res_check > max_hours:
                        self._log_filter(market_id, question_raw, 'TOO_FAR_OUT', market.get('price', 0))
                        continue  # Too far out — too much can change before resolution

                    # FILTER 6b: Skip markets resolving in under 12h — truly last-minute, price locked in
                    # NOTE: BTC range bets resolve in 1-2 days and are our best-performing category.
                    # Keep this floor at 12h only to block bets placed in the final hours before resolution.
                    min_hours = max(self.min_days_to_resolution * 24, 12) if self.min_days_to_resolution > 0 else 12
                    if hours_to_res_check > 0 and hours_to_res_check < min_hours:
                        self._log_filter(market_id, question_raw, 'TOO_CLOSE_TO_RESOLUTION', market.get('price', 0))
                        continue  # Resolves in under 12h — too close to resolution, skip
                        
                    if len(self._positions) + len(self._pending_orders) >= self._risk_limits.max_positions:
                        break

                    # PRE-FILTER: Skip Claude call if we know the market will be rejected anyway.
                    # These checks mirror _analyze_market but cost nothing vs an API call.

                    # Already holding this market (filled) or have a pending buy order for it
                    _held_ids = {p.get('market_id') for p in self._positions.values()}
                    _pending_ids = {o.get('market_id') for o in self._pending_orders.values()}
                    if market_id in _held_ids or market_id in _pending_ids:
                        continue

                    # Recently exited — still in cooldown
                    recent_exit = self._recently_exited.get(market_id)
                    if recent_exit:
                        exit_reason_stored = self._recently_exited_reason.get(market_id, '')
                        if 'PROFIT_LOCK' in exit_reason_stored or 'NEAR_SETTLEMENT' in exit_reason_stored:
                            cooldown_hours = 6.0
                        elif 'TERMINAL_WRITEOFF' in exit_reason_stored:
                            cooldown_hours = 24.0  # Dead markets must not be re-entered until settled
                        else:
                            # STOP_LOSS / SYNC_REMOVED / other forced exits:
                            # 24h cooldown prevents the bot from flipping sides on the same
                            # market the same day (e.g. exit gold NO at 2am, re-enter gold YES
                            # at noon).  The original 2h was too short — gold re-entry today
                            # deployed $211 on a coin-flip after a stop-loss exit.
                            cooldown_hours = 24.0
                        if (datetime.utcnow() - recent_exit).total_seconds() / 3600 < cooldown_hours:
                            continue

                    # Cluster cap reached — no point asking Claude
                    # Count both filled positions AND pending orders so that
                    # markets placed-but-not-yet-filled don't bypass the cap.
                    _pre_cluster_key = self._get_cluster_key(market.get('question', ''))
                    _pre_cluster_count = sum(
                        1 for p in self._positions.values()
                        if self._get_cluster_key(p.get('question', '')) == _pre_cluster_key
                    ) + sum(
                        1 for o in self._pending_orders.values()
                        if self._get_cluster_key(o.get('question', '')) == _pre_cluster_key
                    )
                    MAX_POSITIONS_PER_CLUSTER = int(os.getenv('MAX_CLUSTER_POSITIONS', '3'))
                    # Per-cluster overrides: tighter caps on low-edge speculative categories
                    _CLUSTER_CAPS = {
                        'trump_speech':        1,  # "Will Trump say X" — generic AI reasoning, bad P&L
                        'approval_rating':     1,  # 41.4% and 41.6% are the same bet — cap at 1
                        'weather_new_york':    1,  # Residual; new weather bets already blocked upstream
                        'weather_philadelphia': 1,
                        'weather_generic':     1,
                        'intl_central_banks':  1,  # No news edge on foreign CB markets
                        'macro_economics':     2,  # Allow 2: CPI and unemployment are different data
                                                   # sources on different release days — not the same bet
                    }
                    _cluster_cap = _CLUSTER_CAPS.get(_pre_cluster_key, MAX_POSITIONS_PER_CLUSTER)
                    if _pre_cluster_count >= _cluster_cap:
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

                    # Near-miss fast re-analysis: if last analysis scored high but just
                    # missed a threshold, re-check in 20 min instead of waiting the full
                    # cooldown. Edge can shift as prices move without triggering price_moved.
                    _last_a = next((a for a in self._analyses if a.get('market_id') == market_id), None)
                    if _last_a:
                        _la_edge = _last_a.get('edge', 0)
                        _la_conf = _last_a.get('confidence', 0)
                        _is_near_miss = (_la_edge >= 0.08 and _la_conf >= 0.65)
                        if _is_near_miss:
                            cooldown = min(cooldown, 1200)  # 20 min for near-misses

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

                    try:
                        await self._analyze_market(market)
                    except Exception as _ae:
                        # Catch any unhandled exception that escapes _analyze_market.
                        # We still update _last_analysis so the market is rate-limited
                        # and not re-analyzed every 30 seconds, burning Claude API budget.
                        import traceback as _tb
                        print(f"[Analyze Error] {market_id[:30]}: {_ae}")
                        _tb.print_exc()
                    self._last_analysis[market_id] = datetime.utcnow()
                    self._last_analysis_price[market_id] = current_price
                    await asyncio.sleep(1)  # 1s (was 2s) — halves cycle time, still safe for Brave/Claude rate limits
                    
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
                import traceback as _tb; print(f"[Trading Error] {e}"); _tb.print_exc()
            await asyncio.sleep(30)
    
    def _get_cluster_key(self, question: str) -> str:
        """Map a market question to a correlation cluster key.
        
        Markets in the same cluster move together (same underlying news driver).
        We cap how many positions we hold per cluster to avoid over-concentration.
        """
        q = question.lower()
        # Approval ratings — highly correlated across thresholds (41.4%, 41.6% are the same bet)
        if any(x in q for x in ['approval rating', 'approval rate', 'favorability', 'disapproval']):
            return 'approval_rating'
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

    # Economic data release market keywords — these markets are priced by professionals
    # with Bloomberg/Reuters consensus forecasts. Claude's priors are unreliable here.
    _ECON_DATA_KEYWORDS = [
        'cpi', 'consumer price index', 'core cpi', 'inflation',
        'pce', 'personal consumption', 'core pce',
        'ppi', 'producer price',
        'gdp', 'gross domestic product',
        'nfp', 'non-farm payroll', 'nonfarm payroll', 'jobs report', 'unemployment',
        'retail sales', 'durable goods', 'industrial production',
        'fomc', 'fed funds rate', 'federal reserve', 'interest rate decision',
        'central bank', 'rate decision', 'rate hike', 'rate cut',
        'jobs added', 'payrolls',
    ]
    # Regex to extract percentage numbers from news snippets (e.g. "2.5%", "0.3%")
    _PCT_RE = re.compile(r'(\d+\.?\d*)\s*%')
    # Regex to extract the threshold value from the market question
    _THRESHOLD_RE = re.compile(
        r'\b(above|below|over|under|higher than|lower than|more than|less than|'
        r'exceed|at least|at most)\s+'
        r'(\d+\.?\d*)\s*%',
        re.IGNORECASE,
    )

    def _extract_economic_consensus(
        self, question_full: str, news_items: list, side: str
    ) -> tuple[bool, str]:
        """For economic data release markets, extract consensus forecasts from news
        and check whether they contradict the bet.

        Returns (should_block: bool, reason: str).
        The bet is blocked when the news consensus lands on the OPPOSITE side of the
        threshold — meaning we're betting against what every published forecast says.
        """
        q_lower = question_full.lower()
        if not any(kw in q_lower for kw in self._ECON_DATA_KEYWORDS):
            return False, ''

        # Extract the threshold from the question
        m = self._THRESHOLD_RE.search(question_full)
        if not m:
            return False, ''
        direction = m.group(1).lower()   # "above", "below", "more than", etc.
        threshold = float(m.group(2))
        # "more than" / "exceed" / "over" all mean the threshold is a lower bound (bet_above=True)
        # "less than" / "under" / "below" mean the threshold is an upper bound (bet_above=False)
        bet_above = direction in ('above', 'over', 'higher than', 'exceed', 'at least', 'more than')

        # Collect percentage numbers from news snippets.
        # Critically: filter to the SAME order-of-magnitude as the threshold to avoid
        # YoY percentages (2.4%, 2.5%) polluting the median for MoM markets (0.7%).
        # Example failure: "Will CPI rise more than 0.7%?" threshold=0.7 (MoM).
        # News contains "2.4% YoY", "0.6% MoM forecast", "2.5% core".  Unfiltered
        # median lands at 2.4%, above 0.7%, so the guard thinks YES is correct — but
        # the actual forward-looking MoM consensus is 0.6% which is BELOW 0.7%.
        # Fix: for small thresholds (<2%), only count small percentages (<2%).
        _small_threshold = threshold < 2.0
        forecast_numbers = []
        for item in (news_items or []):
            text = f"{getattr(item, 'title', '')} {getattr(item, 'snippet', '')}"
            for n_str in self._PCT_RE.findall(text):
                n = float(n_str)
                if _small_threshold:
                    if 0.0 < n < 2.0:   # MoM / low-threshold range only
                        forecast_numbers.append(n)
                else:
                    if 0.0 < n < 20.0:  # normal YoY / rate range
                        forecast_numbers.append(n)

        if not forecast_numbers:
            return False, ''

        # Use median of extracted numbers as the consensus estimate
        forecast_numbers.sort()
        median_forecast = forecast_numbers[len(forecast_numbers) // 2]

        # Determine what consensus implies for resolution
        consensus_above_threshold = median_forecast > threshold
        # YES resolves if: question asks "above X" and consensus is above X,
        #                  OR question asks "below X" and consensus is below X
        question_resolves_yes = consensus_above_threshold if bet_above else not consensus_above_threshold

        # Block when consensus says our bet will LOSE:
        # - We bet YES but consensus says NO will win
        # - We bet NO but consensus says YES will win
        if side == 'YES' and not question_resolves_yes:
            return True, (
                f'CONSENSUS_CONTRADICTION_YES_ON_{"ABOVE" if bet_above else "BELOW"}'
                f'_{threshold}%_CONSENSUS_{median_forecast:.1f}%'
            )
        if side == 'NO' and question_resolves_yes:
            return True, (
                f'CONSENSUS_CONTRADICTION_NO_ON_{"ABOVE" if bet_above else "BELOW"}'
                f'_{threshold}%_CONSENSUS_{median_forecast:.1f}%'
            )
        return False, ''

    # Regex to find "Month YYYY" references in economic data market questions
    # e.g. "for the year ending in March 2026", "February 2026 CPI"
    _MONTH_REF_RE = re.compile(
        r'\b(january|february|march|april|may|june|july|august|september|'
        r'october|november|december)\s+(\d{4})\b',
        re.IGNORECASE,
    )
    _MONTH_NAMES: dict[str, int] = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }

    def _check_temporal_data_availability(
        self, question_full: str, close_time_str: str | None
    ) -> tuple[bool, str]:
        """Detect when an economic data market is asking about UNRELEASED data.

        Problem this solves: the CPI NO bet placed on Mar 11 2026 for
        "CPI above 2.8% for year ending March 2026".  The bot fetched news
        showing February 2026 CPI = 2.4% and inferred the answer — but March
        2026 CPI doesn't release until mid-April 2026.  Meanwhile oil surged to
        $110+ from the US-Iran war, making the March print likely 3%+.  The
        MARKET was already pricing YES at 69% when the bot bet NO at 31%.

        For these forward-looking economic data markets:
        - Economic data for month M is released around the 10th of month M+1.
        - If the referenced data period hasn't published yet, the fetched news
          only describes the PREVIOUS release — not what this market resolves on.
        - The market price in this case reflects live Bloomberg/Reuters consensus
          plus futures market signals.  We must trust it over the AI.

        Returns (is_forward_looking: bool, reason_tag: str).
        """
        q_lower = question_full.lower()
        if not any(kw in q_lower for kw in self._ECON_DATA_KEYWORDS):
            return False, ''

        now = datetime.utcnow()

        # Check 1: question explicitly mentions a specific Month YYYY
        month_match = self._MONTH_REF_RE.search(question_full)
        if month_match:
            month_num = self._MONTH_NAMES[month_match.group(1).lower()]
            year = int(month_match.group(2))
            # Economic data for month M releases around the 10th of month M+1
            release_month = month_num % 12 + 1
            release_year = year if month_num < 12 else year + 1
            try:
                estimated_release = datetime(release_year, release_month, 10)
                if estimated_release > now:
                    tag = (
                        f'FORWARD_ECON_{month_match.group(1).upper()}_{year}'
                        f'_RELEASES_{release_month:02d}/{release_year}'
                    )
                    return True, tag
            except ValueError:
                pass

        # Check 2: settlement date is >10 days away — data likely unreleased
        if close_time_str:
            try:
                close_dt = datetime.fromisoformat(close_time_str.replace('Z', ''))
                days_to_close = (close_dt - now).days
                if days_to_close > 10:
                    return True, f'FORWARD_ECON_CLOSES_IN_{days_to_close}D'
            except Exception:
                pass

        return False, ''

    # Real-world tradeable assets whose threshold markets require live price context.
    # Maps lowercase keyword found in the market question → Brave search query for current price.
    _COMMODITY_PRICE_QUERIES: dict[str, str] = {
        # ── Commodities ────────────────────────────────────────────────────────
        'gold':          'gold futures spot price today site:reuters.com OR site:bloomberg.com OR site:cnbc.com',
        'silver':        'silver futures spot price today site:reuters.com OR site:cnbc.com',
        'crude oil':     'WTI crude oil price today site:reuters.com OR site:cnbc.com',
        'wti':           'WTI crude oil price today site:reuters.com',
        'brent':         'Brent crude oil price today site:reuters.com',
        'natural gas':   'natural gas futures price today site:reuters.com OR site:cnbc.com',
        'wheat':         'wheat futures price today site:reuters.com',
        'corn':          'corn futures price today site:reuters.com',
        'copper':        'copper futures price today site:reuters.com',
        # ── Equity indices ─────────────────────────────────────────────────────
        's&p 500':       'S&P 500 index current price today site:cnbc.com OR site:reuters.com',
        'sp500':         'S&P 500 index current price today',
        'dow jones':     'Dow Jones DJIA current price today site:cnbc.com',
        'russell 2000':  'Russell 2000 index current price today',
        'nasdaq':        'NASDAQ 100 index current price today site:cnbc.com',
        'nikkei':        'Nikkei 225 index current price today',
        'ftse':          'FTSE 100 index current price today',
        # ── Inflation ──────────────────────────────────────────────────────────
        # Order matters: more specific phrases first so they match before substrings
        'core pce':      'latest core PCE inflation rate year-over-year 2026 site:bea.gov OR site:reuters.com OR site:cnbc.com',
        'core cpi':      'latest core CPI inflation rate ex food energy year-over-year 2026 site:bls.gov OR site:reuters.com',
        'consumer price index': 'latest CPI consumer price index year-over-year 2026 site:bls.gov OR site:reuters.com',
        'cpi':           'latest US CPI inflation rate year-over-year 2026 site:bls.gov OR site:reuters.com OR site:cnbc.com',
        'pce':           'latest PCE personal consumption expenditures inflation 2026 site:bea.gov OR site:reuters.com',
        'ppi':           'latest US PPI producer price index year-over-year 2026 site:bls.gov OR site:reuters.com',
        'inflation rate':'latest US inflation rate CPI year-over-year 2026 site:bls.gov OR site:reuters.com',
        # ── Employment ─────────────────────────────────────────────────────────
        'nonfarm payroll':   'latest US nonfarm payrolls jobs added 2026 site:bls.gov OR site:reuters.com OR site:cnbc.com',
        'nonfarm':           'latest US nonfarm payrolls jobs added 2026 bureau of labor statistics',
        'payroll':           'latest US nonfarm payrolls jobs report 2026 site:bls.gov OR site:reuters.com',
        'unemployment rate': 'latest US unemployment rate U-3 2026 site:bls.gov OR site:reuters.com OR site:cnbc.com',
        'unemployment':      'latest US unemployment rate 2026 bureau of labor statistics',
        'u-3':               'latest US U-3 unemployment rate 2026 site:bls.gov',
        'u-6':               'latest US U-6 underemployment rate 2026 site:bls.gov OR site:reuters.com',
        'jobless claims':    'latest weekly initial jobless claims 2026 site:dol.gov OR site:reuters.com OR site:cnbc.com',
        'initial claims':    'latest weekly initial jobless claims 2026 department of labor',
        'adp':               'latest ADP private payrolls employment report 2026 site:reuters.com OR site:cnbc.com',
        'labor force':       'latest US labor force participation rate 2026 site:bls.gov',
        # ── GDP & Growth ───────────────────────────────────────────────────────
        'gdp':               'latest US GDP growth rate quarterly annualized 2026 site:bea.gov OR site:reuters.com OR site:cnbc.com',
        'gross domestic':    'latest US GDP growth rate 2026 site:bea.gov OR site:reuters.com',
        # ── Federal Reserve & Rates ────────────────────────────────────────────
        'federal funds rate':'current federal funds rate FOMC 2026 site:federalreserve.gov OR site:reuters.com',
        'fed funds':         'current federal funds rate 2026 site:federalreserve.gov OR site:cnbc.com',
        'fomc':              'latest FOMC meeting decision federal funds rate 2026 site:federalreserve.gov OR site:reuters.com',
        '10-year treasury':  'current 10-year US treasury yield 2026 site:reuters.com OR site:cnbc.com',
        '10 year treasury':  'current 10-year US treasury yield 2026 site:reuters.com',
        '10-year yield':     'current 10-year US treasury yield 2026',
        '2-year treasury':   'current 2-year US treasury yield 2026 site:reuters.com',
        '30-year treasury':  'current 30-year US treasury yield 2026 site:reuters.com',
        # ── Housing ────────────────────────────────────────────────────────────
        'housing starts':    'latest US housing starts 2026 site:census.gov OR site:reuters.com OR site:cnbc.com',
        'existing home':     'latest existing home sales 2026 site:nar.realtor OR site:reuters.com',
        'case-shiller':      'latest Case-Shiller home price index 2026 site:reuters.com OR site:cnbc.com',
        # ── Consumer ───────────────────────────────────────────────────────────
        'retail sales':      'latest US retail sales month-over-month 2026 site:census.gov OR site:reuters.com OR site:cnbc.com',
        'consumer confidence':'latest consumer confidence index 2026 conference board site:reuters.com OR site:cnbc.com',
        'consumer sentiment':'latest University of Michigan consumer sentiment index 2026 site:reuters.com',
        # ── Manufacturing & Services ───────────────────────────────────────────
        'ism manufacturing': 'latest ISM manufacturing PMI index 2026 site:reuters.com OR site:cnbc.com',
        'ism services':      'latest ISM services PMI non-manufacturing index 2026 site:reuters.com',
        'pmi':               'latest US PMI purchasing managers index 2026 site:reuters.com OR site:cnbc.com',
        # ── Trade ──────────────────────────────────────────────────────────────
        'trade deficit':     'latest US trade deficit balance 2026 site:census.gov OR site:reuters.com',
        'trade balance':     'latest US trade balance 2026 site:census.gov OR site:reuters.com',
    }
    # Regex for price/rate threshold patterns in market questions.
    # Matches: "above $1,500"  "below 4.5%"  "exceed 150,000"  "above 50" (ISM PMI)
    # The number group makes % optional so bare integers (NFP, claims, PMI) are caught.
    # False positives are harmless: the keyword lookup is the real gate.
    _PRICE_THRESHOLD_RE = re.compile(
        r'\b(above|below|over|under|exceed|at least|at most|higher than|lower than)'
        r'\s+(\$[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?(?:\s*%)?)',
        re.IGNORECASE,
    )

    async def _fetch_live_data_context(self, question_full: str) -> str | None:
        """Fetch a targeted Brave search for the current value of any real-world
        asset price or economic indicator referenced in a threshold market question,
        so Claude is never forced to rely on stale training-data numbers.
        Returns a short context string to prepend to Claude's news summary, or None."""
        if not self._PRICE_THRESHOLD_RE.search(question_full):
            return None
        q_lower = question_full.lower()
        search_query = None
        matched_asset = None
        for keyword, brave_q in self._COMMODITY_PRICE_QUERIES.items():
            if keyword in q_lower:
                search_query = brave_q
                matched_asset = keyword
                break
        if not search_query:
            return None
        try:
            items = await self._intelligence.news_service._fetch_brave(search_query, max_results=3)
            if not items:
                return None
            snippets = '\n'.join(
                f'• [{it.source}] {it.title}: {it.snippet[:200]}'
                for it in items[:3]
            )
            ctx = (
                f"⚠️  LIVE DATA for '{matched_asset}' (fetched right now — use this number, "
                f"NOT your training-data value which may be months or years out of date):\n{snippets}"
            )
            print(f"[LiveData] Fetched live context for '{matched_asset}': {len(items)} results")
            return ctx
        except Exception as e:
            print(f"[LiveData] Context fetch failed (non-fatal): {e}")
            return None

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
        
        # Step 1: Quant gate for crypto range markets — runs BEFORE intel gathering.
        # Range markets (KXBTC-/KXETH-/etc.) are math problems, not news problems.
        # The log-normal model using Deribit IV + Binance spot is the primary signal.
        # If it sees no edge we skip early without spending any Brave/Claude API budget.
        _ticker_upper_q = market_id.upper()
        _RANGE_PREFIXES_Q = ('KXBTC-', 'KXETH-', 'KXBCH-', 'KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXNASDAQ100-')
        _is_crypto_range_q = any(_ticker_upper_q.startswith(p) for p in _RANGE_PREFIXES_Q)
        # BTC/ETH: invert signal (log-normal is inversely correlated — 13.7% win rate → bet opposite)
        _BTC_ETH_RANGE_PREFIXES = ('KXBTC-', 'KXETH-')
        _is_btc_eth_range = _is_crypto_range_q and any(_ticker_upper_q.startswith(p) for p in _BTC_ETH_RANGE_PREFIXES)
        # DOGE/XRP/SOL/BCH: block entirely — historical defaults only, no validated model
        _ALT_RANGE_PREFIXES = ('KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXBCH-')
        _is_alt_range = _is_crypto_range_q and any(_ticker_upper_q.startswith(p) for p in _ALT_RANGE_PREFIXES)
        _quant_result: CryptoEdgeResult = None
        _quant_override_prob: float = None
        _invert_crypto_range: bool = False  # True when we flip BTC/ETH signal to opposite side

        # Block DOGE/XRP/SOL/BCH range bets — no live vol model, historical defaults only
        if _is_alt_range:
            self._log_filter(market_id, market.get('question', ''), 'ALT_RANGE_NO_VALIDATED_MODEL', current_price)
            print(f"[QuantEdge] BLOCK {market_id[:30]} — alt range (DOGE/XRP/SOL/BCH): no validated quant model")
            return

        if _is_crypto_range_q:
            try:
                hours_to_res = market.get('hours_to_resolution') or 24.0
                _quant_result = await self._crypto_edge.evaluate_range_market(
                    ticker=market_id,
                    question=market.get('question', ''),
                    kalshi_price=current_price,
                    hours_to_expiry=float(hours_to_res),
                )
                if _quant_result:
                    print(
                        f"[QuantEdge] {market_id[:30]} | "
                        f"spot=${_quant_result.spot_price:,.0f} "
                        f"range=[{_quant_result.range_low:,.0f},{_quant_result.range_high:,.0f}] "
                        f"iv={_quant_result.implied_vol*100:.0f}% "
                        f"prob={_quant_result.quant_prob:.1%} "
                        f"kalshi={_quant_result.kalshi_price:.1%} "
                        f"edge={_quant_result.edge:+.1%} "
                        f"[{_quant_result.vol_source}]"
                    )
                    QUANT_MIN_EDGE = float(os.getenv('CRYPTO_QUANT_MIN_EDGE', str(self.min_edge)))
                    if _quant_result.edge < QUANT_MIN_EDGE:
                        reason = f"QUANT_NO_EDGE_{_quant_result.edge*100:+.0f}pct"
                        self._log_filter(market_id, market.get('question', ''), reason, current_price)
                        print(
                            f"[QuantEdge] SKIP — quant edge {_quant_result.edge:+.1%} < "
                            f"threshold {QUANT_MIN_EDGE:.1%}"
                        )
                        return
                    if _is_btc_eth_range:
                        # Invert: empirical data shows log-normal picks the wrong side (13.7% WR).
                        # When quant sees edge for YES (in-range), we bet NO (out-of-range).
                        # Use 1 - quant_prob so the blended signal points toward NO.
                        _quant_override_prob = 1.0 - _quant_result.quant_prob
                        _invert_crypto_range = True
                        print(
                            f"[QuantEdge] INVERTED (BTC/ETH): "
                            f"quant_yes={_quant_result.quant_prob:.1%} → using NO prob={_quant_override_prob:.1%}"
                        )
                    else:
                        _quant_override_prob = _quant_result.quant_prob
                else:
                    # evaluate_range_market returned None — either NASDAQ (different model) or
                    # a single-threshold ticker like KXBTC-26MAR1217-B68750 (no T bound to parse).
                    # For BTC/ETH: these are not range buckets so inversion doesn't apply.
                    # Block them rather than wasting Brave+Claude on unparseable markets.
                    if _is_btc_eth_range:
                        self._log_filter(market_id, market.get('question', ''), 'BTC_ETH_QUANT_UNPARSEABLE', current_price)
                        print(f"[QuantEdge] BLOCK {market_id[:30]} — BTC/ETH quant returned None (single-threshold, not range bucket)")
                        return
            except Exception as _qe:
                print(f"[QuantEdge] Evaluation error (non-fatal): {_qe}")
                _quant_result = None

        # Step 2: Gather market intelligence (news, domain data, overreaction detection).
        # Skipped for crypto range markets where quant succeeded — news is less
        # informative than the vol model for BTC/ETH/DOGE/SOL/XRP/BCH bucket bets.
        # BUT: for KXNASDAQ100- and any range market where quant returned None
        # (no supported asset), we DO want intelligence — Claude is flying blind
        # without it. The `_quant_result is None` fallback re-enables news gathering
        # for any range series that lacks a quant model.
        intel: MarketIntelligence = None
        if self._use_intelligence and (not _is_crypto_range_q or _quant_result is None):
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

        # Step 2c: For real-world asset / economic-indicator threshold markets, fetch live data
        # via Brave so Claude isn't forced to use stale training-data numbers.
        # Only fetched when Claude will actually be called (not on cache hits or inverted crypto).
        _commodity_price_ctx: str | None = None
        _full_question = market.get('question', '')
        # Pre-compute econ flag here so model routing (Haiku vs Sonnet) can use it
        # before the AI call. The same flag is re-used in post-signal risk guards below.
        _is_econ_market = any(kw in _full_question.lower() for kw in self._ECON_DATA_KEYWORDS)

        # Build overreaction info string for AI
        overreaction_info = None
        if intel and intel.overreaction_detected:
            overreaction_info = (
                f"ALERT: Market moved {intel.overreaction_magnitude:.1%} {intel.overreaction_direction} recently. "
                f"Price change 24h: {intel.recent_price_change:+.1%}. "
                f"This could be an overreaction - consider if the move is justified."
            )

        # Step 2b: Get historical performance for learning
        historical = self._get_historical_performance()

        # Step 3: Get AI signal — use probability cache if price is stable (saves API credits)
        # Cache is valid for up to 2h if price moved <1.5¢ since last call
        cached = self._prob_cache.get(market_id)
        use_cache = False
        cache_claude_prob = None  # raw pre-blend Claude prob stored in 5th cache slot
        if cached:
            # Cache tuple: (time, price, blended_prob, conf [, raw_claude_prob])
            # The 5th slot was added to allow correct re-blending on cache hits.
            cache_time, cache_price, cache_prob, cache_conf = cached[:4]
            cache_claude_prob = cached[4] if len(cached) > 4 else cache_prob
            price_drift = abs(current_price - cache_price)
            cache_age = (datetime.utcnow() - cache_time).total_seconds()
            hours_to_res = market.get('hours_to_resolution') or 9999
            max_cache_age = 1200 if hours_to_res <= 24 else 3600  # 20 min for ultra-short (was 5min), 1hr otherwise
            if price_drift < 0.015 and cache_age < max_cache_age:
                use_cache = True

        signal = None  # ensure signal is always defined before use

        # For inverted BTC/ETH range bets: skip Claude entirely.
        # We're betting based on an empirical contrarian signal, not AI probability.
        # Using a synthetic signal avoids wasting API credits and prevents Claude's
        # confidence score from incorrectly blocking an empirically-backed bet.
        if _invert_crypto_range and _quant_result:
            from logic.ai_signal import AISignalOutput
            signal = AISignalOutput(
                raw_prob=_quant_override_prob,  # 1 - quant_prob (inverted)
                confidence=0.85,               # fixed: empirical signal, not model-derived
                key_reasons=[f"INVERTED: log-normal shows {_quant_result.quant_prob:.0%} YES → bet NO (empirical 86% WR)"],
                disconfirming_evidence=["small sample (29 obs)", "market could re-price"],
                what_would_change_mind=["win rate drops below 60% over 50+ settled bets"],
                timeline_sensitivity="no: quant model only",
                failure_modes=["fat-tail move puts price in range despite model signal"],
                base_rate_considered=True,
                information_quality="medium",
            )
            class _InvertedResult:
                success = True
                error = None
                latency_ms = 0
            result = _InvertedResult()
            self._ai_successes += 1
            print(f"[QuantEdge] INVERTED synthetic signal — skipping Claude call")

        elif use_cache:
            # Reconstruct a minimal signal result from cache — no API call needed
            from logic.ai_signal import AISignalOutput
            _cached_raw_prob = cache_prob
            # For range markets: re-apply quant blend using fresh quant prob + stored
            # raw Claude prob (5th cache slot). This gives the correct 70/30 split —
            # previously cache_prob was the already-blended value, giving ~91% quant weight.
            if _quant_result and _quant_override_prob is not None:
                _cached_raw_prob = 0.70 * _quant_override_prob + 0.30 * cache_claude_prob
                print(
                    f"[QuantEdge] Cache re-blend: quant={_quant_override_prob:.1%} "
                    f"claude={cache_claude_prob:.1%} → {_cached_raw_prob:.1%}"
                )
            cached_signal = AISignalOutput(
                raw_prob=_cached_raw_prob,
                confidence=cache_conf,
                key_reasons=["(cached — price unchanged)"],
                disconfirming_evidence=["(cached)"],
                what_would_change_mind=["(cached)"],
                timeline_sensitivity="no: cached signal",
                failure_modes=["(cached)"],
                base_rate_considered=True,
                information_quality="medium",
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
                # Build resolution_date from market end_date
                _res_date = None
                _end_date_str = market.get('end_date')
                if _end_date_str:
                    try:
                        _res_date = datetime.fromisoformat(_end_date_str.replace('Z', '+00:00'))
                    except Exception:
                        pass

                # Fetch live asset/indicator data right before calling Claude.
                # Only runs here (not on cache hits or inverted crypto) to avoid wasted calls.
                if not _is_crypto_range_q:
                    _commodity_price_ctx = await self._fetch_live_data_context(_full_question)

                # Merge quant context + live data + news intel for Claude prompt
                _news_summary_for_claude = intel.news_summary if intel else None
                if _quant_result:
                    quant_block = _quant_result.context_summary
                    _news_summary_for_claude = (
                        quant_block + "\n\n" + _news_summary_for_claude
                        if _news_summary_for_claude
                        else quant_block
                    )
                # Prepend live data so Claude doesn't use stale training numbers
                if _commodity_price_ctx:
                    _news_summary_for_claude = (
                        _commodity_price_ctx + "\n\n" + _news_summary_for_claude
                        if _news_summary_for_claude
                        else _commodity_price_ctx
                    )

                # Model routing: Sonnet only for market types where its synthesis
                # capability genuinely matters. Haiku is used for everything else —
                # same JSON output quality and calibrated probabilities, ~10x cheaper.
                #
                # Sonnet earns its cost when:
                #   1. Economic data markets (CPI, GDP, unemployment, Fed) — complex
                #      numerical reasoning over multiple data series.
                #   2. Commodity/asset price markets (gold, oil, ETH) — live price
                #      context requires synthesis of multiple price sources.
                #
                # Haiku is sufficient for:
                #   • Political / event markets (Trump bets, elections, legislation) —
                #     binary reasoning, no numerical data synthesis required.
                #   • Crypto range markets — quant model provides the primary signal;
                #     Claude is only a secondary sanity-check.
                #   • All other markets (tech layoffs, housing, weather, etc.)
                #
                # Previously the routing was "Haiku when no news" — but Brave returns
                # 5 articles for virtually every market, so Haiku was never used.
                _needs_sonnet = (_is_econ_market or bool(_commodity_price_ctx)) and not _is_crypto_range_q
                result = await self._ai_generator.generate_signal(
                    market_question=market.get('question', ''),
                    current_price=current_price,
                    spread=market.get('spread', 0.02),
                    resolution_rules=market.get('rules', '') or market.get('description', ''),
                    resolution_date=_res_date,
                    volume_24h=market.get('volume_24h', 0),
                    category=market.get('category'),
                    # Intelligence data (includes quant context for range markets)
                    news_summary=_news_summary_for_claude,
                    domain_summary=intel.domain_summary if intel else None,
                    recent_price_change=intel.recent_price_change if intel else 0.0,
                    overreaction_info=overreaction_info,
                    # Historical performance for learning
                    historical_performance=historical.get('summary') if historical.get('total_trades', 0) > 0 else None,
                    # Haiku for political/event/crypto; Sonnet for econ data + commodity markets
                    use_haiku=not _needs_sonnet,
                )
            
                # Check if AI call succeeded
                if not result or not result.success or not result.signal:
                    error_msg = result.error if result else "No response"
                    print(f"[AI] FAILED: {error_msg}")
                    self._analyses.insert(0, {
                        'market_id': market_id,
                        'question': market.get('question') or market.get('title', ''),
                        'market_price': current_price,
                        'ai_probability': None,
                        'confidence': None,
                        'edge': 0,
                        'side': None,
                        'decision': 'NO_TRADE',
                        'reason': f"AI failed: {error_msg}",
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                    self._analyses = self._analyses[:200]
                    await self._broadcast_update()
                    return

                # Extract signal data from result
                signal = result.signal
                self._ai_successes += 1

                # Capture Claude's raw probability BEFORE blending so calibration
                # records the un-blended Claude estimate. This keeps the calibration
                # feedback loop clean: it learns Claude's bias, not the quant blend.
                _claude_raw_prob_for_calibration = signal.raw_prob

                # For crypto range markets: anchor Claude's raw_prob to quant model.
                # Blend: 70% quant (math-grounded) + 30% Claude (news signal).
                if _quant_result and _quant_override_prob is not None:
                    blended_prob = 0.70 * _quant_override_prob + 0.30 * signal.raw_prob
                    print(
                        f"[QuantEdge] Blending prob: quant={_quant_override_prob:.1%} "
                        f"claude={signal.raw_prob:.1%} → blended={blended_prob:.1%}"
                    )
                    signal = signal.__class__(
                        raw_prob=blended_prob,
                        confidence=signal.confidence,
                        key_reasons=signal.key_reasons,
                        disconfirming_evidence=signal.disconfirming_evidence,
                        what_would_change_mind=signal.what_would_change_mind,
                        timeline_sensitivity=signal.timeline_sensitivity,
                        failure_modes=signal.failure_modes,
                        base_rate_considered=signal.base_rate_considered,
                        information_quality=signal.information_quality,
                    )

                # Store blended prob + raw Claude prob in cache.
                # Slot 5 (raw Claude prob) is used on cache hits to re-blend correctly
                # with a fresh quant estimate without compounding the blend.
                self._prob_cache[market_id] = (
                    datetime.utcnow(), current_price,
                    signal.raw_prob,   # blended prob (used for edge/sizing)
                    signal.confidence,
                    _claude_raw_prob_for_calibration,  # raw Claude prob (used for re-blending)
                )
                # Evict oldest entries to keep memory bounded.
                # Each entry ≈200 bytes; 2000 entries ≈ 400KB maximum.
                _PROB_CACHE_MAX = 2000
                if len(self._prob_cache) > _PROB_CACHE_MAX:
                    _oldest_key = min(self._prob_cache, key=lambda k: self._prob_cache[k][0])
                    del self._prob_cache[_oldest_key]
                # Record CLAUDE's raw prob (pre-blend) for calibration learning —
                # the calibrator measures Claude's bias, not the quant formula's.
                if self._calibration and self._db_connected:
                    try:
                        await self._calibration.record_prediction(
                            market_id=market_id,
                            raw_prob=_claude_raw_prob_for_calibration,
                            market_price=current_price,
                            category=market.get('category'),
                            calibrated_prob=None,  # filled in after calibration below
                        )
                    except Exception:
                        pass

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
        
        # Step 4: Determine trade side and edge using FILL prices (what we actually pay).
        # Fill price = mid + bid-ask spread/2 + order slippage.
        # Using mid or even just ask overstates edge — orders fill significantly above ask.
        # Slippage model must match _enter_position exactly: max(2¢, 3% of mid).
        yes_price = market.get('yes_price', current_price)   # midpoint
        no_price = market.get('no_price', 1 - current_price)
        spread = market.get('spread', 0.02)
        half_spread = spread / 2

        # Expected fill prices — ask + realistic slippage (must match _enter_position)
        _slip = max(0.04, yes_price * 0.07)   # 7% or 4¢ min (matches order placement)
        yes_fill = min(0.98, yes_price + half_spread + _slip)
        no_fill  = min(0.98, no_price  + half_spread + _slip)

        if adjusted_prob > yes_fill:
            side = 'YES'
            edge = adjusted_prob - yes_fill   # true edge at actual fill price
            trade_prob = adjusted_prob
            trade_price = yes_fill            # Kelly sizes on what we actually pay
        elif (1 - adjusted_prob) > no_fill:
            side = 'NO'
            no_prob = 1 - adjusted_prob
            edge = no_prob - no_fill
            trade_prob = no_prob
            trade_price = no_fill
        else:
            # No edge even after accounting for fill price
            side = 'YES' if adjusted_prob > yes_price else 'NO'
            edge = 0.0   # forces LOW_EDGE filter below
            trade_prob = adjusted_prob if side == 'YES' else 1 - adjusted_prob
            trade_price = yes_fill if side == 'YES' else no_fill
        
        # BTC/ETH range inversion: force NO side with empirical win rate as trade prob.
        # The blended prob (inverted quant + Claude) may still land above 0.5 in some cases,
        # so we override explicitly here to guarantee the correct contrarian direction.
        # Empirical NO win rate from signal log: ~86% (29 samples). We use 0.80 conservatively.
        if _invert_crypto_range and _quant_result:
            _orig_side = side
            side = 'NO'
            trade_prob = 0.80   # conservative empirical estimate for NO win rate
            trade_price = no_fill
            edge = trade_prob - no_fill  # edge vs actual fill cost
            if edge < 0:
                # NO is too expensive at current price — skip this bet
                print(
                    f"[QuantEdge] INVERTED NO — no edge after fill cost "
                    f"(prob=0.80 < no_fill={no_fill:.2f}), skipping"
                )
                return
            print(
                f"[QuantEdge] INVERTED {_orig_side}→NO | "
                f"empirical_prob=80% no_fill={no_fill:.2f} edge={edge:+.1%}"
            )

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
            'question': market.get('question') or market.get('title', ''),
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
            'model': ('haiku' if not (_is_econ_market or bool(_commodity_price_ctx)) and not _is_crypto_range_q else 'sonnet'),
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
        self._analyses = self._analyses[:200]
        
        # Log signal for backtesting (track ALL signals regardless of trade decision)
        signal_entry = {
            'market_id': market_id,
            'question': question[:80],
            'edge': edge,
            'confidence': signal.confidence,
            'side': side,
            'market_price': current_price,
            'ai_probability': signal.raw_prob,
            'close_time': market.get('end_date') or market.get('close_time'),
            'timestamp': datetime.utcnow().isoformat(),
            'outcome': None,  # To be filled when market settles
            'outcome_checked': False,
        }
        self._signal_log.append(signal_entry)
        
        # Check if we should trade
        should_trade = True
        reasons = []

        # DAILY LOSS CIRCUIT BREAKER: Stop opening new positions if down >15% today.
        # Prefer _ks_reset_baseline (set when kill switch is manually reset) over
        # start_of_day_value — after a kill switch reset the original start_of_day is
        # stale/inflated and incorrectly shows a large loss that blocks all new bets.
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        today_snap = self._daily_snapshots.get(today_str, {})
        start_val = today_snap.get('_ks_reset_baseline') or today_snap.get('start_of_day_value')
        current_val = self._kalshi_total
        if start_val and current_val and start_val > 0:
            daily_loss_pct = (current_val - start_val) / start_val
            if daily_loss_pct < -0.15:
                should_trade = False
                reasons.append(f'DAILY_LOSS_LIMIT_{daily_loss_pct*100:.1f}pct')
                print(f"[Risk] Daily loss circuit breaker: {daily_loss_pct*100:.1f}% today — pausing new bets")

        # DOGE non-range bets: outcome data shows 0/2, -$22.19 — block entirely.
        # DOGE range markets (KXDOGE- series) go through the quant model; don't block those.
        _q_lower_cat = market.get('question', '').lower()
        _ticker_cat = market_id.upper()
        _is_doge_market = (
            not _is_crypto_range_q and
            (
                _ticker_cat.startswith('KXDOGE-') or
                'dogecoin' in _q_lower_cat or
                'doge price' in _q_lower_cat
            )
        )
        if _is_doge_market:
            should_trade = False
            reasons.append('DOGE_BLOCKED_0PCT_WIN_RATE')

        # Global edge floor — applies to all markets not already blocked above.
        # Inverted BTC/ETH range bets use an empirical 80% win rate, not a model probability.
        # Their edge guard is the `edge < 0` check in the inversion block above — we don't
        # apply the standard 12% min_edge here because the formula measures different things.
        if edge < self.min_edge and not _is_doge_market and not _invert_crypto_range:
            should_trade = False
            reasons.append('LOW_EDGE')

        # Confidence check — global minimum.
        # High-edge bypass: if edge is very strong (>= 2x min_edge), allow confidence
        # down to 0.65. A 20%+ edge with 65% confidence is statistically more valuable
        # than a 12% edge with 80% confidence.
        # Inverted bets use a fixed synthetic confidence (0.85) which always passes.
        _conf_floor = 0.65 if edge >= (self.min_edge * 2) else self.min_confidence
        # Persist effective threshold so the dashboard shows the right "need X%" label.
        if self._analyses:
            self._analyses[0]['conf_threshold_effective'] = round(_conf_floor, 4)
        if signal.confidence < _conf_floor and not _invert_crypto_range:
            should_trade = False
            reasons.append('LOW_CONFIDENCE')

        # PRICE SANITY GUARD: For real-world asset price-threshold markets (gold, oil,
        # indices, etc.), if the MARKET prices YES at >55% but our AI says <15% YES,
        # the AI almost certainly has stale training-data prices — not genuine edge.
        # Example that triggered this guard: gold at $5,200, market at 65% YES on
        # "above $5,159.99", Claude said 2% because its training data shows gold at ~$2,700.
        # We don't trust AI vs. real-time market consensus on commodity price levels.
        _full_q_lower = market.get('question', '').lower()
        _is_real_asset_thresh = (
            self._PRICE_THRESHOLD_RE.search(market.get('question', ''))
            and any(kw in _full_q_lower for kw in self._COMMODITY_PRICE_QUERIES)
        )
        if _is_real_asset_thresh and not _is_crypto_range_q:
            _yes_market_price = 1.0 - current_price if side == 'NO' else current_price
            if signal.raw_prob < 0.15 and _yes_market_price > 0.55:
                should_trade = False
                reasons.append(
                    f'PRICE_SANITY_FAIL_AI{signal.raw_prob:.0%}_MKT{_yes_market_price:.0%}'
                )
                print(
                    f"[PriceGuard] BLOCKED {market_id[:35]} — AI={signal.raw_prob:.0%} YES "
                    f"but market={_yes_market_price:.0%} YES. Likely stale AI price data."
                )

        # ECONOMIC DATA CONSENSUS CHECK: For CPI/GDP/jobs/PCE/PPI/FOMC markets,
        # extract the consensus forecast from news and block if it contradicts the bet.
        # Example that triggered this: news said "core CPI forecast 2.5%" but bot bet
        # YES on "above 2.6%" — that is betting directly against every published forecast.
        # Professional traders price these markets using Bloomberg consensus; Claude's
        # priors about inflation/growth are unreliable against hard published forecasts.
        if _is_econ_market and not _is_crypto_range_q and intel and intel.news_items:
            _econ_block, _econ_reason = self._extract_economic_consensus(
                market.get('question', ''), intel.news_items, side
            )
            if _econ_block:
                should_trade = False
                reasons.append(_econ_reason)
                print(f"[ConsensusGuard] BLOCKED {market_id[:35]} — {_econ_reason}")

        # ECONOMIC DATA MARKET PRICE SANITY: For economic data markets, professional
        # traders have Bloomberg/Reuters consensus terminals. When the market prices a
        # side at <20% but AI says >55%, trust the market — it has better information.
        # CPI at 13¢ YES with AI saying 79% YES is a clear sign AI is wrong, not edge.
        # current_price is the YES price; signal.raw_prob is the AI's YES probability.
        # Must adjust both for the bet side before comparing.
        _full_q_lower_econ = market.get('question', '').lower()
        _is_econ_market = any(kw in _full_q_lower_econ for kw in self._ECON_DATA_KEYWORDS)
        if _is_econ_market and not _is_crypto_range_q:
            _our_side_mkt_price = current_price if side == 'YES' else (1.0 - current_price)
            _our_side_ai_prob = signal.raw_prob if side == 'YES' else (1.0 - signal.raw_prob)
            if _our_side_mkt_price < 0.20 and _our_side_ai_prob > 0.55:
                should_trade = False
                reasons.append(
                    f'ECON_MARKET_SANITY_MKT{_our_side_mkt_price:.0%}_AI{_our_side_ai_prob:.0%}'
                )
                print(
                    f"[EconGuard] BLOCKED {market_id[:35]} — our side: market={_our_side_mkt_price:.0%} "
                    f"but AI={_our_side_ai_prob:.0%}. Market has Bloomberg consensus, AI doesn't."
                )

        # TEMPORAL DATA GUARD: For economic data markets where the RELEVANT DATA
        # HAS NOT BEEN RELEASED YET, the news we fetched only describes the PRIOR
        # month's release — not what this market will settle on.
        #
        # Example that triggered this: "CPI above 2.8% for year ending March 2026"
        # settled April 10 2026. Bot fetched news showing Feb CPI = 2.4% and bet NO.
        # But March CPI (released April) will be impacted by oil surging to $110+
        # from the US-Iran war — market correctly priced YES at 69%.  Bot entered
        # NO at 31¢ because it confused Feb data with the March question.
        #
        # Rule: if this is a forward-looking econ market AND the market prices
        # our side at <35% (market "strongly disagrees"), block the trade.
        # The 35% threshold (vs. 20% above) is tighter because for UNRELEASED
        # data the market is the only reliable forward-looking signal we have.
        if _is_econ_market and not _is_crypto_range_q:
            _is_fwd_econ, _fwd_econ_tag = self._check_temporal_data_availability(
                market.get('question', ''), market.get('end_date') or market.get('close_time')
            )
            if _is_fwd_econ:
                _fwd_our_side_mkt = current_price if side == 'YES' else (1.0 - current_price)
                _fwd_our_side_ai = signal.raw_prob if side == 'YES' else (1.0 - signal.raw_prob)
                if _fwd_our_side_mkt < 0.35 and _fwd_our_side_ai > 0.50:
                    should_trade = False
                    reasons.append(
                        f'TEMPORAL_DATA_GUARD_{_fwd_econ_tag}'
                        f'_MKT{_fwd_our_side_mkt:.0%}_AI{_fwd_our_side_ai:.0%}'
                    )
                    print(
                        f"[TemporalGuard] BLOCKED {market_id[:35]} — {_fwd_econ_tag} "
                        f"(unreleased data): market={_fwd_our_side_mkt:.0%} our side, "
                        f"AI={_fwd_our_side_ai:.0%}. News reflects PRIOR release only."
                    )

        # Require meaningful intelligence for non-crypto-range markets.
        # Crypto range markets use the quant model (vol/spot math) as their signal — they
        # intentionally skip the Brave news fetch for efficiency, so has_intel=False is
        # expected and correct for them. For every other market (political, economic, etc.)
        # there is no quant model — Claude is flying blind without news. Historical data:
        # no-intel non-range trades = 50% WR, -$27.34 net across 30 trades.
        if not _is_crypto_range_q:
            _has_intel_flag = analysis_record.get('has_intel', False)
            _news_ct = analysis_record.get('news_count', 0)
            if not _has_intel_flag:
                should_trade = False
                reasons.append('NO_INTEL_NON_RANGE')
            elif _news_ct < 2:
                should_trade = False
                reasons.append(f'WEAK_INTEL_{_news_ct}_articles')

        _held_market_ids = {p.get('market_id') for p in self._positions.values()}
        _pending_market_ids = {o.get('market_id') for o in self._pending_orders.values()}
        # Also deduplicate by question text — two different Kalshi market IDs can have
        # identical question text (e.g. multiple CPI series). Don't hold both.
        _held_questions = {p.get('question', '').strip().lower() for p in self._positions.values()}
        _this_question = market.get('question', '').strip().lower()
        if (market_id in _held_market_ids or market_id in _pending_market_ids
                or _this_question in _held_questions):
            should_trade = False
            reasons.append('ALREADY_IN_POSITION')
        
        # Don't re-enter markets we recently exited
        # Cooldown varies by exit reason:
        #  - PROFIT_LOCK / NEAR_SETTLEMENT: 6h (won, don't re-buy)
        #  - TERMINAL_WRITEOFF: 24h (position is terminal/dead — stay away until resolution)
        #  - STOP_LOSS / other forced exits: 24h — prevents same-day side-flip
        #    (gold NO stop-lossed at 2am → gold YES entered at noon = $211 coin-flip).
        #    The original 2h was too short for this protection to be meaningful.
        recent_exit = self._recently_exited.get(market_id)
        if recent_exit:
            exit_reason_stored = self._recently_exited_reason.get(market_id, '')
            if 'PROFIT_LOCK' in exit_reason_stored or 'NEAR_SETTLEMENT' in exit_reason_stored:
                cooldown_hours = 6.0
            else:
                # TERMINAL_WRITEOFF, STOP_LOSS, SYNC_REMOVED, SETTLED — all 24h
                cooldown_hours = 24.0
            hours_since_exit = (datetime.utcnow() - recent_exit).total_seconds() / 3600
            if hours_since_exit < cooldown_hours:
                should_trade = False
                reasons.append(f'RECENTLY_EXITED_{hours_since_exit:.1f}h_AGO_{exit_reason_stored[:20]}')
        
        if len(self._positions) + len(self._pending_orders) >= self._risk_limits.max_positions:
            should_trade = False
            reasons.append('MAX_POSITIONS')
        
        # CORRELATION CLUSTER CAP: Limit exposure to same news theme
        # Prevents piling into 10 DOGE-cut variants when 2-3 are sufficient.
        # Count pending orders too — avoids bypass when orders are placed but
        # not yet filled (the bug that caused 3 trump_speech bets to slip through).
        MAX_POSITIONS_PER_CLUSTER = int(os.getenv('MAX_CLUSTER_POSITIONS', '3'))
        cluster_key = self._get_cluster_key(market.get('question', ''))
        cluster_count = sum(
            1 for p in self._positions.values()
            if self._get_cluster_key(p.get('question', '')) == cluster_key
        ) + sum(
            1 for o in self._pending_orders.values()
            if self._get_cluster_key(o.get('question', '')) == cluster_key
        )
        # Per-cluster overrides: tighter caps on speculative/low-edge categories
        _EXEC_CLUSTER_CAPS = {
            'trump_speech':        1,
            'approval_rating':     1,  # 41.4% and 41.6% are the same bet — cap at 1
            'weather_new_york':    1,
            'weather_philadelphia': 1,
            'weather_generic':     1,
            'intl_central_banks':  1,
            'macro_economics':     2,  # Allow 2: CPI and unemployment are different release dates
        }
        _exec_cluster_cap = _EXEC_CLUSTER_CAPS.get(cluster_key, MAX_POSITIONS_PER_CLUSTER)
        if cluster_count >= _exec_cluster_cap:
            should_trade = False
            reasons.append(f'CLUSTER_CAP_{cluster_key[:20]}_{cluster_count}')
        
        # Log decision
        decision = 'TRADE' if should_trade else 'NO_TRADE'
        self._analyses[0]['decision'] = decision
        self._analyses[0]['reason'] = ', '.join(reasons) if reasons else 'CRITERIA_MET'

        # Mark the signal log entry with the trade decision so backtest analysis
        # can distinguish "analyzed + skipped" from "analyzed + actually bet on".
        # Uses market_id match as a safety check in case of concurrent edits.
        if self._signal_log and self._signal_log[-1].get('market_id') == market_id:
            self._signal_log[-1]['traded'] = should_trade
            self._signal_log[-1]['skip_reason'] = ', '.join(reasons) if not should_trade else None
        
        if should_trade:
            # ENTRY LOCK: prevents concurrent _analyze_market coroutines from both passing
            # ALREADY_IN_POSITION and placing duplicate orders on the same market.
            # Pattern: acquire lock → re-check → plant placeholder → release lock.
            # The placeholder immediately blocks any other task's ALREADY_IN_POSITION check
            # (it checks _pending_orders by market_id), so the lock only needs to cover
            # the tiny re-check + placeholder-insert window, not the full async API call.
            _placeholder_key = f'_reserving_{market_id}'
            async with self._entry_lock:
                _held_now = {p.get('market_id') for p in self._positions.values()}
                _pending_now = {o.get('market_id') for o in self._pending_orders.values()}
                _questions_now = {p.get('question', '').strip().lower() for p in self._positions.values()}
                if (market_id in _held_now or market_id in _pending_now
                        or _this_question in _questions_now):
                    print(f"[EntryLock] {market_id[:35]} — skipped, another task already entered")
                    return
                # Plant placeholder so any concurrent task sees this market as reserved
                self._pending_orders[_placeholder_key] = {'market_id': market_id, 'size': 0, 'placeholder': True}
            # Lock released — placeholder now blocks other tasks without holding lock during API call

            # Sync total + per-market exposure so both global and per-market limits fire.
            # Include pending orders in both exposure and count so the max_positions
            # check fires correctly before pending orders fill (not just after).
            pending_exposure = sum(o.get('size', 0) for o in self._pending_orders.values())
            real_exposure = sum(p.get('size', 0) for p in self._positions.values()) + pending_exposure
            total_open_count = len(self._positions) + len(self._pending_orders)
            self._risk_engine.sync_open_exposure(real_exposure, position_count=total_open_count)

            market_exposures: dict = {}
            for p in self._positions.values():
                mid = p.get('market_id', '')
                market_exposures[mid] = market_exposures.get(mid, 0.0) + p.get('size', 0.0)
            # Include pending orders so the per-market cap fires for unfilled buys too.
            # Without this, a pending $50 buy on KXBTC-25MAR shows $0 exposure for that
            # market, allowing a second overlapping position on a correlated contract.
            for o in self._pending_orders.values():
                mid = o.get('market_id', '')
                market_exposures[mid] = market_exposures.get(mid, 0.0) + o.get('size', 0.0)
            self._risk_engine.sync_market_exposure(market_exposures)

            # Calculate position size (async with proper params)
            print(f"[Debug] Calculating size: prob={trade_prob:.2f}, price={trade_price:.2f}, edge={edge:.2f}, conf={signal.confidence:.2f}")
            position_size = await self._risk_engine.calculate_position_size(
                adjusted_prob=trade_prob,
                market_price=trade_price,
                edge=edge,
                confidence=signal.confidence,
                market_id=market_id,
            )

            # News-backed bets: only cap sizing on very low confidence (< 0.70) signals.
            # The previous 0.80 threshold was hitting ~90% of bets and negating the Kelly sizing.
            # New bets show 100% win rate with intel — no reason to penalise news-backed bets broadly.
            has_intel = intel is not None
            if has_intel and signal.confidence < 0.70:
                pre_cap = position_size
                position_size = position_size * 0.6
                print(f"[Intel Size Cap] ${pre_cap:.2f} → ${position_size:.2f} (news bet, conf<0.70)")

            # Liquidity cap: don't exceed X% of open interest
            open_interest = market.get('open_interest', 0) or 0
            if open_interest > 0 and self.max_oi_pct > 0:
                # Each contract costs ~trade_price, so max_contracts = OI * max_oi_pct
                max_contracts_by_liquidity = int(open_interest * self.max_oi_pct)
                max_size_by_liquidity = max_contracts_by_liquidity * trade_price
                if position_size > max_size_by_liquidity:
                    print(f"[Liquidity Cap] ${position_size:.2f} → ${max_size_by_liquidity:.2f} (OI={open_interest}, max {self.max_oi_pct*100:.0f}%)")
                    position_size = max_size_by_liquidity

            # BANKROLL % CAP: No single bet can exceed MAX_BET_PCT of the current live
            # account value.  Default 15%.  Uses _kalshi_total (synced from Kalshi API
            # every ~5 minutes) so it tracks real drawdowns, not just the initial bankroll.
            # Fallback to max_position_size if live value not yet synced.
            # Root cause: gold YES entered at $211 (26% of $797 portfolio) on a coin-flip
            # after the gold NO stop-lossed — one bad move could have wiped the account.
            MAX_BET_PCT = float(os.getenv('MAX_BET_PCT', '0.15'))  # 15% default
            _live_bankroll = self._kalshi_total if self._kalshi_total and self._kalshi_total > 10 else self.max_position_size / MAX_BET_PCT
            _bankroll_cap = _live_bankroll * MAX_BET_PCT
            if position_size > _bankroll_cap:
                print(
                    f"[Bankroll Cap] ${position_size:.2f} → ${_bankroll_cap:.2f} "
                    f"({MAX_BET_PCT*100:.0f}% of ${_live_bankroll:.0f} live bankroll)"
                )
                position_size = _bankroll_cap

            # CRYPTO RANGE CAP: BTC/ETH/SOL/XRP range bucket markets are high-variance.
            # Historical data: a single bad range bet ($14.88) wiped out 6 consecutive
            # wins (+$10.25). Cap is kept at $8 regardless of bankroll size — range bets
            # are inherently uncertain (the quant model can be wrong) and should never
            # be large enough to wipe a full day of wins. Overridable via env var.
            _market_id_upper = market_id.upper()
            _RANGE_PREFIXES = ('KXBTC-', 'KXETH-', 'KXNASDAQ100-', 'KXDOGE-', 'KXSOL-', 'KXXRP-', 'KXBCH-')
            _is_range_market = any(_market_id_upper.startswith(p) for p in _RANGE_PREFIXES)
            _q_lower_range = market.get('question', '').lower()
            _is_range_question = any(x in _q_lower_range for x in
                                     ['bitcoin price range', 'ethereum price range', 'btc price range',
                                      'eth price range', 'solana price range', 'nasdaq price range'])
            CRYPTO_RANGE_MAX = float(os.getenv('CRYPTO_RANGE_MAX_SIZE', '16.0'))
            if (_is_range_market or _is_range_question) and position_size > CRYPTO_RANGE_MAX:
                print(f"[Crypto Range Cap] ${position_size:.2f} → ${CRYPTO_RANGE_MAX:.2f} (range market size limit)")
                position_size = CRYPTO_RANGE_MAX

            print(f"[Debug] Position size: ${position_size:.2f}")

            if position_size > 0:
                try:
                    await self._enter_position(market, side, position_size, adjusted_prob, edge, signal.confidence)
                    print(f"[AI] {question}... | ✓ TRADE | Edge: +{edge*100:.1f}% | Conf: {signal.confidence*100:.0f}%")
                finally:
                    # Remove placeholder — real order (if placed) is now in _pending_orders under its order_id
                    self._pending_orders.pop(_placeholder_key, None)
            else:
                self._pending_orders.pop(_placeholder_key, None)
                print(f"[AI] {question}... | ✗ SKIP | Size: $0 (edge={edge:.2%}, conf={signal.confidence:.2%})")
        else:
            print(f"[AI] {question}... | ✗ SKIP | {', '.join(reasons)}")
        
        await self._broadcast_update()
    
    async def _enter_position(self, market: dict, side: str, size: float, prob: float, edge: float, confidence: float):
        """Enter a new position."""
        market_id = market['id']
        pos_id = f"pos_{int(datetime.utcnow().timestamp()*1000)}"
        
        # STRICT LIMITS - prevent runaway orders
        # Contract cap: floor at 200; effectively uncapped since Kelly dollar limit fires first.
        MAX_CONTRACTS_PER_ORDER = max(200, int(self.max_position_size / 0.10))
        MIN_PRICE_CENTS = 10   # Floor at 10¢ — quant-validated range bets at 10-14¢ have excellent
                               # win/loss math (25:1 ratio) and the $8 crypto range cap limits downside
        # Cap at 50¢ for YES bets: at 51% historical hit rate, break-even requires entry < 51¢.
        # At 50¢: win=$0.50, lose=$0.50 → break-even at 50%. At 40¢: win=$0.60, lose=$0.40 →
        # break-even at 40% — the lower the entry price, the better the win/loss ratio.
        # NOTE: This cap applies to YES bets only.  For NO bets the edge gate in _analyze_market
        # (edge = trade_prob - no_fill) already enforces EV > 0 — no separate price cap needed.
        # The inversion strategy bets NO at 60–80¢ (empirical 80% win rate), so blocking those
        # by a 50¢ ceiling was silently killing every inversion trade.
        MAX_PRICE_CENTS = 50
        
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
        # 50¢ cap only applies to YES bets — for NO bets the edge gate already validates EV > 0.
        if side.upper() == 'YES' and price_cents > MAX_PRICE_CENTS:
            print(f"[Order] SKIPPED: YES price {price_cents}¢ too high (max {MAX_PRICE_CENTS}¢) - not enough upside")
            return
        
        # Calculate contracts with STRICT LIMIT
        contracts = int(size / entry_price) if entry_price > 0 else 0
        if contracts > MAX_CONTRACTS_PER_ORDER:
            print(f"[Order] Capping contracts: {contracts} → {MAX_CONTRACTS_PER_ORDER} (max per order)")
            contracts = MAX_CONTRACTS_PER_ORDER
        
        # actual_size and contracts are recalculated at order price inside the live block below.
        # These midpoint-based values are only used as the dry-run fallback.
        actual_size = contracts * entry_price

        # In LIVE mode, actually place the order on Kalshi
        order_id = None
        if not self.dry_run:
            try:
                if contracts < 1:
                    print(f"[Order] Size ${size:.2f} too small for 1 contract at {price_cents}¢")
                    return

                # Slippage: 4¢ minimum or 7% of mid price — must match the fill-price
                # estimate used in the edge gate (_analyze_market step 4).
                # At 7%/4¢, a 50¢ contract orders at 54¢ (~3¢ above a typical 2¢-spread ask)
                # which fills immediately. At 3% the order sat below the ask as a resting
                # limit, waited 10 min, then cancelled — causing near-zero fill rate.
                slippage_cents = max(4, int(price_cents * 0.07))
                order_price_cents = min(price_cents + slippage_cents, 95)

                # Recompute contracts at the actual order price (not midpoint) so the
                # Kelly dollar budget is not exceeded by the slippage premium.
                order_price = order_price_cents / 100
                contracts = int(size / order_price) if order_price > 0 else 0
                if contracts > MAX_CONTRACTS_PER_ORDER:
                    contracts = MAX_CONTRACTS_PER_ORDER
                actual_size = contracts * order_price
                if contracts < 1:
                    print(f"[Order] Size ${size:.2f} too small for 1 contract at {order_price_cents}¢")
                    return

                print(f"[Order] Placing: {contracts} {side} @ {order_price_cents}¢ (market: {price_cents}¢, slippage: {slippage_cents}¢)")

                # Enforce 500ms minimum between order placements (workspace rule)
                _elapsed = (datetime.utcnow() - self._last_order_time).total_seconds()
                if _elapsed < 0.5:
                    await asyncio.sleep(0.5 - _elapsed)

                result = await self._kalshi.place_order(
                    ticker=market_id,
                    side=side.lower(),  # 'yes' or 'no'
                    count=contracts,
                    price=order_price_cents,
                    order_type='limit',
                )
                order_id = result.get('order', {}).get('order_id')
                self._last_order_time = datetime.utcnow()
                if not order_id:
                    print(f"[Order Error] Kalshi returned no order_id — aborting")
                    return  # Can't track an order with no ID
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
            'question': market.get('question') or market.get('title', ''),
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
                'question': market.get('question') or market.get('title', ''),
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
                'question': market.get('question') or market.get('title', ''),
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
    
    async def _check_position_news(self, pos_id: str, pos: dict) -> None:
        """Fetch and cache recent news for an open position (at most once per 2 hours).

        Stores up to 3 headlines in pos['latest_news'] so they appear in the dashboard.
        Logs a brief summary to Railway logs.
        BTC range positions are skipped — the live price monitor already covers them.
        """
        question = pos.get('question', '')
        if not question:
            return

        # BTC range already tracked by the live BTC price monitor — skip news for those only
        q_lower = question.lower()
        _pos_ticker = pos.get('market_id', '').upper()
        _is_btc_range = (
            ('bitcoin' in q_lower and 'range' in q_lower)
            or _pos_ticker.startswith('KXBTC-')
        )
        if _is_btc_range:
            return

        if time.time() - self._news_check_times.get(pos_id, 0) < 7200:
            return  # checked within the last 2 hours

        try:
            news_items = await self._intelligence.news_service.fetch_news(question, max_results=3)
            self._news_check_times[pos_id] = time.time()

            if not news_items:
                return

            # Persist headlines into position dict → flows through to dashboard via _handle_state
            pos['latest_news'] = [
                {'title': item.title[:100], 'source': item.source, 'snippet': item.snippet[:200]}
                for item in news_items[:3]
            ]

            side = pos.get('side', 'YES')
            upnl = pos.get('unrealized_pnl', 0) or 0
            print(f"[Position News] {pos_id[:8]} {side} uPnL=${upnl:+.2f} | {question[:50]}")
            for item in news_items[:3]:
                print(f"  • [{item.source}] {item.title[:90]}")

        except Exception:
            pass  # news failure must never disrupt the position monitor

    async def _fetch_btc_price(self) -> float | None:
        """Fetch live BTC/USD price from CoinGecko (free, no API key)."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": "bitcoin", "vs_currencies": "usd"},
                )
                if r.status_code == 200:
                    return float(r.json()["bitcoin"]["usd"])
        except Exception:
            pass
        return None

    async def _position_monitor_loop(self):
        """Monitor positions for exit conditions."""
        import random
        while self._running:
            try:
                for pos_id, pos in list(self._positions.items()):
                    market = self._markets.get(pos.get('market_id'))
                    if not market:
                        # Position's market is not in the monitored dict (e.g. dropped from
                        # _select_markets or bot just restarted).  Fetch it directly so the
                        # stop-loss and profit-lock can still fire — never skip silently.
                        try:
                            raw = await self._kalshi.get_market(pos.get('market_id', ''))
                            if raw and 'market' in raw:
                                from services.kalshi_client import parse_kalshi_market
                                market = parse_kalshi_market(raw['market'])
                                # Cache it so the next loop cycle is free
                                self._markets[pos['market_id']] = market
                                # Back-fill question if position was stored with empty text
                                if not pos.get('question'):
                                    pos['question'] = market.get('question', '')
                                print(f"[Monitor] Fetched untracked market for {pos_id[:8]}: {pos.get('question','')[:45]}")
                        except Exception as _fetch_err:
                            print(f"[Monitor] Failed to fetch market for {pos_id[:8]} ({pos.get('market_id','')[:25]}): {_fetch_err}")
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
                    contracts = pos.get('contracts', round(pos['size'] / entry_price) if entry_price > 0 else 0)
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

                    # BTC RANGE MONITORING: For open BTC range bets, log live BTC price.
                    # ETH range bets are covered by _check_position_news instead.
                    _q_lower = pos.get('question', '').lower()
                    _pos_ticker_mon = pos.get('market_id', '').upper()
                    _is_btc_range_pos = (
                        ('bitcoin' in _q_lower and 'range' in _q_lower)
                        or _pos_ticker_mon.startswith('KXBTC-')
                    )
                    if _is_btc_range_pos:
                        if time.time() - self._btc_price_cache['fetched_at'] > 900:
                            _btc_live = await self._fetch_btc_price()
                            if _btc_live:
                                self._btc_price_cache = {'price': _btc_live, 'fetched_at': time.time()}
                        _btc_now = self._btc_price_cache.get('price')
                        if _btc_now:
                            _status = "IN RANGE" if current_price > 0.35 else "AT RISK"
                            print(f"[BTC Monitor] {pos_id[:8]} | BTC=${_btc_now:,.0f} | "
                                  f"market={current_price*100:.0f}¢ | uPnL=${unrealized_pnl:+.2f} | "
                                  f"{_status} | {pos.get('question','')[:45]}")

                    # NEWS CHECK: Scan headlines for each non-BTC-range position every 2h
                    await self._check_position_news(pos_id, pos)

                    # TERMINAL WRITE-OFF: If position is down >85% AND current value < $0.50,
                    # it is essentially worthless. A sell order at 1-2¢ will never fill on
                    # Kalshi (no buyers for near-zero contracts). Write it off immediately
                    # rather than letting it hang as a pending exit forever.
                    current_value = contracts * current_price
                    if gain_pct <= -0.85 and current_value < 0.50 and cost_basis > 0:
                        print(f"[Write-Off] {pos_id[:8]} terminal ({gain_pct*100:.0f}%, value=${current_value:.2f}) — recording loss and removing")
                        await self._exit_position(pos_id, current_price, unrealized_pnl, f"TERMINAL_WRITEOFF_{abs(gain_pct)*100:.0f}pct")
                        continue

                    # STOP-LOSS: Exit when position has lost too much
                    # Historical data: avg loss = $7.61, avg win = $3.55 on 40% win rate.
                    # Without a stop, losers go to 100% loss at settlement.
                    # Cutting at -50% halves the average loss and drops break-even
                    # win rate from 67% to ~50%, a dramatically better math profile.
                    # Skip for positions within 24h of resolution — let them settle.
                    stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.50'))
                    near_settlement = days_to_res is not None and days_to_res <= 1.0
                    if (gain_pct <= -stop_loss_pct
                            and cost_basis > 0
                            and not near_settlement):
                        should_exit = True
                        exit_reason_mon = f"STOP_LOSS_{abs(gain_pct)*100:.0f}pct"

                    # PROFIT-LOCK: Exit when position has gained significantly
                    # Logic: if we're up 50%+, the market has already priced in our view —
                    # locking in is better than waiting for a potential reversal.
                    elif gain_pct >= self._profit_lock_pct and cost_basis > 0:
                        should_exit = True
                        exit_reason_mon = f"PROFIT_LOCK_{gain_pct*100:.0f}pct"
                    
                    # NEAR-SETTLEMENT LOCK: Within 4h of resolution, lock any gain > 25%
                    # This captures "approaching certainty" price moves before final settlement
                    elif days_to_res is not None and days_to_res <= 0.17 and gain_pct >= 0.25:
                        should_exit = True
                        exit_reason_mon = f"NEAR_SETTLEMENT_LOCK_{gain_pct*100:.0f}pct"
                    
                    # Logging only (no exit) for monitoring — throttle to once per hour per position
                    elif unrealized_pnl < -0.70 * cost_basis:
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
                import traceback as _tb; print(f"[Monitor Error] {e}"); _tb.print_exc()
            await asyncio.sleep(10)
    
    async def _exit_position(self, pos_id: str, exit_price: float, pnl: float, reason: str):
        """Exit a position."""
        position = self._positions.get(pos_id)
        if not position:
            return

        # Respect retry cooldown set after a failed sell attempt (prevents rapid storm during
        # API outages where the monitor re-fires every 10s on the same position).
        _retry_after = position.get('exit_retry_after')
        if _retry_after:
            try:
                if datetime.utcnow().isoformat() < _retry_after:
                    return
            except Exception:
                pass
            position.pop('exit_retry_after', None)
        
        # In LIVE mode, actually sell the position on Kalshi
        if not self.dry_run:
            try:
                contracts = position.get('contracts', 0)
                if contracts > 0:
                    # Skip if position already has a pending exit order
                    if position.get('pending_exit'):
                        return  # Silently skip - don't spam logs

                    # TERMINAL positions (value < $0.50, price ≤ 5¢): skip the sell entirely.
                    # The monitor fires TERMINAL_WRITEOFF at gain_pct <= -0.85 and value < $0.50.
                    # Limit sells at 1-5¢ rarely fill on Kalshi — no buyers for near-zero contracts.
                    current_side_price = exit_price
                    _is_terminal_writeoff = (
                        current_side_price <= 0.05 and contracts * current_side_price < 0.50
                    )
                    if _is_terminal_writeoff:
                        print(f"[Write-Off] {pos_id[:8]} value ${contracts*current_side_price:.2f} — skipping unfillable sell, falling through to removal")
                        # Fall through — do NOT return — so position removal below runs
                    else:
                        # Normal sell path: limit order at bid minus 2¢, floor at 1¢
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
                        if not exit_order_id:
                            # API accepted the request but returned no order_id.
                            # Setting pending_exit with None would cause _check_pending_exits
                            # to skip it forever (it guards on order_id). Fall through to
                            # immediate removal instead — treat as an immediate write-off.
                            print(f"[LIVE SELL] Warning: no order_id returned — falling through to immediate removal")
                        else:
                            print(f"[LIVE SELL] Order placed @ {sell_price_cents}¢ | Order ID: {exit_order_id}")
                            # Track as pending exit — don't remove position until sell fills
                            position['pending_exit'] = {
                                'order_id': exit_order_id,
                                'exit_price': exit_price,
                                'reason': reason,
                                'placed_time': datetime.utcnow().isoformat(),
                            }
                            self._save_state()
                            return  # Wait for fill confirmation before recording exit
                        # If we reach here, order_id was None — fall through to position removal
                    
            except Exception as e:
                print(f"[Order Error] Failed to place sell order on Kalshi: {e}")
                # Set a 5-min cooldown to prevent rapid retry storms during API outages
                position['exit_retry_after'] = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
                return  # Keep position, retry after cooldown
        
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
    
    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    def _on_ws_ticker(
        self,
        ticker: str,
        yes_bid,   # int cents or None
        yes_ask,   # int cents or None
        last_price,  # int cents or None
        volume,
        open_interest,
    ) -> None:
        """Handle a real-time ticker update from the Kalshi WebSocket.

        Updates the in-memory price caches for the market so that the
        trading loop and performance endpoint always see fresh prices
        without waiting for the 30-second REST poll.
        """
        # Count every WS ticker receipt — Poll badge shows WS is alive even when
        # markets have no active orderbook (null bid/ask is a valid WS message).
        self._price_update_count += 1

        # Compute dollar prices from cent integers
        if yes_bid is not None and yes_ask is not None and yes_ask > yes_bid:
            yes_price = (yes_bid + yes_ask) / 2 / 100
            spread = (yes_ask - yes_bid) / 100
            yes_bid_price = yes_bid / 100   # actual bid — what you'd receive selling YES
            no_bid_price  = 1.0 - yes_ask / 100  # what you'd receive selling NO
        elif last_price is not None:
            yes_price = last_price / 100
            spread = 0.02
            yes_bid_price = yes_price - 0.01   # conservative 1¢ below last
            no_bid_price  = 1.0 - yes_price - 0.01
        else:
            return  # No price data — WS receipt counted above, nothing to cache

        no_price = round(1.0 - yes_price, 6)

        update = {
            'yes_price': yes_price,
            'no_price': no_price,
            'price': yes_price,
            'spread': spread,
            # Bid prices = actual liquidation value (what you receive selling now).
            # Used by the positions page and portfolio valuation so they agree with
            # Kalshi's portfolio_value (which is also bid-based).
            'yes_bid_price': round(yes_bid_price, 4),
            'no_bid_price':  round(max(0.01, no_bid_price), 4),
        }
        if last_price is not None:
            update['last_price'] = last_price / 100
        if volume is not None:
            update['volume'] = volume
        if open_interest is not None:
            update['open_interest'] = open_interest
            update['liquidity'] = open_interest * yes_price

        # Update both caches (same approach as _price_refresh_loop)
        if ticker in self._markets:
            self._markets[ticker].update(update)
        if ticker in self._monitored:
            self._monitored[ticker].update(update)

    def _on_ws_fill(self, fill: dict) -> None:
        """Handle a real-time fill notification from the Kalshi WebSocket.

        Marks the filled order in _pending_orders for immediate processing.
        The position sync loop wakes up via ws_client.fill_event and polls
        the REST API to confirm the fill details (count, exact price).
        """
        order_id = fill.get('order_id', '')
        if not order_id:
            return
        if order_id in self._pending_orders:
            print(f"[WS Fill] Pending entry order filled: {order_id}")
        # fill_event is set inside KalshiWebSocketClient._handle_fill;
        # no duplicate set needed here.

    # ------------------------------------------------------------------
    # Price refresh loop (REST fallback)
    # ------------------------------------------------------------------

    async def _price_refresh_loop(self):
        """Periodically refresh prices from Kalshi API (REST fallback).

        When the WebSocket is connected, this loop skips per-market API calls
        entirely — prices are kept fresh by _on_ws_ticker in real time.  The
        loop still runs on a longer interval to:
          1. Keep WS subscriptions in sync with the current _monitored set.
          2. Catch any markets that may have been missed during a reconnect.

        When WS is disconnected, the loop falls back to the original 30-second
        per-market REST poll so prices never go stale.
        """
        _POLL_INTERVAL_REST = 30    # seconds between full REST refresh
        _POLL_INTERVAL_WS   = 300   # seconds between REST refresh when WS is live

        while self._running:
            try:
                monitored_tickers = list(self._monitored.keys())

                # Keep WS subscriptions aligned with monitored markets AND open positions.
                # Open-position markets can be dropped from _monitored (e.g. expired from
                # the scan filter) but still need real-time prices for the monitor loop.
                _ws_needed = set(monitored_tickers)
                for _p in self._positions.values():
                    _pm = _p.get('market_id')
                    if _pm:
                        _ws_needed.add(_pm)
                if _ws_needed:
                    await self._ws_client.sync_subscriptions(list(_ws_needed))

                ws_live = self._ws_client.is_connected

                if not ws_live:
                    # WS is down — REST poll every market at full rate
                    for market_id in monitored_tickers:
                        try:
                            result = await self._kalshi.get_market(market_id)
                            if result and 'market' in result:
                                updated = parse_kalshi_market(result['market'])
                                # Merge price fields only — do NOT replace the whole dict.
                                # parse_kalshi_market does not return hours_to_resolution;
                                # overwriting wholesale would strip the value computed during
                                # market scanning and break Kelly horizon / cache TTL logic.
                                _price_fields = ('yes_price', 'no_price', 'last_price',
                                                 'price', 'volume', 'open_interest', 'liquidity')
                                for _f in _price_fields:
                                    if _f in updated:
                                        self._markets.setdefault(market_id, {})
                                        self._markets[market_id][_f] = updated[_f]
                                        if market_id in self._monitored:
                                            self._monitored[market_id][_f] = updated[_f]
                                self._price_update_count += 1
                        except Exception as _pref:
                            print(f"[Price Refresh] {market_id[:25]}: {_pref}")
                        await asyncio.sleep(1)  # Rate limit between per-market calls

            except Exception as e:
                print(f"[Price Refresh Error] {e}")

            # Sleep until next cycle — much longer when WS is live
            poll_secs = _POLL_INTERVAL_WS if self._ws_client.is_connected else _POLL_INTERVAL_REST
            await asyncio.sleep(poll_secs)
    
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
                    # Sell order filled - finalize the exit.
                    # Derive fill price from taker+maker costs if available (most accurate),
                    # then try average_fill_price (may not exist in Kalshi API v2),
                    # then fall back to the limit price we placed the order at.
                    _contracts = position.get('contracts', 1) or 1
                    _taker = order_data.get('taker_fill_cost', 0)
                    _maker = order_data.get('maker_fill_cost', 0)
                    if (_taker + _maker) > 0:
                        fill_price = (_taker + _maker) / _contracts / 100
                    else:
                        fill_price = order_data.get('average_fill_price',
                                                    pending_exit.get('exit_price', 0.5) * 100) / 100
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

                elif status in ('resting', 'open', ''):
                    # Sell order still unfilled — enforce a 30-minute staleness timeout.
                    # Without this, a resting exit order for an illiquid market would lock
                    # the position forever (monitor sees pending_exit → returns, never retries).
                    placed_time_str = pending_exit.get('placed_time', '')
                    if placed_time_str:
                        try:
                            placed_dt = datetime.fromisoformat(placed_time_str)
                            age_minutes = (datetime.utcnow() - placed_dt).total_seconds() / 60
                            STALE_EXIT_MINUTES = 30
                            if age_minutes > STALE_EXIT_MINUTES:
                                print(f"[EXIT STALE] Order {exit_order_id} resting {age_minutes:.0f}m — cancelling and retrying")
                                try:
                                    await self._kalshi.cancel_order(exit_order_id)
                                except Exception:
                                    pass
                                # Check for partial fill before treating as clean cancel.
                                # If some contracts already sold, reduce the position's
                                # contract count so the retry doesn't oversell.
                                try:
                                    _stale_result = await self._kalshi.get_order(exit_order_id)
                                    _stale_data = _stale_result.get('order', {})
                                    _stale_partial = (
                                        _stale_data.get('fill_count')
                                        or _stale_data.get('filled_count')
                                        or 0
                                    )
                                    if _stale_partial and _stale_partial > 0:
                                        _remaining = position.get('contracts', 0) - _stale_partial
                                        _entry_price = position.get('entry_price', 0.5)
                                        _exit_px = pending_exit.get('exit_price', 0.5)
                                        _partial_pnl = _stale_partial * (_exit_px - _entry_price)
                                        if _remaining > 0:
                                            position['contracts'] = _remaining
                                            position['size'] = _remaining * _entry_price
                                            print(f"[EXIT STALE] Partial fill: "
                                                  f"{_stale_partial} sold @ {_exit_px:.2f}, "
                                                  f"pnl=${_partial_pnl:+.2f}, {_remaining} remain")
                                        else:
                                            print(f"[EXIT STALE] Order fully filled during cancel check")
                                        # Record partial P&L so bankroll and kill-switch stay accurate
                                        try:
                                            await self._risk_engine.record_trade_result(_partial_pnl)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                position.pop('pending_exit', None)
                                self._save_state()
                        except Exception:
                            pass
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                if '404' in str(e).lower():
                    # Order no longer exists — it was either filled or cancelled externally.
                    # Clear pending_exit so the monitor can retry. _recently_exited is NOT
                    # set here because we don't know the outcome; _sync_positions_with_kalshi
                    # will reconcile the position on its next cycle and record it properly.
                    position.pop('pending_exit', None)
                    self._save_state()
                    print(f"[EXIT ORDER] {exit_order_id} not found — clearing pending_exit, sync will reconcile")
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

                # Kalshi uses both 'settled' and 'finalized' for resolved markets
                if status in ('settled', 'finalized'):
                    result = (market.get('result') or '').lower().strip()
                    signal['outcome_checked'] = True

                    if result in ('yes', 'no'):
                        predicted_side = signal.get('side', '').lower().strip()

                        signal['actual_result'] = result
                        signal['predicted_correct'] = (predicted_side == result)

                        # Theoretical P&L per contract at the signal's market price.
                        # market_price is always the YES price; for a NO bet our
                        # entry cost is (1 - yes_price).
                        _yes_price = float(signal.get('market_price', 0.5))
                        _side = signal.get('side', '').lower().strip()
                        entry_price = _yes_price if _side == 'yes' else (1.0 - _yes_price)
                        entry_price = max(0.01, min(0.99, entry_price))  # clamp to valid range

                        if signal['predicted_correct']:
                            signal['theoretical_pnl'] = round(1.0 - entry_price, 4)
                        else:
                            signal['theoretical_pnl'] = round(-entry_price, 4)

                        signal['outcome'] = 'WIN' if signal['predicted_correct'] else 'LOSS'
                        checked_count += 1
                    else:
                        # Settled but result field empty/unexpected — mark checked so
                        # we don't hammer the API on this market forever
                        signal['outcome'] = 'UNKNOWN'

                elif status in ('closed', 'open', 'active'):
                    # Not settled yet — will check again next cycle
                    pass
                else:
                    # Unknown status — mark checked to avoid infinite retries
                    signal['outcome_checked'] = True
                    signal['outcome'] = f'UNKNOWN_STATUS_{status}'
                    
                await asyncio.sleep(0.3)  # Rate limit
                
            except Exception as e:
                err_str = str(e).lower()
                if '404' in err_str:
                    # Market no longer exists on Kalshi — stop retrying
                    signal['outcome_checked'] = True
                    signal['outcome'] = 'MARKET_NOT_FOUND'
                else:
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
                
                # First, check pending EXIT orders (sells on existing positions).
                # IMPORTANT: this must run BEFORE _sync_positions_with_kalshi.
                # If a sell exit fills between cycles, the full sync sees the position
                # absent on Kalshi and removes it with SYNC_REMOVED — no P&L is recorded
                # and the trade record is lost.  Running exits first means filled sells
                # are processed (and the position removed cleanly) before the sync runs.
                await self._check_pending_exits()

                # Periodically do a full position sync to detect resolved markets
                if sync_counter >= FULL_SYNC_INTERVAL:
                    sync_counter = 0
                    await self._sync_positions_with_kalshi()
                    # NOTE: Do NOT call _cancel_all_resting_orders() here.
                    # That function is startup-only cleanup. Calling it periodically
                    # cancels live exit sell orders before they fill, preventing
                    # profit-locks from ever completing. Stale BUY orders are handled
                    # by the 10-minute timeout below.
                    # Check signal outcomes for backtesting
                    await self._check_signal_outcomes()
                
                # Then check pending BUY orders
                if not self._pending_orders:
                    # Wait up to 10 s, but wake immediately if the WS fires a fill.
                    # This avoids a 10-second delay when there is no live activity.
                    try:
                        await asyncio.wait_for(
                            self._ws_client.fill_event.wait(), timeout=10
                        )
                        self._ws_client.fill_event.clear()
                    except asyncio.TimeoutError:
                        pass
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
                            # Order fully filled.
                            # Try both field names (SDK v1 uses 'filled_count', v2 uses 'fill_count').
                            fill_count = (order_data.get('fill_count')
                                          or order_data.get('filled_count')
                                          or order['contracts'])
                            _taker = order_data.get('taker_fill_cost', 0)
                            _maker = order_data.get('maker_fill_cost', 0)
                            if (_taker + _maker) > 0 and fill_count > 0:
                                fill_price = (_taker + _maker) / fill_count / 100
                            else:
                                fill_price = order_data.get('average_fill_price',
                                                            order['entry_price'] * 100) / 100
                            filled_orders.append({
                                'order_id': order_id,
                                'order': order,
                                'fill_count': fill_count,
                                'fill_price': fill_price,
                            })
                            print(f"[ORDER FILLED] {order_id} | {fill_count} contracts @ {fill_price*100:.0f}¢")
                        
                        elif status == 'canceled' or status == 'cancelled':
                            # B49 fix: a cancel can arrive *after* a partial fill.
                            # Any contracts already filled are real capital deployed —
                            # track them as a position so they aren't orphaned.
                            partial_count = (order_data.get('fill_count')
                                             or order_data.get('filled_count')
                                             or 0)
                            if partial_count and partial_count > 0:
                                _taker = order_data.get('taker_fill_cost', 0)
                                _maker = order_data.get('maker_fill_cost', 0)
                                if (_taker + _maker) > 0 and partial_count > 0:
                                    partial_price = (_taker + _maker) / partial_count / 100
                                else:
                                    partial_price = order_data.get('average_fill_price',
                                                                   order['entry_price'] * 100) / 100
                                filled_orders.append({
                                    'order_id': order_id,
                                    'order': order,
                                    'fill_count': partial_count,
                                    'fill_price': partial_price,
                                    'partial': True,
                                })
                                print(f"[PARTIAL FILL + CANCEL] {order_id} | "
                                      f"{partial_count}/{order['contracts']} contracts @ "
                                      f"{partial_price*100:.0f}¢ — creating position")
                            else:
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
                                            # B49 fix (stale path): check for partial fill before
                                            # treating the cancel as a clean cancel. A resting order
                                            # that was partially filled before we cancelled it needs
                                            # a position created for the already-filled contracts,
                                            # just like the API-reported-cancel path at line ~3773.
                                            stale_partial = (order_data.get('fill_count')
                                                             or order_data.get('filled_count')
                                                             or 0)
                                            if stale_partial > 0:
                                                _st = order_data.get('taker_fill_cost', 0)
                                                _sm = order_data.get('maker_fill_cost', 0)
                                                if (_st + _sm) > 0:
                                                    stale_price = (_st + _sm) / stale_partial / 100
                                                else:
                                                    stale_price = order_data.get('average_fill_price', order['entry_price'] * 100) / 100
                                                filled_orders.append({
                                                    'order_id': order_id,
                                                    'order': order,
                                                    'fill_count': stale_partial,
                                                    'fill_price': stale_price,
                                                })
                                                print(f"[STALE PARTIAL] {order_id} had {stale_partial} filled contracts @ {stale_price*100:.0f}¢ before cancel — creating position")
                                            else:
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
                                            # fills API returns price in DOLLARS (0.0–1.0) — do NOT divide by 100
                                            'fill_price': fill.get('price', order['entry_price']),
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

                    # For partial fills the actual dollar cost is proportional
                    fill_count = filled.get('fill_count', order['contracts'])
                    total_contracts = order.get('contracts', 1) or 1
                    fill_ratio = min(1.0, fill_count / total_contracts) if total_contracts else 1.0
                    # Use actual fill price × contracts for accurate cost basis.
                    # order['size'] uses the midpoint price at analysis time (pre-slippage)
                    # and would understate exposure by up to 15% with configured slippage.
                    actual_size = fill_count * fill_price
                    
                    pos = {
                        'id': pos_id,
                        'order_id': order_id,
                        'market_id': order['market_id'],
                        'question': order.get('question', ''),
                        'side': order['side'],
                        'size': actual_size,
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
                                size=actual_size,
                                raw_prob=order.get('ai_probability', 0.5),
                                adjusted_prob=order.get('ai_probability', 0.5),
                                edge=order.get('edge', 0),
                                confidence=order.get('confidence', 0),
                            )
                            pos['db_trade_id'] = db_trade_id
                        except Exception as e:
                            print(f"[DB] Failed to log filled trade: {e}")
                    
                    partial_flag = " (partial)" if filled.get('partial') else ""
                    print(f"[POSITION OPENED{partial_flag}] {order['side']} ${actual_size:.2f} @ {fill_price*100:.0f}¢ | {order.get('question', '')[:50]}...")
                    
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
                import traceback as _tb; print(f"[Position Sync Error] {e}"); _tb.print_exc()

            # When WS is live, wait up to 10 s OR until a fill event wakes us.
            # This means a fill received via WebSocket is processed with
            # near-zero latency instead of waiting out the full 10-second sleep.
            try:
                await asyncio.wait_for(
                    self._ws_client.fill_event.wait(), timeout=10
                )
                self._ws_client.fill_event.clear()
                print("[Sync] WS fill event received — re-checking orders immediately")
            except asyncio.TimeoutError:
                pass

    async def _broadcast_update(self):
        """Send update to all connected WebSocket clients."""
        if not self._websockets:
            return

        data = json.dumps({
            'type': 'update',
            'stats': self._get_stats(),
            'positions': self._enrich_positions(),
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
    
    async def _handle_healthz(self, request):
        """Railway health check — returns 200 as soon as the HTTP server is up."""
        return web.Response(text='ok', content_type='text/plain')

    async def _handle_force_exit(self, request):
        """Force-exit a position by market_id. Usage: /api/force-exit?market_id=KXU3-26MAR-T4.5"""
        market_id = request.rel_url.query.get('market_id', '').strip().upper()
        if not market_id:
            return web.json_response({'ok': False, 'error': 'market_id param required'}, status=400)

        # Find the position — search internal dict first, then Kalshi raw positions (fix: exit
        # button was returning 404 for positions that exist in Kalshi but not internal state).
        pos_id = None
        for pid, pos in self._positions.items():
            if pos.get('market_id', '').upper() == market_id:
                pos_id = pid
                break

        # Fallback: position exists in Kalshi raw data but not internal dict (e.g. after restart).
        # Synthesise a minimal internal entry so _exit_position can place the sell order.
        if not pos_id:
            kp_match = next(
                (kp for kp in self._kalshi_positions_raw
                 if kp.get('ticker', '').upper() == market_id),
                None
            )
            if kp_match:
                contracts = abs(kp_match.get('position', 0))
                side = 'YES' if kp_match.get('position', 0) > 0 else 'NO'
                cached = self._markets.get(market_id, {})
                cur_price = cached.get('yes_price', 0.5) if side == 'YES' else cached.get('no_price', 0.5)
                exposure = abs(kp_match.get('market_exposure', 0)) / 100
                entry_price = (exposure / contracts) if contracts > 0 else cur_price
                pos_id = f'pos_kals_{market_id}'
                self._positions[pos_id] = {
                    'market_id': market_id,
                    'side': side,
                    'contracts': contracts,
                    'entry_price': entry_price,
                    'current_price': cur_price,
                    'cost': exposure,
                    'size': exposure,
                    'question': cached.get('question', market_id),
                }
                print(f"[ForceExit] Synthesised internal position for Kalshi-only entry {market_id}")

        if not pos_id:
            held = [p.get('market_id') for p in self._positions.values()]
            return web.json_response({'ok': False, 'error': f'{market_id} not in positions', 'held': held}, status=404)

        pos = self._positions[pos_id]

        # Return immediately if a pending exit order is already in flight
        if pos.get('pending_exit'):
            return web.json_response({
                'ok': False,
                'error': 'Exit order already pending — waiting for fill confirmation',
                'order_id': pos['pending_exit'].get('order_id'),
            }, status=409)

        side = pos.get('side', 'YES')
        # Use live cached price if available, otherwise fall back to stored current_price
        cached = self._markets.get(market_id, {})
        live_price = cached.get('yes_price') if side == 'YES' else cached.get('no_price')
        cur_price = live_price or pos.get('current_price', pos.get('entry_price', 0.5))
        exit_price = cur_price
        cost = pos.get('cost') or pos.get('size', 0)
        pnl = (exit_price - pos.get('entry_price', exit_price)) * pos.get('contracts', 0)

        print(f"[ForceExit] Manual exit triggered for {market_id} ({side}) @ {exit_price:.2f}")
        await self._exit_position(pos_id, exit_price, pnl, 'MANUAL_FORCE_EXIT')
        self._save_state()

        # Check if a pending exit was set (limit order placed) vs immediate removal (writeoff)
        still_open = pos_id in self._positions
        return web.json_response({
            'ok': True,
            'message': 'Limit sell order placed — waiting for fill' if still_open else 'Position exited immediately',
            'side': side,
            'exit_price': round(exit_price, 3),
            'estimated_pnl': round(pnl, 2),
            'cost': round(cost, 2),
        })

    async def _handle_reset_killswitch(self, request):
        """Manually clear the kill switch and re-baseline drawdown from current value."""
        live = self._kalshi_total or self._risk_engine.daily_stats.current_bankroll
        self._risk_engine.daily_stats.kill_switch_triggered = False
        # Re-baseline: measure future drawdown from NOW, not from original start of day
        self._risk_engine.daily_stats.starting_bankroll = live
        self._risk_engine.daily_stats.current_bankroll = live
        self._kill_switch_fire_date = ''

        # Also update start_of_day_value in the snapshot so that after a redeploy
        # _load_state restores starting_bankroll = live (not the stale $726 value).
        # Without this, every new deploy re-reads the old baseline and re-triggers.
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        snap = self._daily_snapshots.get(today_str, {})
        snap['start_of_day_value'] = round(live, 2)
        snap['_ks_reset'] = True   # flag so migration guard doesn't overwrite it
        self._daily_snapshots[today_str] = snap
        # Store kill-switch-reset baseline separately so _load_state restores it correctly
        snap['_ks_reset_baseline'] = round(live, 2)

        self._save_state()
        print(f"[KILL SWITCH] Manually reset via /api/reset-killswitch — new baseline ${live:.2f}")
        return web.json_response({'ok': True, 'message': f'Kill switch cleared — drawdown re-baselined from ${live:.2f}.'})

    async def _handle_debug_positions(self, request):
        """Debug endpoint: directly calls Kalshi positions API and returns raw result."""
        try:
            result = await self._kalshi.get_positions()
            raw_count = len(result.get('market_positions') or [])
            event_count = len(result.get('event_positions') or [])
            # Sample first few of each for inspection
            mp_sample = (result.get('market_positions') or [])[:5]
            ep_sample = (result.get('event_positions') or [])[:5]
            return web.json_response({
                'keys': list(result.keys()) if isinstance(result, dict) else [],
                'market_positions_count': raw_count,
                'event_positions_count': event_count,
                'market_positions_sample': mp_sample,
                'event_positions_sample': ep_sample,
                'cached_raw_count': len(self._kalshi_positions_raw),
                'internal_positions': len(self._positions),
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def _handle_positions(self, request):
        """Dedicated positions endpoint — full question text, all positions, live P&L.
        Uses the UNION of _positions (internal, has entry prices) and _kalshi_positions_raw
        (Kalshi API, authoritative for what actually exists) so nothing is ever missed."""
        try:
            if self._kalshi_cash is None or self._kalshi_total is None:
                return web.json_response({'error': 'syncing', 'positions': []})

            # Index both sources by ticker for merging
            internal_by_ticker: dict = {}
            for pos in self._positions.values():
                t = pos.get('market_id', '')
                if t:
                    internal_by_ticker[t] = pos

            kalshi_by_ticker: dict = {}
            for kp in self._kalshi_positions_raw:
                t = kp.get('ticker', '')
                if t and kp.get('position', 0) != 0:
                    kalshi_by_ticker[t] = kp

            # Union: start with all tickers that appear in either source
            all_tickers = set(internal_by_ticker) | set(kalshi_by_ticker)

            positions_detail = []

            for ticker in all_tickers:
                internal = internal_by_ticker.get(ticker, {})
                kpos     = kalshi_by_ticker.get(ticker, {})

                # Side / contracts — prefer internal (has bot context), fall back to Kalshi
                if internal:
                    side      = internal.get('side', 'YES').upper()
                    contracts = internal.get('contracts', 0)
                    entry_price = internal.get('entry_price', 0.5)
                    # Always fall back to _markets cache — internal positions created from
                    # series contracts (e.g. KXCPIYOY-26MAR-T3.1) may have been stored
                    # with question='' because Kalshi returned title='' at order time.
                    _icache = self._markets.get(ticker, {})
                    _stored_q = internal.get('question', '')
                    # Treat question == ticker as "no real question" — use cache or decoder
                    if _stored_q and _stored_q != ticker:
                        question = _stored_q
                    else:
                        _cache_q = _icache.get('question', '') or _icache.get('title', '')
                        if _cache_q and _cache_q != ticker:
                            question = _cache_q
                        else:
                            question = self._decode_kalshi_ticker(ticker)
                    entry_time = internal.get('entry_time')
                elif kpos:
                    raw_c = kpos.get('position', 0)
                    side  = 'YES' if raw_c > 0 else 'NO'
                    contracts = abs(raw_c)
                    entry_price = None  # unknown — will default to current
                    cached_market = self._markets.get(ticker, {})
                    # Try every field name Kalshi uses across API versions before falling
                    # back to the raw ticker so the table shows text, not a code.
                    question = (
                        cached_market.get('question')
                        or kpos.get('title')
                        or kpos.get('market_title')
                        or kpos.get('market_question')
                        or kpos.get('subtitle')
                        or ticker
                    )
                    entry_time = None
                else:
                    continue

                # Current price: prefer live _markets cache, fall back to market_exposure
                cached_market = self._markets.get(ticker, {})
                yes_price     = cached_market.get('yes_price')      # mid-price display
                no_price      = cached_market.get('no_price')
                yes_bid_price = cached_market.get('yes_bid_price')  # actual bid (exit value)
                no_bid_price  = cached_market.get('no_bid_price')

                if yes_price is None and kpos:
                    _exp = abs(kpos.get('market_exposure', 0)) / 100
                    _c   = abs(kpos.get('position', contracts))
                    if _exp > 0 and _c > 0:
                        yes_price = _exp / _c if side == 'YES' else 1 - _exp / _c
                if yes_price is None:
                    yes_price = entry_price or 0.5
                if no_price is None:
                    no_price = 1 - yes_price
                _spread = cached_market.get('spread', 0.02)
                if yes_bid_price is None:
                    yes_bid_price = yes_price - _spread / 2
                if no_bid_price is None:
                    no_bid_price = no_price - _spread / 2

                current_price_mid = yes_price if side == 'YES' else no_price
                current_price_bid = yes_bid_price if side == 'YES' else no_bid_price
                if entry_price is None:
                    entry_price = current_price_mid  # unknown entry — show flat

                cost = contracts * entry_price
                value = contracts * current_price_bid   # bid = true exit value
                unrealized = value - cost
                unrealized_pct = (unrealized / cost * 100) if cost > 0 else 0

                positions_detail.append({
                    'ticker': ticker,
                    'question': question,
                    'side': side,
                    'contracts': contracts,
                    'entry_price': round(entry_price, 4),
                    'current_price': round(current_price_mid, 4),   # mid shown in table
                    'current_price_bid': round(current_price_bid, 4),
                    'cost': round(cost, 2),
                    'value': round(value, 2),  # bid-based
                    'unrealized_pnl': round(unrealized, 2),
                    'unrealized_pct': round(unrealized_pct, 1),
                    'entry_time': entry_time,
                })

            positions_detail.sort(key=lambda p: p['unrealized_pnl'])

            total_cost = sum(p['cost'] for p in positions_detail)
            total_unrealized = sum(p['unrealized_pnl'] for p in positions_detail)
            winning = len([p for p in positions_detail if p['unrealized_pnl'] > 0])
            losing = len([p for p in positions_detail if p['unrealized_pnl'] < 0])

            return web.json_response({
                'positions': positions_detail,
                'summary': {
                    'count': len(positions_detail),
                    'winning': winning,
                    'losing': losing,
                    'flat': len(positions_detail) - winning - losing,
                    'total_cost': round(total_cost, 2),
                    'total_unrealized': round(total_unrealized, 2),
                },
                'timestamp': datetime.utcnow().isoformat(),
            })
        except Exception as e:
            import traceback
            return web.json_response({'error': str(e), 'traceback': traceback.format_exc()}, status=500)

    @staticmethod
    def _decode_kalshi_ticker(ticker: str) -> str:
        """Convert a raw Kalshi ticker to a human-readable label.

        Used as last-resort when Kalshi's API returns no title for a contract.
        Examples:
          KXCPIYOY-26MAR-T3.1  → "CPI YoY: above 3.1 (Mar '26)"
          KXU3-26MAR-T4.5      → "Unemployment U3: above 4.5 (Mar '26)"
          KXNEWTARIFFS-26APR01 → "New Tariffs (Apr 1 '26)"
          KXTX18D-26-AGRE      → "TX 18th District - AGRE ('26)"
        """
        import re as _re
        _SERIES_NAMES: dict[str, str] = {
            'CPIYOY': 'CPI YoY', 'CPI': 'CPI MoM', 'PCE': 'PCE Inflation',
            'PCECORE': 'Core PCE', 'U3': 'Unemployment U3', 'U6': 'Unemployment U6',
            'NONFARM': 'Nonfarm Payrolls', 'JOBSREPORT': 'Jobs Report',
            'GDPQ': 'GDP Growth', 'FED': 'Fed Funds Rate',
            'TECHLAYOFF': 'Tech Layoff Rate', 'LAGODAYS': 'Lag Days',
            'EOWEEK': 'Executive Orders/Week', 'TRUMPMEET': 'Trump Meeting',
            'NEWTARIFFS': 'New Tariffs', 'TX18D': 'TX 18th District',
            'BTC': 'Bitcoin Range', 'ETH': 'Ethereum Range',
            'DOGE': 'Dogecoin Range', 'SOL': 'Solana Range',
            'NASDAQ100': 'Nasdaq 100 Range',
        }
        _MON = {
            'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APR': 'Apr',
            'MAY': 'May', 'JUN': 'Jun', 'JUL': 'Jul', 'AUG': 'Aug',
            'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec',
        }
        t = ticker[2:] if ticker.startswith('KX') else ticker
        parts = t.split('-')
        if not parts:
            return ticker
        series = parts[0]
        name = _SERIES_NAMES.get(series, series)
        date_str = ''
        threshold_str = ''
        extra: list[str] = []
        for part in parts[1:]:
            m = _re.match(
                r'^(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{0,2})$',
                part,
            )
            if m:
                yy, mon, day = m.group(1), _MON[m.group(2)], m.group(3)
                date_str = f"{mon}{' ' + day.lstrip('0') if day else ''} '{yy}"
            elif _re.match(r'^\d{2}$', part):
                date_str = f"'{part}"
            elif part.startswith('T') and len(part) > 1 and _re.match(r'^T[\d.]+$', part):
                threshold_str = f'above {part[1:]}'
            elif _re.match(r'^[\d.]+$', part):
                threshold_str = f'above {part}'
            else:
                extra.append(part)
        label = name
        if threshold_str:
            label += f': {threshold_str}'
        if date_str:
            label += f' ({date_str})'
        if extra:
            label += f' [{"-".join(extra)}]'
        return label

    def _enrich_positions(self) -> list:
        """Return positions list with question text resolved from _markets cache.

        Positions created from Kalshi series contracts (e.g. KXCPIYOY-26MAR-T3.1)
        were sometimes stored with question == market_id (ticker string) because Kalshi
        returns title='' for those contracts and an old fallback wrote the raw ticker.
        We must treat question == ticker as "no readable question" and override it.
        """
        enriched = []
        for pos in self._positions.values():
            ticker = pos.get('market_id', '')
            stored_q = pos.get('question', '')
            # A question that IS the ticker is not a real question — treat as missing
            needs_q = not stored_q or stored_q == ticker
            if needs_q:
                cached = self._markets.get(ticker, {})
                cache_q = cached.get('question', '') or cached.get('title', '')
                # Ignore cache if it also just contains the ticker
                if cache_q and cache_q != ticker:
                    pos = {**pos, 'question': cache_q}
                else:
                    # Last resort: decode ticker into a human-readable label
                    pos = {**pos, 'question': self._decode_kalshi_ticker(ticker)}
            enriched.append(pos)
        return enriched

    async def _handle_state(self, request):
        """API endpoint for current state."""
        return web.json_response({
            'stats': self._get_stats(),
            'positions': self._enrich_positions(),
            'trades': self._trades[:50],
            'analyses': self._analyses[:20],
            'monitored': list(self._monitored.values()),
            'kalshi_positions_raw': self._kalshi_positions_raw[:30],
        })
    
    async def _handle_signals(self, request):
        """API endpoint for signal backtesting analysis."""
        return web.json_response(self._get_signal_performance())

    async def _handle_filters(self, request):
        """API endpoint: shows what markets are being filtered and why."""
        cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        cutoff_1h  = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        log = list(self._filter_log)  # snapshot of deque

        counts_24h: dict = collections.defaultdict(int)
        counts_1h:  dict = collections.defaultdict(int)
        for entry in log:
            reason = entry['reason'] if not entry['reason'].startswith('LOW_VOLUME_') else 'LOW_VOLUME'
            if entry['ts'] >= cutoff_24h:
                counts_24h[reason] += 1
            if entry['ts'] >= cutoff_1h:
                counts_1h[reason] += 1

        return web.json_response({
            'summary': {
                'total_logged': len(log),
                'last_24h': dict(sorted(counts_24h.items(), key=lambda x: -x[1])),
                'last_1h':  dict(sorted(counts_1h.items(),  key=lambda x: -x[1])),
            },
            'recent_50': list(reversed(log))[:50],
            'nightly_report': self._last_nightly_report,
        })

    async def _handle_nightly(self, request):
        """API endpoint: returns latest nightly strategy report (or triggers one now)."""
        force = request.rel_url.query.get('force', '').lower() == 'true'
        if force:
            await self._run_nightly_report()
        return web.json_response(self._last_nightly_report or {'message': 'No report yet — runs at midnight UTC or call with ?force=true'})
    
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
        
        Uses _kalshi_total (cash + market_exposure from sync loop) as the
        canonical account value. This avoids the 0.50-default price bug where
        a cold _markets cache would stamp a wrong start_of_day_value and corrupt
        all of today's P&L, Today's Peak, and chart data for the entire day.
        """
        try:
            # 1. Require both cash and total to be synced.
            if self._kalshi_cash is None or self._kalshi_total is None:
                return web.json_response({'error': 'syncing', 'message': 'Waiting for initial Kalshi sync...'})

            cash = self._kalshi_cash

            # 2. Per-position detail — union of internal _positions and _kalshi_positions_raw
            # so the count is never 0 after restart (positions restored from state file before
            # Kalshi API sync repopulates _kalshi_positions_raw).
            kalshi_by_ticker = {p.get('ticker', ''): p for p in self._kalshi_positions_raw if p.get('position', 0) != 0}
            internal_by_ticker = {p.get('market_id', ''): p for p in self._positions.values() if p.get('market_id')}
            all_tickers = set(internal_by_ticker) | set(kalshi_by_ticker)

            total_position_cost = 0
            total_position_value = 0
            positions_detail = []

            for ticker in all_tickers:
                internal = internal_by_ticker.get(ticker, {})
                kpos     = kalshi_by_ticker.get(ticker, {})

                if internal:
                    side      = internal.get('side', 'YES').lower()
                    contracts = internal.get('contracts', 0)
                    entry_price_internal = internal.get('entry_price', None)
                    question  = internal.get('question', ticker)[:50]
                elif kpos:
                    raw_c = kpos.get('position', 0)
                    side  = 'yes' if raw_c > 0 else 'no'
                    contracts = abs(raw_c)
                    entry_price_internal = None
                    cached_market = self._markets.get(ticker, {})
                    question = cached_market.get('question', ticker)[:50]
                else:
                    continue

                cached_market = self._markets.get(ticker, {})
                yes_price = cached_market.get('yes_price')      # mid-price
                no_price  = cached_market.get('no_price')       # mid-price
                # Bid prices = actual exit/liquidation value.  Consistent with Kalshi's
                # portfolio_value (bid-based), so positions page P&L matches portfolio today.
                yes_bid_price = cached_market.get('yes_bid_price')
                no_bid_price  = cached_market.get('no_bid_price')

                # Fallback: derive price from Kalshi's market_exposure — never default to 0.50
                if yes_price is None and kpos:
                    _exp = abs(kpos.get('market_exposure', 0)) / 100
                    _c   = abs(kpos.get('position', contracts))
                    if _exp > 0 and _c > 0:
                        yes_price = _exp / _c if side == 'yes' else 1 - _exp / _c
                if yes_price is None:
                    yes_price = entry_price_internal or 0.5
                if no_price is None:
                    no_price = 1 - yes_price
                # Bid fallback: subtract half the spread (or 1¢ if spread unknown)
                _spread = cached_market.get('spread', 0.02)
                if yes_bid_price is None:
                    yes_bid_price = yes_price - _spread / 2
                if no_bid_price is None:
                    no_bid_price = no_price - _spread / 2

                # Use bid price for valuation (matches Kalshi's portfolio_value MtM)
                current_price_mid = yes_price if side == 'yes' else no_price
                current_price_bid = yes_bid_price if side == 'yes' else no_bid_price
                entry_price = entry_price_internal if entry_price_internal is not None else current_price_mid

                cost = contracts * entry_price
                value = contracts * current_price_bid   # bid-based = true liquidation value
                unrealized = value - cost

                total_position_cost += cost
                total_position_value += value

                positions_detail.append({
                    'ticker': ticker,
                    'question': question,
                    'side': side.upper(),
                    'contracts': contracts,
                    'entry_price': entry_price,
                    'current_price': current_price_mid,  # show mid in table (more intuitive)
                    'current_price_bid': current_price_bid,  # bid = exit value
                    'cost': cost,
                    'value': value,  # bid-based value
                    'unrealized_pnl': unrealized,  # bid-based so it matches portfolio today
                })

            # 3. Canonical account value: cash + market value of positions (Kalshi API, ~5 min refresh).
            # _kalshi_total = _kalshi_cash + _kalshi_positions_mv — the same value used by
            # _get_stats() and the daily snapshot, so Total Value / Account Value / Graph all match.
            account_value = self._kalshi_total
            total_deposits = float(os.getenv('TOTAL_DEPOSITS', str(int(self.initial_bankroll))))
            total_return = account_value - total_deposits
            return_pct = (total_return / total_deposits * 100) if total_deposits > 0 else 0

            # 4. Unrealized P&L (per-position detail only — for breakdown table)
            unrealized_pnl = sum(p['unrealized_pnl'] for p in positions_detail)

            # 5. Winning/losing position counts
            winning_positions = len([p for p in positions_detail if p['unrealized_pnl'] > 0])
            losing_positions = len([p for p in positions_detail if p['unrealized_pnl'] < 0])

            # 6. Today's activity
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            today_entries = [t for t in self._trades
                           if t.get('action') in ('ENTRY', 'ORDER_PLACED') and
                           t.get('timestamp', '').startswith(today_str)]

            # 7. Exposure breakdown by category
            exposure_by_category = {}
            for p in positions_detail:
                ticker = p['ticker']
                cost = p['cost']
                if 'SNOW' in ticker or 'RAIN' in ticker:
                    cat = 'Weather'
                elif 'BTC' in ticker or 'ETH' in ticker or 'SOL' in ticker:
                    cat = 'Crypto'
                elif 'NBA' in ticker or 'NFL' in ticker or 'MLB' in ticker:
                    cat = 'Sports'
                else:
                    cat = 'Other'
                exposure_by_category[cat] = exposure_by_category.get(cat, 0) + cost

            # 8. NOTE: _save_daily_snapshot is NOT called here.
            # Snapshots are owned exclusively by _sync_positions_with_kalshi (every ~5 min)
            # which uses the same _kalshi_total-based value. Calling it here with potentially
            # cold-cache prices was corrupting start_of_day_value and today_pnl.

            # 8b. Realized P&L today (closed/exited positions that settled today)
            today_exits = [t for t in self._trades
                           if t.get('action') == 'EXIT'
                           and t.get('timestamp', '').startswith(today_str)]
            realized_pnl_today = round(sum(t.get('pnl', 0) for t in today_exits), 2)

            # 9. Today's P&L from snapshot (set by sync loop)
            today_snapshot = self._daily_snapshots.get(today_str, {})
            start_of_day_value = today_snapshot.get('start_of_day_value')
            if start_of_day_value is not None:
                today_pnl = round(account_value - start_of_day_value, 2)
                today_pnl_pct = round((today_pnl / start_of_day_value * 100), 1) if start_of_day_value else 0
            else:
                today_pnl = None
                today_pnl_pct = None

            # 10. Portfolio history
            history_daily, history_today_hourly = self._get_performance_history()
            
            return web.json_response({
                # SINGLE SOURCE OF TRUTH: direct Kalshi API values from sync loop
                'account': {
                    'cash': round(cash, 2),
                    # Market value of positions (same source as total_value — cash + this = total_value)
                    'positions_value': round(self._kalshi_positions_mv or total_position_value, 2),
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
                    # P&L breakdown — explains why "Today" ≠ "Unrealized on open bets"
                    # today_pnl = unrealized_pnl (open bets, bid-based) + realized_pnl_today
                    # (closed bets today) + any Kalshi mark-to-market lag.
                    'unrealized_pnl': round(unrealized_pnl, 2),  # open bets at bid price
                    'realized_pnl_today': realized_pnl_today,     # closed/exited bets today
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
        
        # _daily_snapshots and _portfolio_hourly are always initialised in __init__ and
        # loaded from the state file by _load_state(), so no hasattr guard needed here.

        bot_uptime_secs = (now - self._start_time).total_seconds() if self._start_time else 999

        # Start-of-day value — only lock in after 60s of uptime so cold-start transient
        # prices don't corrupt the baseline used for today's P&L comparison.
        snapshot = self._daily_snapshots.get(today_str, {})

        # Migration guard: snapshots written before the MtM fix used Kalshi's
        # portfolio_value (face value) which inflates positions by ~2x.  If the stored
        # baseline has the old method flag (or no flag at all) and deviates from the
        # current account_value by more than 15%, reset it so today's P&L restarts from
        # a correct MtM baseline rather than reporting a phantom loss.
        _pv_method = snapshot.get('_pv_method')
        if _pv_method != 'mtm':
            _old_sod = snapshot.get('start_of_day_value')
            if _old_sod is not None and val > 0:
                _deviation = abs(_old_sod - val) / max(val, 1)
                if _deviation > 0.15:
                    print(f"[Snapshot] Resetting corrupted start_of_day baseline "
                          f"(old={_old_sod:.2f} vs current={val:.2f}, "
                          f"deviation={_deviation:.1%}) — was written with face-value formula")
                    snapshot['start_of_day_value'] = None
            snapshot['_pv_method'] = 'mtm'

        # Spike guard: reject single-cycle value drops >35% from the previous reading.
        # Kalshi's portfolio_value occasionally returns a transiently bad value (face value,
        # near-zero, etc.) that causes the account total to plunge by hundreds of dollars in
        # a single sync, which sets a false intraday_low and triggers the kill switch.
        _prev_val = snapshot.get('account_value', val)
        _spike_ratio = val / _prev_val if _prev_val > 0 else 1.0
        if _spike_ratio < 0.65 and _prev_val > 50:
            # Reject — looks like a bad API reading, not a real loss this fast
            print(f"[Snapshot] SPIKE GUARD: rejecting val=${val:.2f} (prev=${_prev_val:.2f}, "
                  f"ratio={_spike_ratio:.2f}) — using prev value to protect kill switch")
            val = _prev_val  # use previous value for this snapshot cycle

        if snapshot.get('start_of_day_value') is None and bot_uptime_secs >= 60:
            snapshot['start_of_day_value'] = val
        snapshot['account_value'] = val
        snapshot['cash'] = round(cash, 2)
        snapshot['positions_value'] = round(positions_value, 2)
        snapshot['timestamp'] = now.isoformat()
        
        # Track intraday high watermark — shows "was up X, now at Y" pattern
        current_high = snapshot.get('intraday_high')
        start_val = snapshot.get('start_of_day_value', val)
        if current_high is None:
            # Guard: only initialise the peak after 60s of uptime.
            # The first few syncs can capture transient bid/ask midpoints that don't
            # reflect real tradeable prices. After 60s prices have stabilised.
            if bot_uptime_secs >= 60:
                # Floor at start_of_day so the peak is never below where the day started.
                current_high = max(val, start_val)
                snapshot['intraday_high'] = current_high
                snapshot['intraday_high_time'] = now.isoformat()
        elif val > current_high:
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
        
        if self._last_snapshot_file_save is None or (now - self._last_snapshot_file_save).total_seconds() > 300:
            self._save_state()
            self._last_snapshot_file_save = now
    
    def _get_performance_history(self) -> tuple:
        """Return (daily history for last 14 days, today's hourly points for chart)."""
        
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
        
        Cached for 5 minutes — settlements don't change frequently and the
        calculation requires 1 + N_settled_tickers API calls which is expensive.
        """
        try:
            # Return cached result if fresh enough (5 minute TTL).
            _SETTLEMENTS_TTL = 300
            if (self._settlements_cache is not None
                    and self._settlements_cache_time is not None
                    and (datetime.utcnow() - self._settlements_cache_time).total_seconds() < _SETTLEMENTS_TTL):
                return web.json_response(self._settlements_cache)

            # Get all fills from Kalshi (our trade history)
            result = await self._kalshi.get_fills(limit=500)
            fills = result.get('fills', [])
            
            if not fills:
                _empty = {
                    'settlements': [],
                    'summary': {'total': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0},
                    'today': {'wins': 0, 'losses': 0, 'pnl': 0},
                }
                self._settlements_cache = _empty
                self._settlements_cache_time = datetime.utcnow()
                return web.json_response(_empty)
            
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
                        if fill_side:  # Only update side if field is present (mirrors reconcile fix)
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
                    
                    # Contracts that settled = net position (buys minus pre-settlement manual sells).
                    # Kalshi settlement payouts also appear as 'sell' fills, but those happen
                    # AFTER settlement and should not reduce the contract count here.
                    # We approximate: if total_sold >= total_bought it means all were settled/sold,
                    # otherwise the difference is a pre-settlement manual sell.
                    # Use total_bought as denominator (correct cost basis), net for quantity.
                    net_contracts = max(total_bought - total_sold, 0)
                    contracts_settled = net_contracts if net_contracts > 0 else total_bought
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
            
            total_deposits = float(os.getenv('TOTAL_DEPOSITS', str(int(self.initial_bankroll))))

            _response_data = {
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
            }
            self._settlements_cache = _response_data
            self._settlements_cache_time = datetime.utcnow()
            return web.json_response(_response_data)
            
        except Exception as e:
            import traceback
            # Set a short error-TTL so the dashboard backs off during API outages instead
            # of hammering Kalshi with a fresh traverse every 30 seconds indefinitely.
            self._settlements_cache_time = datetime.utcnow()
            self._settlements_cache = {'error': str(e), 'settlements': [], 'summary': {}, 'today': {}}
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
        try:
            # Send initial state — inside try so disconnect during send still
            # triggers the finally cleanup and removes ws from the set.
            await ws.send_str(json.dumps({
                'type': 'init',
                'stats': self._get_stats(),
                'positions': self._enrich_positions(),
                'trades': self._trades[:50],
                'analyses': self._analyses[:20],
                'monitored': list(self._monitored.values()),
            }))
            async for msg in ws:
                pass
        except Exception:
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
        /* Positions page */
        .pos-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .pos-table-wrap { overflow-x: auto; }
        .pos-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .pos-table th { padding: 8px 12px; text-align: left; font-size: 11px; color: #8b949e; text-transform: uppercase; border-bottom: 1px solid #30363d; white-space: nowrap; }
        .pos-table td { padding: 10px 12px; border-bottom: 1px solid #21262d; vertical-align: top; }
        .pos-table tr:hover td { background: #161b22; }
        .pos-question { max-width: 320px; line-height: 1.4; }
        .pos-ticker { font-size: 11px; color: #8b949e; margin-top: 3px; font-family: monospace; }
        .pnl-pos { color: #3fb950; font-weight: 600; }
        .pnl-neg { color: #f85149; font-weight: 600; }
        .pnl-flat { color: #8b949e; }
        .side-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .side-badge.YES { background: #238636; color: #fff; }
        .side-badge.NO  { background: #6e2929; color: #f85149; border: 1px solid #f85149; }
        .exit-btn { padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; background: #21262d; color: #f85149; border: 1px solid #f85149; white-space: nowrap; }
        .exit-btn:hover { background: #3d1c1c; }
        .pos-empty { text-align: center; padding: 40px; color: #8b949e; font-size: 14px; }
        .pos-last-refresh { font-size: 11px; color: #6e7681; text-align: right; margin-bottom: 8px; }
        .pos-bar { display: flex; gap: 4px; height: 4px; border-radius: 2px; margin-bottom: 20px; overflow: hidden; background: #21262d; }
        .pos-bar-win { background: #3fb950; }
        .pos-bar-loss { background: #f85149; }
        .pos-bar-flat { background: #6e7681; }
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
        <div class="tab" data-tab="positions">Positions</div>
        <div class="tab" data-tab="txlog">Settlements</div>
        <div class="tab" data-tab="activity">Activity</div>
        <div class="tab" data-tab="markets">Markets</div>
    </div>
    <div class="content">
        <div id="posTab" class="tab-content hidden">
            <div class="pos-last-refresh">Auto-refreshes every 3s &nbsp;·&nbsp; Last update: <span id="posRefreshTime">—</span></div>
            <div class="pos-summary">
                <div class="card"><div class="card-label">Open Positions</div><div class="card-value" id="posCount2">—</div></div>
                <div class="card"><div class="card-label">Winning / Losing</div><div class="card-value" id="posWinLoss">—</div><div class="card-sub" id="posFlat"></div></div>
                <div class="card"><div class="card-label">Total Deployed</div><div class="card-value yellow" id="posTotalCost">—</div></div>
                <div class="card"><div class="card-label">Unrealized P&L</div><div class="card-value" id="posTotalPnl">—</div></div>
            </div>
            <div class="pos-bar" id="posBar"></div>
            <div class="pos-table-wrap">
                <table class="pos-table">
                    <thead>
                        <tr>
                            <th>Market</th>
                            <th>Side</th>
                            <th>Contracts</th>
                            <th>Entry ¢</th>
                            <th>Current ¢</th>
                            <th>Move</th>
                            <th>Deployed</th>
                            <th>Value</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody id="posTableBody">
                        <tr><td colspan="11" class="pos-empty">Loading positions…</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

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
                <div class="card"><div class="card-label">Total vs <span id="depositsLabel">$150</span> deposited</div><div class="card-value" id="totalReturn">$0.00</div><div class="card-sub" id="returnPctSub">0%</div></div>
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
                <div class="card"><div class="card-label">Horizon Window</div><div class="card-value" id="cfgMaxDays">—</div><div class="card-sub" id="cfgDaysSub">must resolve within X days</div></div>
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

            <div class="section-title" style="margin-top:20px">Filter Activity
                <span id="filterBadge" style="float:right;background:#30363d;padding:2px 8px;border-radius:10px;font-size:11px;">loading…</span>
            </div>
            <div id="filterActivity" style="font-size:12px;color:#8b949e;padding:8px 0;">Loading filter data…</div>

            <div class="section-title" style="margin-top:20px">Nightly Strategy Report
                <span style="float:right;font-size:10px;color:#6e7681;" id="nightlyTs"></span>
            </div>
            <div id="nightlyReport" style="font-size:12px;color:#8b949e;padding:8px 0;">Runs at midnight UTC · <a href="/api/nightly?force=true" target="_blank" style="color:#58a6ff;">run now ↗</a></div>
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
        let positionsInterval = null;
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
                const tabId = tab.dataset.tab === 'positions' ? 'posTab' : tab.dataset.tab;
                document.getElementById(tabId).classList.remove('hidden');
                if (tab.dataset.tab === 'txlog') fetchSettlements();
                if (tab.dataset.tab === 'positions') {
                    fetchPositions();
                    if (!positionsInterval) positionsInterval = setInterval(fetchPositions, 3000);
                } else {
                    if (positionsInterval) { clearInterval(positionsInterval); positionsInterval = null; }
                }
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

                // Debounce: only trigger a performance refresh once per 3s of WS activity.
                // Without this, a burst of fill events fires N parallel fetchPerformance()
                // calls, which can complete out-of-order and overwrite newer DOM values with stale ones.
                if (!ws._perfDebounce) {
                    ws._perfDebounce = setTimeout(() => {
                        ws._perfDebounce = null;
                        fetchPerformance();
                    }, 3000);
                }
            };
        }
        
        // Escape HTML special characters before inserting user-supplied strings into innerHTML.
        function esc(str) {
            return String(str || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        }

        function updateStats(s) {
            // Legacy stats (At Risk section).
            // available and total_value are 0 when Kalshi hasn't synced yet — show '--'
            // instead of '$0.00' so the user knows data is loading, not that they have $0.
            document.getElementById('available').textContent = s.kalshi_synced ? '$' + s.available.toFixed(2) : '--';
            document.getElementById('atRisk').textContent = '$' + s.at_risk.toFixed(2);
            document.getElementById('posCount').textContent = s.open_positions + ' positions + ' + (s.pending_orders || 0) + ' resting';
            document.getElementById('positionsAtRisk').textContent = '$' + (s.positions_at_risk || 0).toFixed(2);
            document.getElementById('filledCount').textContent = s.open_positions + ' filled';
            document.getElementById('pendingAtRisk').textContent = '$' + (s.pending_at_risk || 0).toFixed(2);
            document.getElementById('restingCount').textContent = (s.pending_orders || 0) + ' resting';
            document.getElementById('atRiskUltra').textContent = '$' + (s.at_risk_ultra_short || 0).toFixed(2);
            document.getElementById('atRiskShort').textContent = '$' + (s.at_risk_short || 0).toFixed(2);
            document.getElementById('atRiskMedium').textContent = '$' + (s.at_risk_medium || 0).toFixed(2);
            document.getElementById('totalValue').textContent = s.kalshi_synced ? '$' + s.total_value.toFixed(2) : '--';
            
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
            const fDaysSub = document.getElementById('cfgDaysSub');
            if (fDaysSub && s.min_days_to_resolution != null) fDaysSub.textContent = s.min_days_to_resolution + 'd min · ' + s.max_days_to_resolution + 'd max';
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
                ('drawdown ' + (s.daily_drawdown != null ? s.daily_drawdown.toFixed(1) + '%' : '0%'));
        }
        
        // Fetch performance first so dashboard always shows Kalshi numbers; then state + signals for readout
        async function fetchPerformance() {
            try {
                const perfRes = await fetch('/api/performance');
                const p = await perfRes.json();
                if (p.error) {
                    if (p.error === 'syncing') {
                        // Bot just started — sync not complete yet. Retry in 5s.
                        document.getElementById('accountValue') && (document.getElementById('accountValue').textContent = 'Syncing…');
                        document.getElementById('totalReturn') && (document.getElementById('totalReturn').textContent = '—');
                        setTimeout(fetchPerformance, 5000);
                    } else {
                        console.error('Performance API error:', p.error);
                    }
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
                document.getElementById('returnPctSub').textContent = (returnPctVal >= 0 ? '+' : '') + returnPctVal.toFixed(1) + '%';
                document.getElementById('returnPctSub').className = 'card-sub ' + (returnPctVal >= 0 ? 'green' : 'red');
                // Also update the Account section "Return" card (same value, different element)
                document.getElementById('returnPct').textContent = (returnPctVal >= 0 ? '+' : '') + returnPctVal.toFixed(1) + '%';
                document.getElementById('returnPct').className = 'card-value ' + (returnPctVal >= 0 ? 'green' : 'red');
                
                // Today's change = account value now - start of day (from Kalshi)
                const todayPn  = p.performance.today_pnl;
                const todayPct = p.performance.today_pnl_pct;
                const unrealPn = p.performance.unrealized_pnl;
                const realPn   = p.performance.realized_pnl_today;
                const todayEl    = document.getElementById('todayPnl');
                const todaySubEl = document.getElementById('todayPnlSub');
                if (todayPn != null && todayPct != null) {
                    todayEl.textContent = (todayPn >= 0 ? '+' : '') + '$' + todayPn.toFixed(2);
                    todayEl.className = 'card-value ' + (todayPn >= 0 ? 'green' : 'red');
                    // Sub-line shows breakdown so user can see why today ≠ positions P&L
                    let breakdownParts = [];
                    if (unrealPn != null) {
                        const uSign = unrealPn >= 0 ? '+' : '';
                        breakdownParts.push(`open: ${uSign}$${unrealPn.toFixed(2)}`);
                    }
                    if (realPn != null && realPn !== 0) {
                        const rSign = realPn >= 0 ? '+' : '';
                        breakdownParts.push(`closed: ${rSign}$${realPn.toFixed(2)}`);
                    }
                    const pctStr = (todayPct >= 0 ? '+' : '') + todayPct.toFixed(1) + '%';
                    todaySubEl.textContent = breakdownParts.length
                        ? pctStr + ' · ' + breakdownParts.join(' · ')
                        : pctStr + ' vs start of day';
                    todaySubEl.className = 'card-sub';
                } else {
                    todayEl.textContent = '—';
                    todayEl.className = 'card-value';
                    todaySubEl.textContent = 'vs start of day (set tomorrow)';
                }
                
                document.getElementById('positionStatus').textContent = p.positions.winning + ' winning / ' + p.positions.losing + ' losing';
                document.getElementById('positionStatusSub').textContent = 'at bid (exit) price';
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
            const html = positions.map(p => {
                const displayQ = p.question || p.market_id || '—';
                return `
                <div class="position">
                    <div class="position-header">
                        <div class="position-title">${esc(displayQ)}</div>
                        <span class="position-side ${p.side.toLowerCase()}">${p.side}</span>
                    </div>
                    <div class="position-details">
                        <div><div class="position-detail-label">ENTRY</div><div class="position-detail-value">${(p.entry_price*100).toFixed(0)}¢</div></div>
                        <div><div class="position-detail-label">CURRENT</div><div class="position-detail-value">${(p.current_price*100).toFixed(0)}¢</div></div>
                        <div><div class="position-detail-label">SIZE</div><div class="position-detail-value">$${p.size.toFixed(2)}</div></div>
                        <div><div class="position-detail-label">P&L</div><div class="position-detail-value ${p.unrealized_pnl >= 0 ? 'green' : 'red'}">${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)}</div></div>
                    </div>
                </div>
            `;
            }).join('');
            document.getElementById('positions').innerHTML = html || '<div style="color:#8b949e;font-size:13px;padding:12px;">No open positions</div>';
        }
        
        function renderTrades(trades) {
            document.getElementById('tradeCount').textContent = trades.length;
            const html = trades.slice(0, 10).map(t => {
                const isExit = t.action === 'EXIT';
                const isLoss = isExit && t.pnl < 0;
                // Normalize action class: ORDER_PLACED should use 'entry' styling
                const actionClass = t.action === 'ORDER_PLACED' ? 'entry' : t.action.toLowerCase();
                // EXIT trades store entry_price/exit_price, not price; use exit_price for display
                const displayPrice = (t.exit_price != null ? t.exit_price : (t.price != null ? t.price : (t.entry_price || 0)));
                return `
                    <div class="trade ${actionClass} ${isLoss ? 'loss' : ''}">
                        <div class="trade-info">
                            <div>${esc(t.question)}</div>
                            <div style="color:#8b949e;font-size:11px;">${t.side} @ ${(displayPrice*100).toFixed(0)}¢ · $${t.size.toFixed(2)} · ${new Date(t.timestamp).toLocaleTimeString()}${t.reason ? ' · ' + t.reason : ''}</div>
                        </div>
                        ${isExit ? `<div class="${t.pnl >= 0 ? 'green' : 'red'}">${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}</div>` : ''}
                        <span class="trade-action ${actionClass}">${t.action}</span>
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
                // Use server-supplied effective threshold when present (high-edge bypass lowers it).
                // Without this, UI shows "need 72%" but the bot trades at 65% — looks like a bug.
                const effConf   = a.conf_threshold_effective != null ? a.conf_threshold_effective : threshConf;
                const edgePass  = edge >= threshEdge;
                const confPass  = conf == null || conf >= effConf;
                const edgeCol   = edgePass ? '#3fb950' : '#f85149';
                const confCol   = confPass ? '#3fb950' : '#f85149';
                // Show "65%" when bypass active, "72%" otherwise
                const confNeedPct = (effConf * 100).toFixed(0);
                // Label the bypass so the user knows why threshold is lower
                const bypassLabel = (effConf < threshConf) ? ` <span style="color:#e3b341;font-size:10px;">(high-edge bypass)</span>` : '';

                // Rejection reason — humanised.
                // Note: edge ✓ + conf ✓ + SKIP means the trade qualified on signals but was
                // blocked by a risk/portfolio rule (cluster cap, daily loss, position limit…).
                const rawReason = a.reason || '';
                let rejLabel = '';
                if (!isTrade && rawReason) {
                    const parts = [];
                    if (rawReason.includes('LOW_EDGE'))           parts.push(`Edge ${(edge*100).toFixed(1)}% < ${(threshEdge*100).toFixed(0)}% min`);
                    if (rawReason.includes('LOW_CONFIDENCE'))     parts.push(`Conf ${conf != null ? (conf*100).toFixed(0)+'%' : '?'} < ${confNeedPct}% min`);
                    if (rawReason.includes('YES_BETS_DISABLED'))  parts.push('YES side disabled');
                    if (rawReason.includes('NO_BETS_DISABLED'))   parts.push('NO side disabled');
                    if (rawReason.includes('MAX_POSITIONS'))      parts.push('Position limit reached');
                    if (rawReason.includes('CLUSTER_CAP'))        parts.push('Cluster cap hit (too many correlated bets)');
                    if (rawReason.includes('RECENTLY_EXITED'))    parts.push('Cooldown after recent exit');
                    if (rawReason.includes('ALREADY_IN'))         parts.push('Already holding this market');
                    if (rawReason.includes('DAILY_LOSS_LIMIT'))   {
                        const m = rawReason.match(/DAILY_LOSS_LIMIT_([\-\d.]+)pct/);
                        parts.push('Daily loss limit hit' + (m ? ` (${m[1]}% today)` : ''));
                    }
                    if (rawReason.includes('DOGE_BLOCKED'))       parts.push('DOGE blocked (0% win rate historically)');
                    if (rawReason.includes('INVERTED_BET_BLOCKED')) parts.push('Inverted range bet blocked');
                    if (rawReason.includes('EXPOSURE_LIMIT'))     parts.push('Exposure limit reached');
                    if (rawReason.includes('KILL_SWITCH'))        parts.push('Kill-switch active — trading halted');
                    if (rawReason === 'CRITERIA_MET')             { /* should_trade=true, handled by isTrade */ }
                    if (parts.length === 0)                       parts.push(rawReason.slice(0, 80));
                    rejLabel = parts.join(' · ');
                }

                return `
                <div class="analysis">
                    <div class="analysis-header">
                        <div style="flex:1;font-size:13px;">${esc(a.question)}</div>
                        <span class="analysis-decision ${isTrade ? 'trade' : 'skip'}">${isTrade ? '✓ TRADE' : '✗ SKIP'}</span>
                    </div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:6px 0 4px 0;">
                        <span>AI: <strong>${a.ai_probability != null ? (a.ai_probability*100).toFixed(0)+'%' : '—'}</strong></span>
                        <span>Market: <strong>${(a.market_price*100).toFixed(0)}¢</strong></span>
                        <span style="color:${edgeCol};">Edge: <strong>${edge >= 0 ? '+' : ''}${(edge*100).toFixed(1)}%</strong> ${edgePass ? '✓' : '✗ need '+((threshEdge)*100).toFixed(0)+'%'}</span>
                        ${conf != null ? `<span style="color:${confCol};">Conf: <strong>${(conf*100).toFixed(0)}%</strong> ${confPass ? '✓' : `✗ need ${confNeedPct}%`}${bypassLabel}</span>` : ''}
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

        // REST fallback for main stats — fires every 5 s regardless of WS status.
        // The WS is the real-time layer; this baseline poll ensures the stats section
        // never stays at $0.00 just because Railway's proxy is slow to upgrade the
        // WebSocket connection (or the connection dropped between reconnect attempts).
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                if (!res.ok) return;
                const d = await res.json();
                if (d.stats) {
                    updateStats(d.stats);
                    renderPositions(d.positions || []);
                    renderTrades(d.trades || []);
                    renderAnalyses(d.analyses || []);
                    renderMarkets(d.monitored || []);
                }
            } catch(e) { /* ignore — WS may already be handling it */ }
        }
        fetchState();
        setInterval(fetchState, 5000);

        // Initial performance fetch and regular updates
        fetchPerformance();
        fetchSettlements();
        fetchFilters();
        setInterval(fetchPerformance, 15000); // Update every 15 seconds
        setInterval(fetchSettlements, 30000); // Update settlements every 30 seconds
        setInterval(fetchFilters, 60000);     // Update filter/nightly data every 60 seconds

        async function fetchFilters() {
            try {
                const res = await fetch('/api/filters');
                if (!res.ok) return;
                const d = await res.json();

                // Filter Activity badge + breakdown
                const s24 = d.summary && d.summary.last_24h ? d.summary.last_24h : {};
                const total24 = Object.values(s24).reduce((a,b)=>a+b,0);
                const badge = document.getElementById('filterBadge');
                if (badge) badge.textContent = total24 + ' filtered (24h)';

                const fa = document.getElementById('filterActivity');
                if (fa) {
                    if (total24 === 0) {
                        fa.innerHTML = '<span style="color:#6e7681;">No markets filtered yet — bot may still be starting up.</span>';
                    } else {
                        const rows = Object.entries(s24)
                            .sort((a,b)=>b[1]-a[1])
                            .map(([k,v]) => `<span style="display:inline-block;margin:3px 6px 3px 0;padding:2px 8px;background:#161b22;border:1px solid #30363d;border-radius:10px;color:#c9d1d9;">${k} <b style="color:#58a6ff;">${v}</b></span>`)
                            .join('');
                        fa.innerHTML = rows;
                    }
                }

                // Nightly Report
                const nr = d.nightly_report;
                const nrEl = document.getElementById('nightlyReport');
                const nrTs = document.getElementById('nightlyTs');
                if (nrEl && nr && nr.generated_at) {
                    if (nrTs) nrTs.textContent = new Date(nr.generated_at).toLocaleString();
                    const cats = nr.category_stats || {};
                    const catRows = Object.entries(cats)
                        .sort((a,b) => b[1].pnl - a[1].pnl)
                        .map(([k,v]) => {
                            const tot = v.wins + v.losses;
                            const wr = tot > 0 ? Math.round(v.win_rate*100) : 0;
                            const col = v.pnl >= 0 ? '#3fb950' : '#f85149';
                            return `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d;">
                                <span style="color:#c9d1d9;">${k}</span>
                                <span>${v.wins}W/${v.losses}L &nbsp; ${wr}% WR &nbsp; <b style="color:${col};">${v.pnl >= 0 ? '+' : ''}$${v.pnl.toFixed(2)}</b></span>
                            </div>`;
                        }).join('');
                    const pnl7 = nr.pnl_7d || 0;
                    const pnlCol = pnl7 >= 0 ? '#3fb950' : '#f85149';
                    // Market gaps section
                    const gapsRaw = nr.market_gaps;  // undefined = report not yet run; [] = no gaps found
                    let gapsHtml = '';
                    if (gapsRaw && gapsRaw.length > 0) {
                        const gapRows = gapsRaw.slice(0,8).map(g => {
                            const bid = Math.round((g.yes_bid||0)*100);
                            const oi = (g.open_interest||0).toLocaleString();
                            const q = (g.example_question||'').replace(/[<>&]/g,'');
                            return `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d;">
                                <span style="color:#e3b341;font-weight:600;">${g.series}</span>
                                <span style="color:#8b949e;font-size:11px;flex:1;margin:0 8px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">${q}</span>
                                <span style="color:#c9d1d9;white-space:nowrap;">bid=${bid}¢ oi=${oi}</span>
                            </div>`;
                        }).join('');
                        gapsHtml = `<div style="margin-top:12px;padding:8px 10px;background:rgba(227,179,65,0.07);border:1px solid rgba(227,179,65,0.3);border-radius:6px;">
                            <div style="font-size:10px;letter-spacing:1px;color:#e3b341;margin-bottom:6px;">⚠ MARKET GAPS — LIQUID SERIES NOT YET ANALYSED (${gapsRaw.length})</div>
                            ${gapRows}
                        </div>`;
                    } else if (Array.isArray(gapsRaw)) {
                        // market_gaps field exists but is empty array → report ran, no gaps
                        gapsHtml = `<div style="margin-top:8px;font-size:11px;color:#3fb950;">✓ All liquid series covered — no gaps detected.</div>`;
                    } else {
                        // market_gaps field absent → nightly report hasn't run yet today
                        gapsHtml = `<div style="margin-top:8px;font-size:11px;color:#6e7681;">Gap check runs at midnight UTC — <a href="/api/nightly?force=true" target="_blank" style="color:#58a6ff;">run now ↗</a></div>`;
                    }
                    nrEl.innerHTML = `<div style="margin-bottom:8px;color:#c9d1d9;">7-day P&L: <b style="color:${pnlCol};">${pnl7 >= 0 ? '+' : ''}$${pnl7.toFixed(2)}</b> &nbsp;|&nbsp; 30d settled: ${nr.total_settled_30d || 0} &nbsp;|&nbsp; <a href="/api/nightly?force=true" target="_blank" style="color:#58a6ff;font-size:11px;">run now ↗</a></div>${catRows || '<span style="color:#6e7681;">No settled trades in 30 days yet.</span>'}${gapsHtml}`;
                } else if (nrEl) {
                    nrEl.innerHTML = 'Runs at midnight UTC &nbsp;·&nbsp; <a href="/api/nightly?force=true" target="_blank" style="color:#58a6ff;">run now ↗</a>';
                }
            } catch(e) { console.warn('fetchFilters error', e); }
        }

        // ── Positions page ────────────────────────────────────────────────────
        function fmtCents(v) { return Math.round(v * 100) + '¢'; }
        function fmtDollar(v) {
            const s = (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2);
            return s;
        }
        function pnlClass(v) { return v > 0.005 ? 'pnl-pos' : (v < -0.005 ? 'pnl-neg' : 'pnl-flat'); }

        async function fetchPositions() {
            try {
                const r = await fetch('/api/positions');
                const d = await r.json();
                if (d.error) { return; }

                const s = d.summary;
                document.getElementById('posCount2').textContent = s.count;

                const wl = document.getElementById('posWinLoss');
                wl.innerHTML = `<span class="pnl-pos">${s.winning}W</span> / <span class="pnl-neg">${s.losing}L</span>`;
                const flat = document.getElementById('posFlat');
                flat.textContent = s.flat > 0 ? `${s.flat} flat` : 'vs entry price';

                document.getElementById('posTotalCost').textContent = '$' + s.total_cost.toFixed(2);
                const pnlEl = document.getElementById('posTotalPnl');
                pnlEl.textContent = (s.total_unrealized >= 0 ? '+' : '') + '$' + s.total_unrealized.toFixed(2);
                pnlEl.className = 'card-value ' + pnlClass(s.total_unrealized);

                // Win/loss bar
                const bar = document.getElementById('posBar');
                if (s.count > 0) {
                    const wp = (s.winning / s.count * 100).toFixed(1);
                    const lp = (s.losing  / s.count * 100).toFixed(1);
                    const fp = (s.flat    / s.count * 100).toFixed(1);
                    bar.innerHTML = `<div class="pos-bar-win" style="width:${wp}%"></div><div class="pos-bar-loss" style="width:${lp}%"></div><div class="pos-bar-flat" style="width:${fp}%"></div>`;
                } else {
                    bar.innerHTML = '';
                }

                // Table rows — sorted losing-first (worst P&L at top) so problems are obvious
                const tbody = document.getElementById('posTableBody');
                if (d.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="11" class="pos-empty">No open positions</td></tr>';
                } else {
                    const rows = d.positions.map(p => {
                        const move = p.current_price - p.entry_price;
                        const moveClass = pnlClass(move);
                        const moveTxt = (move >= 0 ? '+' : '') + Math.round(move * 100) + '¢';
                        const pnlTxt  = (p.unrealized_pnl >= 0 ? '+$' : '-$') + Math.abs(p.unrealized_pnl).toFixed(2);
                        const pnlPct  = (p.unrealized_pct >= 0 ? '+' : '') + p.unrealized_pct.toFixed(1) + '%';
                        const tickerShort = p.ticker.length > 22 ? p.ticker.slice(0, 22) + '…' : p.ticker;
                        const entryDate = p.entry_time ? new Date(p.entry_time).toLocaleDateString('en-US', {month:'short', day:'numeric'}) : '—';
                        // Always show something in the question cell — fall back to ticker if empty
                        const displayQuestion = p.question || p.ticker;
                        return `<tr>
                            <td class="pos-question">
                                <div>${displayQuestion}</div>
                                <div class="pos-ticker">${tickerShort} · entered ${entryDate}</div>
                            </td>
                            <td><span class="side-badge ${p.side}">${p.side}</span></td>
                            <td>${p.contracts}</td>
                            <td>${fmtCents(p.entry_price)}</td>
                            <td>${fmtCents(p.current_price)}</td>
                            <td class="${moveClass}">${moveTxt}</td>
                            <td>$${p.cost.toFixed(2)}</td>
                            <td>$${p.value.toFixed(2)}</td>
                            <td class="${pnlClass(p.unrealized_pnl)}">${pnlTxt}</td>
                            <td class="${pnlClass(p.unrealized_pct)}">${pnlPct}</td>
                            <td><button class="exit-btn" onclick="forceExit('${p.ticker}', this)">Exit</button></td>
                        </tr>`;
                    }).join('');

                    const totalCost  = d.summary.total_cost;
                    const totalPnl   = d.summary.total_unrealized;
                    const totalValue = d.positions.reduce((a, p) => a + p.value, 0);
                    const totPnlTxt  = (totalPnl >= 0 ? '+$' : '-$') + Math.abs(totalPnl).toFixed(2);
                    const totPnlPct  = totalCost > 0 ? ((totalPnl / totalCost) * 100).toFixed(1) : '0.0';
                    const totClass   = pnlClass(totalPnl);
                    const totalRow   = `<tr style="border-top:2px solid #30363d;font-weight:600;background:#161b22;">
                        <td colspan="6" style="color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">Total (${d.positions.length} positions)</td>
                        <td>$${totalCost.toFixed(2)}</td>
                        <td>$${totalValue.toFixed(2)}</td>
                        <td class="${totClass}">${totPnlTxt}</td>
                        <td class="${totClass}">${totPnlPct >= 0 ? '+' : ''}${totPnlPct}%</td>
                        <td></td>
                    </tr>`;

                    tbody.innerHTML = rows + totalRow;
                }

                const ts = new Date(d.timestamp);
                document.getElementById('posRefreshTime').textContent = ts.toLocaleTimeString();
            } catch(e) { console.warn('fetchPositions error', e); }
        }

        async function forceExit(marketId, btn) {
            if (!confirm('Force-exit ' + marketId + '?')) return;
            btn.textContent = '…';
            btn.disabled = true;
            try {
                const r = await fetch('/api/force-exit?market_id=' + encodeURIComponent(marketId));
                const d = await r.json();
                if (d.ok) {
                    btn.textContent = 'Done';
                    btn.style.color = '#3fb950';
                    btn.style.borderColor = '#3fb950';
                    setTimeout(fetchPositions, 1500);
                } else if (r.status === 409) {
                    // Exit order already in flight — show pending state
                    btn.textContent = 'Pending';
                    btn.style.color = '#e3b341';
                    btn.style.borderColor = '#e3b341';
                    btn.disabled = false;
                    alert('An exit order is already pending for ' + marketId + '. Waiting for fill.');
                } else {
                    btn.textContent = 'Exit';
                    btn.disabled = false;
                    alert('Exit failed: ' + (d.error || JSON.stringify(d)));
                }
            } catch(e) {
                btn.textContent = 'Exit';
                btn.disabled = false;
                alert('Exit error: ' + e);
            }
        }

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
