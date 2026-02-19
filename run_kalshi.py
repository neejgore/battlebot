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
        self.min_edge = float(os.getenv('MIN_EDGE', 0.03))
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
        
        # Trading state (persisted)
        self._state_file = f"{self._storage_dir}/kalshi_state.json"
        self._positions: dict[str, dict] = {}
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
            max_positions=5,
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
        
        # Start background tasks
        asyncio.create_task(self._market_loop())
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._price_refresh_loop())
        
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
        """Fetch open markets from Kalshi."""
        try:
            result = await self._kalshi.get_markets(status='open', limit=100)
            markets = result.get('markets', [])
            
            for m in markets:
                try:
                    market = parse_kalshi_market(m)
                    if market['id']:
                        self._markets[market['id']] = market
                except Exception as e:
                    continue
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(self._markets)} Kalshi markets")
        except Exception as e:
            print(f"[Kalshi API Error] {e}")
    
    async def _select_markets(self):
        """Filter markets using eligibility criteria."""
        eligible = []
        rejection_counts = {
            'no_end_date': 0, 'too_far_out': 0, 'low_volume': 0,
            'wide_spread': 0, 'extreme_price': 0, 'low_liquidity': 0,
        }
        
        for m in self._markets.values():
            # Must have end date
            if not m.get('end_date'):
                rejection_counts['no_end_date'] += 1
                continue
            
            # Time to resolution check
            try:
                end_date = datetime.fromisoformat(m['end_date'].replace('Z', '+00:00'))
                days_to_resolution = (end_date.replace(tzinfo=None) - datetime.utcnow()).days
                if days_to_resolution > 30:
                    rejection_counts['too_far_out'] += 1
                    continue
                if days_to_resolution < 0:
                    continue
            except:
                pass
            
            # Spread check (≤4 cents)
            if m.get('spread', 0.04) > 0.04:
                rejection_counts['wide_spread'] += 1
                continue
            
            # Price range (not too extreme)
            price = m.get('price', 0.5)
            if price < 0.05 or price > 0.95:
                rejection_counts['extreme_price'] += 1
                continue
            
            eligible.append(m)
        
        # Take top 15 by volume
        eligible.sort(key=lambda x: x.get('volume_24h', 0), reverse=True)
        self._monitored = {m['id']: m for m in eligible[:15]}
        
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
        
        # Determine trade side and edge
        if adjusted_prob > current_price:
            side = 'YES'
            edge = adjusted_prob - current_price
            trade_prob = adjusted_prob
            trade_price = current_price
        else:
            side = 'NO'
            edge = (1 - adjusted_prob) - (1 - current_price)
            trade_prob = 1 - adjusted_prob
            trade_price = 1 - current_price
        
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
            position_size = await self._risk_engine.calculate_position_size(
                adjusted_prob=trade_prob,
                market_price=trade_price,
                edge=edge,
                confidence=signal.confidence,
                market_id=market_id,
            )
            
            if position_size > 0:
                await self._enter_position(market, side, position_size, adjusted_prob, edge, signal.confidence)
                print(f"[AI] {question}... | ✓ TRADE | Edge: +{edge*100:.1f}% | Conf: {signal.confidence*100:.0f}%")
            else:
                print(f"[AI] {question}... | ✗ SKIP | Size: $0")
        else:
            print(f"[AI] {question}... | ✗ SKIP | Edge: +{edge*100:.1f}%")
        
        await self._broadcast_update()
    
    async def _enter_position(self, market: dict, side: str, size: float, prob: float, edge: float, confidence: float):
        """Enter a new position."""
        market_id = market['id']
        pos_id = f"pos_{int(datetime.utcnow().timestamp()*1000)}"
        
        pos = {
            'id': pos_id,
            'market_id': market_id,
            'question': market.get('question', ''),
            'side': side,
            'size': size,
            'entry_price': market['price'],
            'current_price': market['price'],
            'ai_probability': prob,
            'edge': edge,
            'confidence': confidence,
            'entry_time': datetime.utcnow().isoformat(),
            'unrealized_pnl': 0.0,
        }
        
        # Log to database
        if self._db_connected:
            try:
                db_trade_id = await self._db.log_trade_entry(
                    decision_id=0,  # We don't track this linking yet
                    market_id=market_id,
                    token_id=market.get('token_id', market_id),
                    entry_price=market['price'],
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
        
        # Record trade
        trade = {
            'id': pos_id,
            'market_id': market_id,
            'question': market.get('question', ''),
            'action': 'ENTRY',
            'side': side,
            'price': market['price'],
            'size': size,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.insert(0, trade)
        
        mode = "[DRY RUN]" if self.dry_run else "[LIVE]"
        print(f"{mode} ENTERED: {side} ${size:.2f} @ {int(market['price']*100)}¢ | {market.get('question', '')[:50]}...")
        
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
                    
                    current_price = market['price']
                    entry_price = pos['entry_price']
                    side = pos['side']
                    
                    # Simulate price movement if enabled
                    if self.simulate_prices:
                        drift = random.gauss(0, 0.02)
                        simulated_price = max(0.05, min(0.95, entry_price + drift))
                        current_price = simulated_price
                        market['price'] = simulated_price
                    
                    # Calculate P&L
                    if side == 'YES':
                        price_change = current_price - entry_price
                    else:
                        price_change = entry_price - current_price
                    
                    unrealized_pnl = price_change * pos['size']
                    pos['current_price'] = current_price
                    pos['unrealized_pnl'] = unrealized_pnl
                    
                    # Exit conditions
                    profit_take_pct = self._risk_limits.profit_take_pct
                    stop_loss_pct = self._risk_limits.stop_loss_pct
                    
                    profit_target = pos['size'] * profit_take_pct
                    stop_loss = -pos['size'] * stop_loss_pct
                    
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
        position = self._positions.pop(pos_id, None)
        if not position:
            return
        
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
        
        trade = {
            'id': pos_id,
            'market_id': position['market_id'],
            'question': position.get('question', ''),
            'action': 'EXIT',
            'side': position['side'],
            'price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._trades.insert(0, trade)
        
        mode = "[DRY RUN]" if self.dry_run else "[LIVE]"
        print(f"{mode} EXITED: ${pnl:+.2f} ({reason}) | {position.get('question', '')[:50]}...")
        
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
    
    async def _broadcast_update(self):
        """Send update to all connected WebSocket clients."""
        if not self._websockets:
            return
        
        data = json.dumps({
            'type': 'update',
            'stats': self._get_stats(),
            'positions': list(self._positions.values()),
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
