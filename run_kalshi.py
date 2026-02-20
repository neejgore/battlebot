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
        self.initial_bankroll = float(os.getenv('INITIAL_BANKROLL', 1000))
        self.min_edge = float(os.getenv('MIN_EDGE', 0.10))  # 10% min edge - only trade with real edge
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.70))  # 70% confidence required
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 15))  # $15 max - bigger bets on high confidence
        self.kelly_fraction = float(os.getenv('FRACTIONAL_KELLY', 0.1))
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
        # Actual Kalshi values from API (synced every 5 min)
        self._kalshi_cash = None       # Available cash
        self._kalshi_portfolio = None  # Current positions value
        self._kalshi_total = None      # Total portfolio value
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
        
        # Risk Engine - selective but confident
        self._risk_limits = RiskLimits(
            max_daily_drawdown=0.15,
            max_position_size=self.max_position_size,  # $15 max
            max_percent_bankroll_per_market=0.15,  # 15% per market max
            max_total_open_risk=0.90,  # 90% max exposure - 10% reserve
            max_positions=10,  # Max 10 positions - focus, don't spread thin
            profit_take_pct=0.20,  # 20% profit target - let winners run
            stop_loss_pct=0.15,  # 15% stop loss - give room but limit damage
            time_stop_hours=720,  # 30 days - let bets ride to settlement
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
            phantom_count = 0
            for pos_id in list(self._positions.keys()):
                pos = self._positions[pos_id]
                market_id = pos.get('market_id', '')
                
                if market_id not in kalshi_by_ticker:
                    print(f"[Sync] Removing phantom position: {pos.get('question', '')[:40]}...")
                    self._positions.pop(pos_id)
                    phantom_count += 1
            
            if phantom_count:
                print(f"[Sync] Removed {phantom_count} phantom positions")
            
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
            
            if added_count or updated_count or phantom_count:
                self._save_state()
                print(f"[Sync] Complete: {added_count} added, {updated_count} updated, {phantom_count} removed")
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
        lines = ["=== NEWS INTELLIGENCE EFFECTIVENESS ==="]
        
        if with_news['trades'] > 0:
            lines.append(f"\nWITH NEWS ({with_news['trades']} trades):")
            lines.append(f"  Win Rate: {with_news['win_rate']*100:.0f}% ({with_news['wins']}W/{with_news['losses']}L)")
            lines.append(f"  P&L: ${with_news['pnl']:+.2f}")
        else:
            lines.append("\nWITH NEWS: No trades yet")
        
        if without_news['trades'] > 0:
            lines.append(f"\nWITHOUT NEWS ({without_news['trades']} trades):")
            lines.append(f"  Win Rate: {without_news['win_rate']*100:.0f}% ({without_news['wins']}W/{without_news['losses']}L)")
            lines.append(f"  P&L: ${without_news['pnl']:+.2f}")
        else:
            lines.append("\nWITHOUT NEWS: No trades yet")
        
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
            brave_key = self._intelligence._news_service._brave_api_key
            if brave_key:
                print(f"  News Source: BRAVE SEARCH API (configured)")
            else:
                print(f"  News Source: Google News RSS (fallback)")
        except:
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
        
        max_days = int(os.getenv('MAX_DAYS_TO_RESOLUTION', '365'))
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
        # Balance: inefficiency × 0.6 + normalized_oi × 0.4
        def market_score(m):
            ineff = m.get('inefficiency_score', 0) * 0.6
            # Normalize OI to 0-1 scale (cap at 1000)
            oi = min(m.get('open_interest', 0) / 1000, 1.0) * 0.4
            return ineff + oi
        
        if self._prefer_inefficient:
            ultra_short.sort(key=market_score, reverse=True)
            short_term.sort(key=market_score, reverse=True)
            medium_term.sort(key=market_score, reverse=True)
        else:
            # Original behavior: sort by OI only
            ultra_short.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
            short_term.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
            medium_term.sort(key=lambda x: x.get('open_interest', 0) or 0, reverse=True)
        
        # PRIORITIZE: Ultra-short first, then short, then medium
        selected = ultra_short[:8] + short_term[:5] + medium_term[:2]
        
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
                    # Weight: time matters most, but boost high-news-value markets
                    # Lower score = higher priority
                    time_score = min(hours, 168) / 168  # Normalize to 0-1 (cap at 1 week)
                    return time_score - (news_score * 0.3)  # News relevance reduces score
                
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
                        
                    if len(self._positions) >= self._risk_limits.max_positions:
                        break
                    
                    # Check cooldown and price movement
                    last = self._last_analysis.get(market_id)
                    last_price = self._last_analysis_price.get(market_id, 0)
                    current_price = market.get('price', 0.5)
                    
                    price_moved = abs(current_price - last_price) >= self._price_change_threshold if last_price else False
                    
                    # Shorter cooldown for ultra-short markets (analyze more frequently)
                    hours_to_res = market.get('hours_to_resolution', 9999)
                    cooldown = 300 if hours_to_res <= 24 else self._analysis_cooldown  # 5 min for ultra-short
                    
                    if last and not price_moved:
                        if (datetime.utcnow() - last).total_seconds() < cooldown:
                            continue
                    
                    # Log time horizon and news relevance for visibility
                    news_relevance = NewsService.score_news_relevance(
                        market.get('question', ''),
                        market.get('category')
                    )
                    time_label = "ULTRA-SHORT" if hours_to_res <= 24 else "SHORT" if hours_to_res <= 168 else "MEDIUM"
                    if hours_to_res <= 168 or news_relevance > 0.6:  # Log short-term or high news value
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
                        news_stats = self._intelligence._news_service.get_usage_stats()
                        intel_eff = self._get_intel_effectiveness()
                        print(f"\n[INTEL STATUS] Brave searches: {news_stats.get('brave_searches', 0)}, Cache size: {news_stats.get('cache_size', 0)}")
                        if intel_eff.get('with_news', {}).get('trades', 0) > 0 or intel_eff.get('without_news', {}).get('trades', 0) > 0:
                            print(intel_eff['summary'])
                    except Exception as e:
                        print(f"[INTEL STATUS] Error: {e}")
                    
            except Exception as e:
                print(f"[Trading Error] {e}")
            await asyncio.sleep(30)
    
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
        
        # Step 3: Get AI signal with intelligence data and historical performance
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
        MAX_CONTRACTS_PER_ORDER = 10  # Max 10 contracts per order
        MIN_PRICE_CENTS = 5  # Don't trade below 5¢ (too risky)
        MAX_PRICE_CENTS = 90  # Don't trade above 90¢ (not enough upside)
        
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
                'category': market.get('category', 'unknown'),  # Track for learning
                'has_intel': has_intel,  # Track if news was used
                'news_count': news_count,  # Track how many news items
                'timestamp': datetime.utcnow().isoformat(),
            }
            self._trades.insert(0, trade)
            news_indicator = f" [NEWS:{news_count}]" if news_count > 0 else ""
            print(f"[DRY RUN] ENTERED: {side} ${size:.2f} @ {int(entry_price*100)}¢{news_indicator} | {market.get('question', '')[:50]}...")
        else:
            # In LIVE mode, track as pending until we confirm fill from Kalshi
            # Get intel data from most recent analysis for this market
            recent_analysis = next((a for a in self._analyses if a.get('market_id') == market_id), None)
            has_intel = recent_analysis.get('has_intel', False) if recent_analysis else False
            news_count = recent_analysis.get('news_count', 0) if recent_analysis else 0
            
            # Store intel info in order_data so it persists through fill
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
                'has_intel': has_intel,
                'news_count': news_count,
                'timestamp': datetime.utcnow().isoformat(),
                'order_id': order_id,
                'status': 'pending',
            }
            self._trades.insert(0, trade)
            news_indicator = f" [NEWS:{news_count}]" if news_count > 0 else ""
            print(f"[PENDING] Order {order_id} placed{news_indicator}, waiting to fill...")
        
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
                    
                    # Dynamic exit thresholds based on market spread
                    # Tighter spreads = tighter thresholds, wider spreads = more room
                    spread = market.get('spread', 0.04)  # Default 4¢ if unknown
                    
                    if spread <= 0.02:  # Tight spread (≤2¢)
                        profit_take_pct = 0.05   # 5% profit target
                        stop_loss_pct = 0.05     # 5% stop loss
                    elif spread <= 0.04:  # Medium spread (2-4¢)
                        profit_take_pct = 0.10   # 10% profit target
                        stop_loss_pct = 0.08     # 8% stop loss
                    else:  # Wide spread (>4¢)
                        profit_take_pct = 0.15   # 15% profit target
                        stop_loss_pct = 0.12     # 12% stop loss
                    
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
                        # Don't place another sell order if one is already pending
                        if 'pending_exit' in pos:
                            continue
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
                    # Skip if position already has a pending exit order
                    if position.get('pending_exit'):
                        return  # Silently skip - don't spam logs
                    
                    # Skip exit for positions > 10 contracts (likely illiquid, wait for settlement)
                    if contracts > 10:
                        print(f"[LIVE SELL] Position ({contracts} contracts) too large for liquid exit - waiting for settlement")
                        return
                    
                    # Use limit order at 1¢ for immediate fill
                    sell_price_cents = 1
                    
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
            document.getElementById('posCount').textContent = s.open_positions + ' positions + ' + (s.pending_orders || 0) + ' resting';
            // At Risk Breakdown
            document.getElementById('positionsAtRisk').textContent = '$' + (s.positions_at_risk || 0).toFixed(2);
            document.getElementById('filledCount').textContent = s.open_positions + ' filled';
            document.getElementById('pendingAtRisk').textContent = '$' + (s.pending_at_risk || 0).toFixed(2);
            document.getElementById('restingCount').textContent = (s.pending_orders || 0) + ' resting';
            // Time Horizon
            document.getElementById('atRiskUltra').textContent = '$' + (s.at_risk_ultra_short || 0).toFixed(2);
            document.getElementById('atRiskShort').textContent = '$' + (s.at_risk_short || 0).toFixed(2);
            document.getElementById('atRiskMedium').textContent = '$' + (s.at_risk_medium || 0).toFixed(2);
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
