#!/usr/bin/env python3
"""Backtest/Simulation Harness for Polymarket Battle-Bot V2.1.

Replays historical price data through the strategy to evaluate performance.
Supports both recorded price data from SQLite and synthetic/mocked data.

Usage:
    python scripts/backtest.py --db data/battlebot.db --market 0x1234...
    python scripts/backtest.py --generate --days 30 --output data/mock_prices.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data.models import Market, PriceUpdate, MarketOutcome
from data.database import TelemetryDB
from logic.strategy_v2 import StrategyV2, StrategyConfig
from logic.risk_engine import RiskEngine, RiskLimits
from logic.calibration import CalibrationEngine


# ============================================================================
# Mock Components for Backtesting
# ============================================================================

class MockWebSocketClient:
    """Mock WebSocket client for backtesting."""
    
    def __init__(self):
        self._callbacks = []
        self._subscribed = set()
    
    def register_callback(self, callback):
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def subscribe(self, token_ids: list[str]):
        self._subscribed.update(token_ids)
    
    async def unsubscribe(self, token_ids: list[str]):
        self._subscribed -= set(token_ids)
    
    async def emit_price_update(self, update: PriceUpdate):
        """Emit a price update to all callbacks."""
        for callback in self._callbacks:
            await callback(update)


class MockExecutionEngine:
    """Mock execution engine for backtesting.
    
    Simulates fills with configurable slippage.
    """
    
    def __init__(self, slippage_bps: float = 10.0):
        """Initialize mock execution.
        
        Args:
            slippage_bps: Slippage in basis points (10 = 0.1%)
        """
        self.slippage_bps = slippage_bps
        self._order_id = 0
        self._orders = []
    
    async def create_order(self, token_id, side, price, size):
        """Simulate order execution with slippage."""
        self._order_id += 1
        
        # Apply slippage
        slippage = self.slippage_bps / 10000
        if side.value == "BUY":
            fill_price = price * (1 + slippage)  # Pay more
        else:
            fill_price = price * (1 - slippage)  # Receive less
        
        order = type('Order', (), {
            'order_id': f"MOCK_{self._order_id}",
            'token_id': token_id,
            'side': side,
            'price': fill_price,
            'size': size,
            'status': 'FILLED',
        })()
        
        self._orders.append({
            'order_id': order.order_id,
            'token_id': token_id,
            'side': side.value,
            'price': price,
            'fill_price': fill_price,
            'size': size,
            'slippage': fill_price - price,
        })
        
        return order


class MockAISignalGenerator:
    """Mock AI signal generator for backtesting.
    
    Uses predetermined signals or random generation.
    """
    
    def __init__(self, signals: Optional[dict] = None, noise_std: float = 0.05):
        """Initialize mock generator.
        
        Args:
            signals: Dict mapping (market_id, timestamp) -> signal dict
            noise_std: Standard deviation of noise to add to "true" probability
        """
        self._signals = signals or {}
        self.noise_std = noise_std
        self.is_available = True
    
    async def generate_signal(self, market_question, current_price, **kwargs):
        """Generate a mock signal."""
        from logic.ai_signal import AISignalResult, AISignalOutput
        
        # Add noise to current price to simulate AI estimate
        noise = random.gauss(0, self.noise_std)
        raw_prob = max(0.01, min(0.99, current_price + noise))
        
        # Confidence inversely related to how far from 0.5
        confidence = 0.5 + 0.3 * (1 - abs(raw_prob - 0.5) * 2)
        
        signal = AISignalOutput(
            raw_prob=raw_prob,
            confidence=confidence,
            key_reasons=["Mock signal for backtesting"],
            disconfirming_evidence=["This is a simulation"],
            what_would_change_mind=["Real data"],
            timeline_sensitivity="no: mock",
            failure_modes=["Simulation limitations"],
            base_rate_considered=True,
            information_quality="medium",
        )
        
        return AISignalResult(
            success=True,
            signal=signal,
            latency_ms=50,
        )


# ============================================================================
# Data Generation
# ============================================================================

@dataclass
class PriceTick:
    """A single price tick for backtesting."""
    timestamp: datetime
    market_id: str
    token_id: str
    mid_price: float
    bid: float
    ask: float
    spread: float
    volume: float = 0.0


def generate_mock_price_data(
    market_id: str,
    token_id: str,
    days: int = 30,
    ticks_per_day: int = 1440,  # 1 per minute
    initial_price: float = 0.5,
    volatility: float = 0.02,
    drift: float = 0.0,
    spread: float = 0.02,
) -> list[PriceTick]:
    """Generate synthetic price data using geometric Brownian motion.
    
    Args:
        market_id: Market identifier
        token_id: Token identifier
        days: Number of days to generate
        ticks_per_day: Ticks per day (1440 = 1 per minute)
        initial_price: Starting price
        volatility: Daily volatility (std dev)
        drift: Daily drift (mean return)
        spread: Fixed bid-ask spread
        
    Returns:
        List of PriceTick objects
    """
    ticks = []
    dt = 1.0 / ticks_per_day  # Time step in days
    price = initial_price
    
    start_time = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days * ticks_per_day):
        timestamp = start_time + timedelta(minutes=i)
        
        # Geometric Brownian motion step
        random_shock = random.gauss(0, 1)
        price_return = drift * dt + volatility * (dt ** 0.5) * random_shock
        price = price * (1 + price_return)
        
        # Clamp to valid range
        price = max(0.01, min(0.99, price))
        
        # Calculate bid/ask
        half_spread = spread / 2
        bid = max(0.01, price - half_spread)
        ask = min(0.99, price + half_spread)
        
        ticks.append(PriceTick(
            timestamp=timestamp,
            market_id=market_id,
            token_id=token_id,
            mid_price=price,
            bid=bid,
            ask=ask,
            spread=ask - bid,
        ))
    
    return ticks


def save_ticks_to_jsonl(ticks: list[PriceTick], filepath: str) -> None:
    """Save ticks to JSONL file."""
    with open(filepath, 'w') as f:
        for tick in ticks:
            f.write(json.dumps({
                'timestamp': tick.timestamp.isoformat(),
                'market_id': tick.market_id,
                'token_id': tick.token_id,
                'mid_price': tick.mid_price,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.spread,
                'volume': tick.volume,
            }) + '\n')
    
    logger.info(f"Saved {len(ticks)} ticks to {filepath}")


def load_ticks_from_jsonl(filepath: str) -> list[PriceTick]:
    """Load ticks from JSONL file."""
    ticks = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            ticks.append(PriceTick(
                timestamp=datetime.fromisoformat(data['timestamp']),
                market_id=data['market_id'],
                token_id=data['token_id'],
                mid_price=data['mid_price'],
                bid=data['bid'],
                ask=data['ask'],
                spread=data['spread'],
                volume=data.get('volume', 0),
            ))
    return ticks


# ============================================================================
# Backtest Engine
# ============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time_hours: float = 0.0
    sharpe_ratio: float = 0.0
    
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': round(self.total_pnl, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_hold_time_hours': round(self.avg_hold_time_hours, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
        }


class Backtester:
    """Backtest engine for strategy evaluation."""
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        slippage_bps: float = 10.0,
        strategy_config: Optional[StrategyConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """Initialize backtester.
        
        Args:
            initial_bankroll: Starting capital
            slippage_bps: Slippage in basis points
            strategy_config: Strategy configuration
            risk_limits: Risk limits configuration
        """
        self.initial_bankroll = initial_bankroll
        self.slippage_bps = slippage_bps
        self.strategy_config = strategy_config or StrategyConfig()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Components (will be initialized per run)
        self._ws_client: Optional[MockWebSocketClient] = None
        self._execution_engine: Optional[MockExecutionEngine] = None
        self._risk_engine: Optional[RiskEngine] = None
        self._strategy: Optional[StrategyV2] = None
    
    async def run(
        self,
        ticks: list[PriceTick],
        market: Market,
        resolution_rules: str = "Mock resolution rules for backtesting",
    ) -> BacktestResult:
        """Run backtest on price data.
        
        Args:
            ticks: List of price ticks to replay
            market: Market being tested
            resolution_rules: Resolution rules text
            
        Returns:
            BacktestResult with metrics
        """
        logger.info(f"Starting backtest | {len(ticks)} ticks | Market: {market.question[:50]}...")
        
        # Initialize components
        self._ws_client = MockWebSocketClient()
        self._execution_engine = MockExecutionEngine(self.slippage_bps)
        self._risk_engine = RiskEngine(
            initial_bankroll=self.initial_bankroll,
            limits=self.risk_limits,
        )
        
        # Use mock AI generator
        mock_ai = MockAISignalGenerator()
        
        # Create in-memory calibration engine (no persistence)
        calibration_engine = CalibrationEngine(db=None)
        
        self._strategy = StrategyV2(
            risk_engine=self._risk_engine,
            execution_engine=self._execution_engine,
            ws_client=self._ws_client,
            config=self.strategy_config,
            db=None,  # No DB persistence for backtest
            ai_generator=mock_ai,
            calibration_engine=calibration_engine,
        )
        
        # Add market
        await self._strategy.add_market(
            market=market,
            resolution_rules=resolution_rules,
        )
        
        # Start strategy (but don't start background tasks)
        self._strategy._running = True
        self._ws_client.register_callback(self._strategy.on_price_update)
        
        # Track equity curve
        equity_curve = [(ticks[0].timestamp, self.initial_bankroll)]
        
        # Replay ticks
        for i, tick in enumerate(ticks):
            update = PriceUpdate(
                token_id=tick.token_id,
                price=tick.mid_price,
                bid=tick.bid,
                ask=tick.ask,
                spread=tick.spread,
                timestamp=tick.timestamp,
            )
            
            # Emit update
            await self._ws_client.emit_price_update(update)
            
            # Track equity
            current_equity = self._risk_engine.bankroll + self._risk_engine.daily_stats.unrealized_pnl
            equity_curve.append((tick.timestamp, current_equity))
            
            # Progress logging
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(ticks)} ticks | Equity: ${current_equity:.2f}")
        
        # Compute results
        result = self._compute_results(equity_curve)
        
        logger.info(
            f"Backtest complete | Trades: {result.total_trades} | "
            f"P&L: ${result.total_pnl:+.2f} | Win rate: {result.win_rate:.1f}%"
        )
        
        return result
    
    def _compute_results(self, equity_curve: list) -> BacktestResult:
        """Compute backtest metrics."""
        result = BacktestResult()
        result.equity_curve = equity_curve
        
        # Get orders from mock execution
        orders = self._execution_engine._orders
        result.total_trades = len(orders)
        
        # Get final equity
        if equity_curve:
            final_equity = equity_curve[-1][1]
            result.total_pnl = final_equity - self.initial_bankroll
        
        # Compute max drawdown
        peak = self.initial_bankroll
        max_dd = 0.0
        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd * 100
        
        # Get trade stats from risk engine
        stats = self._risk_engine.daily_stats
        result.winning_trades = stats.winning_trades
        result.losing_trades = stats.losing_trades
        result.win_rate = stats.win_rate
        
        # Profit factor (gross profit / gross loss)
        # Simplified calculation
        if stats.losing_trades > 0 and result.winning_trades > 0:
            avg_win = result.total_pnl / max(1, result.winning_trades) if result.total_pnl > 0 else 0
            avg_loss = abs(result.total_pnl) / max(1, result.losing_trades) if result.total_pnl < 0 else 0
            if avg_loss > 0:
                result.profit_factor = avg_win / avg_loss
        
        return result
    
    def save_results(self, result: BacktestResult, filepath: str) -> None:
        """Save backtest results to CSV."""
        # Save summary
        with open(filepath, 'w') as f:
            f.write("Backtest Results\n")
            f.write("================\n")
            for key, value in result.to_dict().items():
                f.write(f"{key}: {value}\n")
        
        # Save equity curve
        curve_path = filepath.replace('.csv', '_equity.csv')
        with open(curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'equity'])
            for ts, equity in result.equity_curve:
                writer.writerow([ts.isoformat(), round(equity, 2)])
        
        logger.info(f"Results saved to {filepath}")


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description='Backtest Strategy V2.1')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Generate mock data command
    gen_parser = subparsers.add_parser('generate', help='Generate mock price data')
    gen_parser.add_argument('--days', type=int, default=30, help='Days of data')
    gen_parser.add_argument('--output', type=str, default='data/mock_prices.jsonl')
    gen_parser.add_argument('--volatility', type=float, default=0.02, help='Daily volatility')
    gen_parser.add_argument('--drift', type=float, default=0.0, help='Daily drift')
    gen_parser.add_argument('--initial-price', type=float, default=0.5)
    
    # Run backtest command
    run_parser = subparsers.add_parser('run', help='Run backtest')
    run_parser.add_argument('--data', type=str, required=True, help='JSONL price data file')
    run_parser.add_argument('--bankroll', type=float, default=1000.0)
    run_parser.add_argument('--slippage', type=float, default=10.0, help='Slippage in bps')
    run_parser.add_argument('--output', type=str, default='data/backtest_results.csv')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        logger.info(f"Generating {args.days} days of mock price data...")
        
        ticks = generate_mock_price_data(
            market_id="MOCK_MARKET_001",
            token_id="MOCK_TOKEN_001",
            days=args.days,
            initial_price=args.initial_price,
            volatility=args.volatility,
            drift=args.drift,
        )
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_ticks_to_jsonl(ticks, args.output)
        
    elif args.command == 'run':
        logger.info(f"Loading price data from {args.data}...")
        
        ticks = load_ticks_from_jsonl(args.data)
        
        if not ticks:
            logger.error("No ticks loaded!")
            return
        
        # Create mock market
        market = Market(
            condition_id=ticks[0].market_id,
            question="Mock market for backtesting",
            token_id=ticks[0].token_id,
            outcome=MarketOutcome.YES,
            current_price=ticks[0].mid_price,
            volume_24h=50000.0,
            liquidity=100000.0,
        )
        
        # Run backtest
        backtester = Backtester(
            initial_bankroll=args.bankroll,
            slippage_bps=args.slippage,
        )
        
        result = await backtester.run(ticks, market)
        
        # Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        backtester.save_results(result, args.output)
        
        # Print summary
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        for key, value in result.to_dict().items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    asyncio.run(main())
