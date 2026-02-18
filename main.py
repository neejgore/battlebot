"""Main entry point for Polymarket Battle-Bot V2.0.

An asynchronous, event-driven trading bot for Polymarket that uses
Claude AI for trade signals based on real-time WebSocket data.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from data.models import Market, MarketOutcome
from services.websocket_client import WebSocketClient
from services.execution_engine import ExecutionEngine
from services.market_manager import MarketManager, MarketManagerConfig, DiscoveredMarket
from services.dashboard import Dashboard
from logic.risk_engine import RiskEngine, RiskLimits
from logic.strategy_v2 import StrategyV2, StrategyConfig


# Load environment variables
load_dotenv()


class DryRunExecutionEngine:
    """Mock execution engine for dry run mode.
    
    Logs what would be traded without placing real orders.
    """
    
    def __init__(self):
        self._order_id = 0
        self._orders = []
        logger.info("DryRunExecutionEngine initialized - no real orders will be placed")
    
    async def create_order(self, token_id, side, price, size):
        """Simulate order creation."""
        self._order_id += 1
        order_id = f"DRY_RUN_{self._order_id}"
        
        logger.info(
            f"[DRY RUN] Would place order: {side.value} {size:.2f} @ {price:.4f} | "
            f"Token: {token_id[:20]}..."
        )
        
        self._orders.append({
            'order_id': order_id,
            'token_id': token_id,
            'side': side.value,
            'price': price,
            'size': size,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # Return mock order object
        return type('Order', (), {
            'order_id': order_id,
            'status': 'FILLED',
        })()
    
    async def cancel_order(self, order_id):
        """Simulate order cancellation."""
        logger.info(f"[DRY RUN] Would cancel order: {order_id}")
        return True
    
    async def cancel_all_orders(self):
        """Simulate cancelling all orders."""
        logger.info("[DRY RUN] Would cancel all orders")
        return len(self._orders)
    
    def get_stats(self):
        return {
            'dry_run': True,
            'simulated_orders': len(self._orders),
        }


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )


class BattleBot:
    """Main battle bot orchestrator.
    
    Coordinates all components: WebSocket client, execution engine,
    risk engine, trading strategy, and automatic market discovery.
    """
    
    def __init__(
        self,
        initial_bankroll: float,
        fractional_kelly: float = 0.1,
        max_position_size: float = 50.0,
        min_price_change: float = 0.03,
        min_edge: float = 0.05,
        market_refresh_minutes: int = 60,
        dry_run: bool = False,
    ):
        """Initialize the battle bot.
        
        Args:
            initial_bankroll: Starting bankroll in USDC
            fractional_kelly: Kelly fraction (default 10%)
            max_position_size: Maximum position size in USDC
            min_price_change: Minimum price change to trigger analysis
            min_edge: Minimum edge required to trade
            market_refresh_minutes: How often to refresh markets (default 60)
            dry_run: If True, don't place real orders
        """
        self.initial_bankroll = initial_bankroll
        self.dry_run = dry_run
        
        # Initialize components
        self.ws_client = WebSocketClient()
        
        # Use mock execution engine for dry run
        if dry_run:
            self.execution_engine = DryRunExecutionEngine()
            logger.warning("DRY RUN MODE - No real orders will be placed")
        else:
            self.execution_engine = ExecutionEngine()
        
        # Risk limits
        risk_limits = RiskLimits(
            max_position_size=max_position_size,
        )
        
        self.risk_engine = RiskEngine(
            initial_bankroll=initial_bankroll,
            fractional_kelly=fractional_kelly,
            limits=risk_limits,
        )
        
        # Strategy config
        strategy_config = StrategyConfig(
            min_price_change=min_price_change,
            min_edge=min_edge,
        )
        
        self.strategy = StrategyV2(
            risk_engine=self.risk_engine,
            execution_engine=self.execution_engine,
            ws_client=self.ws_client,
            config=strategy_config,
        )
        
        # Market manager for automatic discovery
        market_config = MarketManagerConfig(
            refresh_interval_minutes=market_refresh_minutes,
        )
        
        self.market_manager = MarketManager(
            config=market_config,
            on_market_change=self._on_market_change,
        )
        
        # Dashboard for real-time monitoring
        self.dashboard = Dashboard(port=8080)
        
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
        mode_str = "[DRY RUN] " if dry_run else ""
        logger.info(
            f"{mode_str}BattleBot initialized | Bankroll: ${initial_bankroll:.2f} | "
            f"Kelly: {fractional_kelly*100:.0f}% | Max pos: ${max_position_size:.2f} | "
            f"Market refresh: {market_refresh_minutes}min"
        )
    
    async def _on_market_change(self, market: DiscoveredMarket, action: str) -> None:
        """Handle market add/remove from MarketManager.
        
        Args:
            market: The discovered market
            action: 'add' or 'remove'
        """
        if action == 'add':
            # Convert to Market model and add to strategy
            market_model = Market(
                condition_id=market.condition_id,
                question=market.question,
                token_id=market.token_id,
                outcome=MarketOutcome.YES,
                current_price=market.current_price,
                volume_24h=market.volume_24h,
                liquidity=market.liquidity,
                end_date=market.end_date,
            )
            
            await self.strategy.add_market(
                market=market_model,
                resolution_rules=market.resolution_rules,
                category=market.category,
            )
            
            # Notify dashboard
            await self.dashboard.market_added({
                'token_id': market.token_id,
                'question': market.question,
                'current_price': market.current_price,
                'volume_24h': market.volume_24h,
                'category': market.category,
                'url': market.url,
            })
            
        elif action == 'remove':
            await self.strategy.remove_market(market.token_id)
            await self.dashboard.market_removed(market.token_id, "No longer eligible")
    
    async def add_market(
        self,
        condition_id: str,
        question: str,
        token_id: str,
        outcome: MarketOutcome = MarketOutcome.YES,
        current_price: float = 0.5,
        end_date: Optional[datetime] = None,
        volume_24h: float = 0.0,
        liquidity: float = 0.0,
    ) -> None:
        """Add a market to monitor.
        
        Args:
            condition_id: Market condition ID
            question: Market question
            token_id: Token ID for the outcome
            outcome: YES or NO
            current_price: Current market price
            end_date: Market resolution date
            volume_24h: 24h trading volume
            liquidity: Market liquidity
        """
        market = Market(
            condition_id=condition_id,
            question=question,
            token_id=token_id,
            outcome=outcome,
            current_price=current_price,
            end_date=end_date,
            volume_24h=volume_24h,
            liquidity=liquidity,
        )
        
        await self.strategy.add_market(market)
    
    async def start(self) -> None:
        """Start the battle bot."""
        if self._running:
            logger.warning("BattleBot already running")
            return
        
        self._running = True
        
        mode_str = "[DRY RUN] " if self.dry_run else ""
        logger.info("=" * 60)
        logger.info(f"{mode_str}POLYMARKET BATTLE-BOT V2.1 STARTING")
        logger.info("=" * 60)
        
        # Start dashboard first
        await self.dashboard.start()
        await self.dashboard.update_stats({
            'dry_run': self.dry_run,
            'running': True,
            'bankroll': self.initial_bankroll,
        })
        
        # Start strategy
        await self.strategy.start()
        
        # Start WebSocket client in background
        ws_task = asyncio.create_task(self.ws_client.run())
        self._tasks.append(ws_task)
        
        # Start market manager (auto-discovers markets)
        await self.market_manager.start()
        
        # Start monitoring loop
        monitor_task = asyncio.create_task(self._monitoring_loop())
        self._tasks.append(monitor_task)
        
        logger.info(f"{mode_str}BattleBot started successfully")
        logger.info(f"Markets will refresh every {self.market_manager.config.refresh_interval_minutes} minutes")
        logger.info(f"Dashboard: http://localhost:8080")
    
    async def stop(self) -> None:
        """Stop the battle bot gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping BattleBot...")
        self._running = False
        
        # Update dashboard
        await self.dashboard.update_stats({'running': False})
        
        # Stop market manager
        await self.market_manager.stop()
        
        # Stop strategy
        await self.strategy.stop()
        
        # Cancel all open orders
        await self.execution_engine.cancel_all_orders()
        
        # Disconnect WebSocket
        await self.ws_client.disconnect()
        
        # Stop dashboard
        await self.dashboard.stop()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        logger.info("BattleBot stopped")
        self._print_final_stats()
    
    async def _monitoring_loop(self) -> None:
        """Periodic monitoring and logging loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Update dashboard every 30 seconds
                
                if self._running:
                    await self._log_status()
                    
                    # Check for daily reset (midnight UTC)
                    now = datetime.utcnow()
                    if now.hour == 0 and now.minute == 0:
                        await self.risk_engine.reset_daily_stats()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _log_status(self) -> None:
        """Log current bot status and update dashboard."""
        portfolio = self.risk_engine.get_portfolio_summary()
        strategy_stats = self.strategy.get_stats()
        market_stats = self.market_manager.get_stats()
        
        mode_str = "[DRY RUN] " if self.dry_run else ""
        
        logger.info(
            f"{mode_str}Status | Bankroll: ${portfolio['bankroll']:.2f} | "
            f"Daily P&L: ${portfolio['daily_pnl']:+.2f} | "
            f"Drawdown: {portfolio['current_drawdown_pct']:.1f}% | "
            f"Positions: {portfolio['open_positions']} | "
            f"Markets: {market_stats['active_markets']} | "
            f"Trades: {strategy_stats['trades_executed']}"
        )
        
        # Update dashboard
        await self.dashboard.update_stats({
            'bankroll': portfolio['bankroll'],
            'daily_pnl': portfolio['daily_pnl'],
            'active_positions': portfolio['open_positions'],
            'markets_monitored': market_stats['active_markets'],
            'total_trades': strategy_stats['trades_executed'],
            'win_rate': strategy_stats.get('win_rate', 0),
            'dry_run': self.dry_run,
            'running': self._running,
        })
        
        if portfolio['kill_switch_triggered']:
            logger.critical("KILL SWITCH ACTIVE - TRADING HALTED")
    
    def _print_final_stats(self) -> None:
        """Print final statistics on shutdown."""
        portfolio = self.risk_engine.get_portfolio_summary()
        strategy_stats = self.strategy.get_stats()
        
        logger.info("=" * 60)
        logger.info("FINAL SESSION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Starting Bankroll: ${self.initial_bankroll:.2f}")
        logger.info(f"Ending Bankroll:   ${portfolio['bankroll']:.2f}")
        logger.info(f"Total P&L:         ${portfolio['bankroll'] - self.initial_bankroll:+.2f}")
        logger.info(f"Daily P&L:         ${portfolio['daily_pnl']:+.2f}")
        logger.info(f"Max Drawdown:      {self.risk_engine.daily_stats.max_drawdown_pct:.1f}%")
        logger.info(f"Trades Executed:   {strategy_stats['trades_executed']}")
        logger.info(f"Signals Generated: {strategy_stats['signals_generated']}")
        logger.info(f"AI Calls:          {strategy_stats['ai_calls']}")
        logger.info(f"Win Rate:          {portfolio['win_rate']:.1f}%")
        logger.info("=" * 60)


async def main():
    """Main async entry point."""
    # Configuration from environment
    initial_bankroll = float(os.getenv("INITIAL_BANKROLL", "1000"))
    fractional_kelly = float(os.getenv("FRACTIONAL_KELLY", "0.1"))
    max_position_size = float(os.getenv("MAX_POSITION_SIZE", "50"))
    min_price_change = float(os.getenv("MIN_PRICE_CHANGE", "0.02"))
    min_edge = float(os.getenv("MIN_EDGE", "0.03"))
    market_refresh_minutes = int(os.getenv("MARKET_REFRESH_MINUTES", "15"))
    dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    
    # Setup logging
    setup_logging(log_level, log_file)
    
    # Create bot
    bot = BattleBot(
        initial_bankroll=initial_bankroll,
        fractional_kelly=fractional_kelly,
        max_position_size=max_position_size,
        min_price_change=min_price_change,
        min_edge=min_edge,
        market_refresh_minutes=market_refresh_minutes,
        dry_run=dry_run,
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def handle_shutdown():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown)
    
    try:
        # Start bot - markets are now auto-discovered!
        await bot.start()
        
        # Keep running until stopped
        while bot._running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
