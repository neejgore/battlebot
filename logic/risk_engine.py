"""Risk Engine for Polymarket Battle-Bot V2.1.

Implements the Fractional Kelly Criterion for optimal position sizing
and manages daily risk limits including the 15% max drawdown kill-switch.

V2.1 Additions:
- Edge-based throttle for position sizing
- Per-market position limits
- Global exposure limits
- Enhanced position tracking with exit rules
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from data.models import DailyStats, TradeSignal, Position, OrderSide


@dataclass
class PositionWithExitRules:
    """Position with exit rule tracking."""
    position: Position
    entry_time: datetime
    entry_edge: float
    entry_confidence: float
    
    # Exit thresholds
    profit_take_price: float
    stop_loss_price: float
    time_stop_hours: float
    
    # State
    trade_id: Optional[int] = None
    calibration_sample_id: Optional[int] = None
    
    @property
    def is_time_stopped(self) -> bool:
        """Check if position has exceeded time limit."""
        elapsed = datetime.utcnow() - self.entry_time
        return elapsed.total_seconds() / 3600 > self.time_stop_hours


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_daily_drawdown: float = 0.15  # 15%
    max_position_size: float = 50.0  # $50 per position
    max_percent_bankroll_per_market: float = 0.10  # 10% of bankroll per market
    max_total_open_risk: float = 0.30  # 30% of bankroll in open positions
    max_positions: int = 10  # Maximum concurrent positions
    
    # Exit rules
    profit_take_pct: float = 0.15  # 15% profit target
    stop_loss_pct: float = 0.10  # 10% stop loss
    time_stop_hours: float = 72.0  # 3 days max hold
    
    # Edge throttle
    edge_scale: float = 0.10  # Scale factor for edge-based throttle
    min_edge: float = 0.03  # Minimum edge to trade


def calculate_kelly_size(
    bankroll: float,
    true_prob: float,
    market_price: float,
    fractional_multiplier: float = 0.1,
    max_pos: float = 50.0,
    edge_throttle: float = 1.0,
) -> float:
    """Calculate optimal position size using Fractional Kelly Criterion.
    
    The Kelly Criterion determines the optimal bet size to maximize
    long-term growth while managing risk. We use a fractional multiplier
    (default 10%) to be more conservative.
    
    Args:
        bankroll: Current available bankroll in USDC
        true_prob: Estimated true probability (0-1)
        market_price: Current market price (0-1)
        fractional_multiplier: Kelly fraction (default 0.1 = 10%)
        max_pos: Maximum position size in USDC
        edge_throttle: Additional throttle based on edge (0-1)
        
    Returns:
        Optimal position size in USDC, capped at max_pos
    """
    # Validate inputs
    if market_price <= 0 or market_price >= 1:
        return 0.0
    
    # Calculate edge (expected advantage)
    edge = true_prob - market_price
    if edge <= 0:
        return 0.0
    
    # Calculate odds (b = potential profit / stake)
    b = (1 / market_price) - 1
    
    # Kelly formula: f* = (b*p - q) / b
    # where p = probability of winning, q = probability of losing
    p = true_prob
    q = 1 - true_prob
    
    full_kelly = (b * p - q) / b
    
    # Apply fractional multiplier for conservative sizing
    bet_size = bankroll * (full_kelly * fractional_multiplier)
    
    # Apply edge-based throttle
    bet_size *= edge_throttle
    
    # Ensure non-negative and cap at maximum
    return round(max(0.0, min(bet_size, max_pos)), 2)


class RiskEngine:
    """Manages risk for the trading bot.
    
    V2.1 Responsibilities:
    - Calculate position sizes using Fractional Kelly Criterion
    - Apply edge-based position throttling
    - Track daily P&L and enforce 15% max drawdown kill-switch
    - Validate trades against risk limits
    - Monitor portfolio exposure
    - Enforce per-market and global position limits
    - Track positions with exit rules
    """
    
    def __init__(
        self,
        initial_bankroll: float,
        fractional_kelly: float = 0.1,
        limits: Optional[RiskLimits] = None,
    ):
        """Initialize the risk engine.
        
        Args:
            initial_bankroll: Starting bankroll in USDC
            fractional_kelly: Kelly fraction multiplier
            limits: Risk limits configuration
        """
        self.bankroll = initial_bankroll
        self.fractional_kelly = fractional_kelly
        self.limits = limits or RiskLimits()
        
        self._positions: dict[str, PositionWithExitRules] = {}
        self._market_exposure: dict[str, float] = {}  # market_id -> exposure
        self._exit_cooldowns: dict[str, datetime] = {}  # token_id -> cooldown expiry
        self._lock = asyncio.Lock()
        
        # Initialize daily stats
        self.daily_stats = DailyStats(
            starting_bankroll=initial_bankroll,
            current_bankroll=initial_bankroll,
        )
        
        logger.info(
            f"RiskEngine V2.1 initialized | Bankroll: ${initial_bankroll:.2f} | "
            f"Kelly: {fractional_kelly*100:.0f}% | Max position: ${self.limits.max_position_size:.2f}"
        )
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on risk limits."""
        return not self.daily_stats.kill_switch_triggered
    
    @property
    def current_drawdown(self) -> float:
        """Get current drawdown as decimal."""
        return self.daily_stats.current_drawdown_pct / 100
    
    @property
    def available_capital(self) -> float:
        """Get available capital for new positions."""
        total_exposure = sum(
            pos.position.cost_basis for pos in self._positions.values()
        )
        return max(0.0, self.bankroll - total_exposure)
    
    @property
    def total_exposure(self) -> float:
        """Get total open position exposure."""
        return sum(pos.position.cost_basis for pos in self._positions.values())
    
    @property
    def exposure_ratio(self) -> float:
        """Get exposure as fraction of bankroll."""
        if self.bankroll <= 0:
            return 1.0
        return self.total_exposure / self.bankroll
    
    def _compute_edge_throttle(self, edge: float) -> float:
        """Compute edge-based position size throttle.
        
        Higher edge = less throttling (up to 1.0)
        Edge at or below min_edge = heavy throttling
        
        Args:
            edge: Calculated edge (adjusted_prob - market_price)
            
        Returns:
            Throttle factor (0-1)
        """
        if edge <= self.limits.min_edge:
            return 0.0
        
        # Linear scaling: full size at edge = edge_scale
        throttle = min(1.0, edge / self.limits.edge_scale)
        return throttle
    
    async def calculate_position_size(
        self,
        adjusted_prob: float,
        market_price: float,
        edge: float,
        confidence: float,
        market_id: str,
        custom_max_pos: Optional[float] = None,
    ) -> float:
        """Calculate optimal position size for a trade.
        
        Applies multiple constraints:
        1. Kelly criterion sizing
        2. Edge-based throttle
        3. Per-market limit
        4. Global exposure limit
        5. Max position size
        
        Args:
            adjusted_prob: Calibrated and adjusted probability
            market_price: Current market price
            edge: Calculated edge
            confidence: AI confidence score
            market_id: Market identifier
            custom_max_pos: Optional custom max position
            
        Returns:
            Recommended position size in USDC
        """
        async with self._lock:
            # Check if trading is allowed
            if not self.is_trading_allowed:
                logger.warning("Trading halted - kill switch triggered")
                return 0.0
            
            # Check position count
            if len(self._positions) >= self.limits.max_positions:
                logger.warning(f"Max positions ({self.limits.max_positions}) reached")
                return 0.0
            
            # Get available capital
            available = self.available_capital
            if available <= 0:
                logger.warning("No available capital for new positions")
                return 0.0
            
            # Check global exposure limit
            max_additional = self.bankroll * self.limits.max_total_open_risk - self.total_exposure
            if max_additional <= 0:
                logger.warning("Global exposure limit reached")
                return 0.0
            
            # Per-market limit
            current_market_exposure = self._market_exposure.get(market_id, 0.0)
            max_market_exposure = self.bankroll * self.limits.max_percent_bankroll_per_market
            remaining_market_limit = max_market_exposure - current_market_exposure
            
            if remaining_market_limit <= 0:
                logger.warning(f"Per-market exposure limit reached for {market_id[:16]}...")
                return 0.0
            
            # Compute edge throttle
            edge_throttle = self._compute_edge_throttle(edge)
            if edge_throttle <= 0:
                logger.debug(f"Edge {edge:.2%} below threshold - no trade")
                return 0.0
            
            # Base max position
            max_pos = custom_max_pos or self.limits.max_position_size
            
            # Apply all limits
            effective_max = min(
                max_pos,
                available,
                max_additional,
                remaining_market_limit,
            )
            
            # Calculate Kelly size with edge throttle
            size = calculate_kelly_size(
                bankroll=available,
                true_prob=adjusted_prob,
                market_price=market_price,
                fractional_multiplier=self.fractional_kelly,
                max_pos=effective_max,
                edge_throttle=edge_throttle,
            )
            
            logger.debug(
                f"Position size calculated | Edge: {edge:.2%} | Throttle: {edge_throttle:.2f} | "
                f"Size: ${size:.2f} | Max: ${effective_max:.2f}"
            )
            
            return size
    
    async def validate_trade(
        self,
        size: float,
        price: float,
        token_id: str,
    ) -> tuple[bool, list[str]]:
        """Validate a proposed trade against risk limits.
        
        Args:
            size: Position size in USDC
            price: Entry price
            token_id: Token ID
            
        Returns:
            Tuple of (is_valid, list of reason codes)
        """
        reasons = []
        
        async with self._lock:
            # Check kill switch
            if not self.is_trading_allowed:
                reasons.append("KILL_SWITCH_TRIGGERED")
            
            # Check position size
            if size > self.limits.max_position_size:
                reasons.append("EXCEEDS_MAX_POSITION")
            
            # Check available capital
            if size > self.available_capital:
                reasons.append("INSUFFICIENT_CAPITAL")
            
            # Check if this trade would push us over drawdown limit
            potential_loss = size
            potential_drawdown = (
                self.daily_stats.starting_bankroll - 
                (self.daily_stats.current_bankroll - potential_loss)
            ) / self.daily_stats.starting_bankroll
            
            if potential_drawdown >= self.limits.max_daily_drawdown:
                reasons.append("WOULD_EXCEED_DRAWDOWN_LIMIT")
            
            # Check exit cooldown
            if token_id in self._exit_cooldowns:
                if datetime.utcnow() < self._exit_cooldowns[token_id]:
                    reasons.append("EXIT_COOLDOWN_ACTIVE")
            
            is_valid = len(reasons) == 0
            
            if not is_valid:
                logger.debug(f"Trade validation failed: {reasons}")
            
            return is_valid, reasons
    
    async def add_position(
        self,
        position: Position,
        entry_edge: float,
        entry_confidence: float,
        trade_id: Optional[int] = None,
        calibration_sample_id: Optional[int] = None,
    ) -> PositionWithExitRules:
        """Add a new position with exit rules.
        
        Args:
            position: Position to track
            entry_edge: Edge at entry
            entry_confidence: Confidence at entry
            trade_id: Database trade ID
            calibration_sample_id: Calibration sample ID
            
        Returns:
            PositionWithExitRules object
        """
        async with self._lock:
            # Calculate exit prices
            entry_price = position.avg_entry_price
            
            # Profit take: entry_price + profit_take_pct
            profit_take_price = entry_price * (1 + self.limits.profit_take_pct)
            
            # Stop loss: entry_price - stop_loss_pct
            stop_loss_price = entry_price * (1 - self.limits.stop_loss_pct)
            
            pos_with_rules = PositionWithExitRules(
                position=position,
                entry_time=datetime.utcnow(),
                entry_edge=entry_edge,
                entry_confidence=entry_confidence,
                profit_take_price=min(0.99, profit_take_price),
                stop_loss_price=max(0.01, stop_loss_price),
                time_stop_hours=self.limits.time_stop_hours,
                trade_id=trade_id,
                calibration_sample_id=calibration_sample_id,
            )
            
            self._positions[position.token_id] = pos_with_rules
            
            # Update market exposure
            market_id = position.condition_id
            self._market_exposure[market_id] = (
                self._market_exposure.get(market_id, 0.0) + position.cost_basis
            )
            
            logger.info(
                f"Position added | {position.token_id[:16]}... | Size: {position.size:.2f} @ {entry_price:.4f} | "
                f"TP: {profit_take_price:.4f} | SL: {stop_loss_price:.4f}"
            )
            
            return pos_with_rules
    
    async def check_exit_conditions(
        self,
        token_id: str,
        current_price: float,
        current_adjusted_prob: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if position should be exited.
        
        Args:
            token_id: Token ID
            current_price: Current market price
            current_adjusted_prob: Current adjusted probability (for signal flip)
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        async with self._lock:
            if token_id not in self._positions:
                return False, None
            
            pos = self._positions[token_id]
            entry_price = pos.position.avg_entry_price
            
            # Check profit take
            if current_price >= pos.profit_take_price:
                return True, "PROFIT_TAKE"
            
            # Check stop loss
            if current_price <= pos.stop_loss_price:
                return True, "STOP_LOSS"
            
            # Check time stop
            if pos.is_time_stopped:
                return True, "TIME_STOP"
            
            # Check signal flip (if we have updated probability)
            if current_adjusted_prob is not None:
                current_edge = current_adjusted_prob - current_price
                # If edge has flipped negative or significantly degraded
                if current_edge < -0.02:  # Edge flipped by >2%
                    return True, "SIGNAL_FLIP"
            
            return False, None
    
    async def remove_position(
        self,
        token_id: str,
        exit_reason: str = "MANUAL",
        cooldown_hours: float = 1.0,
    ) -> Optional[PositionWithExitRules]:
        """Remove a position and apply exit cooldown.
        
        Args:
            token_id: Token ID of position to remove
            exit_reason: Reason for exit
            cooldown_hours: Hours to wait before re-entering
            
        Returns:
            Removed position or None if not found
        """
        async with self._lock:
            if token_id not in self._positions:
                return None
            
            pos = self._positions.pop(token_id)
            
            # Update market exposure
            market_id = pos.position.condition_id
            if market_id in self._market_exposure:
                self._market_exposure[market_id] -= pos.position.cost_basis
                if self._market_exposure[market_id] <= 0:
                    del self._market_exposure[market_id]
            
            # Apply cooldown
            self._exit_cooldowns[token_id] = datetime.utcnow() + timedelta(hours=cooldown_hours)
            
            logger.info(f"Position removed | {token_id[:16]}... | Reason: {exit_reason}")
            
            return pos
    
    async def update_position_price(self, token_id: str, new_price: float) -> None:
        """Update a position's current price and P&L.
        
        Args:
            token_id: Token ID of position to update
            new_price: New market price
        """
        async with self._lock:
            if token_id in self._positions:
                self._positions[token_id].position.update_unrealized_pnl(new_price)
                
                # Update total unrealized P&L
                self.daily_stats.unrealized_pnl = sum(
                    pos.position.unrealized_pnl for pos in self._positions.values()
                )
    
    async def record_trade_result(self, pnl: float) -> None:
        """Record the result of a completed trade.
        
        Args:
            pnl: Profit/loss from the trade in USDC
        """
        async with self._lock:
            is_winner = pnl > 0
            self.daily_stats.update_stats(pnl, is_winner)
            self.bankroll += pnl
            
            logger.info(
                f"Trade recorded | P&L: ${pnl:+.2f} | "
                f"Daily P&L: ${self.daily_stats.realized_pnl:+.2f} | "
                f"Drawdown: {self.daily_stats.current_drawdown_pct:.1f}%"
            )
            
            if self.daily_stats.kill_switch_triggered:
                logger.critical(
                    f"KILL SWITCH TRIGGERED! Drawdown: {self.daily_stats.current_drawdown_pct:.1f}% | "
                    "All trading halted."
                )
    
    async def reset_daily_stats(self) -> None:
        """Reset daily statistics for a new trading day."""
        async with self._lock:
            self.daily_stats = DailyStats(
                starting_bankroll=self.bankroll,
                current_bankroll=self.bankroll,
            )
            logger.info(f"Daily stats reset | Starting bankroll: ${self.bankroll:.2f}")
    
    async def get_positions_to_check(self) -> list[PositionWithExitRules]:
        """Get all positions for exit checking.
        
        Returns:
            List of positions with exit rules
        """
        async with self._lock:
            return list(self._positions.values())
    
    def get_portfolio_summary(self) -> dict:
        """Get a summary of current portfolio state.
        
        Returns:
            Dictionary containing portfolio metrics
        """
        return {
            "bankroll": self.bankroll,
            "available_capital": self.available_capital,
            "total_exposure": self.total_exposure,
            "exposure_ratio": round(self.exposure_ratio * 100, 1),
            "open_positions": len(self._positions),
            "unrealized_pnl": self.daily_stats.unrealized_pnl,
            "daily_pnl": self.daily_stats.total_pnl,
            "current_drawdown_pct": self.daily_stats.current_drawdown_pct,
            "kill_switch_triggered": self.daily_stats.kill_switch_triggered,
            "trades_today": self.daily_stats.trades_count,
            "win_rate": self.daily_stats.win_rate,
        }
