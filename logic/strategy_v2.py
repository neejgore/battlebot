"""Event-Driven Strategy V2.1 for Polymarket Battle-Bot.

Calibration-first, overtrade-resistant trading strategy with:
1. Market eligibility filtering
2. Evidence-gated LLM probability estimation
3. Calibration layer for probability adjustment
4. Edge gating with liquidity checks
5. Fractional Kelly position sizing
6. Trade lifecycle management (entry -> monitor -> exit)
7. Full telemetry logging
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from loguru import logger

from data.models import (
    Market,
    PriceUpdate,
    OrderSide,
    Position,
    MarketOutcome,
)
from data.database import TelemetryDB, get_db
from logic.ai_signal import AISignalGenerator, AISignalResult, get_ai_generator
from logic.calibration import CalibrationEngine, get_calibration_engine
from logic.risk_engine import RiskEngine, RiskLimits
from services.websocket_client import WebSocketClient
from services.execution_engine import ExecutionEngine


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    
    # Market eligibility
    max_days_to_resolution: int = 30
    min_volume_24h: float = 5000.0  # $5k minimum volume
    max_spread: float = 0.04  # 4 cents max spread
    min_depth_notional: float = 1000.0  # $1k minimum depth
    
    # Price bounds - avoid extreme prices unless justified
    min_tradeable_price: float = 0.05
    max_tradeable_price: float = 0.95
    
    # Headline market filter (high-volume efficient markets to avoid)
    headline_volume_threshold: float = 500000.0  # $500k marks "headline" markets
    allow_headline_markets: bool = False
    
    # Event triggering
    min_price_change: float = 0.02  # 2% price move triggers analysis
    volatility_window_seconds: int = 300  # 5 min window for volatility
    volatility_threshold: float = 0.03  # 3% std dev triggers analysis
    
    # Cooldowns
    analysis_cooldown_seconds: int = 600  # 10 min between analyses per market
    exit_cooldown_hours: float = 2.0  # 2 hours before re-entry after exit
    
    # Edge gating
    min_edge: float = 0.03  # 3% minimum edge
    min_confidence: float = 0.50  # 50% minimum confidence
    min_information_quality: str = "medium"  # Require at least medium info quality
    
    # Position monitoring
    position_check_interval_seconds: int = 60  # Check positions every minute


@dataclass
class MarketState:
    """Tracked state for a monitored market."""
    market: Market
    last_price: float
    last_analysis_time: Optional[datetime] = None
    price_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Resolution rules (required for trading)
    resolution_rules: Optional[str] = None
    category: Optional[str] = None
    
    # Eligibility cache
    is_eligible: bool = False
    ineligibility_reasons: list[str] = field(default_factory=list)
    
    def add_price(self, price: float, timestamp: datetime) -> None:
        """Add price to history."""
        self.price_history.append((timestamp, price))
        self.last_price = price
    
    def get_recent_prices(self, seconds: int = 300) -> list[float]:
        """Get prices from the last N seconds."""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [p for t, p in self.price_history if t >= cutoff]
    
    def get_volatility(self, seconds: int = 300) -> float:
        """Calculate price volatility (std dev) over window."""
        prices = self.get_recent_prices(seconds)
        if len(prices) < 2:
            return 0.0
        
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return variance ** 0.5


# ============================================================================
# Strategy V2.1
# ============================================================================

class StrategyV2:
    """Event-driven trading strategy with calibration and risk management.
    
    Pipeline:
    1. Price update received
    2. Check market eligibility
    3. Check trigger conditions (price move, volatility)
    4. Generate AI signal (structured output)
    5. Apply calibration
    6. Check edge gates
    7. Calculate position size
    8. Execute trade
    9. Log everything
    """
    
    def __init__(
        self,
        risk_engine: RiskEngine,
        execution_engine: ExecutionEngine,
        ws_client: WebSocketClient,
        config: Optional[StrategyConfig] = None,
        db: Optional[TelemetryDB] = None,
        ai_generator: Optional[AISignalGenerator] = None,
        calibration_engine: Optional[CalibrationEngine] = None,
    ):
        """Initialize the strategy.
        
        Args:
            risk_engine: Risk management engine
            execution_engine: Order execution engine
            ws_client: WebSocket client for price feeds
            config: Strategy configuration
            db: Telemetry database
            ai_generator: AI signal generator
            calibration_engine: Calibration engine
        """
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.ws_client = ws_client
        self.config = config or StrategyConfig()
        
        self._db = db
        self._ai_generator = ai_generator
        self._calibration_engine = calibration_engine
        
        # Market tracking
        self._markets: dict[str, MarketState] = {}
        self._running = False
        self._lock = asyncio.Lock()
        self._position_monitor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "signals_generated": 0,
            "signals_rejected": 0,
            "trades_executed": 0,
            "ai_calls": 0,
            "ai_failures": 0,
        }
        
        logger.info(
            f"StrategyV2.1 initialized | Min edge: {self.config.min_edge:.1%} | "
            f"Min confidence: {self.config.min_confidence:.1%} | "
            f"Cooldown: {self.config.analysis_cooldown_seconds}s"
        )
    
    async def _ensure_dependencies(self) -> None:
        """Ensure all dependencies are initialized."""
        if self._db is None:
            self._db = await get_db()
        if self._ai_generator is None:
            self._ai_generator = get_ai_generator()
        if self._calibration_engine is None:
            self._calibration_engine = await get_calibration_engine()
    
    # ========================================================================
    # Market Management
    # ========================================================================
    
    async def add_market(
        self,
        market: Market,
        resolution_rules: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[bool, list[str]]:
        """Add a market to monitor.
        
        Args:
            market: Market to add
            resolution_rules: Market resolution rules text
            category: Market category for calibration
            
        Returns:
            Tuple of (is_eligible, list of ineligibility reasons)
        """
        await self._ensure_dependencies()
        
        async with self._lock:
            state = MarketState(
                market=market,
                last_price=market.current_price,
                resolution_rules=resolution_rules,
                category=category,
            )
            
            # Check eligibility
            is_eligible, reasons = self._check_market_eligibility(market, resolution_rules)
            state.is_eligible = is_eligible
            state.ineligibility_reasons = reasons
            
            self._markets[market.token_id] = state
            
            # Subscribe to WebSocket updates
            await self.ws_client.subscribe([market.token_id])
            
            status = "ELIGIBLE" if is_eligible else f"INELIGIBLE ({', '.join(reasons)})"
            logger.info(f"Market added: {market.question[:50]}... | {status}")
            
            return is_eligible, reasons
    
    async def remove_market(self, token_id: str) -> None:
        """Remove a market from monitoring."""
        async with self._lock:
            if token_id in self._markets:
                del self._markets[token_id]
                await self.ws_client.unsubscribe([token_id])
                logger.info(f"Market removed: {token_id[:16]}...")
    
    def _check_market_eligibility(
        self,
        market: Market,
        resolution_rules: Optional[str],
    ) -> tuple[bool, list[str]]:
        """Check if market is eligible for trading.
        
        Args:
            market: Market to check
            resolution_rules: Resolution rules text
            
        Returns:
            Tuple of (is_eligible, list of reason codes)
        """
        reasons = []
        
        # Must have resolution rules
        if not resolution_rules:
            reasons.append("NO_RESOLUTION_RULES")
        
        # Time to resolution check
        if market.end_date:
            days_remaining = (market.end_date - datetime.utcnow()).days
            if days_remaining > self.config.max_days_to_resolution:
                reasons.append(f"TOO_FAR_OUT_{days_remaining}d")
        
        # Volume check
        if market.volume_24h < self.config.min_volume_24h:
            reasons.append(f"LOW_VOLUME_{market.volume_24h:.0f}")
        
        # Headline market check (avoid efficient high-volume markets)
        if not self.config.allow_headline_markets:
            if market.volume_24h >= self.config.headline_volume_threshold:
                reasons.append("HEADLINE_MARKET")
        
        # Price bounds
        price = market.current_price
        if price < self.config.min_tradeable_price:
            reasons.append(f"PRICE_TOO_LOW_{price:.2f}")
        elif price > self.config.max_tradeable_price:
            reasons.append(f"PRICE_TOO_HIGH_{price:.2f}")
        
        # Liquidity check (if we have depth info)
        if market.liquidity < self.config.min_depth_notional:
            reasons.append(f"LOW_LIQUIDITY_{market.liquidity:.0f}")
        
        return len(reasons) == 0, reasons
    
    # ========================================================================
    # Event Handling
    # ========================================================================
    
    async def on_price_update(self, update: PriceUpdate) -> None:
        """Handle incoming price update from WebSocket.
        
        This is the main event handler that triggers the strategy logic.
        """
        token_id = update.token_id
        
        async with self._lock:
            if token_id not in self._markets:
                return
            
            state = self._markets[token_id]
            now = datetime.utcnow()
            
            # Record price tick for backtesting
            if self._db:
                await self._db.log_price_tick(
                    market_id=state.market.condition_id,
                    token_id=token_id,
                    mid_price=update.price,
                    bid=update.bid,
                    ask=update.ask,
                    spread=update.spread,
                )
            
            # Update state
            old_price = state.last_price
            state.add_price(update.price, now)
            state.market.current_price = update.price
            state.market.updated_at = now
            
            # Update positions
            await self.risk_engine.update_position_price(token_id, update.price)
            
            # Check if market is eligible
            if not state.is_eligible:
                return
            
            # Check trigger conditions
            should_analyze, trigger_reason = self._check_triggers(state, old_price, update)
            
            if not should_analyze:
                return
            
            # Check cooldown
            if state.last_analysis_time:
                elapsed = (now - state.last_analysis_time).total_seconds()
                if elapsed < self.config.analysis_cooldown_seconds:
                    logger.debug(
                        f"Cooldown active for {token_id[:16]}... "
                        f"({elapsed:.0f}s / {self.config.analysis_cooldown_seconds}s)"
                    )
                    return
            
            # Check if trading is allowed
            if not self.risk_engine.is_trading_allowed:
                logger.warning("Trading halted - skipping analysis")
                return
        
        # Trigger analysis (outside lock)
        logger.info(
            f"Analysis triggered | {trigger_reason} | "
            f"Market: {state.market.question[:40]}... | Price: {update.price:.4f}"
        )
        
        asyncio.create_task(self._analyze_and_trade(state, update))
    
    def _check_triggers(
        self,
        state: MarketState,
        old_price: float,
        update: PriceUpdate,
    ) -> tuple[bool, str]:
        """Check if analysis should be triggered.
        
        Returns:
            Tuple of (should_trigger, trigger_reason)
        """
        # Price movement trigger
        price_change = abs(update.price - old_price)
        if price_change >= self.config.min_price_change:
            return True, f"PRICE_MOVE_{price_change:.2%}"
        
        # Volatility spike trigger
        volatility = state.get_volatility(self.config.volatility_window_seconds)
        if volatility >= self.config.volatility_threshold:
            return True, f"VOLATILITY_SPIKE_{volatility:.2%}"
        
        # Spread compression + depth increase (liquidity improving)
        if update.spread is not None and update.spread < self.config.max_spread * 0.5:
            # Spread is tight - potential opportunity
            return True, f"SPREAD_TIGHT_{update.spread:.3f}"
        
        return False, ""
    
    # ========================================================================
    # Analysis Pipeline
    # ========================================================================
    
    async def _analyze_and_trade(self, state: MarketState, update: PriceUpdate) -> None:
        """Full analysis and trade pipeline.
        
        Pipeline:
        1. Get AI signal
        2. Apply calibration
        3. Check edge gates
        4. Calculate position size
        5. Validate trade
        6. Execute
        7. Log decision
        """
        start_time = time.monotonic()
        market = state.market
        token_id = market.token_id
        
        # Initialize decision log
        decision_data = {
            "market_id": market.condition_id,
            "token_id": token_id,
            "price": update.price,
            "bid": update.bid,
            "ask": update.ask,
            "spread": update.spread,
            "volume_24h": market.volume_24h,
        }
        reason_codes = []
        
        try:
            # Update last analysis time
            state.last_analysis_time = datetime.utcnow()
            
            # ================================================================
            # Step 1: Get AI Signal
            # ================================================================
            self._stats["ai_calls"] += 1
            
            recent_prices = state.get_recent_prices(300)
            
            ai_result = await self._ai_generator.generate_signal(
                market_question=market.question,
                current_price=update.price,
                spread=update.spread or 0.02,
                resolution_rules=state.resolution_rules,
                resolution_date=market.end_date,
                volume_24h=market.volume_24h,
                liquidity=market.liquidity,
                recent_price_path=recent_prices,
                category=state.category,
            )
            
            decision_data["ai_latency_ms"] = ai_result.latency_ms
            
            if not ai_result.success or ai_result.signal is None:
                self._stats["ai_failures"] += 1
                reason_codes.append(f"AI_FAILED:{ai_result.error}")
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            signal = ai_result.signal
            decision_data["raw_prob"] = signal.raw_prob
            decision_data["confidence"] = signal.confidence
            
            logger.debug(
                f"AI Signal | Prob: {signal.raw_prob:.2%} | Conf: {signal.confidence:.2%} | "
                f"Info quality: {signal.information_quality}"
            )
            
            # ================================================================
            # Step 2: Check AI output quality gates
            # ================================================================
            
            # Confidence gate
            if signal.confidence < self.config.min_confidence:
                reason_codes.append(f"LOW_CONFIDENCE_{signal.confidence:.2f}")
            
            # Information quality gate
            quality_order = {"high": 3, "medium": 2, "low": 1}
            min_quality = quality_order.get(self.config.min_information_quality, 2)
            actual_quality = quality_order.get(signal.information_quality, 1)
            if actual_quality < min_quality:
                reason_codes.append(f"LOW_INFO_QUALITY_{signal.information_quality}")
            
            # Base rate check
            if not signal.base_rate_considered:
                reason_codes.append("NO_BASE_RATE")
            
            if reason_codes:
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            # ================================================================
            # Step 3: Apply Calibration
            # ================================================================
            
            calib_result = await self._calibration_engine.calibrate(
                raw_prob=signal.raw_prob,
                market_price=update.price,
                confidence=signal.confidence,
                category=state.category,
            )
            
            decision_data["calibrated_prob"] = calib_result.calibrated_prob
            decision_data["adjusted_prob"] = calib_result.adjusted_prob
            
            # Determine side and calculate edge
            adjusted_prob = calib_result.adjusted_prob
            
            if adjusted_prob > update.price:
                side = OrderSide.BUY
                edge = adjusted_prob - update.price
            else:
                side = OrderSide.SELL
                edge = (1 - adjusted_prob) - (1 - update.price)
            
            decision_data["edge"] = edge
            decision_data["side"] = side.value
            
            logger.info(
                f"Calibration | Raw: {signal.raw_prob:.2%} -> Adj: {adjusted_prob:.2%} | "
                f"Edge: {edge:+.2%} | Shrinkage: {calib_result.shrinkage_weight:.2f}"
            )
            
            # ================================================================
            # Step 4: Edge Gating
            # ================================================================
            
            if edge < self.config.min_edge:
                reason_codes.append(f"EDGE_TOO_LOW_{edge:.3f}")
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            # Spread check (edge must exceed spread to be profitable)
            spread = update.spread or 0.02
            if edge < spread:
                reason_codes.append(f"EDGE_BELOW_SPREAD_{edge:.3f}<{spread:.3f}")
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            # ================================================================
            # Step 5: Position Sizing
            # ================================================================
            
            position_size = await self.risk_engine.calculate_position_size(
                adjusted_prob=adjusted_prob,
                market_price=update.price,
                edge=edge,
                confidence=signal.confidence,
                market_id=market.condition_id,
            )
            
            if position_size <= 0:
                reason_codes.append("ZERO_POSITION_SIZE")
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            decision_data["size"] = position_size
            
            # ================================================================
            # Step 6: Trade Validation
            # ================================================================
            
            is_valid, validation_reasons = await self.risk_engine.validate_trade(
                size=position_size,
                price=update.price,
                token_id=token_id,
            )
            
            if not is_valid:
                reason_codes.extend(validation_reasons)
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            # ================================================================
            # Step 7: Execute Trade
            # ================================================================
            
            self._stats["signals_generated"] += 1
            
            # Calculate shares from USDC size
            shares = position_size / update.price
            
            logger.info(
                f"Executing trade | {side.value} {shares:.2f} shares @ {update.price:.4f} | "
                f"Size: ${position_size:.2f} | Edge: {edge:.2%}"
            )
            
            order = await self.execution_engine.create_order(
                token_id=token_id,
                side=side,
                price=update.price,
                size=shares,
            )
            
            if not order.order_id:
                reason_codes.append(f"ORDER_FAILED_{order.status}")
                await self._log_no_trade(decision_data, reason_codes, start_time)
                return
            
            decision_data["order_id"] = order.order_id
            self._stats["trades_executed"] += 1
            
            # ================================================================
            # Step 8: Record Trade and Position
            # ================================================================
            
            # Log decision
            total_latency = int((time.monotonic() - start_time) * 1000)
            decision_data["total_latency_ms"] = total_latency
            
            decision_id = await self._db.log_decision(
                decision="TRADE",
                reason_codes=["TRADE_EXECUTED"],
                **decision_data
            )
            
            # Log trade entry
            trade_id = await self._db.log_trade_entry(
                decision_id=decision_id,
                market_id=market.condition_id,
                token_id=token_id,
                entry_price=update.price,
                entry_side=side.value,
                size=shares,
                raw_prob=signal.raw_prob,
                adjusted_prob=adjusted_prob,
                edge=edge,
                confidence=signal.confidence,
            )
            
            # Record calibration sample
            calib_sample_id = await self._calibration_engine.record_prediction(
                market_id=market.condition_id,
                raw_prob=signal.raw_prob,
                market_price=update.price,
                category=state.category,
                calibrated_prob=calib_result.calibrated_prob,
            )
            
            # Create position with exit rules
            position = Position(
                token_id=token_id,
                condition_id=market.condition_id,
                outcome=market.outcome,
                size=shares,
                avg_entry_price=update.price,
                current_price=update.price,
            )
            
            await self.risk_engine.add_position(
                position=position,
                entry_edge=edge,
                entry_confidence=signal.confidence,
                trade_id=trade_id,
                calibration_sample_id=calib_sample_id,
            )
            
            logger.info(
                f"Trade executed | Order: {order.order_id} | "
                f"Reasoning: {signal.key_reasons[0][:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            reason_codes.append(f"EXCEPTION:{str(e)[:50]}")
            await self._log_no_trade(decision_data, reason_codes, start_time)
    
    async def _log_no_trade(
        self,
        decision_data: dict,
        reason_codes: list[str],
        start_time: float,
    ) -> None:
        """Log a no-trade decision."""
        self._stats["signals_rejected"] += 1
        
        total_latency = int((time.monotonic() - start_time) * 1000)
        decision_data["total_latency_ms"] = total_latency
        
        if self._db:
            await self._db.log_decision(
                decision="NO_TRADE",
                reason_codes=reason_codes,
                **decision_data
            )
        
        logger.debug(f"NO_TRADE | Reasons: {reason_codes}")
    
    # ========================================================================
    # Position Monitoring
    # ========================================================================
    
    async def _position_monitor_loop(self) -> None:
        """Monitor positions for exit conditions."""
        while self._running:
            try:
                await asyncio.sleep(self.config.position_check_interval_seconds)
                
                positions = await self.risk_engine.get_positions_to_check()
                
                for pos_with_rules in positions:
                    token_id = pos_with_rules.position.token_id
                    current_price = pos_with_rules.position.current_price
                    
                    # Check exit conditions
                    should_exit, exit_reason = await self.risk_engine.check_exit_conditions(
                        token_id=token_id,
                        current_price=current_price,
                    )
                    
                    if should_exit:
                        await self._execute_exit(pos_with_rules, exit_reason)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
    
    async def _execute_exit(
        self,
        pos_with_rules,
        exit_reason: str,
    ) -> None:
        """Execute position exit.
        
        Args:
            pos_with_rules: Position with exit rules
            exit_reason: Reason for exit
        """
        position = pos_with_rules.position
        token_id = position.token_id
        current_price = position.current_price
        
        logger.info(
            f"Exiting position | {token_id[:16]}... | Reason: {exit_reason} | "
            f"Entry: {position.avg_entry_price:.4f} -> Current: {current_price:.4f}"
        )
        
        try:
            # Create exit order (opposite side)
            exit_side = OrderSide.SELL  # Assuming we were long
            
            order = await self.execution_engine.create_order(
                token_id=token_id,
                side=exit_side,
                price=current_price,
                size=position.size,
            )
            
            if order.order_id:
                # Calculate P&L
                pnl = (current_price - position.avg_entry_price) * position.size
                
                # Log trade exit
                if pos_with_rules.trade_id and self._db:
                    await self._db.log_trade_exit(
                        trade_id=pos_with_rules.trade_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl=pnl,
                    )
                
                # Record trade result
                await self.risk_engine.record_trade_result(pnl)
                
                # Remove position with cooldown
                await self.risk_engine.remove_position(
                    token_id=token_id,
                    exit_reason=exit_reason,
                    cooldown_hours=self.config.exit_cooldown_hours,
                )
                
                logger.info(f"Position exited | P&L: ${pnl:+.2f} | Reason: {exit_reason}")
            else:
                logger.warning(f"Exit order failed: {order.status}")
                
        except Exception as e:
            logger.error(f"Exit execution failed: {e}")
    
    # ========================================================================
    # Lifecycle
    # ========================================================================
    
    async def start(self) -> None:
        """Start the strategy."""
        await self._ensure_dependencies()
        
        self._running = True
        
        # Register callback for price updates
        self.ws_client.register_callback(self.on_price_update)
        
        # Start position monitor
        self._position_monitor_task = asyncio.create_task(self._position_monitor_loop())
        
        logger.info("StrategyV2.1 started")
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self._running = False
        
        # Stop position monitor
        if self._position_monitor_task:
            self._position_monitor_task.cancel()
            try:
                await self._position_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Unregister callback
        self.ws_client.unregister_callback(self.on_price_update)
        
        logger.info("StrategyV2.1 stopped")
    
    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            "running": self._running,
            "markets_monitored": len(self._markets),
            "eligible_markets": sum(1 for s in self._markets.values() if s.is_eligible),
            **self._stats,
            "config": {
                "min_edge": self.config.min_edge,
                "min_confidence": self.config.min_confidence,
                "analysis_cooldown": self.config.analysis_cooldown_seconds,
            }
        }
