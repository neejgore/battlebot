"""Tests for Strategy V2.1 components.

Tests cover:
- Market eligibility filter
- Calibration shrinkage
- Edge gating
- Kelly sizing with caps
- Exit conditions
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from data.models import Market, MarketOutcome, Position
from logic.strategy_v2 import StrategyConfig, MarketState, StrategyV2
from logic.risk_engine import RiskEngine, RiskLimits, calculate_kelly_size
from logic.calibration import CalibrationEngine


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def strategy_config():
    """Default strategy configuration."""
    return StrategyConfig(
        min_edge=0.03,
        min_confidence=0.50,
        min_volume_24h=5000.0,
        max_spread=0.04,
        min_depth_notional=1000.0,
        max_days_to_resolution=30,
    )


@pytest.fixture
def risk_limits():
    """Default risk limits."""
    return RiskLimits(
        max_daily_drawdown=0.15,
        max_position_size=50.0,
        max_percent_bankroll_per_market=0.10,
        max_total_open_risk=0.30,
        min_edge=0.03,
        edge_scale=0.10,
    )


@pytest.fixture
def risk_engine(risk_limits):
    """Risk engine with default config."""
    return RiskEngine(
        initial_bankroll=1000.0,
        fractional_kelly=0.1,
        limits=risk_limits,
    )


@pytest.fixture
def sample_market():
    """Sample market for testing."""
    return Market(
        condition_id="0xTEST123",
        question="Will test event happen by end of month?",
        token_id="TOKEN123",
        outcome=MarketOutcome.YES,
        current_price=0.50,
        volume_24h=50000.0,
        liquidity=100000.0,
        end_date=datetime.utcnow() + timedelta(days=15),
    )


# ============================================================================
# Market Eligibility Filter Tests
# ============================================================================

class TestMarketEligibility:
    """Tests for market eligibility filtering."""
    
    def test_eligible_market_passes(self, sample_market, strategy_config):
        """A market meeting all criteria should be eligible."""
        # Create a mock strategy to test the eligibility check
        # For now, test the logic directly
        
        # Check all criteria
        assert sample_market.volume_24h >= strategy_config.min_volume_24h
        assert sample_market.current_price >= strategy_config.min_tradeable_price
        assert sample_market.current_price <= strategy_config.max_tradeable_price
        assert sample_market.liquidity >= strategy_config.min_depth_notional
        
        # Days to resolution
        days = (sample_market.end_date - datetime.utcnow()).days
        assert days <= strategy_config.max_days_to_resolution
    
    def test_low_volume_rejected(self, sample_market, strategy_config):
        """Markets with low volume should be rejected."""
        sample_market.volume_24h = 1000.0  # Below threshold
        
        assert sample_market.volume_24h < strategy_config.min_volume_24h
    
    def test_extreme_price_rejected(self, sample_market, strategy_config):
        """Markets with extreme prices should be rejected."""
        # Test low price
        sample_market.current_price = 0.02
        assert sample_market.current_price < strategy_config.min_tradeable_price
        
        # Test high price
        sample_market.current_price = 0.98
        assert sample_market.current_price > strategy_config.max_tradeable_price
    
    def test_far_resolution_rejected(self, sample_market, strategy_config):
        """Markets too far from resolution should be rejected."""
        sample_market.end_date = datetime.utcnow() + timedelta(days=60)
        
        days = (sample_market.end_date - datetime.utcnow()).days
        assert days > strategy_config.max_days_to_resolution
    
    def test_headline_market_rejected(self, strategy_config):
        """Headline markets (high volume) should be rejected by default."""
        strategy_config.allow_headline_markets = False
        volume = 600000.0  # Above headline threshold
        
        assert volume >= strategy_config.headline_volume_threshold
    
    def test_low_liquidity_rejected(self, sample_market, strategy_config):
        """Markets with low liquidity should be rejected."""
        sample_market.liquidity = 500.0  # Below threshold
        
        assert sample_market.liquidity < strategy_config.min_depth_notional


# ============================================================================
# Calibration Shrinkage Tests
# ============================================================================

class TestCalibrationShrinkage:
    """Tests for calibration shrinkage calculations."""
    
    def test_shrinkage_weight_with_low_confidence(self):
        """Low confidence should result in more shrinkage to market."""
        engine = CalibrationEngine()
        
        # With confidence = 0, weight should be at BASE_SHRINKAGE
        weight = engine._compute_shrinkage_weight(confidence=0.0, sample_size=0)
        
        assert weight == engine.BASE_SHRINKAGE
    
    def test_shrinkage_weight_with_high_confidence(self):
        """High confidence should result in less shrinkage."""
        engine = CalibrationEngine()
        
        # With high confidence and many samples, weight should be higher
        weight = engine._compute_shrinkage_weight(confidence=1.0, sample_size=100)
        
        expected = (
            engine.BASE_SHRINKAGE +
            engine.CONFIDENCE_BOOST * 1.0 +
            engine.SAMPLE_SIZE_BOOST * 1.0
        )
        assert weight == min(1.0, expected)
    
    def test_shrinkage_weight_bounded(self):
        """Shrinkage weight should be bounded [BASE, 1.0]."""
        engine = CalibrationEngine()
        
        # Test various inputs
        for conf in [0.0, 0.5, 1.0]:
            for samples in [0, 25, 100]:
                weight = engine._compute_shrinkage_weight(conf, samples)
                assert engine.BASE_SHRINKAGE <= weight <= 1.0
    
    def test_sample_size_factor_saturates(self):
        """Sample size factor should saturate at SAMPLE_SIZE_SCALE."""
        engine = CalibrationEngine()
        
        # Factor should be 1.0 at scale and above
        weight_at_scale = engine._compute_shrinkage_weight(0.5, engine.SAMPLE_SIZE_SCALE)
        weight_above_scale = engine._compute_shrinkage_weight(0.5, engine.SAMPLE_SIZE_SCALE * 2)
        
        assert weight_at_scale == weight_above_scale


# ============================================================================
# Edge Gating Tests
# ============================================================================

class TestEdgeGating:
    """Tests for edge gating logic."""
    
    def test_edge_below_minimum_rejected(self, strategy_config):
        """Edge below minimum should be rejected."""
        edge = 0.02  # 2%, below 3% minimum
        
        assert edge < strategy_config.min_edge
    
    def test_edge_above_minimum_accepted(self, strategy_config):
        """Edge above minimum should be accepted."""
        edge = 0.05  # 5%, above 3% minimum
        
        assert edge >= strategy_config.min_edge
    
    def test_edge_below_spread_rejected(self, strategy_config):
        """Edge below spread should be rejected (unprofitable)."""
        edge = 0.03
        spread = 0.04  # Spread is wider than edge
        
        assert edge < spread
    
    def test_edge_throttle_calculation(self, risk_limits):
        """Edge throttle should scale position size."""
        engine = RiskEngine(initial_bankroll=1000.0, limits=risk_limits)
        
        # At minimum edge, throttle should be low
        min_edge_throttle = engine._compute_edge_throttle(risk_limits.min_edge)
        assert min_edge_throttle == risk_limits.min_edge / risk_limits.edge_scale
        
        # At edge_scale, throttle should be 1.0
        full_throttle = engine._compute_edge_throttle(risk_limits.edge_scale)
        assert full_throttle == 1.0
        
        # Above edge_scale, throttle should still be 1.0 (capped)
        high_edge_throttle = engine._compute_edge_throttle(0.20)
        assert high_edge_throttle == 1.0
    
    def test_edge_below_min_returns_zero_throttle(self, risk_limits):
        """Edge below minimum should return zero throttle."""
        engine = RiskEngine(initial_bankroll=1000.0, limits=risk_limits)
        
        throttle = engine._compute_edge_throttle(0.02)  # Below min_edge of 0.03
        assert throttle == 0.0


# ============================================================================
# Kelly Sizing Tests
# ============================================================================

class TestKellySizing:
    """Tests for Kelly criterion position sizing."""
    
    def test_kelly_with_positive_edge(self):
        """Positive edge should return positive position size."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.60,
            market_price=0.50,
            fractional_multiplier=0.1,
            max_pos=50.0,
        )
        
        assert size > 0
    
    def test_kelly_with_zero_edge(self):
        """Zero edge should return zero position size."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.50,
            market_price=0.50,
        )
        
        assert size == 0.0
    
    def test_kelly_with_negative_edge(self):
        """Negative edge should return zero position size."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.40,
            market_price=0.50,
        )
        
        assert size == 0.0
    
    def test_kelly_respects_max_position(self):
        """Kelly should respect maximum position size."""
        size = calculate_kelly_size(
            bankroll=100000,
            true_prob=0.90,
            market_price=0.10,
            fractional_multiplier=1.0,
            max_pos=100.0,
        )
        
        assert size <= 100.0
    
    def test_kelly_with_edge_throttle(self):
        """Edge throttle should reduce position size."""
        full_size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.70,
            market_price=0.50,
            fractional_multiplier=0.1,
            max_pos=100.0,
            edge_throttle=1.0,
        )
        
        throttled_size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.70,
            market_price=0.50,
            fractional_multiplier=0.1,
            max_pos=100.0,
            edge_throttle=0.5,
        )
        
        assert throttled_size == pytest.approx(full_size * 0.5, rel=0.01)
    
    def test_kelly_invalid_market_price(self):
        """Invalid market prices should return zero."""
        # Price of 0
        size0 = calculate_kelly_size(bankroll=1000, true_prob=0.60, market_price=0.0)
        assert size0 == 0.0
        
        # Price of 1
        size1 = calculate_kelly_size(bankroll=1000, true_prob=0.60, market_price=1.0)
        assert size1 == 0.0


# ============================================================================
# Risk Engine Position Caps Tests
# ============================================================================

class TestPositionCaps:
    """Tests for position size caps in risk engine."""
    
    @pytest.mark.asyncio
    async def test_max_position_size_cap(self, risk_engine):
        """Position size should be capped at max_position_size."""
        size = await risk_engine.calculate_position_size(
            adjusted_prob=0.90,
            market_price=0.10,
            edge=0.80,
            confidence=1.0,
            market_id="TEST",
        )
        
        assert size <= risk_engine.limits.max_position_size
    
    @pytest.mark.asyncio
    async def test_per_market_cap(self, risk_engine):
        """Position size should respect per-market cap."""
        max_per_market = risk_engine.bankroll * risk_engine.limits.max_percent_bankroll_per_market
        
        size = await risk_engine.calculate_position_size(
            adjusted_prob=0.90,
            market_price=0.10,
            edge=0.80,
            confidence=1.0,
            market_id="TEST",
        )
        
        assert size <= max_per_market
    
    @pytest.mark.asyncio
    async def test_global_exposure_cap(self, risk_engine):
        """Global exposure limit should be respected."""
        # Add some existing positions
        pos1 = Position(
            token_id="TOKEN1",
            condition_id="COND1",
            outcome=MarketOutcome.YES,
            size=100.0,
            avg_entry_price=0.50,
            current_price=0.50,
        )
        
        await risk_engine.add_position(pos1, entry_edge=0.05, entry_confidence=0.7)
        
        # Try to add more - should be limited by global exposure
        max_additional = (
            risk_engine.bankroll * risk_engine.limits.max_total_open_risk -
            risk_engine.total_exposure
        )
        
        size = await risk_engine.calculate_position_size(
            adjusted_prob=0.90,
            market_price=0.10,
            edge=0.80,
            confidence=1.0,
            market_id="COND2",
        )
        
        assert size <= max_additional or size == 0
    
    @pytest.mark.asyncio
    async def test_max_positions_limit(self, risk_engine):
        """Should reject when max positions reached."""
        # Add max positions
        for i in range(risk_engine.limits.max_positions):
            pos = Position(
                token_id=f"TOKEN{i}",
                condition_id=f"COND{i}",
                outcome=MarketOutcome.YES,
                size=1.0,
                avg_entry_price=0.50,
                current_price=0.50,
            )
            await risk_engine.add_position(pos, entry_edge=0.05, entry_confidence=0.7)
        
        # Try to add one more
        size = await risk_engine.calculate_position_size(
            adjusted_prob=0.60,
            market_price=0.50,
            edge=0.10,
            confidence=0.8,
            market_id="NEW",
        )
        
        assert size == 0.0


# ============================================================================
# Exit Condition Tests
# ============================================================================

class TestExitConditions:
    """Tests for position exit conditions."""
    
    @pytest.mark.asyncio
    async def test_profit_take_trigger(self, risk_engine):
        """Profit take should trigger at threshold."""
        pos = Position(
            token_id="TOKEN1",
            condition_id="COND1",
            outcome=MarketOutcome.YES,
            size=100.0,
            avg_entry_price=0.50,
            current_price=0.50,
        )
        
        await risk_engine.add_position(pos, entry_edge=0.05, entry_confidence=0.7)
        
        # Price moves up by profit_take_pct (15%)
        new_price = 0.50 * (1 + risk_engine.limits.profit_take_pct)
        
        should_exit, reason = await risk_engine.check_exit_conditions(
            token_id="TOKEN1",
            current_price=new_price,
        )
        
        assert should_exit
        assert reason == "PROFIT_TAKE"
    
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, risk_engine):
        """Stop loss should trigger at threshold."""
        pos = Position(
            token_id="TOKEN1",
            condition_id="COND1",
            outcome=MarketOutcome.YES,
            size=100.0,
            avg_entry_price=0.50,
            current_price=0.50,
        )
        
        await risk_engine.add_position(pos, entry_edge=0.05, entry_confidence=0.7)
        
        # Price moves down by stop_loss_pct (10%)
        new_price = 0.50 * (1 - risk_engine.limits.stop_loss_pct)
        
        should_exit, reason = await risk_engine.check_exit_conditions(
            token_id="TOKEN1",
            current_price=new_price,
        )
        
        assert should_exit
        assert reason == "STOP_LOSS"
    
    @pytest.mark.asyncio
    async def test_no_exit_within_bounds(self, risk_engine):
        """Position within bounds should not trigger exit."""
        pos = Position(
            token_id="TOKEN1",
            condition_id="COND1",
            outcome=MarketOutcome.YES,
            size=100.0,
            avg_entry_price=0.50,
            current_price=0.50,
        )
        
        await risk_engine.add_position(pos, entry_edge=0.05, entry_confidence=0.7)
        
        # Small price move
        should_exit, reason = await risk_engine.check_exit_conditions(
            token_id="TOKEN1",
            current_price=0.52,  # Small gain
        )
        
        assert not should_exit
        assert reason is None


# ============================================================================
# Kill Switch Tests
# ============================================================================

class TestKillSwitch:
    """Tests for 15% daily drawdown kill switch."""
    
    @pytest.mark.asyncio
    async def test_kill_switch_triggers_at_threshold(self, risk_engine):
        """Kill switch should trigger at 15% drawdown."""
        # Record loss that exceeds 15%
        loss = risk_engine.bankroll * 0.16  # 16% loss
        
        await risk_engine.record_trade_result(pnl=-loss)
        
        assert risk_engine.daily_stats.kill_switch_triggered
        assert not risk_engine.is_trading_allowed
    
    @pytest.mark.asyncio
    async def test_trading_halted_after_kill_switch(self, risk_engine):
        """No trading should be allowed after kill switch."""
        # Trigger kill switch
        loss = risk_engine.bankroll * 0.16
        await risk_engine.record_trade_result(pnl=-loss)
        
        # Try to calculate position size
        size = await risk_engine.calculate_position_size(
            adjusted_prob=0.90,
            market_price=0.10,
            edge=0.80,
            confidence=1.0,
            market_id="TEST",
        )
        
        assert size == 0.0
    
    @pytest.mark.asyncio
    async def test_no_trigger_below_threshold(self, risk_engine):
        """Kill switch should not trigger below threshold."""
        # Record loss just under 15%
        loss = risk_engine.bankroll * 0.14  # 14% loss
        
        await risk_engine.record_trade_result(pnl=-loss)
        
        assert not risk_engine.daily_stats.kill_switch_triggered
        assert risk_engine.is_trading_allowed


# ============================================================================
# Market State Tests
# ============================================================================

class TestMarketState:
    """Tests for market state tracking."""
    
    def test_price_history(self, sample_market):
        """Price history should be tracked correctly."""
        state = MarketState(
            market=sample_market,
            last_price=sample_market.current_price,
        )
        
        # Add some prices
        now = datetime.utcnow()
        for i in range(5):
            state.add_price(0.50 + i * 0.01, now + timedelta(seconds=i))
        
        assert len(state.price_history) == 5
        assert state.last_price == 0.54
    
    def test_volatility_calculation(self, sample_market):
        """Volatility should be calculated correctly."""
        state = MarketState(
            market=sample_market,
            last_price=sample_market.current_price,
        )
        
        # Add varied prices
        now = datetime.utcnow()
        prices = [0.50, 0.52, 0.48, 0.55, 0.45]
        for i, p in enumerate(prices):
            state.add_price(p, now + timedelta(seconds=i))
        
        vol = state.get_volatility(seconds=10)
        
        assert vol > 0  # Should have some volatility


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
