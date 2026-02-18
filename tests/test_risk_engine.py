"""Tests for the Risk Engine and Kelly Criterion calculations."""

import pytest
from logic.risk_engine import calculate_kelly_size, RiskEngine
from data.models import TradeSignal, OrderSide


class TestKellyCriterion:
    """Test suite for the Fractional Kelly Criterion calculator."""

    def test_positive_edge_returns_bet_size(self):
        """With positive edge, should return a positive bet size."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.60,
            market_price=0.50,
            fractional_multiplier=0.1,
            max_pos=50.0,
        )
        assert size > 0
        assert size <= 50.0

    def test_zero_edge_returns_zero(self):
        """With zero edge, should return 0."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.50,
            market_price=0.50,
        )
        assert size == 0.0

    def test_negative_edge_returns_zero(self):
        """With negative edge (true prob < market price), should return 0."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.40,
            market_price=0.50,
        )
        assert size == 0.0

    def test_respects_max_position(self):
        """Bet size should never exceed max_pos."""
        size = calculate_kelly_size(
            bankroll=100000,
            true_prob=0.90,
            market_price=0.10,
            fractional_multiplier=1.0,  # Full Kelly
            max_pos=100.0,
        )
        assert size <= 100.0

    def test_invalid_market_price_zero(self):
        """Market price of 0 should return 0."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.60,
            market_price=0.0,
        )
        assert size == 0.0

    def test_invalid_market_price_one(self):
        """Market price of 1 should return 0."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.60,
            market_price=1.0,
        )
        assert size == 0.0

    def test_fractional_kelly_reduces_size(self):
        """Fractional Kelly should reduce bet size proportionally."""
        full_kelly = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.70,
            market_price=0.50,
            fractional_multiplier=1.0,
            max_pos=1000.0,
        )
        half_kelly = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.70,
            market_price=0.50,
            fractional_multiplier=0.5,
            max_pos=1000.0,
        )
        assert half_kelly == pytest.approx(full_kelly / 2, rel=0.01)

    def test_small_bankroll(self):
        """Should work correctly with small bankrolls."""
        size = calculate_kelly_size(
            bankroll=10,
            true_prob=0.60,
            market_price=0.40,
            fractional_multiplier=0.1,
            max_pos=50.0,
        )
        assert size >= 0
        assert size <= 10  # Can't bet more than bankroll

    def test_returns_rounded_value(self):
        """Result should be rounded to 2 decimal places."""
        size = calculate_kelly_size(
            bankroll=1000,
            true_prob=0.65,
            market_price=0.45,
        )
        assert size == round(size, 2)


class TestRiskEngine:
    """Test suite for the Risk Engine."""

    @pytest.fixture
    def risk_engine(self):
        """Create a RiskEngine instance for testing."""
        return RiskEngine(
            initial_bankroll=1000.0,
            fractional_kelly=0.1,
            max_position_size=50.0,
        )

    def test_initialization(self, risk_engine):
        """RiskEngine should initialize with correct values."""
        assert risk_engine.bankroll == 1000.0
        assert risk_engine.fractional_kelly == 0.1
        assert risk_engine.max_position_size == 50.0
        assert risk_engine.is_trading_allowed is True

    def test_kill_switch_not_triggered_initially(self, risk_engine):
        """Kill switch should not be triggered on init."""
        assert risk_engine.daily_stats.kill_switch_triggered is False

    @pytest.mark.asyncio
    async def test_record_trade_updates_stats(self, risk_engine):
        """Recording a trade should update daily stats."""
        await risk_engine.record_trade_result(pnl=50.0)
        
        assert risk_engine.daily_stats.realized_pnl == 50.0
        assert risk_engine.daily_stats.trades_count == 1
        assert risk_engine.daily_stats.winning_trades == 1
        assert risk_engine.bankroll == 1050.0

    @pytest.mark.asyncio
    async def test_kill_switch_triggers_on_large_loss(self, risk_engine):
        """Kill switch should trigger when drawdown exceeds 15%."""
        # Lose more than 15% of starting bankroll
        await risk_engine.record_trade_result(pnl=-160.0)  # 16% loss
        
        assert risk_engine.daily_stats.kill_switch_triggered is True
        assert risk_engine.is_trading_allowed is False

    @pytest.mark.asyncio
    async def test_validate_trade_passes_for_valid_trade(self, risk_engine):
        """Valid trades should pass validation."""
        is_valid, reason = await risk_engine.validate_trade(size=25.0, price=0.50)
        
        assert is_valid is True
        assert reason == "Trade validated"

    @pytest.mark.asyncio
    async def test_validate_trade_fails_when_size_exceeds_max(self, risk_engine):
        """Trades exceeding max position should fail validation."""
        is_valid, reason = await risk_engine.validate_trade(size=100.0, price=0.50)
        
        assert is_valid is False
        assert "exceeds max" in reason

    @pytest.mark.asyncio
    async def test_validate_trade_fails_when_kill_switch_triggered(self, risk_engine):
        """Trades should fail when kill switch is triggered."""
        risk_engine.daily_stats.kill_switch_triggered = True
        
        is_valid, reason = await risk_engine.validate_trade(size=25.0, price=0.50)
        
        assert is_valid is False
        assert "Kill switch" in reason

    @pytest.mark.asyncio
    async def test_daily_stats_reset(self, risk_engine):
        """Daily stats should reset correctly."""
        await risk_engine.record_trade_result(pnl=-50.0)
        await risk_engine.reset_daily_stats()
        
        assert risk_engine.daily_stats.realized_pnl == 0.0
        assert risk_engine.daily_stats.trades_count == 0
        assert risk_engine.daily_stats.kill_switch_triggered is False

    def test_portfolio_summary(self, risk_engine):
        """Portfolio summary should include all key metrics."""
        summary = risk_engine.get_portfolio_summary()
        
        assert "bankroll" in summary
        assert "available_capital" in summary
        assert "open_positions" in summary
        assert "daily_pnl" in summary
        assert "current_drawdown_pct" in summary
        assert "kill_switch_triggered" in summary
