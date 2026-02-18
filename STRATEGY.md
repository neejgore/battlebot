# Strategy V2.1: Calibration-First Trading

This document describes the trading strategy implementation, decision gates, tunable parameters, and evaluation methodology.

## Philosophy

The V2.1 strategy is designed to address common failure modes of LLM-based prediction trading:

1. **Overconfidence** - LLMs often express high confidence without strong evidence
2. **Narrative bias** - Compelling stories lead to overweighting recent/dramatic events
3. **Overtrading** - Acting on noise rather than signal
4. **Poor calibration** - Raw probabilities don't match actual outcomes

Our solution: A **calibration-first, evidence-gated pipeline** that requires multiple conditions to align before trading.

## Decision Pipeline

```
Price Update
    │
    ▼
┌─────────────────────────────────────┐
│  1. MARKET ELIGIBILITY FILTER       │
│  - Resolution rules present?        │
│  - Time to resolution ≤ 30 days?    │
│  - Volume ≥ $5,000?                 │
│  - Not headline market?             │
│  - Price in [0.05, 0.95]?           │
│  - Liquidity ≥ $1,000?              │
└─────────────────────────────────────┘
    │ PASS
    ▼
┌─────────────────────────────────────┐
│  2. TRIGGER CONDITIONS              │
│  - Price move ≥ 2%?                 │
│  - OR volatility spike ≥ 3%?        │
│  - OR spread tight (< 2%)?          │
│  - Cooldown expired?                │
└─────────────────────────────────────┘
    │ PASS
    ▼
┌─────────────────────────────────────┐
│  3. AI SIGNAL GENERATION            │
│  - Structured JSON output           │
│  - Probability + confidence         │
│  - Key reasons + disconfirming      │
│  - Base rate considered?            │
│  - Information quality?             │
└─────────────────────────────────────┘
    │ VALID
    ▼
┌─────────────────────────────────────┐
│  4. AI QUALITY GATES                │
│  - Confidence ≥ 50%?                │
│  - Info quality ≥ medium?           │
│  - Base rate considered?            │
└─────────────────────────────────────┘
    │ PASS
    ▼
┌─────────────────────────────────────┐
│  5. CALIBRATION                     │
│  - Apply learned calibration curve  │
│  - Shrink toward market price       │
│  - Weight by confidence + samples   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  6. EDGE GATING                     │
│  - Edge ≥ 3%?                       │
│  - Edge > spread?                   │
└─────────────────────────────────────┘
    │ PASS
    ▼
┌─────────────────────────────────────┐
│  7. POSITION SIZING                 │
│  - Fractional Kelly (10%)           │
│  - Edge-based throttle              │
│  - Per-market limit (10% bankroll)  │
│  - Global exposure (30% bankroll)   │
│  - Max position ($50)               │
└─────────────────────────────────────┘
    │ SIZE > 0
    ▼
┌─────────────────────────────────────┐
│  8. RISK VALIDATION                 │
│  - Kill switch not triggered?       │
│  - Sufficient capital?              │
│  - Within drawdown limits?          │
│  - No exit cooldown?                │
└─────────────────────────────────────┘
    │ VALID
    ▼
┌─────────────────────────────────────┐
│  9. EXECUTE TRADE                   │
│  - Place order                      │
│  - Record position with exit rules  │
│  - Log to telemetry                 │
└─────────────────────────────────────┘
```

## Configuration Parameters

### Market Eligibility (`StrategyConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_days_to_resolution` | 30 | Maximum days until market resolves |
| `min_volume_24h` | $5,000 | Minimum 24h trading volume |
| `max_spread` | 0.04 | Maximum bid-ask spread (4 cents) |
| `min_depth_notional` | $1,000 | Minimum orderbook depth |
| `min_tradeable_price` | 0.05 | Minimum price to trade |
| `max_tradeable_price` | 0.95 | Maximum price to trade |
| `headline_volume_threshold` | $500,000 | Volume threshold for "headline" markets |
| `allow_headline_markets` | false | Whether to trade headline markets |

### Trigger Conditions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_price_change` | 0.02 | Price move to trigger analysis (2%) |
| `volatility_window_seconds` | 300 | Window for volatility calculation |
| `volatility_threshold` | 0.03 | Volatility spike threshold (3% std dev) |
| `analysis_cooldown_seconds` | 600 | Cooldown between analyses (10 min) |
| `exit_cooldown_hours` | 2.0 | Cooldown before re-entry after exit |

### Edge Gating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge` | 0.03 | Minimum edge to trade (3%) |
| `min_confidence` | 0.50 | Minimum AI confidence (50%) |
| `min_information_quality` | "medium" | Minimum info quality rating |

### Risk Limits (`RiskLimits`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_drawdown` | 0.15 | Kill switch at 15% daily drawdown |
| `max_position_size` | $50 | Maximum single position |
| `max_percent_bankroll_per_market` | 0.10 | 10% of bankroll per market |
| `max_total_open_risk` | 0.30 | 30% of bankroll in open positions |
| `max_positions` | 10 | Maximum concurrent positions |

### Exit Rules

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profit_take_pct` | 0.15 | Take profit at 15% gain |
| `stop_loss_pct` | 0.10 | Stop loss at 10% loss |
| `time_stop_hours` | 72 | Exit after 3 days max |

### Position Sizing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fractional_kelly` | 0.10 | Use 10% of full Kelly |
| `edge_scale` | 0.10 | Edge for full position size |

## Calibration System

### Overview

The calibration layer maps raw LLM probabilities to calibrated probabilities using historical performance data.

```
adjusted_prob = w × calibrated_prob + (1-w) × market_price
```

Where `w` (shrinkage weight) depends on:
- **Confidence**: Higher confidence = more weight to our estimate
- **Sample size**: More historical data = more weight to calibration

### Shrinkage Weight Formula

```
w = BASE + CONFIDENCE_BOOST × confidence + SAMPLE_BOOST × sample_factor
```

| Component | Value | Description |
|-----------|-------|-------------|
| `BASE_SHRINKAGE` | 0.30 | Minimum weight (always some shrinkage) |
| `CONFIDENCE_BOOST` | 0.40 | Weight from confidence score |
| `SAMPLE_SIZE_BOOST` | 0.30 | Weight from sample size |
| `SAMPLE_SIZE_SCALE` | 50 | Samples needed for full boost |

### Calibration Methods

1. **Passthrough** (< 20 samples): Use raw probability
2. **Platt scaling** (default): Binned historical accuracy
3. **Isotonic regression** (optional): Monotonic calibration curve

### Evaluation Metrics

- **Brier Score**: `mean((prediction - outcome)²)` - lower is better
- **Calibration Curve**: 10-bin comparison of predicted vs actual
- **Edge vs Realized P&L**: Correlation of expected edge with actual returns

## Position Sizing: Fractional Kelly

```python
def calculate_kelly_size(bankroll, true_prob, market_price, fractional_multiplier=0.1):
    edge = true_prob - market_price
    if edge <= 0:
        return 0.0
    
    b = (1 / market_price) - 1  # Odds
    p, q = true_prob, 1 - true_prob
    
    full_kelly = (b * p - q) / b
    bet_size = bankroll * full_kelly * fractional_multiplier
    
    return max(0.0, min(bet_size, max_position))
```

### Edge-Based Throttle

Position size is further scaled by edge:

```
throttle = min(1.0, edge / EDGE_SCALE)
final_size = kelly_size × throttle
```

This means:
- Edge = 3% (minimum): ~30% of Kelly size
- Edge = 10%: Full Kelly size
- Edge = 15%: Full Kelly size (capped at 1.0)

## Exit Rules

### Deterministic Exits

1. **Profit Take**: Price moves in favor by `profit_take_pct` (15%)
2. **Stop Loss**: Price moves against by `stop_loss_pct` (10%)
3. **Time Stop**: Position held > `time_stop_hours` (72h)
4. **Signal Flip**: Adjusted probability flips against position by >2%

### Anti-Churn

After exit, a cooldown prevents re-entry to the same market for `exit_cooldown_hours` (2h).

## Telemetry

### Decision Table

Every decision (trade or no-trade) is logged with:

- Timestamp, market ID, token ID
- Price, bid, ask, spread, depth
- Raw prob, confidence, calibrated prob, adjusted prob
- Edge, decision, reason codes
- Order ID (if traded), AI latency, total latency

### Trade Table

Each trade records:

- Entry: time, price, side, size, probabilities
- Exit: time, price, reason, P&L
- Market resolution (if applicable)

### Metrics

- Brier score (raw and calibrated)
- Calibration curve (10 bins)
- Win rate, profit factor
- Max drawdown
- Average hold time

## Backtesting

### Generate Mock Data

```bash
python scripts/backtest.py generate --days 30 --output data/mock_prices.jsonl
```

### Run Backtest

```bash
python scripts/backtest.py run --data data/mock_prices.jsonl --bankroll 1000
```

### Backtest Assumptions

- **Buy fills**: At ask price + slippage
- **Sell fills**: At bid price - slippage
- **Default slippage**: 10 basis points (0.1%)

## Tuning Guide

### Conservative Settings (Less Trading)

```python
config = StrategyConfig(
    min_edge=0.05,           # Require 5% edge
    min_confidence=0.60,     # Higher confidence threshold
    analysis_cooldown_seconds=900,  # 15 min cooldown
    min_volume_24h=10000,    # Higher volume requirement
)

limits = RiskLimits(
    fractional_kelly=0.05,   # 5% of Kelly
    max_position_size=25.0,  # Smaller positions
    max_total_open_risk=0.20,  # 20% max exposure
)
```

### Aggressive Settings (More Trading)

```python
config = StrategyConfig(
    min_edge=0.02,           # Lower edge threshold
    min_confidence=0.45,     # Lower confidence OK
    analysis_cooldown_seconds=300,  # 5 min cooldown
    allow_headline_markets=True,
)

limits = RiskLimits(
    fractional_kelly=0.15,   # 15% of Kelly
    max_position_size=100.0, # Larger positions
    max_total_open_risk=0.40,  # 40% max exposure
)
```

## Common Failure Modes & Mitigations

| Failure Mode | Mitigation |
|--------------|------------|
| Overtrading on noise | Cooldowns, volatility filtering |
| Overconfidence | Shrinkage to market, confidence gates |
| Poor calibration | Historical calibration, Brier tracking |
| Excessive drawdown | 15% kill switch, position limits |
| Chasing headlines | Headline market filter |
| Extreme prices | Price bounds [0.05, 0.95] |
| Flip-flopping | Exit cooldowns |
| Illiquid markets | Volume/depth requirements |
| Narrative bias | Base rate requirement, disconfirming evidence prompt |

## Development Checklist

- [ ] Monitor Brier score over time
- [ ] Analyze calibration curve monthly
- [ ] Review no-trade reason codes
- [ ] Track edge vs realized P&L correlation
- [ ] Audit position sizing distribution
- [ ] Check kill switch never false-triggered
