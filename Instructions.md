MASTER SPEC: POLYMARKET AI BATTLE-BOT V2.0
1. PROJECT OVERVIEW
An asynchronous, event-driven trading bot for Polymarket that uses Claude 3.5/3.7 for trade signals based on real-time WebSocket data.
2. PROJECT RULES (.cursorrules)
Async-First: Use asyncio and aiosqlite for all I/O.
Risk Management: Maintain a strict 15% Max Daily Drawdown kill-switch.
Security: Never hardcode private keys; use environment variables via a .env file.
Execution: Enforce a 500ms delay between all order placements.
3. ARCHITECTURE & DIRECTORY STRUCTURE
services/websocket_client.py: Real-time listener for wss://://clob.polymarket.com.
services/execution_engine.py: EIP-712 authenticated order placement using ClobClient.
logic/strategy_v2.py: Event-driven logic that triggers AI analysis upon price movements.
logic/risk_engine.py: Calculates position sizes using the Fractional Kelly Criterion.
data/models.py: Pydantic V2 models for Market, Order, and Position objects.
4. CORE ALGORITHM: FRACTIONAL KELLY CRITERION
python
def calculate_kelly_size(bankroll, true_prob, market_price, fractional_multiplier=0.1, max_pos=50.0):
    if market_price <= 0 or market_price >= 1: return 0.0
    edge = true_prob - market_price
    if edge <= 0: return 0.0
    b = (1 / market_price) - 1
    p, q = true_prob, 1 - true_prob
    full_kelly = (b * p - q) / b
    bet_size = bankroll * (full_kelly * fractional_multiplier)
    return round(max(0.0, min(bet_size, max_pos)), 2)
Use code with caution.

5. TECH STACK
Language: Python 3.11+
Libraries: py-clob-client, websockets, pydantic, loguru, pandas, aiosqlite
Infrastructure: Deployment-ready for a Linux VPS via a systemd service.