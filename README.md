# Polymarket Battle-Bot V2.0

An asynchronous, event-driven trading bot for Polymarket that uses Claude AI (3.5/3.7) for trade signals based on real-time WebSocket data.

## Features

- **Real-time Price Monitoring**: WebSocket connection to Polymarket CLOB for instant price updates
- **AI-Powered Analysis**: Claude AI estimates true probabilities when significant price movements occur
- **Fractional Kelly Criterion**: Optimal position sizing based on calculated edge
- **Risk Management**: 15% max daily drawdown kill-switch protects capital
- **Async Architecture**: Non-blocking I/O throughout for maximum performance
- **EIP-712 Authentication**: Secure order signing for Polymarket

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BattleBot (main.py)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  WebSocket  │  │  Execution  │  │    Strategy V2          │  │
│  │   Client    │──│   Engine    │──│  (AI Signal Generation) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          │                                      │
│                   ┌──────┴──────┐                               │
│                   │ Risk Engine │                               │
│                   │  (Kelly +   │                               │
│                   │ Kill-switch)│                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11 or higher
- A Polymarket account with API credentials
- An Anthropic API key for Claude

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd battlebot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

Edit your `.env` file with the following required variables:

```env
# Required
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Trading Parameters
INITIAL_BANKROLL=1000        # Starting capital in USDC
FRACTIONAL_KELLY=0.1         # 10% of full Kelly (conservative)
MAX_POSITION_SIZE=50         # Max $50 per position
MIN_PRICE_CHANGE=0.03        # 3% move triggers analysis
MIN_EDGE=0.05                # 5% minimum edge to trade
```

## Usage

### Running the Bot

```bash
python main.py
```

### Adding Markets to Monitor

In your code, add markets using the `BattleBot.add_market()` method:

```python
await bot.add_market(
    condition_id="0x...",
    question="Will Bitcoin reach $100k by end of 2026?",
    token_id="12345",
    outcome=MarketOutcome.YES,
    current_price=0.65,
    volume_24h=150000.0,
    liquidity=500000.0,
)
```

### Graceful Shutdown

The bot handles `SIGTERM` and `SIGINT` signals for graceful shutdown:
- Cancels all open orders
- Closes WebSocket connection
- Prints final session statistics

## Core Algorithm: Fractional Kelly Criterion

The bot uses the Fractional Kelly Criterion for position sizing:

```python
def calculate_kelly_size(bankroll, true_prob, market_price, 
                         fractional_multiplier=0.1, max_pos=50.0):
    if market_price <= 0 or market_price >= 1: 
        return 0.0
    
    edge = true_prob - market_price
    if edge <= 0: 
        return 0.0
    
    b = (1 / market_price) - 1
    p, q = true_prob, 1 - true_prob
    full_kelly = (b * p - q) / b
    bet_size = bankroll * (full_kelly * fractional_multiplier)
    
    return round(max(0.0, min(bet_size, max_pos)), 2)
```

## Risk Management

### Kill-Switch

The bot enforces a **15% maximum daily drawdown**. If losses exceed this threshold:
- All trading immediately halts
- Open orders are cancelled
- A critical alert is logged

### Position Limits

- Maximum single position: $50 (configurable)
- Fractional Kelly sizing: 10% of full Kelly (configurable)
- Order rate limiting: 500ms between orders

## Project Structure

```
battlebot/
├── main.py                     # Entry point & orchestration
├── data/
│   ├── __init__.py
│   └── models.py               # Pydantic V2 models
├── logic/
│   ├── __init__.py
│   ├── risk_engine.py          # Kelly sizing & kill-switch
│   └── strategy_v2.py          # AI-driven strategy
├── services/
│   ├── __init__.py
│   ├── websocket_client.py     # Real-time data feed
│   └── execution_engine.py     # Order execution
├── .env.example                # Environment template
├── .cursorrules                # Development guidelines
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=.
```

### Code Formatting

```bash
black .
ruff check . --fix
```

### Type Checking

```bash
mypy .
```

## Deployment

### Systemd Service (Linux VPS)

Create `/etc/systemd/system/battlebot.service`:

```ini
[Unit]
Description=Polymarket Battle-Bot V2.0
After=network.target

[Service]
Type=simple
User=battlebot
WorkingDirectory=/opt/battlebot
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/battlebot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable battlebot
sudo systemctl start battlebot
```

## Security Considerations

- **Never commit your `.env` file** - it contains sensitive credentials
- Private keys should have minimal permissions (only what's needed for trading)
- Consider using a dedicated wallet with limited funds
- Monitor the bot's activity regularly
- Set up alerting for kill-switch triggers

## Disclaimer

This software is for educational purposes only. Trading on prediction markets involves risk. Use at your own risk and never trade with funds you cannot afford to lose.

## License

MIT License - See LICENSE file for details.
