"""Pydantic V2 models for Polymarket Battle-Bot V2.0.

This module defines the core data models used throughout the trading system:
- Market: Represents a Polymarket prediction market
- Order: Represents a trading order
- Position: Represents an open position
- TradeSignal: AI-generated trading signal
- PriceUpdate: Real-time price update from WebSocket
- DailyStats: Daily trading statistics for risk management
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, computed_field


class OrderSide(str, Enum):
    """Order side enum for buy/sell orders."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enum."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    GTC = "GTC"  # Good-til-cancelled
    FOK = "FOK"  # Fill-or-kill


class OrderStatus(str, Enum):
    """Order status enum."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class MarketOutcome(str, Enum):
    """Market outcome for binary markets."""
    YES = "YES"
    NO = "NO"


class Market(BaseModel):
    """Represents a Polymarket prediction market.
    
    Attributes:
        condition_id: Unique identifier for the market condition
        question: The market question/description
        token_id: Token ID for the specific outcome
        outcome: YES or NO outcome
        end_date: Market resolution date
        min_tick_size: Minimum price increment
        current_price: Current market price (0-1)
        volume_24h: 24-hour trading volume in USDC
        liquidity: Total liquidity in the market
        active: Whether the market is currently active
    """
    condition_id: str = Field(..., description="Unique market condition identifier")
    question: str = Field(..., description="Market question/description")
    token_id: str = Field(..., description="Token ID for the outcome")
    outcome: MarketOutcome = Field(..., description="YES or NO outcome")
    end_date: Optional[datetime] = Field(None, description="Market resolution date")
    min_tick_size: Decimal = Field(default=Decimal("0.01"), description="Minimum price increment")
    current_price: float = Field(..., ge=0.0, le=1.0, description="Current price (0-1)")
    volume_24h: float = Field(default=0.0, ge=0.0, description="24h volume in USDC")
    liquidity: float = Field(default=0.0, ge=0.0, description="Total liquidity")
    active: bool = Field(default=True, description="Whether market is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("current_price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price is within valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Price must be between 0 and 1")
        return round(v, 4)

    model_config = {
        "json_schema_extra": {
            "example": {
                "condition_id": "0x1234...",
                "question": "Will Bitcoin reach $100k by end of 2026?",
                "token_id": "12345",
                "outcome": "YES",
                "current_price": 0.65,
                "volume_24h": 150000.0,
                "liquidity": 500000.0,
            }
        }
    }


class Order(BaseModel):
    """Represents a trading order on Polymarket.
    
    Attributes:
        order_id: Unique order identifier (assigned after submission)
        token_id: Token ID being traded
        side: BUY or SELL
        order_type: Order type (LIMIT, MARKET, etc.)
        price: Order price (0-1)
        size: Order size in shares
        filled_size: Amount filled so far
        status: Current order status
        created_at: Order creation timestamp
        updated_at: Last update timestamp
    """
    order_id: Optional[str] = Field(None, description="Order ID from exchange")
    token_id: str = Field(..., description="Token ID being traded")
    side: OrderSide = Field(..., description="BUY or SELL")
    order_type: OrderType = Field(default=OrderType.LIMIT, description="Order type")
    price: float = Field(..., ge=0.0, le=1.0, description="Order price (0-1)")
    size: float = Field(..., gt=0.0, description="Order size in shares")
    filled_size: float = Field(default=0.0, ge=0.0, description="Filled amount")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Round price to 4 decimal places."""
        return round(v, 4)

    @field_validator("size", "filled_size")
    @classmethod
    def validate_size(cls, v: float) -> float:
        """Round size to 2 decimal places."""
        return round(v, 2)

    @computed_field
    @property
    def remaining_size(self) -> float:
        """Calculate remaining unfilled size."""
        return round(self.size - self.filled_size, 2)

    @computed_field
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_size >= self.size

    model_config = {
        "json_schema_extra": {
            "example": {
                "token_id": "12345",
                "side": "BUY",
                "order_type": "LIMIT",
                "price": 0.45,
                "size": 100.0,
            }
        }
    }


class Position(BaseModel):
    """Represents an open position in a market.
    
    Attributes:
        token_id: Token ID for the position
        condition_id: Market condition ID
        outcome: YES or NO
        size: Position size in shares
        avg_entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss
        opened_at: When position was opened
        updated_at: Last update timestamp
    """
    token_id: str = Field(..., description="Token ID")
    condition_id: str = Field(..., description="Market condition ID")
    outcome: MarketOutcome = Field(..., description="YES or NO")
    size: float = Field(..., description="Position size in shares")
    avg_entry_price: float = Field(..., ge=0.0, le=1.0, description="Average entry price")
    current_price: float = Field(..., ge=0.0, le=1.0, description="Current market price")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L in USDC")
    realized_pnl: float = Field(default=0.0, description="Realized P&L in USDC")
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return round(self.size * self.current_price, 2)

    @computed_field
    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis of position."""
        return round(self.size * self.avg_entry_price, 2)

    @computed_field
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return round((self.unrealized_pnl / self.cost_basis) * 100, 2)

    def update_unrealized_pnl(self, new_price: float) -> None:
        """Update unrealized P&L based on new price."""
        self.current_price = new_price
        self.unrealized_pnl = round((new_price - self.avg_entry_price) * self.size, 2)
        self.updated_at = datetime.utcnow()

    model_config = {
        "json_schema_extra": {
            "example": {
                "token_id": "12345",
                "condition_id": "0x1234...",
                "outcome": "YES",
                "size": 100.0,
                "avg_entry_price": 0.45,
                "current_price": 0.52,
            }
        }
    }


class TradeSignal(BaseModel):
    """AI-generated trading signal from Claude analysis.
    
    Attributes:
        token_id: Token ID for the signal
        condition_id: Market condition ID
        side: Recommended side (BUY or SELL)
        confidence: AI confidence score (0-1)
        true_probability: AI's estimated true probability
        market_price: Current market price at signal time
        edge: Calculated edge (true_prob - market_price)
        recommended_size: Kelly-calculated position size
        reasoning: AI's reasoning for the signal
        created_at: Signal generation timestamp
    """
    token_id: str = Field(..., description="Token ID")
    condition_id: str = Field(..., description="Market condition ID")
    side: OrderSide = Field(..., description="Recommended side")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence (0-1)")
    true_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated true probability")
    market_price: float = Field(..., ge=0.0, le=1.0, description="Market price at signal time")
    edge: float = Field(..., description="Calculated edge")
    recommended_size: float = Field(..., ge=0.0, description="Kelly-calculated size in USDC")
    reasoning: str = Field(..., description="AI reasoning for the signal")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("edge")
    @classmethod
    def validate_edge(cls, v: float) -> float:
        """Round edge to 4 decimal places."""
        return round(v, 4)

    @computed_field
    @property
    def is_actionable(self) -> bool:
        """Check if signal has positive edge and sufficient confidence."""
        return self.edge > 0.02 and self.confidence >= 0.6

    model_config = {
        "json_schema_extra": {
            "example": {
                "token_id": "12345",
                "condition_id": "0x1234...",
                "side": "BUY",
                "confidence": 0.75,
                "true_probability": 0.70,
                "market_price": 0.55,
                "edge": 0.15,
                "recommended_size": 25.0,
                "reasoning": "Strong fundamentals suggest higher probability...",
            }
        }
    }


class PriceUpdate(BaseModel):
    """Real-time price update from WebSocket feed.
    
    Attributes:
        token_id: Token ID
        asset_id: Asset identifier
        price: New price
        timestamp: Update timestamp
        bid: Best bid price
        ask: Best ask price
        spread: Bid-ask spread
        volume: Recent volume
    """
    token_id: str = Field(..., description="Token ID")
    asset_id: Optional[str] = Field(None, description="Asset identifier")
    price: float = Field(..., ge=0.0, le=1.0, description="Current price")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bid: Optional[float] = Field(None, ge=0.0, le=1.0, description="Best bid")
    ask: Optional[float] = Field(None, ge=0.0, le=1.0, description="Best ask")
    spread: Optional[float] = Field(None, ge=0.0, description="Bid-ask spread")
    volume: Optional[float] = Field(None, ge=0.0, description="Recent volume")

    @computed_field
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return round((self.bid + self.ask) / 2, 4)
        return None

    model_config = {
        "json_schema_extra": {
            "example": {
                "token_id": "12345",
                "price": 0.55,
                "bid": 0.54,
                "ask": 0.56,
                "spread": 0.02,
            }
        }
    }


class DailyStats(BaseModel):
    """Daily trading statistics for risk management.
    
    Used to track daily P&L and enforce the 15% max drawdown kill-switch.
    
    Attributes:
        date: Trading date
        starting_bankroll: Bankroll at start of day
        current_bankroll: Current bankroll
        realized_pnl: Total realized P&L for the day
        unrealized_pnl: Total unrealized P&L
        trades_count: Number of trades executed
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        max_drawdown_pct: Maximum drawdown percentage reached
        kill_switch_triggered: Whether kill-switch was triggered
    """
    date: datetime = Field(default_factory=lambda: datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0))
    starting_bankroll: float = Field(..., gt=0.0, description="Starting bankroll")
    current_bankroll: float = Field(..., ge=0.0, description="Current bankroll")
    realized_pnl: float = Field(default=0.0, description="Daily realized P&L")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    trades_count: int = Field(default=0, ge=0, description="Number of trades")
    winning_trades: int = Field(default=0, ge=0, description="Winning trades count")
    losing_trades: int = Field(default=0, ge=0, description="Losing trades count")
    max_drawdown_pct: float = Field(default=0.0, ge=0.0, description="Max drawdown %")
    kill_switch_triggered: bool = Field(default=False, description="Kill-switch status")

    # Maximum allowed daily drawdown (15%)
    MAX_DAILY_DRAWDOWN: float = 0.15

    @computed_field
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return round(self.realized_pnl + self.unrealized_pnl, 2)

    @computed_field
    @property
    def current_drawdown_pct(self) -> float:
        """Calculate current drawdown as percentage."""
        if self.starting_bankroll == 0:
            return 0.0
        drawdown = (self.starting_bankroll - self.current_bankroll) / self.starting_bankroll
        return round(max(0.0, drawdown) * 100, 2)

    @computed_field
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return round((self.winning_trades / total) * 100, 2)

    def check_kill_switch(self) -> bool:
        """Check if kill-switch should be triggered.
        
        Returns:
            True if drawdown exceeds 15%, False otherwise.
        """
        drawdown_pct = self.current_drawdown_pct / 100  # Convert to decimal
        if drawdown_pct >= self.MAX_DAILY_DRAWDOWN:
            self.kill_switch_triggered = True
            return True
        return False

    def update_stats(self, pnl: float, is_winner: bool) -> None:
        """Update daily statistics after a trade.
        
        Args:
            pnl: Profit/loss from the trade
            is_winner: Whether the trade was profitable
        """
        self.realized_pnl += pnl
        self.current_bankroll += pnl
        self.trades_count += 1
        
        if is_winner:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update max drawdown if current is higher
        if self.current_drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = self.current_drawdown_pct
        
        # Check kill-switch
        self.check_kill_switch()

    model_config = {
        "json_schema_extra": {
            "example": {
                "starting_bankroll": 10000.0,
                "current_bankroll": 9500.0,
                "realized_pnl": -500.0,
                "trades_count": 15,
                "winning_trades": 8,
                "losing_trades": 7,
            }
        }
    }
