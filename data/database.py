"""Async SQLite database for telemetry and calibration data.

Stores all trading decisions, trade outcomes, and calibration history
for evaluation and strategy improvement.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import aiosqlite
from loguru import logger


# SQL schema for all tables
SCHEMA = """
-- Trading decisions log (every decision point, trade or no-trade)
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    
    -- Market state at decision time
    price REAL NOT NULL,
    bid REAL,
    ask REAL,
    spread REAL,
    depth_bid REAL,
    depth_ask REAL,
    volume_24h REAL,
    
    -- AI signal outputs
    raw_prob REAL,
    confidence REAL,
    calibrated_prob REAL,
    adjusted_prob REAL,
    edge REAL,
    
    -- Decision outcome
    decision TEXT NOT NULL,  -- 'TRADE', 'NO_TRADE'
    reason_codes TEXT,  -- JSON array of reason codes
    
    -- If traded
    order_id TEXT,
    side TEXT,
    size REAL,
    
    -- Metadata
    ai_latency_ms INTEGER,
    total_latency_ms INTEGER
);

-- Completed trades with outcomes
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER REFERENCES decisions(id),
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    
    -- Entry
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_side TEXT NOT NULL,
    size REAL NOT NULL,
    
    -- Signal at entry
    raw_prob_entry REAL,
    adjusted_prob_entry REAL,
    edge_entry REAL,
    confidence_entry REAL,
    
    -- Exit
    exit_time TEXT,
    exit_price REAL,
    exit_reason TEXT,  -- 'PROFIT_TAKE', 'STOP_LOSS', 'TIME_STOP', 'SIGNAL_FLIP', 'MANUAL'
    
    -- Outcome
    pnl REAL,
    pnl_percent REAL,
    fees_estimate REAL,
    outcome INTEGER,  -- 1 = win, 0 = loss, NULL = unresolved
    
    -- Market resolution (if resolved)
    resolved INTEGER DEFAULT 0,
    resolution_value INTEGER  -- 1 = YES, 0 = NO
);

-- Calibration history for probability estimates
CREATE TABLE IF NOT EXISTS calibration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    category TEXT,  -- Market category for grouped calibration
    
    -- Probability estimates
    raw_prob REAL NOT NULL,
    calibrated_prob REAL,
    market_price REAL NOT NULL,
    
    -- Outcome (filled when market resolves)
    outcome INTEGER,  -- 1 = correct, 0 = incorrect
    resolved_at TEXT
);

-- Price history for backtesting
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    
    mid_price REAL NOT NULL,
    bid REAL,
    ask REAL,
    spread REAL,
    depth_bid REAL,
    depth_ask REAL,
    volume REAL
);

-- Daily performance metrics
CREATE TABLE IF NOT EXISTS daily_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    
    starting_bankroll REAL NOT NULL,
    ending_bankroll REAL NOT NULL,
    
    total_pnl REAL,
    realized_pnl REAL,
    unrealized_pnl REAL,
    
    trades_count INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    
    signals_generated INTEGER,
    signals_rejected INTEGER,
    
    max_drawdown_pct REAL,
    avg_edge REAL,
    avg_confidence REAL,
    
    brier_score REAL,
    calibration_error REAL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_decisions_market ON decisions(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions(decision, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome, exit_time);
CREATE INDEX IF NOT EXISTS idx_calibration_category ON calibration_history(category, outcome);
CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_id, timestamp);
"""


class TelemetryDB:
    """Async SQLite database for trading telemetry.
    
    Handles all database operations for logging decisions, trades,
    and calibration data. All operations are async-safe.
    """
    
    def __init__(self, db_path: str = "data/battlebot.db"):
        """Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        
    async def connect(self) -> None:
        """Establish database connection and create tables."""
        async with self._lock:
            if self._connection is None:
                self._connection = await aiosqlite.connect(str(self.db_path))
                self._connection.row_factory = aiosqlite.Row
                await self._connection.executescript(SCHEMA)
                await self._connection.commit()
                logger.info(f"TelemetryDB connected: {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None
                logger.info("TelemetryDB closed")
    
    async def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure database is connected."""
        if self._connection is None:
            await self.connect()
        return self._connection
    
    # =========================================================================
    # Decision Logging
    # =========================================================================
    
    async def log_decision(
        self,
        market_id: str,
        token_id: str,
        price: float,
        decision: str,
        reason_codes: list[str],
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        spread: Optional[float] = None,
        depth_bid: Optional[float] = None,
        depth_ask: Optional[float] = None,
        volume_24h: Optional[float] = None,
        raw_prob: Optional[float] = None,
        confidence: Optional[float] = None,
        calibrated_prob: Optional[float] = None,
        adjusted_prob: Optional[float] = None,
        edge: Optional[float] = None,
        order_id: Optional[str] = None,
        side: Optional[str] = None,
        size: Optional[float] = None,
        ai_latency_ms: Optional[int] = None,
        total_latency_ms: Optional[int] = None,
    ) -> int:
        """Log a trading decision.
        
        Args:
            market_id: Market condition ID
            token_id: Token ID
            price: Current market price
            decision: 'TRADE' or 'NO_TRADE'
            reason_codes: List of reason codes explaining decision
            ... other optional fields
            
        Returns:
            Decision ID
        """
        conn = await self._ensure_connected()
        
        sql = """
        INSERT INTO decisions (
            timestamp, market_id, token_id, price, bid, ask, spread,
            depth_bid, depth_ask, volume_24h, raw_prob, confidence,
            calibrated_prob, adjusted_prob, edge, decision, reason_codes,
            order_id, side, size, ai_latency_ms, total_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        async with self._lock:
            cursor = await conn.execute(sql, (
                datetime.utcnow().isoformat(),
                market_id, token_id, price, bid, ask, spread,
                depth_bid, depth_ask, volume_24h, raw_prob, confidence,
                calibrated_prob, adjusted_prob, edge, decision,
                json.dumps(reason_codes),
                order_id, side, size, ai_latency_ms, total_latency_ms
            ))
            await conn.commit()
            
            decision_id = cursor.lastrowid
            logger.debug(f"Decision logged: {decision_id} | {decision} | {reason_codes}")
            return decision_id
    
    # =========================================================================
    # Trade Logging
    # =========================================================================
    
    async def log_trade_entry(
        self,
        decision_id: int,
        market_id: str,
        token_id: str,
        entry_price: float,
        entry_side: str,
        size: float,
        raw_prob: Optional[float] = None,
        adjusted_prob: Optional[float] = None,
        edge: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Log a trade entry.
        
        Args:
            decision_id: ID of the decision that triggered this trade
            market_id: Market condition ID
            token_id: Token ID
            entry_price: Entry price
            entry_side: 'BUY' or 'SELL'
            size: Position size
            
        Returns:
            Trade ID
        """
        conn = await self._ensure_connected()
        
        sql = """
        INSERT INTO trades (
            decision_id, market_id, token_id, entry_time, entry_price,
            entry_side, size, raw_prob_entry, adjusted_prob_entry,
            edge_entry, confidence_entry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        async with self._lock:
            cursor = await conn.execute(sql, (
                decision_id, market_id, token_id,
                datetime.utcnow().isoformat(),
                entry_price, entry_side, size,
                raw_prob, adjusted_prob, edge, confidence
            ))
            await conn.commit()
            
            trade_id = cursor.lastrowid
            logger.debug(f"Trade entry logged: {trade_id}")
            return trade_id
    
    async def log_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        fees_estimate: float = 0.0,
    ) -> None:
        """Log a trade exit.
        
        Args:
            trade_id: Trade ID to update
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl: Profit/loss
            fees_estimate: Estimated fees
        """
        conn = await self._ensure_connected()
        
        # Get entry price for pnl_percent calculation
        async with self._lock:
            cursor = await conn.execute(
                "SELECT entry_price, size FROM trades WHERE id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                entry_price = row['entry_price']
                size = row['size']
                cost_basis = entry_price * size
                pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                outcome = 1 if pnl > 0 else 0
                
                await conn.execute("""
                    UPDATE trades SET
                        exit_time = ?,
                        exit_price = ?,
                        exit_reason = ?,
                        pnl = ?,
                        pnl_percent = ?,
                        fees_estimate = ?,
                        outcome = ?
                    WHERE id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    exit_price, exit_reason, pnl, pnl_percent,
                    fees_estimate, outcome, trade_id
                ))
                await conn.commit()
                
                logger.debug(f"Trade exit logged: {trade_id} | PnL: ${pnl:+.2f}")
    
    async def log_trade_resolution(
        self,
        trade_id: int,
        resolution_value: int,  # 1 = YES, 0 = NO
    ) -> None:
        """Log market resolution for a trade.
        
        Args:
            trade_id: Trade ID
            resolution_value: 1 for YES, 0 for NO
        """
        conn = await self._ensure_connected()
        
        async with self._lock:
            await conn.execute("""
                UPDATE trades SET
                    resolved = 1,
                    resolution_value = ?
                WHERE id = ?
            """, (resolution_value, trade_id))
            await conn.commit()
    
    # =========================================================================
    # Calibration Logging
    # =========================================================================
    
    async def log_calibration_sample(
        self,
        market_id: str,
        raw_prob: float,
        market_price: float,
        category: Optional[str] = None,
        calibrated_prob: Optional[float] = None,
    ) -> int:
        """Log a calibration sample.
        
        Args:
            market_id: Market ID
            raw_prob: Raw LLM probability
            market_price: Market price at time of prediction
            category: Market category for grouped calibration
            calibrated_prob: Calibrated probability (if available)
            
        Returns:
            Sample ID
        """
        conn = await self._ensure_connected()
        
        sql = """
        INSERT INTO calibration_history (
            timestamp, market_id, category, raw_prob,
            calibrated_prob, market_price
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        
        async with self._lock:
            cursor = await conn.execute(sql, (
                datetime.utcnow().isoformat(),
                market_id, category, raw_prob,
                calibrated_prob, market_price
            ))
            await conn.commit()
            return cursor.lastrowid
    
    async def update_calibration_outcome(
        self,
        sample_id: int,
        outcome: int,  # 1 = event occurred, 0 = did not occur
    ) -> None:
        """Update calibration sample with outcome.
        
        Args:
            sample_id: Sample ID
            outcome: 1 if event occurred, 0 otherwise
        """
        conn = await self._ensure_connected()
        
        async with self._lock:
            await conn.execute("""
                UPDATE calibration_history SET
                    outcome = ?,
                    resolved_at = ?
                WHERE id = ?
            """, (outcome, datetime.utcnow().isoformat(), sample_id))
            await conn.commit()
    
    async def get_calibration_data(
        self,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get calibration history for analysis.
        
        Args:
            category: Optional category filter
            limit: Maximum records to return
            
        Returns:
            List of calibration records
        """
        conn = await self._ensure_connected()
        
        if category:
            sql = """
                SELECT * FROM calibration_history
                WHERE category = ? AND outcome IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (category, limit)
        else:
            sql = """
                SELECT * FROM calibration_history
                WHERE outcome IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """
            params = (limit,)
        
        async with self._lock:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Price History
    # =========================================================================
    
    async def log_price_tick(
        self,
        market_id: str,
        token_id: str,
        mid_price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        spread: Optional[float] = None,
        depth_bid: Optional[float] = None,
        depth_ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> None:
        """Log a price tick for backtesting.
        
        Args:
            market_id: Market ID
            token_id: Token ID
            mid_price: Mid price
            bid: Best bid
            ask: Best ask
            spread: Bid-ask spread
            depth_bid: Bid depth
            depth_ask: Ask depth
            volume: Recent volume
        """
        conn = await self._ensure_connected()
        
        sql = """
        INSERT INTO price_history (
            timestamp, market_id, token_id, mid_price,
            bid, ask, spread, depth_bid, depth_ask, volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        async with self._lock:
            await conn.execute(sql, (
                datetime.utcnow().isoformat(),
                market_id, token_id, mid_price,
                bid, ask, spread, depth_bid, depth_ask, volume
            ))
            await conn.commit()
    
    async def get_price_history(
        self,
        market_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> list[dict]:
        """Get price history for backtesting.
        
        Args:
            market_id: Market ID
            start_time: Optional start time (ISO format)
            end_time: Optional end time (ISO format)
            
        Returns:
            List of price records
        """
        conn = await self._ensure_connected()
        
        sql = "SELECT * FROM price_history WHERE market_id = ?"
        params = [market_id]
        
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        
        sql += " ORDER BY timestamp ASC"
        
        async with self._lock:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Metrics & Analytics
    # =========================================================================
    
    async def compute_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compute trading metrics for a time period.
        
        Args:
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
            
        Returns:
            Dictionary of metrics
        """
        conn = await self._ensure_connected()
        
        # Build date filter
        date_filter = ""
        params = []
        if start_date:
            date_filter += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND entry_time <= ?"
            params.append(end_date)
        
        async with self._lock:
            # Trade statistics
            cursor = await conn.execute(f"""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN outcome = 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    AVG(edge_entry) as avg_edge,
                    AVG(confidence_entry) as avg_confidence,
                    AVG(julianday(exit_time) - julianday(entry_time)) * 24 as avg_hold_hours
                FROM trades
                WHERE exit_time IS NOT NULL {date_filter}
            """, params)
            trade_stats = dict(await cursor.fetchone())
            
            # Decision statistics
            cursor = await conn.execute(f"""
                SELECT
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN decision = 'TRADE' THEN 1 ELSE 0 END) as trade_decisions,
                    SUM(CASE WHEN decision = 'NO_TRADE' THEN 1 ELSE 0 END) as no_trade_decisions
                FROM decisions
                WHERE 1=1 {date_filter.replace('entry_time', 'timestamp')}
            """, params)
            decision_stats = dict(await cursor.fetchone())
            
            # Calibration metrics (Brier score)
            cursor = await conn.execute("""
                SELECT
                    AVG((raw_prob - outcome) * (raw_prob - outcome)) as brier_raw,
                    AVG((calibrated_prob - outcome) * (calibrated_prob - outcome)) as brier_calibrated,
                    COUNT(*) as sample_count
                FROM calibration_history
                WHERE outcome IS NOT NULL
            """)
            calib_stats = dict(await cursor.fetchone())
            
            # Win rate and profit factor
            win_rate = 0.0
            profit_factor = 0.0
            if trade_stats['total_trades'] and trade_stats['total_trades'] > 0:
                win_rate = (trade_stats['winning_trades'] or 0) / trade_stats['total_trades'] * 100
                
                if trade_stats['avg_loss'] and trade_stats['avg_loss'] < 0:
                    gross_profit = (trade_stats['avg_win'] or 0) * (trade_stats['winning_trades'] or 0)
                    gross_loss = abs(trade_stats['avg_loss'] or 0) * (trade_stats['losing_trades'] or 0)
                    if gross_loss > 0:
                        profit_factor = gross_profit / gross_loss
            
            return {
                "trades": trade_stats,
                "decisions": decision_stats,
                "calibration": calib_stats,
                "win_rate": round(win_rate, 2),
                "profit_factor": round(profit_factor, 2),
            }
    
    async def get_calibration_curve(
        self,
        n_bins: int = 10,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Compute calibration curve bins.
        
        Args:
            n_bins: Number of probability bins
            category: Optional category filter
            
        Returns:
            List of bin statistics
        """
        conn = await self._ensure_connected()
        
        # Get all resolved calibration samples
        if category:
            sql = """
                SELECT raw_prob, outcome FROM calibration_history
                WHERE outcome IS NOT NULL AND category = ?
            """
            params = (category,)
        else:
            sql = """
                SELECT raw_prob, outcome FROM calibration_history
                WHERE outcome IS NOT NULL
            """
            params = ()
        
        async with self._lock:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        
        if not rows:
            return []
        
        # Bin the data
        bins = [{"bin_start": i/n_bins, "bin_end": (i+1)/n_bins, 
                 "count": 0, "sum_pred": 0.0, "sum_actual": 0.0}
                for i in range(n_bins)]
        
        for row in rows:
            prob = row['raw_prob']
            outcome = row['outcome']
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bins[bin_idx]['count'] += 1
            bins[bin_idx]['sum_pred'] += prob
            bins[bin_idx]['sum_actual'] += outcome
        
        # Compute averages
        result = []
        for b in bins:
            if b['count'] > 0:
                result.append({
                    "bin_start": b['bin_start'],
                    "bin_end": b['bin_end'],
                    "count": b['count'],
                    "mean_predicted": round(b['sum_pred'] / b['count'], 4),
                    "mean_actual": round(b['sum_actual'] / b['count'], 4),
                })
        
        return result


# Singleton instance
_db_instance: Optional[TelemetryDB] = None


async def get_db(db_path: str = "data/battlebot.db") -> TelemetryDB:
    """Get or create the database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        TelemetryDB instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = TelemetryDB(db_path)
        await _db_instance.connect()
    return _db_instance
