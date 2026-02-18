"""Calibration Layer for Polymarket Battle-Bot V2.1.

Maps raw LLM probabilities to calibrated probabilities using historical
performance data. Implements shrinkage toward market price based on
confidence and sample size.
"""

import asyncio
import math
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from data.database import TelemetryDB, get_db


@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    raw_prob: float
    calibrated_prob: float
    adjusted_prob: float  # After shrinkage toward market
    shrinkage_weight: float  # Weight given to calibrated vs market (0-1)
    sample_size: int  # Historical samples used
    method: str  # Calibration method used


class CalibrationEngine:
    """Calibrates LLM probability estimates using historical data.
    
    Implements:
    1. Simple shrinkage toward market price (default)
    2. Beta-binomial shrinkage (improved with more data)
    3. Isotonic regression (future, behind flag)
    
    The key insight: raw LLM probabilities are systematically biased.
    We learn the bias from historical (prediction, outcome) pairs and
    correct for it.
    """
    
    # Minimum samples before trusting calibration
    MIN_SAMPLES_FOR_CALIBRATION = 20
    
    # Prior parameters for beta-binomial (weakly informative)
    PRIOR_ALPHA = 1.0  # Pseudo-count for successes
    PRIOR_BETA = 1.0   # Pseudo-count for failures
    
    # Shrinkage parameters
    BASE_SHRINKAGE = 0.3  # Minimum weight to market price
    CONFIDENCE_BOOST = 0.4  # Additional weight from confidence
    SAMPLE_SIZE_BOOST = 0.3  # Additional weight from sample size
    SAMPLE_SIZE_SCALE = 50  # Samples needed for full boost
    
    def __init__(self, db: Optional[TelemetryDB] = None):
        """Initialize the calibration engine.
        
        Args:
            db: Database instance for historical data
        """
        self._db = db
        self._cache: dict[str, dict] = {}  # Cache calibration params by category
        self._lock = asyncio.Lock()
        
        logger.info("CalibrationEngine initialized")
    
    async def _ensure_db(self) -> TelemetryDB:
        """Ensure database is available."""
        if self._db is None:
            self._db = await get_db()
        return self._db
    
    async def calibrate(
        self,
        raw_prob: float,
        market_price: float,
        confidence: float,
        category: Optional[str] = None,
        use_isotonic: bool = False,
    ) -> CalibrationResult:
        """Calibrate a raw probability estimate.
        
        The calibration pipeline:
        1. Apply learned calibration curve (if sufficient data)
        2. Shrink toward market price based on confidence and sample size
        
        Args:
            raw_prob: Raw LLM probability estimate (0-1)
            market_price: Current market price (0-1)
            confidence: LLM confidence score (0-1)
            category: Market category for grouped calibration
            use_isotonic: Use isotonic regression (requires sklearn)
            
        Returns:
            CalibrationResult with all probability stages
        """
        # Get historical calibration data
        db = await self._ensure_db()
        
        async with self._lock:
            # Get calibration history for this category
            history = await db.get_calibration_data(category=category, limit=500)
            sample_size = len(history)
            
            # Step 1: Apply calibration curve
            if sample_size >= self.MIN_SAMPLES_FOR_CALIBRATION:
                if use_isotonic:
                    calibrated_prob = await self._isotonic_calibrate(raw_prob, history)
                    method = "isotonic"
                else:
                    calibrated_prob = self._platt_calibrate(raw_prob, history)
                    method = "platt"
            else:
                # Not enough data - use raw probability
                calibrated_prob = raw_prob
                method = "passthrough"
            
            # Step 2: Shrink toward market price
            shrinkage_weight = self._compute_shrinkage_weight(
                confidence=confidence,
                sample_size=sample_size,
            )
            
            # adjusted_prob = w * calibrated_prob + (1-w) * market_price
            adjusted_prob = (
                shrinkage_weight * calibrated_prob +
                (1 - shrinkage_weight) * market_price
            )
            
            # Clamp to valid range
            adjusted_prob = max(0.001, min(0.999, adjusted_prob))
            
            logger.debug(
                f"Calibration | Raw: {raw_prob:.3f} -> Calibrated: {calibrated_prob:.3f} "
                f"-> Adjusted: {adjusted_prob:.3f} | Weight: {shrinkage_weight:.2f} | "
                f"Samples: {sample_size} | Method: {method}"
            )
            
            return CalibrationResult(
                raw_prob=raw_prob,
                calibrated_prob=calibrated_prob,
                adjusted_prob=adjusted_prob,
                shrinkage_weight=shrinkage_weight,
                sample_size=sample_size,
                method=method,
            )
    
    def _compute_shrinkage_weight(
        self,
        confidence: float,
        sample_size: int,
    ) -> float:
        """Compute weight for calibrated probability vs market price.
        
        Higher weight = more trust in our calibrated estimate.
        Lower weight = shrink toward market price.
        
        Formula:
        w = BASE + CONFIDENCE_BOOST * confidence + SAMPLE_BOOST * sample_factor
        
        Args:
            confidence: LLM confidence (0-1)
            sample_size: Number of historical samples
            
        Returns:
            Weight in range [BASE_SHRINKAGE, 1.0]
        """
        # Sample size factor: saturates at SAMPLE_SIZE_SCALE samples
        sample_factor = min(1.0, sample_size / self.SAMPLE_SIZE_SCALE)
        
        # Compute weight
        weight = (
            self.BASE_SHRINKAGE +
            self.CONFIDENCE_BOOST * confidence +
            self.SAMPLE_SIZE_BOOST * sample_factor
        )
        
        # Clamp to valid range
        return min(1.0, max(self.BASE_SHRINKAGE, weight))
    
    def _platt_calibrate(
        self,
        raw_prob: float,
        history: list[dict],
    ) -> float:
        """Simple Platt-style calibration using binned averages.
        
        Groups historical predictions into bins and uses the actual
        outcome rate in each bin to adjust predictions.
        
        Args:
            raw_prob: Raw probability to calibrate
            history: Historical (raw_prob, outcome) pairs
            
        Returns:
            Calibrated probability
        """
        if not history:
            return raw_prob
        
        # Create bins
        n_bins = 10
        bins = [[] for _ in range(n_bins)]
        
        for record in history:
            prob = record.get('raw_prob', 0.5)
            outcome = record.get('outcome')
            if outcome is not None:
                bin_idx = min(int(prob * n_bins), n_bins - 1)
                bins[bin_idx].append(outcome)
        
        # Find which bin the raw_prob falls into
        bin_idx = min(int(raw_prob * n_bins), n_bins - 1)
        
        # Get outcomes in this bin
        bin_outcomes = bins[bin_idx]
        
        if len(bin_outcomes) >= 3:
            # Use beta-binomial posterior mean
            successes = sum(bin_outcomes)
            trials = len(bin_outcomes)
            
            # Posterior mean with prior
            calibrated = (
                (self.PRIOR_ALPHA + successes) /
                (self.PRIOR_ALPHA + self.PRIOR_BETA + trials)
            )
            return calibrated
        
        # Not enough data in this bin - interpolate from neighbors
        return self._interpolate_from_neighbors(raw_prob, bins, n_bins)
    
    def _interpolate_from_neighbors(
        self,
        raw_prob: float,
        bins: list[list],
        n_bins: int,
    ) -> float:
        """Interpolate calibration from neighboring bins.
        
        Args:
            raw_prob: Raw probability
            bins: Binned outcomes
            n_bins: Number of bins
            
        Returns:
            Interpolated calibration
        """
        bin_idx = min(int(raw_prob * n_bins), n_bins - 1)
        
        # Collect data from nearby bins
        outcomes = []
        for offset in range(-2, 3):
            idx = bin_idx + offset
            if 0 <= idx < n_bins:
                outcomes.extend(bins[idx])
        
        if len(outcomes) >= 3:
            successes = sum(outcomes)
            trials = len(outcomes)
            return (self.PRIOR_ALPHA + successes) / (self.PRIOR_ALPHA + self.PRIOR_BETA + trials)
        
        # Fall back to raw probability
        return raw_prob
    
    async def _isotonic_calibrate(
        self,
        raw_prob: float,
        history: list[dict],
    ) -> float:
        """Isotonic regression calibration (monotonic).
        
        Requires sklearn. Falls back to Platt if not available.
        
        Args:
            raw_prob: Raw probability
            history: Historical data
            
        Returns:
            Calibrated probability
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Prepare data
            X = [r['raw_prob'] for r in history if r.get('outcome') is not None]
            y = [r['outcome'] for r in history if r.get('outcome') is not None]
            
            if len(X) < 10:
                return self._platt_calibrate(raw_prob, history)
            
            # Fit isotonic regression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(X, y)
            
            # Predict
            calibrated = iso.predict([raw_prob])[0]
            return float(calibrated)
            
        except ImportError:
            logger.warning("sklearn not available - using Platt calibration")
            return self._platt_calibrate(raw_prob, history)
    
    async def record_prediction(
        self,
        market_id: str,
        raw_prob: float,
        market_price: float,
        category: Optional[str] = None,
        calibrated_prob: Optional[float] = None,
    ) -> int:
        """Record a prediction for future calibration.
        
        Args:
            market_id: Market identifier
            raw_prob: Raw LLM probability
            market_price: Market price at prediction time
            category: Market category
            calibrated_prob: Calibrated probability (if computed)
            
        Returns:
            Sample ID for later outcome update
        """
        db = await self._ensure_db()
        return await db.log_calibration_sample(
            market_id=market_id,
            raw_prob=raw_prob,
            market_price=market_price,
            category=category,
            calibrated_prob=calibrated_prob,
        )
    
    async def record_outcome(
        self,
        sample_id: int,
        outcome: int,
    ) -> None:
        """Record the outcome for a prediction.
        
        Args:
            sample_id: Sample ID from record_prediction
            outcome: 1 if event occurred, 0 otherwise
        """
        db = await self._ensure_db()
        await db.update_calibration_outcome(sample_id, outcome)
    
    async def get_calibration_curve(
        self,
        category: Optional[str] = None,
        n_bins: int = 10,
    ) -> list[dict]:
        """Get calibration curve for visualization.
        
        Args:
            category: Market category filter
            n_bins: Number of probability bins
            
        Returns:
            List of bin statistics
        """
        db = await self._ensure_db()
        return await db.get_calibration_curve(n_bins=n_bins, category=category)
    
    async def compute_brier_score(
        self,
        category: Optional[str] = None,
    ) -> dict:
        """Compute Brier score for predictions.
        
        Brier score = mean((prediction - outcome)^2)
        Lower is better. Perfect = 0, random = 0.25
        
        Args:
            category: Market category filter
            
        Returns:
            Dictionary with raw and calibrated Brier scores
        """
        db = await self._ensure_db()
        history = await db.get_calibration_data(category=category)
        
        if not history:
            return {"raw_brier": None, "calibrated_brier": None, "sample_count": 0}
        
        raw_brier_sum = 0.0
        calibrated_brier_sum = 0.0
        count = 0
        calibrated_count = 0
        
        for record in history:
            outcome = record.get('outcome')
            if outcome is None:
                continue
            
            raw_prob = record.get('raw_prob', 0.5)
            raw_brier_sum += (raw_prob - outcome) ** 2
            count += 1
            
            calibrated_prob = record.get('calibrated_prob')
            if calibrated_prob is not None:
                calibrated_brier_sum += (calibrated_prob - outcome) ** 2
                calibrated_count += 1
        
        raw_brier = raw_brier_sum / count if count > 0 else None
        calibrated_brier = calibrated_brier_sum / calibrated_count if calibrated_count > 0 else None
        
        return {
            "raw_brier": round(raw_brier, 4) if raw_brier else None,
            "calibrated_brier": round(calibrated_brier, 4) if calibrated_brier else None,
            "sample_count": count,
        }


# Singleton instance
_calibration_engine: Optional[CalibrationEngine] = None


async def get_calibration_engine() -> CalibrationEngine:
    """Get or create the calibration engine singleton.
    
    Returns:
        CalibrationEngine instance
    """
    global _calibration_engine
    if _calibration_engine is None:
        _calibration_engine = CalibrationEngine()
    return _calibration_engine
