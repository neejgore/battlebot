"""AI Signal Generation for Polymarket Battle-Bot V2.1.

Structured LLM probability estimation with uncertainty quantification,
citations, and failure mode awareness. Uses strict JSON schema validation.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
from loguru import logger

import anthropic
from pydantic import BaseModel, Field, ValidationError


# ============================================================================
# Structured Output Schema
# ============================================================================

class AISignalOutput(BaseModel):
    """Strict schema for AI signal output.
    
    The LLM must produce output matching this schema exactly.
    Invalid outputs are rejected.
    """
    raw_prob: float = Field(..., ge=0.0, le=1.0, description="Estimated true probability (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Epistemic confidence in estimate (0-1)")
    
    key_reasons: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=5,
        description="Top reasons supporting the probability estimate"
    )
    disconfirming_evidence: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=3,
        description="Evidence that could make this estimate wrong"
    )
    what_would_change_mind: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=3,
        description="What information would significantly change the estimate"
    )
    
    timeline_sensitivity: str = Field(
        ..., 
        description="Does timing matter? 'yes' or 'no' with brief explanation"
    )
    failure_modes: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=3,
        description="Ways this prediction could fail"
    )
    
    base_rate_considered: bool = Field(
        ..., 
        description="Whether historical base rates were considered"
    )
    information_quality: str = Field(
        ..., 
        description="Quality of available information: 'high', 'medium', 'low'"
    )


@dataclass
class AISignalResult:
    """Result from AI signal generation."""
    success: bool
    signal: Optional[AISignalOutput] = None
    error: Optional[str] = None
    latency_ms: int = 0
    raw_response: Optional[str] = None
    retries: int = 0


# ============================================================================
# System Prompt - Calibration-Focused
# ============================================================================

SYSTEM_PROMPT = """You are a calibrated probability estimator for prediction markets.

YOUR CORE TASK:
Estimate the TRUE probability of an event, independent of the current market price.
The market may be mispriced - your job is to find edge, not confirm the market.

CRITICAL RULES FOR CALIBRATION:

1. BASE RATES FIRST
   - Start with historical base rates for similar events
   - Only adjust from base rate with strong, specific evidence
   - "This time is different" is usually wrong

2. CONFIDENCE REFLECTS INFORMATION QUALITY
   - High confidence (>0.7): Multiple independent, reliable sources; clear precedent
   - Medium confidence (0.4-0.7): Some evidence but uncertainty remains
   - Low confidence (<0.4): Speculation, limited information, novel situation

3. AVOID NARRATIVE TRAPS
   - Compelling stories ≠ high probability
   - Recent events are overweighted (recency bias)
   - Dramatic outcomes are overweighted (availability bias)
   
4. PENALIZE WEAK EVIDENCE
   - Single source: reduce confidence
   - Correlated sources: count as ~1 source
   - "Experts say" without specifics: weak evidence
   - Social media sentiment: very weak evidence

5. EXTREME PROBABILITIES REQUIRE EXTREME EVIDENCE
   - Prob > 0.90: Need overwhelming, diverse evidence
   - Prob < 0.10: Need strong evidence of impossibility
   - When uncertain, stay closer to 0.50

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object matching this exact schema:
{
    "raw_prob": <float 0-1>,
    "confidence": <float 0-1>,
    "key_reasons": ["<reason1>", "<reason2>", "<reason3>"],
    "disconfirming_evidence": ["<evidence1>", "<evidence2>"],
    "what_would_change_mind": ["<factor1>", "<factor2>"],
    "timeline_sensitivity": "<yes/no: brief explanation>",
    "failure_modes": ["<mode1>", "<mode2>"],
    "base_rate_considered": <true/false>,
    "information_quality": "<high/medium/low>"
}

NO text before or after the JSON. The JSON must be parseable."""


# ============================================================================
# AI Signal Generator
# ============================================================================

class AISignalGenerator:
    """Generates calibrated probability signals using Claude.
    
    Features:
    - Strict JSON schema validation
    - Automatic retries with timeout
    - Latency tracking
    - Graceful degradation on failure
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 1
    TIMEOUT_SECONDS = 30.0
    MAX_TOKENS = 800
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ):
        """Initialize the AI signal generator.
        
        Args:
            api_key: Anthropic API key (from env if not provided)
            model: Model to use
            max_retries: Maximum retry attempts
            timeout_seconds: Timeout per request
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No Anthropic API key - AI signals disabled")
            self._client = None
        else:
            self._client = anthropic.Anthropic(api_key=api_key)
        
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Statistics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._total_latency_ms = 0
        
        logger.info(f"AISignalGenerator initialized | Model: {model}")
    
    @property
    def is_available(self) -> bool:
        """Check if AI signals are available."""
        return self._client is not None
    
    async def generate_signal(
        self,
        market_question: str,
        current_price: float,
        spread: float,
        resolution_rules: Optional[str] = None,
        resolution_date: Optional[datetime] = None,
        volume_24h: float = 0.0,
        liquidity: float = 0.0,
        recent_price_path: Optional[list[float]] = None,
        depth_summary: Optional[str] = None,
        category: Optional[str] = None,
    ) -> AISignalResult:
        """Generate a probability signal for a market.
        
        Args:
            market_question: The market question text
            current_price: Current market price (0-1)
            spread: Current bid-ask spread
            resolution_rules: Market resolution rules text
            resolution_date: Expected resolution date
            volume_24h: 24h trading volume
            liquidity: Total market liquidity
            recent_price_path: Recent price history
            depth_summary: Orderbook depth summary
            category: Market category
            
        Returns:
            AISignalResult with signal or error
        """
        if not self.is_available:
            return AISignalResult(
                success=False,
                error="AI signals unavailable - no API key",
                latency_ms=0
            )
        
        # Require resolution rules for trading
        if not resolution_rules:
            return AISignalResult(
                success=False,
                error="NO_RESOLUTION_RULES",
                latency_ms=0
            )
        
        start_time = time.monotonic()
        self._total_calls += 1
        
        # Build the prompt
        prompt = self._build_prompt(
            market_question=market_question,
            current_price=current_price,
            spread=spread,
            resolution_rules=resolution_rules,
            resolution_date=resolution_date,
            volume_24h=volume_24h,
            liquidity=liquidity,
            recent_price_path=recent_price_path,
            depth_summary=depth_summary,
        )
        
        # Try to get a valid response
        last_error = None
        raw_response = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Call the API with timeout
                response = await asyncio.wait_for(
                    self._call_api(prompt),
                    timeout=self.timeout_seconds
                )
                
                raw_response = response
                
                # Parse and validate the response
                signal = self._parse_response(response)
                
                latency_ms = int((time.monotonic() - start_time) * 1000)
                self._successful_calls += 1
                self._total_latency_ms += latency_ms
                
                logger.debug(
                    f"AI signal generated | Prob: {signal.raw_prob:.2%} | "
                    f"Conf: {signal.confidence:.2%} | Latency: {latency_ms}ms"
                )
                
                return AISignalResult(
                    success=True,
                    signal=signal,
                    latency_ms=latency_ms,
                    raw_response=raw_response,
                    retries=attempt
                )
                
            except asyncio.TimeoutError:
                last_error = "TIMEOUT"
                logger.warning(f"AI signal timeout (attempt {attempt + 1})")
                
            except ValidationError as e:
                last_error = f"VALIDATION_ERROR: {e}"
                logger.warning(f"AI signal validation failed: {e}")
                
            except json.JSONDecodeError as e:
                last_error = f"JSON_PARSE_ERROR: {e}"
                logger.warning(f"AI signal JSON parse failed: {e}")
                
            except Exception as e:
                last_error = f"API_ERROR: {e}"
                logger.error(f"AI signal API error: {e}")
        
        # All attempts failed
        latency_ms = int((time.monotonic() - start_time) * 1000)
        self._failed_calls += 1
        
        return AISignalResult(
            success=False,
            error=last_error,
            latency_ms=latency_ms,
            raw_response=raw_response,
            retries=self.max_retries
        )
    
    def _build_prompt(
        self,
        market_question: str,
        current_price: float,
        spread: float,
        resolution_rules: Optional[str],
        resolution_date: Optional[datetime],
        volume_24h: float,
        liquidity: float,
        recent_price_path: Optional[list[float]],
        depth_summary: Optional[str],
    ) -> str:
        """Build the user prompt for the AI.
        
        Args:
            ... market data fields
            
        Returns:
            Formatted prompt string
        """
        # Format price path
        price_path_str = "N/A"
        if recent_price_path and len(recent_price_path) > 0:
            price_path_str = " -> ".join(f"{p:.2%}" for p in recent_price_path[-10:])
        
        # Format resolution date
        resolution_str = "Unknown"
        days_remaining = "Unknown"
        if resolution_date:
            resolution_str = resolution_date.strftime("%Y-%m-%d %H:%M UTC")
            delta = resolution_date - datetime.utcnow()
            days_remaining = f"{delta.days} days, {delta.seconds // 3600} hours"
        
        prompt = f"""MARKET ANALYSIS REQUEST

=== MARKET QUESTION ===
{market_question}

=== RESOLUTION RULES ===
{resolution_rules or 'NOT PROVIDED - Exercise caution'}

=== CURRENT MARKET STATE ===
Current Price: {current_price:.4f} ({current_price*100:.1f}%)
Bid-Ask Spread: {spread:.4f} ({spread*100:.2f}%)
24h Volume: ${volume_24h:,.0f}
Liquidity: ${liquidity:,.0f}

=== TIMING ===
Resolution Date: {resolution_str}
Time Remaining: {days_remaining}

=== RECENT PRICE HISTORY ===
{price_path_str}

=== ORDERBOOK DEPTH ===
{depth_summary or 'Not available'}

=== YOUR TASK ===
Estimate the TRUE probability that this market resolves YES.

REMINDERS:
- Your probability must be base-rate aware
- Avoid narrative overreaction
- Penalize weak evidence
- The market price ({current_price:.1%}) may be wrong - that's what we're checking
- If information quality is low, confidence should be low
- Extreme probabilities (>90% or <10%) require extreme evidence

Respond with ONLY a valid JSON object. No other text."""

        return prompt
    
    async def _call_api(self, prompt: str) -> str:
        """Call the Anthropic API.
        
        Args:
            prompt: User prompt
            
        Returns:
            Raw response text
        """
        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.content[0].text
    
    def _parse_response(self, response_text: str) -> AISignalOutput:
        """Parse and validate the AI response.
        
        Args:
            response_text: Raw response from API
            
        Returns:
            Validated AISignalOutput
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
            ValidationError: If schema validation fails
        """
        # Try to extract JSON from response
        # First, try the whole response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            # Find the JSON content
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                text = match.group(1).strip()
        
        # Parse JSON
        data = json.loads(text)
        
        # Validate with Pydantic
        return AISignalOutput.model_validate(data)
    
    def get_stats(self) -> dict[str, Any]:
        """Get generator statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_latency = 0
        if self._successful_calls > 0:
            avg_latency = self._total_latency_ms / self._successful_calls
        
        success_rate = 0.0
        if self._total_calls > 0:
            success_rate = self._successful_calls / self._total_calls * 100
        
        return {
            "available": self.is_available,
            "model": self.model,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "success_rate": round(success_rate, 1),
            "avg_latency_ms": round(avg_latency, 0),
        }


# ============================================================================
# Convenience function
# ============================================================================

_generator: Optional[AISignalGenerator] = None


def get_ai_generator() -> AISignalGenerator:
    """Get or create the AI signal generator singleton.
    
    Returns:
        AISignalGenerator instance
    """
    global _generator
    if _generator is None:
        _generator = AISignalGenerator()
    return _generator
