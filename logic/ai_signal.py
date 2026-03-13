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
        max_length=10,
        description="Top reasons supporting the probability estimate"
    )
    disconfirming_evidence: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Evidence that could make this estimate wrong"
    )
    what_would_change_mind: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="What information would significantly change the estimate"
    )
    
    timeline_sensitivity: str = Field(
        ..., 
        description="Does timing matter? 'yes' or 'no' with brief explanation"
    )
    failure_modes: list[str] = Field(
        ..., 
        min_length=1, 
        max_length=10,
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

SYSTEM_PROMPT = """You are a disciplined prediction market analyst. Your job is to identify genuine information asymmetry — situations where YOU have specific knowledge the market consensus does not.

CORE BELIEF: Markets are hard to beat.
The market price already reflects the views of thousands of sophisticated participants with real money at stake. You should NOT be contrarian by default. Most of the time, the market is right. Your job is to identify the RARE cases where you have specific information the market hasn't incorporated.

WHEN TO DEVIATE FROM THE MARKET:
Only justify a significant probability deviation (>10 percentage points from market) if you can clearly state:
  1. WHAT specific fact or data point gives you an edge (not just "the news suggests")
  2. WHY that fact is not yet priced in (why don't other traders know this?)
  3. HOW CONFIDENT you are in that specific fact (not just the narrative)

If you cannot answer all three concretely, your estimate should be CLOSE to the market price (within 10%), and your confidence should be LOW (0.3-0.5).

CALIBRATION RULES:

1. START WITH THE MARKET AS YOUR PRIOR
   - The market price is your Bayesian prior, not your adversary
   - You need SPECIFIC evidence to move away from it
   - "I think X is likely" without data support → stay near market

2. INFORMATION QUALITY GATE
   - News from major outlets that is clearly on-topic: can justify moving ±15% from market
   - Generic/tangentially related news: insufficient to diverge
   - No relevant news/data: your estimate should be within ±5% of market, confidence ≤ 0.4
   - "I know from training data that X typically happens": weak unless very specific base rate

3. NEVER MANUFACTURE CONFIDENCE
   - Low confidence (≤0.5) is the honest default when information is limited
   - High confidence (>0.7) REQUIRES multiple independent, specific, verifiable facts
   - It is perfectly fine — and often correct — to output a probability close to the market price

4. ANTI-OVERCONFIDENCE RULES
   - If your estimate is >15pp above market: your confidence must be ≤ 0.6 unless you cite direct, verifiable data
   - If your estimate is >25pp above market: confidence must be ≤ 0.5
   - Single news source alone cannot justify >15pp deviation
   - Narrative reasoning ("this feels likely") ≠ evidence

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
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Used when live news/data context is present
    HAIKU_MODEL   = "claude-3-haiku-20240307"    # Used when no context — ~10x cheaper, quality parity without news
    MAX_RETRIES = 1
    TIMEOUT_SECONDS = 30.0
    MAX_TOKENS = 1500  # Raised from 800 — full 9-field JSON schema with list fields can exceed 800 tokens and truncate silently
    
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
        # Intelligence data for information advantage
        news_summary: Optional[str] = None,
        domain_summary: Optional[str] = None,
        recent_price_change: float = 0.0,
        overreaction_info: Optional[str] = None,
        # Historical performance for learning
        historical_performance: Optional[str] = None,
        # Model selection: caller sets True when no live news/data context was fetched
        use_haiku: bool = False,
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
            news_summary: Recent news about this market topic
            domain_summary: Domain-specific data (prices, stats, etc.)
            recent_price_change: Price change in last 24h
            overreaction_info: Info about potential market overreaction
            historical_performance: Summary of past trading performance by category
            
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
        
        # Build the prompt with intelligence data and historical performance
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
            news_summary=news_summary,
            domain_summary=domain_summary,
            recent_price_change=recent_price_change,
            overreaction_info=overreaction_info,
            historical_performance=historical_performance,
        )
        
        # Select model: Haiku when no live context was fetched (no news, no live data).
        # Sonnet's advantage is news synthesis — without context it's no better than Haiku
        # but ~10x more expensive. Haiku still produces valid JSON and calibrated probabilities.
        active_model = self.HAIKU_MODEL if use_haiku else self.model
        if use_haiku:
            logger.debug(f"[AI] No context → Haiku ({self.HAIKU_MODEL})")

        # Try to get a valid response
        last_error = None
        raw_response = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Call the API with timeout
                response = await asyncio.wait_for(
                    self._call_api(prompt, model=active_model),
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
        news_summary: Optional[str] = None,
        domain_summary: Optional[str] = None,
        recent_price_change: float = 0.0,
        overreaction_info: Optional[str] = None,
        historical_performance: Optional[str] = None,
    ) -> str:
        """Build the user prompt for the AI.
        
        Args:
            ... market data fields
            news_summary: Recent news about this topic
            domain_summary: Domain-specific data
            recent_price_change: 24h price change
            overreaction_info: Overreaction detection info
            historical_performance: Summary of past trading performance
            
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
            # Ensure both sides of the subtraction are timezone-aware (UTC) or both naive
            from datetime import timezone as _tz
            now_utc = datetime.now(_tz.utc)
            res_aware = resolution_date if resolution_date.tzinfo else resolution_date.replace(tzinfo=_tz.utc)
            delta = res_aware - now_utc
            days_remaining = f"{delta.days} days, {delta.seconds // 3600} hours"
        
        # Format price change
        price_change_str = f"{recent_price_change:+.1%}" if recent_price_change != 0 else "N/A"
        
        # Build intelligence section
        intelligence_section = ""
        if news_summary or domain_summary:
            intelligence_section = "\n=== CURRENT INTELLIGENCE (USE THIS!) ===\n"
            if news_summary:
                intelligence_section += f"RECENT NEWS:\n{news_summary}\n\n"
            if domain_summary:
                intelligence_section += f"DOMAIN DATA:\n{domain_summary}\n"
            intelligence_section += "\nIMPORTANT: The above intelligence may contain information NOT yet priced into the market. Use it!\n"
        
        # Build overreaction section
        # Keep framing neutral — large moves represent market consensus updating, not automatic
        # contrarian opportunities. The system prompt already says: don't be contrarian by default.
        overreaction_section = ""
        if overreaction_info:
            overreaction_section = f"""
=== MARKET DYNAMICS NOTE ===
{overreaction_info}

CAUTION: Large price moves usually reflect genuine new information. Do NOT assume it is an overreaction without specific evidence. Verify whether the move is justified before deviating from the updated market price.
"""
        
        # Build historical performance section (learning from past trades)
        historical_section = ""
        if historical_performance:
            historical_section = f"""
=== LEARNING FROM PAST TRADES ===
{historical_performance}

IMPORTANT: Use this historical data to inform your confidence. If this market category has performed poorly, be MORE skeptical and consider LOWERING your confidence or SKIPPING the bet entirely.
"""
        
        prompt = f"""MARKET ANALYSIS REQUEST

=== MARKET QUESTION ===
{market_question}

=== RESOLUTION RULES ===
{resolution_rules or 'NOT PROVIDED - Exercise caution'}

=== CURRENT MARKET STATE ===
Market Consensus Price: {current_price:.4f} ({current_price*100:.1f}%) ← thousands of traders agree on this
Bid-Ask Spread: {spread:.4f} ({spread*100:.2f}%)
24h Volume: ${volume_24h:,.0f}
Liquidity: ${liquidity:,.0f}
24h Price Change: {price_change_str}

=== TIMING ===
Resolution Date: {resolution_str}
Time Remaining: {days_remaining}

=== RECENT PRICE HISTORY ===
{price_path_str}

=== ORDERBOOK DEPTH ===
{depth_summary or 'Not available'}
{intelligence_section}{overreaction_section}{historical_section}
=== YOUR TASK ===
Step 1: What is your INDEPENDENT probability estimate for this event based on base rates and facts?
Step 2: The MARKET CONSENSUS is {current_price*100:.1f}%. Do you have SPECIFIC information that explains why your estimate should differ by more than 10 percentage points?

DECISION GUIDE:
- If you have NO specific on-topic news/data above: your estimate should be within ±5% of {current_price*100:.1f}%, confidence ≤ 0.4
- If you have RELEVANT news that directly relates to this specific outcome: you may move up to ±15%, confidence ≤ 0.6
- If you have DIRECT verifiable data (statistics, official reports, concrete facts): you may move up to ±20%, confidence ≤ 0.75
- Deviation >25pp from market ({current_price*100:.1f}%): ONLY if you can explicitly name the market's specific mistake

IMPORTANT: A probability close to the market price ({current_price*100:.1f}%) is often the CORRECT answer. Do not manufacture a different number to seem useful. If the market is right, say so.

Respond with ONLY a valid JSON object. No other text."""

        return prompt
    
    async def _call_api(self, prompt: str, model: Optional[str] = None) -> str:
        """Call the Anthropic API.
        
        Args:
            prompt: User prompt
            model: Model override (defaults to self.model)
            
        Returns:
            Raw response text
        """
        response = await asyncio.to_thread(
            self._client.messages.create,
            model=model or self.model,
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
        text = response_text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                text = match.group(1).strip()
            else:
                # Truncated response: opening fence but no closing fence.
                # Strip the opening fence line and attempt to parse the rest.
                text = re.sub(r'^```(?:json)?\s*', '', text).strip()

        # Fallback: if text still can't parse directly, try to extract the first
        # {...} object from a prose-wrapped response (e.g. "Here is my analysis: {...}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            obj_match = re.search(r'\{[\s\S]*\}', text)
            if obj_match:
                data = json.loads(obj_match.group(0))
            else:
                raise  # Re-raise original error if no JSON object found at all
        
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
