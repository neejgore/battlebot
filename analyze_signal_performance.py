#!/usr/bin/env python3
"""
Signal Performance Analyzer
============================
Analyzes the signal_log from bot state to answer:
1. How did signals we DIDN'T participate in perform? (shadow portfolio)
2. How did signals we DID participate in perform vs expected?
3. What optimal edge/confidence thresholds does the data suggest?
4. What would "selective profit-lock" exits have done to performance?

Run with: python analyze_signal_performance.py
"""

import json
import os
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()


def load_state(state_path: str) -> dict:
    """Load bot state file."""
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {}


def analyze_signal_log(signals: list, trades: list, positions: dict) -> None:
    """Full analysis of signal log performance."""
    
    print("=" * 80)
    print("SIGNAL LOG PERFORMANCE ANALYSIS")
    print("(Since 'No-Exit / Let-Settle' Strategy Was Implemented)")
    print("=" * 80)

    if not signals:
        print("\n⚠️  No signals logged yet.")
        print("   Signals are logged every time the AI analyzes a market.")
        print("   Wait for more markets to settle, then re-run.")
        return

    total = len(signals)
    settled = [s for s in signals if s.get('outcome')]
    pending = [s for s in signals if not s.get('outcome_checked')]
    checked_no_outcome = [s for s in signals if s.get('outcome_checked') and not s.get('outcome')]

    print(f"\n📊 SIGNAL OVERVIEW:")
    print(f"   Total signals logged:     {total}")
    print(f"   Settled (have outcome):   {len(settled)}")
    print(f"   Still pending:            {len(pending)}")
    print(f"   Checked but not settled:  {len(checked_no_outcome)}")

    if not settled:
        print("\n⏳ No settled signals yet. Markets need to resolve first.")
        print("\nPending signal topics:")
        for s in signals[:10]:
            ts = s.get('timestamp', '')[:10]
            print(f"  [{ts}] {s.get('question', '')[:60]} | edge={s.get('edge', 0)*100:.1f}% | side={s.get('side', '?')}")
        return

    # =========================================================================
    # 1. OVERALL SIGNAL PERFORMANCE
    # =========================================================================
    wins = [s for s in settled if s.get('outcome') == 'WIN']
    losses = [s for s in settled if s.get('outcome') == 'LOSS']
    win_rate = len(wins) / len(settled) * 100

    total_pnl = sum(s.get('theoretical_pnl', 0) for s in settled)
    total_invested = sum(s.get('market_price', 0.5) for s in settled)
    roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    print(f"\n{'=' * 80}")
    print("1. OVERALL SIGNAL PERFORMANCE (ALL Signals, Trades + No-Trades)")
    print("=" * 80)
    print(f"   Win Rate:        {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} losses)")
    print(f"   Theoretical P&L: ${total_pnl:+.2f}")
    print(f"   ROI:             {roi:+.1f}%")
    print(f"   Breakeven need:  ~{total_invested/len(settled)*100:.0f}% win rate")

    # =========================================================================
    # 2. PARTICIPATED VS NON-PARTICIPATED SIGNALS
    # =========================================================================
    traded_market_ids = set()
    for t in trades:
        if t.get('action') == 'ENTRY':
            traded_market_ids.add(t.get('market_id', ''))
    for pos in positions.values():
        traded_market_ids.add(pos.get('market_id', ''))

    traded_signals = [s for s in settled if s.get('market_id') in traded_market_ids]
    skipped_signals = [s for s in settled if s.get('market_id') not in traded_market_ids]

    print(f"\n{'=' * 80}")
    print("2. PARTICIPATED vs NON-PARTICIPATED SIGNALS")
    print("(Non-participated = AI analyzed but bot chose NO_TRADE)")
    print("=" * 80)

    def summarize_group(label, group):
        if not group:
            print(f"\n   {label}: No data")
            return
        w = [s for s in group if s.get('outcome') == 'WIN']
        wr = len(w) / len(group) * 100
        pnl = sum(s.get('theoretical_pnl', 0) for s in group)
        avg_edge = sum(s.get('edge', 0) for s in group) / len(group) * 100
        avg_conf = sum(s.get('confidence', 0) for s in group) / len(group) * 100
        print(f"\n   {label} ({len(group)} signals):")
        print(f"     Win Rate:     {wr:.1f}%")
        print(f"     Theory P&L:   ${pnl:+.2f}")
        print(f"     Avg Edge:     {avg_edge:.1f}%")
        print(f"     Avg Conf:     {avg_conf:.1f}%")

    summarize_group("PARTICIPATED (bot traded)", traded_signals)
    summarize_group("NON-PARTICIPATED (bot skipped)", skipped_signals)

    if skipped_signals:
        skipped_wins = [s for s in skipped_signals if s.get('outcome') == 'WIN']
        if len(skipped_wins) / len(skipped_signals) > 0.55:
            print(f"\n   ⚠️  INSIGHT: Skipped signals won at {len(skipped_wins)/len(skipped_signals)*100:.0f}% - edge threshold may be too strict!")
        elif len(skipped_wins) / len(skipped_signals) < 0.45:
            print(f"\n   ✓  INSIGHT: Skipped signals won only {len(skipped_wins)/len(skipped_signals)*100:.0f}% - filters are working!")

    # =========================================================================
    # 3. BY EDGE THRESHOLD (What threshold maximizes profit?)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("3. BY EDGE THRESHOLD (Should we raise/lower our 10% minimum?)")
    print("=" * 80)
    print(f"   {'Threshold':<12} {'Signals':<10} {'Win Rate':<12} {'Theory P&L':<14} {'ROI'}")
    print(f"   {'-'*60}")

    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        above = [s for s in settled if abs(s.get('edge', 0)) >= threshold]
        if len(above) < 3:
            continue
        w = len([s for s in above if s.get('outcome') == 'WIN'])
        wr = w / len(above) * 100
        pnl = sum(s.get('theoretical_pnl', 0) for s in above)
        inv = sum(abs(s.get('market_price', 0.5)) for s in above)
        roi_t = (pnl / inv * 100) if inv > 0 else 0
        marker = " ← CURRENT" if abs(threshold - 0.10) < 0.001 else ""
        print(f"   {threshold*100:.0f}%+{'':<7} {len(above):<10} {wr:.1f}%{'':<6} ${pnl:+.2f}{'':<8} {roi_t:+.1f}%{marker}")

    # =========================================================================
    # 4. BY CONFIDENCE THRESHOLD
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("4. BY CONFIDENCE THRESHOLD (Should we raise/lower our 30% minimum?)")
    print("=" * 80)
    print(f"   {'Min Conf':<12} {'Signals':<10} {'Win Rate':<12} {'Theory P&L':<14} {'ROI'}")
    print(f"   {'-'*60}")

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        above = [s for s in settled if s.get('confidence', 0) >= threshold]
        if len(above) < 3:
            continue
        w = len([s for s in above if s.get('outcome') == 'WIN'])
        wr = w / len(above) * 100
        pnl = sum(s.get('theoretical_pnl', 0) for s in above)
        inv = sum(abs(s.get('market_price', 0.5)) for s in above)
        roi_t = (pnl / inv * 100) if inv > 0 else 0
        marker = " ← CURRENT" if abs(threshold - 0.30) < 0.001 else ""
        print(f"   {threshold*100:.0f}%+{'':<7} {len(above):<10} {wr:.1f}%{'':<6} ${pnl:+.2f}{'':<8} {roi_t:+.1f}%{marker}")

    # =========================================================================
    # 5. SIDE ANALYSIS (YES vs NO signals)
    # =========================================================================
    yes_signals = [s for s in settled if s.get('side', '').upper() == 'YES']
    no_signals = [s for s in settled if s.get('side', '').upper() == 'NO']

    print(f"\n{'=' * 80}")
    print("5. YES vs NO SIGNAL PERFORMANCE")
    print("(Bot trades NO-only — this shows if that's still the right call)")
    print("=" * 80)

    for label, group in [("YES signals", yes_signals), ("NO signals", no_signals)]:
        if not group:
            print(f"\n   {label}: No data")
            continue
        w = len([s for s in group if s.get('outcome') == 'WIN'])
        wr = w / len(group) * 100
        pnl = sum(s.get('theoretical_pnl', 0) for s in group)
        print(f"\n   {label} ({len(group)} signals):")
        print(f"     Win Rate:    {wr:.1f}%")
        print(f"     Theory P&L:  ${pnl:+.2f}")

    # =========================================================================
    # 6. CATEGORY BREAKDOWN
    # =========================================================================
    categories = defaultdict(list)
    for s in settled:
        q = s.get('question', '').lower()
        if any(x in q for x in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'score', 'game']):
            cat = 'Sports'
        elif any(x in q for x in ['bitcoin', 'eth', 'crypto', 'btc']):
            cat = 'Crypto'
        elif any(x in q for x in ['weather', 'rain', 'temperature', 'snow']):
            cat = 'Weather'
        elif any(x in q for x in ['trump', 'doge', 'elon', 'congress', 'senate', 'deport', 'budget', 'spending']):
            cat = 'Politics/DOGE'
        elif any(x in q for x in ['fed', 'cpi', 'gdp', 'inflation', 'rate']):
            cat = 'Economics'
        else:
            cat = 'Other'
        categories[cat].append(s)

    if any(len(v) >= 3 for v in categories.values()):
        print(f"\n{'=' * 80}")
        print("6. BY CATEGORY")
        print("=" * 80)
        for cat, group in sorted(categories.items(), key=lambda x: -len(x[1])):
            if len(group) < 3:
                continue
            w = len([s for s in group if s.get('outcome') == 'WIN'])
            wr = w / len(group) * 100
            pnl = sum(s.get('theoretical_pnl', 0) for s in group)
            print(f"\n   {cat} ({len(group)} signals):")
            print(f"     Win Rate:   {wr:.1f}%")
            print(f"     Theory P&L: ${pnl:+.2f}")
            rec = "✓ GOOD" if wr >= 55 else ("⚠️  MARGINAL" if wr >= 45 else "✗ AVOID")
            print(f"     Signal:     {rec}")

    # =========================================================================
    # 7. PROFIT-LOCK SIMULATION
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("7. PROFIT-LOCK SIMULATION")
    print("(What if we exited positions up 40%+ instead of waiting to settle?)")
    print("This directly addresses 'was up, gave it back' problem")
    print("=" * 80)
    print("""
   NOTE: This simulation requires position price history (time-series).
   The current signal_log captures entry edge but not intraday price peaks.

   WHAT WE CAN INFER from the data:
   - Positions with high initial edge (20%+) that LOST → price reversed
   - These are the "was up, gave it back" candidates
   - A 40% profit-lock would have captured: (exit_price - entry_price) × contracts
     where exit_price = entry_price × 1.4

   Signals with high edge (20%+) that LOST (would have benefited from profit-lock):
""")
    high_edge_losses = [
        s for s in settled
        if abs(s.get('edge', 0)) >= 0.15 and s.get('outcome') == 'LOSS'
    ]
    if high_edge_losses:
        for s in high_edge_losses[:10]:
            entry = s.get('market_price', 0.5)
            simulated_take = entry * 0.4  # 40% gain captured
            print(f"   [{s.get('timestamp', '')[:10]}] {s.get('question', '')[:55]}")
            print(f"     Edge={s.get('edge', 0)*100:.0f}% | Entry≈{entry*100:.0f}¢ | "
                  f"Actual loss=${s.get('theoretical_pnl', 0):+.2f} | "
                  f"40%-lock would have saved: ${simulated_take:.2f}")
    else:
        print("   No high-edge losses found yet. Need more settled data.")

    # =========================================================================
    # 8. CURRENT OPEN POSITIONS RISK ASSESSMENT
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("8. CURRENT OPEN POSITIONS")
    print("=" * 80)

    if positions:
        total_exposure = sum(p.get('size', 0) for p in positions.values())
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions.values())
        
        print(f"\n   Open positions: {len(positions)}")
        print(f"   Total exposure: ${total_exposure:.2f}")
        print(f"   Unrealized P&L: ${total_unrealized:+.2f}")
        print(f"\n   {'Question':<50} {'Side':<5} {'Edge':<8} {'Entry':<8} {'Cur':<8} {'uPnL'}")
        print(f"   {'-'*90}")
        
        for pos in sorted(positions.values(), key=lambda x: x.get('unrealized_pnl', 0)):
            q = pos.get('question', '')[:48]
            side = pos.get('side', '?')
            edge = pos.get('edge', 0) * 100
            entry = pos.get('entry_price', 0) * 100
            cur = pos.get('current_price', 0) * 100
            upnl = pos.get('unrealized_pnl', 0)
            contracts = pos.get('contracts', int(pos.get('size', 0) / (pos.get('entry_price', 0.5) or 0.5)))
            print(f"   {q:<50} {side:<5} {edge:+.0f}%   {entry:.0f}¢     {cur:.0f}¢     ${upnl:+.2f}")

        # Assess risk for long-duration bets
        long_duration = []
        for pos in positions.values():
            end_date = pos.get('end_date')
            if end_date:
                try:
                    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    days_left = (end - datetime.now(end.tzinfo)).days
                    if days_left > 60:
                        long_duration.append((pos, days_left))
                except:
                    pass

        if long_duration:
            print(f"\n   ⚠️  LONG-DURATION POSITIONS (>60 days to settlement):")
            for pos, days in long_duration:
                print(f"     {pos.get('question', '')[:60]} | {days} days remaining")
            print(f"\n   These positions will oscillate with news cycles and are")
            print(f"   most at risk of 'was up, gave it back' pattern.")
    else:
        print("   No open positions.")

    # =========================================================================
    # 9. KEY RECOMMENDATIONS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("9. STRATEGY RECOMMENDATIONS BASED ON SIGNAL DATA")
    print("=" * 80)

    # Compute if NO-only is still correct
    if no_signals and yes_signals:
        no_wr = len([s for s in no_signals if s.get('outcome') == 'WIN']) / len(no_signals) * 100
        yes_wr = len([s for s in yes_signals if s.get('outcome') == 'WIN']) / len(yes_signals) * 100
        if no_wr > yes_wr + 10:
            print(f"\n   ✓ NO-only strategy VALIDATED: NO wins {no_wr:.0f}% vs YES wins {yes_wr:.0f}%")
        elif yes_wr > no_wr + 10:
            print(f"\n   ⚠️  Consider enabling YES bets: YES wins {yes_wr:.0f}% vs NO wins {no_wr:.0f}%")
        else:
            print(f"\n   📊 Mixed results: NO wins {no_wr:.0f}% vs YES wins {yes_wr:.0f}% (not conclusive)")

    print("""
   ADDRESSING "WAS UP YESTERDAY, DOWN TODAY" PROBLEM:
   
   Root Cause: Long-duration markets (DOGE cuts, deportation quotas, GDP) 
   fluctuate daily with news but won't settle until year-end 2025.
   With exits disabled, ALL paper gains get given back.

   RECOMMENDED FIXES:
   
   A) RE-ENABLE SELECTIVE PROFIT-LOCK for long-duration markets only:
      - If position is up 40%+ AND days_to_settlement > 30: take profit
      - This locks in "was up" scenarios without cutting short-term winners
      - Code change: profit_take_pct = 0.40 (not 999.0) when time > 30 days
   
   B) TIME-DECAY EXIT: For positions held > 14 days without settling,
      if unrealized gain > 25%, exit half the position to lock in profits
      while keeping skin in the game.
   
   C) NEWS-TRIGGERED RE-EVALUATION: When major news hits (DOGE budget cuts
      announced, deportation numbers released), force re-analysis and
      exit if edge has flipped negative.
   
   D) SEPARATE LONG vs SHORT DURATION strategies:
      - Short-duration (< 7 days): let settle (current behavior is correct)
      - Long-duration (> 30 days): use profit targets (40%) to avoid
        giving back daily swings
""")


async def main():
    """Main analysis entry point."""
    
    print("\n" + "=" * 80)
    print("BATTLEBOT SIGNAL PERFORMANCE ANALYZER")
    print("=" * 80)
    
    # Try multiple state file locations
    state_paths = [
        'storage/kalshi_state.json',
        'data/bot_state.json',
        'battlebot_state.json',
    ]
    
    state = {}
    loaded_from = None
    
    for path in state_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            state = load_state(full_path)
            loaded_from = path
            print(f"\n✓ Loaded state from: {path}")
            break
    
    if not state:
        print("\n⚠️  No state file found. Bot may not have run yet.")
        print(f"   Checked: {', '.join(state_paths)}")
        return
    
    signals = state.get('signal_log', [])
    trades = state.get('trades', [])
    positions = state.get('positions', {})
    daily_snapshots = state.get('daily_snapshots', {})
    
    print(f"   Signals: {len(signals)} | Trades: {len(trades)} | Open Positions: {len(positions)}")
    
    # Analyze daily P&L trend if snapshots exist
    if daily_snapshots:
        print(f"\n{'=' * 80}")
        print("DAILY PERFORMANCE SNAPSHOTS")
        print("=" * 80)
        for date, snap in sorted(daily_snapshots.items(), reverse=True)[:14]:
            pnl = snap.get('daily_pnl', snap.get('pnl', 0))
            bankroll = snap.get('bankroll', snap.get('total', 0))
            print(f"   {date}: P&L ${pnl:+.2f} | Bankroll ${bankroll:.2f}")
    
    # Main analysis
    analyze_signal_log(signals, trades, positions)
    
    # Trade history summary
    if trades:
        exits = [t for t in trades if t.get('action') == 'EXIT']
        entries = [t for t in trades if t.get('action') == 'ENTRY']
        
        print(f"\n{'=' * 80}")
        print("TRADE HISTORY SUMMARY")
        print("=" * 80)
        print(f"   Total entries: {len(entries)}")
        print(f"   Total exits:   {len(exits)}")
        
        completed = [t for t in exits if t.get('pnl') is not None]
        if completed:
            wins = [t for t in completed if t.get('pnl', 0) > 0]
            losses = [t for t in completed if t.get('pnl', 0) < 0]
            breakeven = [t for t in completed if t.get('pnl', 0) == 0]
            total_pnl = sum(t.get('pnl', 0) for t in completed)
            
            print(f"\n   Completed exits: {len(completed)}")
            print(f"   Wins: {len(wins)} | Losses: {len(losses)} | Breakeven: {len(breakeven)}")
            print(f"   Total realized P&L: ${total_pnl:+.2f}")
            
            if completed:
                win_rate = len(wins) / len(completed) * 100
                print(f"   Win rate: {win_rate:.1f}%")
            
            # Exit reason breakdown
            reasons = defaultdict(int)
            for t in exits:
                reasons[t.get('reason', 'UNKNOWN')] += 1
            print(f"\n   Exit reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"     {reason}: {count}")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
