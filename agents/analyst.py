"""
Analyst agent: turns match statistics into explainable insights.

Step 1 (Python): compute exact facts from the event stream (match_stats.py).
Step 2 (model):  explain the facts as insights, citing event IDs as evidence.
Step 3 (Python): verify every cited event ID exists in the match, and drop
                 any insight that has no valid evidence.

Usage:
    python -m agents.analyst --match data/match_002.json
    python -m agents.analyst --match data/match_002.json --stats-only   # no AI call, free
"""

import argparse
import json
from pathlib import Path

from agents.match_stats import compute_match_facts

SYSTEM_PROMPT = """You are the Analyst agent in a football match intelligence system.
You receive exact statistics computed from a synthetic match's event stream.
Your job is to explain what really happened in the match, beyond the scoreline.

Rules:
- Use ONLY the facts provided. Never invent players, events, minutes or numbers.
- Every insight must cite evidence_ids: event IDs taken from the facts
  (fields named id, evidence_ids, build_up_ids or save_ids).
- Momentum runs from +100 (home team on top) to -100 (away team on top).
- Always use team names, never the words "home" or "away".
- If standout_goalkeeper is not null, one insight MUST be about that goalkeeper,
  of type "goalkeeping", citing their save_ids.
- Prefer insights a fan would not get from the scoreline alone: momentum shifts,
  spells of pressure, goalkeeping, how goals were built, who controlled the ball.

Reply with JSON in exactly this shape:
{
  "headline": "one-sentence story of the match",
  "insights": [
    {
      "title": "short title",
      "type": "momentum | key_moment | pressure | goalkeeping | build_up | control",
      "minute": 61,
      "team": "team name or null",
      "explanation": "2-3 sentences explaining what happened and why it mattered",
      "evidence_ids": [809, 813]
    }
  ]
}
Give 6 to 8 insights in match order."""


def verify_citations(analysis, match_data):
    """Keep only event IDs that exist; drop insights left with no evidence."""
    valid_ids = {e["id"] for e in match_data["events"]}
    verified, removed_ids = [], 0
    for insight in analysis.get("insights", []):
        cited = insight.get("evidence_ids", [])
        kept = list(dict.fromkeys(i for i in cited if isinstance(i, int) and i in valid_ids))
        removed_ids += len(cited) - len(kept)
        if kept:
            insight["evidence_ids"] = kept
            verified.append(insight)
    # Keep insights in match order, whatever order the model wrote them in.
    analysis["insights"] = sorted(verified, key=lambda i: i.get("minute") or 0)
    analysis["verification"] = {
        "insights_kept": len(verified),
        "invalid_ids_removed": removed_ids,
    }
    return analysis


def analyse(match_data, ask=None):
    """Run the Analyst agent on a loaded match. `ask` can be swapped out in tests."""
    if ask is None:
        from agents.llm import ask_json as ask
    facts = compute_match_facts(match_data)
    analysis = ask(SYSTEM_PROMPT, json.dumps(facts))
    return verify_citations(analysis, match_data)


def print_analysis(analysis, match_data):
    events = {e["id"]: e for e in match_data["events"]}
    print(f"\n{analysis.get('headline', '')}\n")
    for insight in analysis["insights"]:
        print(f"[{insight.get('minute', '?')}'] {insight['title']}  ({insight.get('type', '')})")
        print(f"   {insight['explanation']}")
        sample = [events[i] for i in insight["evidence_ids"][:3]]
        cited = ", ".join(f"#{e['id']} {e['player']} {e['type']}" for e in sample)
        print(f"   Evidence: {cited}\n")
    v = analysis["verification"]
    print(f"Verified: {v['insights_kept']} insights kept, {v['invalid_ids_removed']} invalid event IDs removed")


def main():
    parser = argparse.ArgumentParser(description="Run the Analyst agent on a match.")
    parser.add_argument("--match", default="data/match_002.json")
    parser.add_argument("--out", help="where to save the analysis (default: data/analysis_<match>.json)")
    parser.add_argument("--stats-only", action="store_true", help="print the computed facts without calling the model")
    args = parser.parse_args()

    match_data = json.loads(Path(args.match).read_text())

    if args.stats_only:
        print(json.dumps(compute_match_facts(match_data), indent=2))
        return

    analysis = analyse(match_data)
    out = Path(args.out or Path(args.match).with_name("analysis_" + Path(args.match).name.replace("match_", "")))
    out.write_text(json.dumps(analysis, indent=2))
    print_analysis(analysis, match_data)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
