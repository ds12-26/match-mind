"""
Replay a generated match as if it were a live feed.

Events are emitted one at a time on a sped-up match clock, so the agents
and the app can treat the match exactly like a real-time stream. In
production, this module is the piece you'd swap for a real data feed.

Usage:
    python generator/replay.py                      # 90 minutes in ~3 minutes
    python generator/replay.py --speed 120          # faster
    python generator/replay.py --key-only           # print only key moments

In code:
    from generator.replay import replay
    for event in replay("data/match_001.json", speed=30):
        ...
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATCH = ROOT / "data" / "match_001.json"


def match_seconds(event):
    """Seconds on the match clock (period 2 starts at 45:00)."""
    return event["minute"] * 60 + event["second"]


def load_match(path=DEFAULT_MATCH):
    return json.loads(Path(path).read_text())


def replay(path=DEFAULT_MATCH, speed=30.0):
    """Yield events in order, sleeping between them to mimic a live match.

    speed is how many match seconds pass per real second:
    30 plays 90 minutes in about 3 minutes; 0 means no waiting.
    """
    events = load_match(path)["events"]
    previous = None
    for event in events:
        if speed and previous is not None and event["period"] == previous["period"]:
            gap = match_seconds(event) - match_seconds(previous)
            if gap > 0:
                time.sleep(gap / speed)
        previous = event
        yield event


def describe(event):
    clock = f"{event['minute']:02d}:{event['second']:02d}"
    text = f"[{clock}] {event['team']:<20} {event['player']:<14} {event['type']} ({event['outcome']})"
    if event.get("score"):
        text += f"   SCORE {event['score']['home']}-{event['score']['away']}"
    return text


def main():
    parser = argparse.ArgumentParser(description="Replay a synthetic match as a live stream.")
    parser.add_argument("--match", default=DEFAULT_MATCH, help="generated match JSON file")
    parser.add_argument("--speed", type=float, default=30, help="match seconds per real second (0 = instant)")
    parser.add_argument("--key-only", action="store_true", help="only print key moments, goals and cards")
    args = parser.parse_args()

    for event in replay(args.match, args.speed):
        if args.key_only and not (event.get("key_moment") or event["type"] in ("card", "kick_off")):
            continue
        print(describe(event), flush=True)


if __name__ == "__main__":
    main()
