"""
Narrator agent: live commentary written as match events stream in.

It watches every event from the replay feed but only speaks when something
is worth saying:
  - goals, own goals, saves, cards, big chances and the start of each period
  - a short "state of play" line if nothing notable has happened for a while

Each line is written by the model from the events that triggered it, cites
their IDs, and is checked so it can only cite events it was actually shown.
Without an Azure connection (--no-ai) it falls back to simple templates, so
the live feed never stops.

Usage:
    python -m agents.narrator --match data/match_002.json
    python -m agents.narrator --match data/match_002.json --speed 120
    python -m agents.narrator --match data/match_002.json --no-ai   # free, offline
"""

import argparse
import json
import re
from collections import deque
from pathlib import Path

from generator.replay import load_match, replay

QUIET_SPELL_MINUTES = 6   # speak at least this often
CONTEXT_EVENTS = 10       # how many recent events the model sees

SYSTEM_PROMPT = """You are the Narrator agent: a live football commentator.
You receive the latest moment in a match plus the events leading up to it.
Write the next line of live commentary.

Rules:
- Present tense, like a TV commentator. Two sentences for goals and red cards,
  ONE sentence for everything else.
- Match your energy to the moment: huge for goals, warm for great saves, calm for a lull.
- "what_happened" is the exact truth about the latest moment. Follow it; never contradict it.
- "phase" says which part of the match it is (e.g. first period of extra time).
  Only say "stoppage time" if the phase says so.
- For "state_of_play", describe the flow since the last line using "since_last_line"
  (who has the ball, who is pressing). Don't narrate a single routine pass.
- Only mention the score after a goal, or occasionally in a quiet moment.
  "score" is the score right now, already including the latest moment.
  Never write any other scoreline.
- Mention the phase only at the start of a period or when it adds drama.
- Never put event IDs or brackets in the text; they go in evidence_ids only.
- Use ONLY the facts provided. Never invent players, events, numbers or crowd details.
- Use team and player names. Never say "home" or "away".
- Don't repeat the phrasing of the previous lines you are shown.
- Cite the event IDs your line is based on.

Reply with JSON: {"text": "...", "evidence_ids": [123, 124]}"""


PERIOD_END = {1: 45, 2: 90, 3: 105, 4: 120}


def clock(event):
    """Match clock, with stoppage time shown football-style: 45+1:20."""
    end = PERIOD_END.get(event["period"])
    if end and event["minute"] >= end:
        return f"{end}+{event['minute'] - end}:{event['second']:02d}"
    return f"{event['minute']}:{event['second']:02d}"


PHASE_NAMES = {1: "first half", 2: "second half",
               3: "first period of extra time", 4: "second period of extra time"}


def phase(event):
    """Which part of the match an event is in, in words the model can't misread."""
    name = PHASE_NAMES.get(event["period"], "match")
    end = PERIOD_END.get(event["period"])
    if end and event["minute"] >= end:
        return f"stoppage time at the end of the {name}"
    if event["type"] == "kick_off" and event["minute"] in (0, 45, 90, 105):
        return f"start of the {name}"
    return name


def other_team(match_info, team_name):
    home, away = match_info["home"]["name"], match_info["away"]["name"]
    return away if team_name == home else home


def what_happened(event, context, match_info):
    """A plain-English statement of the latest moment, written by code, not the model."""
    player, team, etype, outcome = event["player"], event["team"], event["type"], event["outcome"]
    if etype == "save":
        shot = next((e for e in reversed(context) if e["type"] == "shot" and e["id"] < event["id"]), None)
        shooter = f"{shot['player']} of {shot['team']}" if shot else other_team(match_info, team)
        return (f"{shooter} has a shot saved by {player}, the {team} goalkeeper. "
                f"{other_team(match_info, team)} were attacking.")
    if etype == "own_goal":
        return (f"{player} of {team} puts the ball into his own net. "
                f"The goal counts for {event.get('credited_to', other_team(match_info, team))}.")
    if etype == "shot" and outcome == "goal":
        extra = f", assisted by {event['assist']}" if event.get("assist") else ""
        how = f" from a {event['set_piece'].replace('_', ' ')}" if event.get("set_piece") else ""
        head = " with a header" if event.get("body_part") == "head" else ""
        return f"GOAL for {team}: {player} scores{head}{how}{extra}."
    if etype == "shot":
        return f"{player} of {team} shoots; outcome: {outcome.replace('_', ' ')}."
    if etype == "card":
        return f"{player} of {team} is shown a {outcome.replace('_', ' ')} card."
    if etype == "kick_off":
        return f"{team} kick off."
    return f"{player} of {team}: {etype} ({outcome})."


SCORELINE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b(?!\s*-?\s*pass)")
BRACKETED_IDS = re.compile(r"\s*\[[\d,\s]+\]")


def check_scorelines(text, score):
    """Raise if the text states a scoreline that isn't the real one."""
    real = {(score["home"], score["away"]), (score["away"], score["home"])}
    for a, b in SCORELINE.findall(text):
        if (int(a), int(b)) not in real:
            raise ValueError(f"wrong score {a}-{b} in commentary")


def clean(text):
    """Remove any event IDs the model leaked into the text."""
    return BRACKETED_IDS.sub("", text).strip()


def compact(event):
    """The fields of an event the model needs, nothing more."""
    keep = ("id", "minute", "second", "team", "player", "type", "outcome",
            "assist", "set_piece", "body_part", "xg", "credited_to", "fouled_player")
    return {k: event[k] for k in keep if k in event}


class Narrator:
    def __init__(self, match_info, ask=None, use_ai=True):
        self.match = match_info
        self.use_ai = use_ai
        if use_ai and ask is None:
            from agents.llm import ask_json as ask
        self.ask = ask
        self.recent = deque(maxlen=CONTEXT_EVENTS)
        self.lines = []
        self.score = {"home": 0, "away": 0}
        self.period = None
        self.last_spoke_minute = 0
        self.spell_events = []

    # ---------- deciding when to speak ----------

    def trigger(self, event):
        """Return why this event deserves commentary, or None."""
        etype, outcome = event["type"], event["outcome"]
        if event["period"] != self.period:
            return "period_start"
        if etype == "own_goal" or (etype == "shot" and outcome == "goal"):
            return "goal"
        if etype == "save":
            return "save"  # the saved shot is included as context
        if etype == "shot" and outcome != "saved" and (event.get("key_moment") or event.get("xg", 0) >= 0.15):
            return "chance"
        if etype == "card":
            return "red_card" if outcome == "red" else "booking"
        if event["minute"] - self.last_spoke_minute >= QUIET_SPELL_MINUTES:
            return "state_of_play"
        return None

    def observe(self, event):
        """Feed one event in; returns a commentary line or None."""
        self.recent.append(event)
        self.spell_events.append(event)
        reason = self.trigger(event)
        self.period = event["period"]
        if event.get("score"):
            self.score = event["score"]
        if reason is None:
            return None

        line = self.speak(reason, event)
        self.lines.append(line)
        self.last_spoke_minute = event["minute"]
        self.spell_events = []
        return line

    # ---------- writing the line ----------

    def spell_summary(self):
        """Who has been on top since the last line (used for state_of_play)."""
        summary = {}
        for side in ("home", "away"):
            mine = [e for e in self.spell_events if e["side"] == side]
            summary[self.match[side]["name"]] = {
                "passes": sum(e["type"] == "pass" for e in mine),
                "passes_into_final_third": sum(e["type"] == "pass" and e["outcome"] == "complete"
                                               and e["end"] and e["end"][0] > 66 for e in mine),
                "ball_wins": sum(e["type"] in ("tackle", "interception") for e in mine),
            }
        return summary

    def speak(self, reason, event):
        line = {"clock": clock(event), "minute": event["minute"], "period": event["period"],
                "kind": reason, "event_id": event["id"], "score": dict(self.score)}
        home, away = self.match["home"]["name"], self.match["away"]["name"]
        score_text = f"{home} {self.score['home']}-{self.score['away']} {away}"

        if self.use_ai:
            prompt = {
                "teams": {"home": home, "away": away},
                "score": score_text,
                "moment": reason,
                "phase": phase(event),
                "previous_lines": [l["text"] for l in self.lines[-3:]],
            }
            if reason == "state_of_play":
                # Describe the flow of play, not the single event that triggered the line.
                evidence = [e for e in self.spell_events
                            if e["type"] in ("tackle", "interception", "shot")
                            or (e["type"] == "pass" and e["end"] and e["end"][0] > 66)][-8:]
                allowed_ids = {e["id"] for e in evidence} | {event["id"]}
                prompt["what_happened"] = "A quiet spell. Describe the general flow of play in your own words."
                prompt["since_last_line"] = self.spell_summary()
                prompt["example_events"] = [compact(e) for e in evidence]
            else:
                context = list(self.recent) if reason != "period_start" else [event]
                allowed_ids = {e["id"] for e in context}
                happened = what_happened(event, context, self.match)
                if reason == "goal":
                    happened += f" The score is now {score_text}."
                prompt["what_happened"] = happened
                prompt["latest_event"] = compact(event)
                prompt["events_leading_up"] = [compact(e) for e in context]
            try:
                reply = self.ask(SYSTEM_PROMPT, json.dumps(prompt))
                text = clean(reply["text"])
                check_scorelines(text, self.score)
                cited = [i for i in reply.get("evidence_ids", []) if i in allowed_ids]
                line["text"] = text
                line["evidence_ids"] = list(dict.fromkeys(cited)) or [event["id"]]
                line["source"] = "ai"
                return line
            except Exception as error:  # wrong score, network problem, bad reply: keep the feed alive
                line["fallback_reason"] = str(error) or type(error).__name__

        line["text"] = self.template(reason, event)
        line["evidence_ids"] = [event["id"]]
        line["source"] = "template"
        return line

    def template(self, reason, event):
        home, away = self.match["home"]["name"], self.match["away"]["name"]
        score = f"{home} {self.score['home']}-{self.score['away']} {away}"
        player, team = event["player"], event["team"]
        if reason == "period_start":
            names = {1: "We're under way", 2: "The second half is under way",
                     3: "Extra time begins", 4: "The second period of extra time begins"}
            return f"{names.get(event['period'], 'Play resumes')}. {score}."
        if reason == "goal":
            if event["type"] == "own_goal":
                return f"Own goal! {player} turns it into his own net. {score}."
            return f"GOAL! {player} scores for {team}. {score}."
        if reason == "save":
            return f"Great save by {player} to keep {team} in it."
        if reason == "chance":
            return f"Chance for {team}, but {player} can't find the target."
        if reason == "red_card":
            return f"Red card! {player} is sent off and {team} are down to ten."
        if reason == "booking":
            return f"{player} goes into the book."
        return f"{score} as play continues."


def print_line(line):
    marker = {"goal": "⚽", "save": "🧤", "red_card": "🟥", "booking": "🟨"}.get(line["kind"], "  ")
    print(f"[{line['clock']:>8}] {marker} {line['text']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Live commentary from a replayed match.")
    parser.add_argument("--match", default="data/match_002.json")
    parser.add_argument("--speed", type=float, default=60, help="match seconds per real second (0 = instant)")
    parser.add_argument("--no-ai", action="store_true", help="use templates instead of the model (free, offline)")
    parser.add_argument("--out", help="where to save the commentary (default: data/commentary_<match>.json)")
    args = parser.parse_args()

    match_data = load_match(args.match)
    narrator = Narrator(match_data["match"], use_ai=not args.no_ai)
    home, away = match_data["match"]["home"]["name"], match_data["match"]["away"]["name"]
    print(f"\n{home} vs {away}: live commentary\n")

    try:
        for event in replay(args.match, args.speed):
            line = narrator.observe(event)
            if line:
                print_line(line)
    except KeyboardInterrupt:
        print("\nCommentary stopped.")

    out = Path(args.out or Path(args.match).with_name(
        "commentary_" + Path(args.match).name.replace("match_", "")))
    out.write_text(json.dumps(narrator.lines, indent=2))
    ai_lines = sum(l["source"] == "ai" for l in narrator.lines)
    fallbacks = [l for l in narrator.lines if l.get("fallback_reason")]
    print(f"\n{len(narrator.lines)} lines ({ai_lines} written by AI). Saved to {out}")
    for l in fallbacks:
        print(f"  Safety fallback at {l['clock']}: {l['fallback_reason']}")


if __name__ == "__main__":
    main()
