"""
Synthetic football match generator.

Reads a match story (phases + key moments) from match_script.json and
generates a full, realistic stream of match events around it: passes,
tackles, interceptions, fouls, throw-ins, shots, cards and goals.

Usage:
    python generator/generate_match.py
    python generator/generate_match.py --script generator/match_script.json --out data/match_001.json

Coordinate system:
    Every event's coordinates are from the point of view of the team
    performing it. x runs 0-100 from that team's own goal to the
    opponent's goal; y runs 0-100 from left touchline to right.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = ROOT / "generator" / "match_script.json"
DEFAULT_OUT = ROOT / "data" / "match_001.json"

# Which positions tend to have the ball in each third of the pitch.
ZONE_WEIGHTS = {
    "defensive": {"GK": 2, "RB": 4, "CB": 6, "LB": 4, "DM": 5, "CM": 3, "AM": 1, "RW": 1, "LW": 1, "ST": 0.5},
    "middle":    {"GK": 0, "RB": 3, "CB": 2, "LB": 3, "DM": 5, "CM": 6, "AM": 5, "RW": 3, "LW": 3, "ST": 2},
    "attacking": {"GK": 0, "RB": 2, "CB": 0.3, "LB": 2, "DM": 1, "CM": 3, "AM": 5, "RW": 5, "LW": 5, "ST": 6},
}


def clamp(value, low=1.0, high=99.0):
    return max(low, min(high, value))


def flip(point):
    """Convert a location to the other team's point of view."""
    return [round(100 - point[0], 1), round(100 - point[1], 1)]


class MatchGenerator:
    def __init__(self, script):
        self.script = script
        self.rng = random.Random(script.get("seed", 0))
        self.teams = {"home": script["home"], "away": script["away"]}
        self.sent_off = {"home": set(), "away": set()}
        self.booked = set()
        self.score = {"home": 0, "away": 0}
        self.events = []
        self.possession_id = 0

    # ---------- helpers ----------

    def other(self, side):
        return "away" if side == "home" else "home"

    def phase_at(self, period, minute):
        for phase in self.script["phases"]:
            if phase["period"] == period and phase["start"] <= minute < phase["end"]:
                return phase
        # Fall back to the last phase of the period.
        return [p for p in self.script["phases"] if p["period"] == period][-1]

    def territory(self, phase, side):
        return phase["home_territory"] if side == "home" else phase["away_territory"]

    def pick_player(self, side, x, exclude=None):
        zone = "defensive" if x < 35 else "middle" if x < 70 else "attacking"
        weights = ZONE_WEIGHTS[zone]
        candidates = [
            p for p in self.teams[side]["players"]
            if p["name"] not in self.sent_off[side] and p["name"] != exclude
        ]
        return self.rng.choices(candidates, [weights.get(p["position"], 1) for p in candidates])[0]

    def new_possession(self):
        self.possession_id += 1

    def add(self, period, t, side, player, etype, outcome, start, end=None, **extra):
        event = {
            "id": len(self.events) + 1,
            "period": period,
            "minute": int(t // 60),
            "second": int(t % 60),
            "team": self.teams[side]["name"],
            "side": side,
            "player": player["name"] if isinstance(player, dict) else player,
            "type": etype,
            "outcome": outcome,
            "start": [round(start[0], 1), round(start[1], 1)],
            "end": [round(end[0], 1), round(end[1], 1)] if end else None,
            "possession_id": self.possession_id,
            "under_pressure": extra.pop("under_pressure", False),
        }
        event.update(extra)
        self.events.append(event)
        return event

    # ---------- key moments ----------

    def player(self, side, name):
        return next(p for p in self.teams[side]["players"] if p["name"] == name)

    def goalkeeper(self, side):
        return next(p for p in self.teams[side]["players"] if p["position"] == "GK")

    def record_save(self, period, t, shooting_side):
        """The defending goalkeeper makes a save."""
        keeper_side = self.other(shooting_side)
        return self.add(period, t, keeper_side, self.goalkeeper(keeper_side), "save", "saved", [4, 50])

    def build_up(self, period, t, key_t, side, ball, target, passes, assist=None, exclude=None):
        """Passes that carry the ball from its current spot to the key moment's location."""
        times = [t + (key_t - t) * (i + 1) / (passes + 1) for i in range(passes)]
        for i, pass_t in enumerate(times):
            last = i == passes - 1
            if last and assist:
                passer = self.player(side, assist)
            else:
                passer = self.pick_player(side, ball["pos"][0], exclude=exclude)
            if last:
                end = list(target)
            else:
                step = (target[0] - ball["pos"][0]) / (passes - i)
                end = [clamp(ball["pos"][0] + step + self.rng.gauss(0, 4)),
                       clamp(ball["pos"][1] + self.rng.gauss(0, 14))]
            self.add(period, pass_t, side, passer, "pass", "complete", ball["pos"], end,
                     under_pressure=self.rng.random() < 0.4)
            ball["pos"] = end

    def score_goal(self, event, scoring_side):
        self.score[scoring_side] += 1
        event["score"] = dict(self.score)
        self.new_possession()
        # Conceding team kicks off a minute later.
        return {"side": self.other(scoring_side), "pos": [50, 50]}

    def play_key_event(self, period, t, key, ball):
        """Build up to and play a scripted key moment. Returns (ball, t) for the restart.

        Supported key event types:
          shot      outcome goal / saved / hit_post / off_target
                    optional: assist, set_piece ("corner" or "free_kick"),
                    body_part, build_up (number of passes before the shot)
          own_goal  "team" is the side of the player who scores into their own net
          card      outcome yellow / red
        """
        side = key["team"]
        key_t = key["minute"] * 60 + key["second"]
        target = [key["x"], key["y"]]
        set_piece = key.get("set_piece")

        if key["type"] == "own_goal":
            # The opponent attacks; the cross is turned in by a defender.
            attackers = self.other(side)
            if ball["side"] != attackers:
                self.new_possession()
                ball = {"side": attackers, "pos": [55, 50]}
            cross_from = [92, 8 if target[1] < 50 else 92]
            self.build_up(period, t, key_t - 2, attackers, ball, cross_from, key.get("build_up", 3))
            self.add(period, key_t - 1, attackers, key.get("assist") or self.pick_player(attackers, 92),
                     "cross", "complete", cross_from, flip(target), under_pressure=True)
            og = self.add(period, key_t, side, key["player"], "own_goal", "goal", target, [0, 50],
                          under_pressure=True, credited_to=self.teams[attackers]["name"], key_moment=True)
            return self.score_goal(og, attackers), key_t + 60

        if ball["side"] != side:
            self.new_possession()
            ball = {"side": side, "pos": [clamp(target[0] - 35), 50]}

        if key["type"] == "card":
            # The carded player commits a foul; the opponent gets a free kick.
            victim = self.pick_player(self.other(side), 100 - target[0])
            self.add(period, key_t - 1, side, key["player"], "foul", "committed", target,
                     fouled_player=victim["name"])
            self.add(period, key_t, side, key["player"], "card", key["outcome"], target, key_moment=True)
            if key["outcome"] == "red":
                self.sent_off[side].add(key["player"])
            self.new_possession()
            return {"side": self.other(side), "pos": flip(target)}, key_t + 45

        if key["type"] != "shot":
            raise ValueError(f"Unsupported key event type: {key['type']}")

        extra = {}
        if set_piece == "corner":
            # A blocked attempt wins a corner; the shot comes from the delivery.
            corner_y = 1 if target[1] < 50 else 99
            blocked_t = t + (key_t - t) * 0.6
            self.build_up(period, t, blocked_t, side, ball, [80, 50], 2, exclude=key["player"])
            self.add(period, blocked_t, side, self.pick_player(side, 80, exclude=key["player"]),
                     "shot", "blocked", [80, 50], [100, 50], under_pressure=True, xg=self.estimate_xg([80, 50]))
            taker = self.player(side, key["assist"]) if key.get("assist") else self.pick_player(side, 90)
            self.add(period, key_t - 2, side, taker, "corner", "complete", [99, corner_y], target)
            extra["set_piece"] = "corner"
        elif set_piece == "free_kick":
            # The defending team gives away a foul at the spot of the free kick.
            fouler = self.pick_player(self.other(side), 100 - target[0])
            self.add(period, t, self.other(side), fouler, "foul", "committed", flip(target),
                     fouled_player=key["player"])
            extra["set_piece"] = "free_kick"
        else:
            self.build_up(period, t, key_t, side, ball, target, key.get("build_up", 3),
                          assist=key.get("assist"), exclude=key["player"])

        if key.get("body_part"):
            extra["body_part"] = key["body_part"]
        if key.get("assist"):
            extra["assist"] = key["assist"]

        shot = self.add(period, key_t, side, key["player"], "shot", key["outcome"], target, [100, 50],
                        under_pressure=set_piece != "free_kick", xg=self.estimate_xg(target),
                        key_moment=True, **extra)
        if key["outcome"] == "goal":
            return self.score_goal(shot, side), key_t + 60
        if key["outcome"] == "saved":
            save = self.record_save(period, key_t + 1, side)
            save["key_moment"] = True
        # Saved / off target / post: goalkeeper restarts.
        self.new_possession()
        return {"side": self.other(side), "pos": [6, 50]}, key_t + 8

    def estimate_xg(self, pos):
        """Rough expected-goals estimate from shot location."""
        distance = ((100 - pos[0]) ** 2 + ((pos[1] - 50) * 0.7) ** 2) ** 0.5
        return round(max(0.02, min(0.75, 0.9 - distance * 0.035)), 2)

    # ---------- routine play ----------

    def play_action(self, period, t, ball, phase):
        """Play one on-the-ball action. Returns (ball, seconds_elapsed)."""
        side = ball["side"]
        x, y = ball["pos"]
        player = self.pick_player(side, x)
        pressure = self.rng.random() < 0.15 + x / 250

        # Shot from a dangerous position (non-scripted shots never score,
        # so the scoreline always follows the story).
        # A phase can set one shot_rate for both teams, or home_shot_rate / away_shot_rate.
        shot_rate = phase.get(f"{side}_shot_rate", phase["shot_rate"])
        if x > 72 and self.rng.random() < shot_rate * (1 + (x - 72) / 15):
            outcome = self.rng.choices(["off_target", "saved", "blocked"], [45, 30, 25])[0]
            self.add(period, t, side, player, "shot", outcome, [x, y], [100, 50],
                     under_pressure=pressure, xg=self.estimate_xg([x, y]))
            if outcome == "saved":
                self.record_save(period, t + 1, side)
            self.new_possession()
            if outcome == "blocked":
                return {"side": self.other(side), "pos": [clamp(100 - x + 8), 100 - y]}, 5
            return {"side": self.other(side), "pos": [6, 50]}, 10

        # Occasional foul by the defending team.
        if self.rng.random() < 0.025:
            fouler = self.pick_player(self.other(side), 100 - x)
            self.add(period, t, self.other(side), fouler, "foul", "committed", flip([x, y]),
                     fouled_player=player["name"])
            # Random bookings never hit a player already on a yellow, so the
            # only sending-offs are the ones scripted in match_script.json.
            if self.rng.random() < 0.12 and fouler["name"] not in self.booked:
                self.add(period, t + 1, self.other(side), fouler, "card", "yellow", flip([x, y]))
                self.booked.add(fouler["name"])
            return ball, 25

        # Pass forward-ish, drifting towards the team's preferred territory.
        target_x = self.territory(phase, side)
        dx = self.rng.gauss((target_x - x) * 0.25 + 3, 12)
        end = [clamp(x + dx), clamp(y + self.rng.gauss(0, 18))]
        duration = self.rng.uniform(2.5, 5.5)

        # The team meant to dominate this phase loses the ball less often.
        share = phase["home_possession"] if side == "home" else 1 - phase["home_possession"]
        turnover_chance = phase["turnover_rate"] * ((1 - share) / 0.5) ** 1.5 * (1.4 if pressure else 1.0)
        roll = self.rng.random()

        if roll < 0.03:
            # Ball goes out of play: throw-in to the opponent.
            self.add(period, t, side, player, "pass", "out", [x, y], [end[0], 0 if y < 50 else 100],
                     under_pressure=pressure)
            self.new_possession()
            opp = self.other(side)
            thrower = self.pick_player(opp, 100 - end[0])
            pos = [clamp(100 - end[0]), 100 if y < 50 else 0]
            self.add(period, t + 15, opp, thrower, "throw_in", "complete", pos, [clamp(pos[0] + 5), 50])
            return {"side": opp, "pos": [clamp(pos[0] + 5), 50]}, 20

        if roll < 0.03 + turnover_chance:
            opp = self.other(side)
            win_pos = flip(end)
            defender = self.pick_player(opp, win_pos[0])
            if self.rng.random() < 0.5:
                self.add(period, t, side, player, "pass", "intercepted", [x, y], end, under_pressure=pressure)
                self.new_possession()
                self.add(period, t + 1, opp, defender, "interception", "won", win_pos)
            else:
                self.add(period, t, side, player, "dribble", "lost", [x, y], [clamp(x + 3), y],
                         under_pressure=True)
                self.new_possession()
                self.add(period, t + 1, opp, defender, "tackle", "won", flip([clamp(x + 3), y]))
                win_pos = flip([clamp(x + 3), y])
            return {"side": opp, "pos": win_pos}, duration

        self.add(period, t, side, player, "pass", "complete", [x, y], end, under_pressure=pressure)
        return {"side": side, "pos": end}, duration

    # ---------- main loop ----------

    def play_period(self, period, kickoff_side):
        phases = [p for p in self.script["phases"] if p["period"] == period]
        t = phases[0]["start"] * 60
        end_t = phases[-1]["end"] * 60
        keys = sorted(
            (k for k in self.script["key_events"] if k["period"] == period),
            key=lambda k: k["minute"] * 60 + k["second"],
        )

        self.new_possession()
        ball = {"side": kickoff_side, "pos": [50, 50]}
        self.add(period, t, kickoff_side, self.pick_player(kickoff_side, 50), "kick_off", "complete", [50, 50])
        t += 2

        while t < end_t:
            # Start building up early enough that the move never runs past the scripted time
            # (a single routine action can use up to 25 seconds).
            lead = 0
            if keys:
                lead = 25 + (60 if keys[0].get("set_piece") else max(15, 5 * keys[0].get("build_up", 3)))
            if keys and t >= keys[0]["minute"] * 60 + keys[0]["second"] - lead:
                ball, t = self.play_key_event(period, t, keys.pop(0), ball)
                if ball["pos"] == [50, 50]:
                    self.add(period, t, ball["side"], self.pick_player(ball["side"], 50), "kick_off", "complete", [50, 50])
                    t += 2
                continue
            phase = self.phase_at(period, int(t // 60))
            ball, elapsed = self.play_action(period, t, ball, phase)
            t += elapsed

    def generate(self):
        periods = sorted({p["period"] for p in self.script["phases"]})
        for period in periods:
            # Kick-offs alternate between the teams each period.
            self.play_period(period, "home" if period % 2 else "away")
        return {
            "match": {
                "match_id": self.script["match_id"],
                "home": {"name": self.teams["home"]["name"], "short": self.teams["home"]["short"],
                         "players": self.teams["home"]["players"]},
                "away": {"name": self.teams["away"]["name"], "short": self.teams["away"]["short"],
                         "players": self.teams["away"]["players"]},
                "final_score": dict(self.score),
                "after_extra_time": max(periods) > 2,
                "periods": periods,
                "event_count": len(self.events),
                "synthetic": True,
                "coordinate_system": "x 0-100 from acting team's own goal to opponent goal; y 0-100 across the pitch",
            },
            "events": self.events,
        }


def print_summary(match):
    events = match["events"]
    home, away = match["match"]["home"]["name"], match["match"]["away"]["name"]
    aet = " (after extra time)" if match["match"].get("after_extra_time") else ""
    print(f"\n{home} {match['match']['final_score']['home']} - "
          f"{match['match']['final_score']['away']} {away}{aet}")
    print(f"Total events: {len(events)}\n")

    for side, name in (("home", home), ("away", away)):
        mine = [e for e in events if e["side"] == side]
        passes = [e for e in mine if e["type"] == "pass"]
        shots = [e for e in mine if e["type"] == "shot"]
        print(f"{name}:")
        print(f"  passes {len(passes)} ({sum(e['outcome'] == 'complete' for e in passes)} complete)")
        on_target = sum(e["outcome"] in ("goal", "saved") for e in shots)
        print(f"  shots {len(shots)} ({on_target} on target), xG {sum(e.get('xg', 0) for e in shots):.2f}")
        print(f"  saves by goalkeeper {sum(e['type'] == 'save' for e in mine)}")
        print(f"  tackles+interceptions {sum(e['type'] in ('tackle', 'interception') for e in mine)}")
    total_passes = sum(e["type"] == "pass" for e in events)
    home_passes = sum(e["type"] == "pass" and e["side"] == "home" for e in events)
    print(f"\nPossession (by passes): {home} {home_passes / total_passes:.0%}")
    print("\nKey moments:")
    for e in events:
        if e.get("key_moment"):
            score = f"  {e['score']['home']}-{e['score']['away']}" if e.get("score") else ""
            print(f"  P{e['period']} {e['minute']:02d}:{e['second']:02d}  #{e['id']:<5} "
                  f"{e['team']:<20} {e['player']:<14} {e['type']} ({e['outcome']}){score}")


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic football match.")
    parser.add_argument("--script", default=DEFAULT_SCRIPT, help="match story JSON file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the generated match")
    args = parser.parse_args()

    script = json.loads(Path(args.script).read_text())
    match = MatchGenerator(script).generate()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(match, indent=2))
    print_summary(match)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
