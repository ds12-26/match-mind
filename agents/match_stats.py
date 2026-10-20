"""
Exact match statistics computed from the event stream.

No AI here: this is plain Python, so every number is correct and every
finding keeps the IDs of the events behind it. The Analyst agent hands
these facts to the model, which explains them in plain language.
"""

from collections import defaultdict

# Clock minute at which each period starts (3 and 4 are extra time).
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105}
PERIOD_LENGTH = {1: 45, 2: 45, 3: 15, 4: 15}
WINDOW = 5  # minutes per momentum window

BALL_WINS = {"tackle", "interception"}


def is_goal(event):
    return (event["type"] == "shot" and event["outcome"] == "goal") or event["type"] == "own_goal"


def scoring_side(event):
    """The side credited with a goal (an own goal counts for the opponent)."""
    if event["type"] == "own_goal":
        return "away" if event["side"] == "home" else "home"
    return event["side"]


def clock(event):
    return f"{event['minute']}:{event['second']:02d}"


def window_key(event):
    period = event["period"]
    start = PERIOD_START[period]
    # Stoppage time (e.g. 45+2) joins the period's last window.
    offset = min(max(0, event["minute"] - start), PERIOD_LENGTH[period] - 1)
    return (period, start + WINDOW * (offset // WINDOW))


def team_totals(events, side):
    mine = [e for e in events if e["side"] == side]
    passes = [e for e in mine if e["type"] == "pass"]
    shots = [e for e in mine if e["type"] == "shot"]
    return {
        "passes": len(passes),
        "pass_accuracy": round(100 * sum(e["outcome"] == "complete" for e in passes) / max(1, len(passes))),
        "shots": len(shots),
        "shots_on_target": sum(e["outcome"] in ("goal", "saved") for e in shots),
        "xg": round(sum(e.get("xg", 0) for e in shots), 2),
        "ball_wins": sum(e["type"] in BALL_WINS for e in mine),
        "fouls": sum(e["type"] == "foul" for e in mine),
        "saves": sum(e["type"] == "save" for e in mine),
    }


def momentum_windows(events):
    """Split the match into 5-minute windows and score which team was on top.

    Threat for a team in a window = passes into the final third
    + 4 per shot + 10 per expected goal + 2 per ball won in the opponent's half.
    Momentum runs from -100 (away completely on top) to +100 (home on top).
    A small baseline is added to both sides so quiet windows don't jump to the extremes.
    """
    baseline = 4
    buckets = defaultdict(lambda: {"home": 0.0, "away": 0.0, "evidence": [], "passes": {"home": 0, "away": 0}})
    for e in events:
        b = buckets[window_key(e)]
        side = e["side"]
        if e["type"] == "pass":
            b["passes"][side] += 1
            if e["outcome"] == "complete" and e["end"] and e["end"][0] > 66:
                b[side] += 1
        elif e["type"] == "shot":
            b[side] += 4 + 10 * e.get("xg", 0)
            b["evidence"].append(e["id"])
        elif e["type"] in BALL_WINS and e["start"][0] > 50:
            b[side] += 2
            b["evidence"].append(e["id"])
        if is_goal(e) and e["id"] not in b["evidence"]:
            b["evidence"].append(e["id"])

    windows = []
    for (period, start), b in sorted(buckets.items()):
        total = b["home"] + b["away"]
        last_window_start = PERIOD_START[period] + PERIOD_LENGTH[period] - WINDOW
        end = start + WINDOW if start < last_window_start else PERIOD_START[period] + PERIOD_LENGTH[period]
        total_passes = b["passes"]["home"] + b["passes"]["away"]
        windows.append({
            "period": period,
            "from_minute": start,
            "to_minute": end,
            "momentum": round(100 * (b["home"] - b["away"]) / (total + 2 * baseline)),
            "home_threat": round(b["home"], 1),
            "away_threat": round(b["away"], 1),
            "home_pass_share": round(100 * b["passes"]["home"] / total_passes) if total_passes else 50,
            "evidence_ids": b["evidence"][:12],
        })
    return windows


def dominant_spells(windows, threshold=35, min_windows=2):
    """Runs of consecutive windows where one team clearly had the upper hand."""
    spells, current = [], None
    for w in windows:
        side = "home" if w["momentum"] >= threshold else "away" if w["momentum"] <= -threshold else None
        if current and side == current["side"]:
            current["windows"].append(w)
        else:
            if current and len(current["windows"]) >= min_windows:
                spells.append(current)
            current = {"side": side, "windows": [w]} if side else None
    if current and len(current["windows"]) >= min_windows:
        spells.append(current)

    return [{
        "side": s["side"],
        "from_minute": s["windows"][0]["from_minute"],
        "to_minute": s["windows"][-1]["to_minute"],
        "average_momentum": round(sum(w["momentum"] for w in s["windows"]) / len(s["windows"])),
        "evidence_ids": [i for w in s["windows"] for i in w["evidence_ids"]][:15],
    } for s in spells]


def momentum_swings(windows, min_change=60):
    """Points where momentum flipped sharply between consecutive windows."""
    swings = []
    for before, after in zip(windows, windows[1:]):
        change = after["momentum"] - before["momentum"]
        if abs(change) >= min_change and (before["momentum"] > 0) != (after["momentum"] > 0):
            swings.append({
                "at_minute": after["from_minute"],
                "towards": "home" if change > 0 else "away",
                "momentum_before": before["momentum"],
                "momentum_after": after["momentum"],
                "evidence_ids": after["evidence_ids"],
            })
    return swings


def key_moments(events):
    """Goals, red cards and scripted big moments, each with its build-up."""
    by_possession = defaultdict(list)
    for e in events:
        by_possession[e["possession_id"]].append(e)

    moments = []
    for e in events:
        if not (is_goal(e) or e.get("key_moment") or (e["type"] == "card" and e["outcome"] == "red")):
            continue
        if e["type"] == "save":
            continue  # saves are listed with their shot
        chain = [x for x in by_possession[e["possession_id"]] if x["id"] < e["id"]]
        passes = [x for x in chain if x["type"] in ("pass", "cross", "corner")]
        moment = {
            "id": e["id"],
            "clock": clock(e),
            "period": e["period"],
            "team": e["team"],
            "player": e["player"],
            "type": e["type"],
            "outcome": e["outcome"],
            "build_up_passes": len(passes),
            "build_up_ids": [x["id"] for x in chain][-12:],
        }
        for field in ("assist", "set_piece", "body_part", "xg", "score", "credited_to"):
            if field in e:
                moment[field] = e[field]
        moments.append(moment)
    return moments


def goalkeepers(events, match):
    keepers = []
    for side in ("home", "away"):
        keeper = next(p["name"] for p in match[side]["players"] if p["position"] == "GK")
        saves = [e for e in events if e["type"] == "save" and e["side"] == side]
        keepers.append({"side": side, "player": keeper, "saves": len(saves),
                        "save_ids": [e["id"] for e in saves]})
    return keepers


def standout_goalkeeper(keepers, min_difference=3):
    """The keeper who made clearly more saves than the other one, if any."""
    busiest, other = sorted(keepers, key=lambda k: -k["saves"])
    if busiest["saves"] - other["saves"] >= min_difference:
        return {**busiest, "saves_more_than_opponent_keeper": busiest["saves"] - other["saves"]}
    return None


def top_players(events):
    completed = defaultdict(int)
    ball_wins = defaultdict(int)
    teams = {}
    for e in events:
        teams[e["player"]] = e["team"]
        if e["type"] == "pass" and e["outcome"] == "complete":
            completed[e["player"]] += 1
        if e["type"] in BALL_WINS:
            ball_wins[e["player"]] += 1
    top = lambda counts: [{"player": p, "team": teams[p], "count": c}
                          for p, c in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
    return {"most_completed_passes": top(completed), "most_ball_wins": top(ball_wins)}


def compute_match_facts(match_data):
    """Everything the Analyst agent needs, as one dictionary."""
    match, events = match_data["match"], match_data["events"]
    windows = momentum_windows(events)
    keepers = goalkeepers(events, match)
    return {
        "teams": {"home": match["home"]["name"], "away": match["away"]["name"]},
        "final_score": match["final_score"],
        "after_extra_time": match.get("after_extra_time", False),
        "totals": {"home": team_totals(events, "home"), "away": team_totals(events, "away")},
        "key_moments": key_moments(events),
        "goalkeepers": keepers,
        "standout_goalkeeper": standout_goalkeeper(keepers),
        "dominant_spells": dominant_spells(windows),
        "momentum_swings": momentum_swings(windows),
        "momentum_windows": windows,
        "top_players": top_players(events),
        "momentum_scale": "+100 = home team completely on top, -100 = away team completely on top",
    }
