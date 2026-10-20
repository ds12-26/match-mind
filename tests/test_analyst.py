"""
Tests for the match statistics and the Analyst agent.

These use a fake model, so they run instantly, cost nothing and don't need
an Azure key. Run with:  pytest
"""

import json
from pathlib import Path

from agents.analyst import analyse
from agents.match_stats import compute_match_facts

MATCH = json.loads((Path(__file__).resolve().parent.parent / "data" / "match_002.json").read_text())


def test_final_score_matches_goal_events():
    facts = compute_match_facts(MATCH)
    goals = [m for m in facts["key_moments"] if m.get("score")]
    assert goals[-1]["score"] == facts["final_score"] == {"home": 3, "away": 2}


def test_own_goal_is_credited_to_the_other_team():
    facts = compute_match_facts(MATCH)
    own_goal = next(m for m in facts["key_moments"] if m["type"] == "own_goal")
    assert own_goal["team"] == "Saltmarsh Rovers"
    assert own_goal["credited_to"] == "Aurora United"


def test_momentum_stays_in_range():
    facts = compute_match_facts(MATCH)
    assert all(-100 <= w["momentum"] <= 100 for w in facts["momentum_windows"])


def test_underdog_keeper_made_the_most_saves():
    keepers = {k["side"]: k for k in compute_match_facts(MATCH)["goalkeepers"]}
    assert keepers["away"]["saves"] > keepers["home"]["saves"]


def test_standout_goalkeeper_is_the_underdog_keeper():
    standout = compute_match_facts(MATCH)["standout_goalkeeper"]
    assert standout["player"] == "V. Amado"
    assert standout["saves"] == len(standout["save_ids"])


def test_insights_are_sorted_by_minute():
    def fake_model(system_prompt, user_content):
        return {"insights": [
            {"title": "Later", "minute": 61, "explanation": "x", "evidence_ids": [809]},
            {"title": "Earlier", "minute": 53, "explanation": "x", "evidence_ids": [706]},
        ]}

    analysis = analyse(MATCH, ask=fake_model)
    assert [i["title"] for i in analysis["insights"]] == ["Earlier", "Later"]


def test_invented_event_ids_are_removed():
    def fake_model(system_prompt, user_content):
        return {
            "headline": "Test match",
            "insights": [
                {"title": "Real evidence", "explanation": "x", "evidence_ids": [809, 999999]},
                {"title": "Made-up evidence only", "explanation": "x", "evidence_ids": [999998]},
            ],
        }

    analysis = analyse(MATCH, ask=fake_model)
    assert [i["title"] for i in analysis["insights"]] == ["Real evidence"]
    assert analysis["insights"][0]["evidence_ids"] == [809]
    assert analysis["verification"]["invalid_ids_removed"] == 2
