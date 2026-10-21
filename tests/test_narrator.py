"""
Tests for the Narrator agent, using a fake model (free, no Azure key needed).
Run with:  pytest
"""

import json
from pathlib import Path

from agents.narrator import Narrator, check_scorelines, clean, clock, phase, what_happened

MATCH = json.loads((Path(__file__).resolve().parent.parent / "data" / "match_002.json").read_text())
EVENTS = MATCH["events"]


def run(narrator):
    return [line for e in EVENTS if (line := narrator.observe(e))]


def test_every_goal_gets_a_line():
    lines = run(Narrator(MATCH["match"], use_ai=False))
    goal_lines = [l for l in lines if l["kind"] == "goal"]
    assert len(goal_lines) == 5
    assert goal_lines[-1]["score"] == {"home": 3, "away": 2}


def test_routine_passes_are_not_narrated():
    lines = run(Narrator(MATCH["match"], use_ai=False))
    narrated = {l["event_id"] for l in lines}
    passes = {e["id"] for e in EVENTS if e["type"] == "pass"}
    # Only "state of play" lines may land on a pass; nothing else should.
    assert all(l["kind"] == "state_of_play" for l in lines if l["event_id"] in passes)
    assert len(lines) < len(EVENTS) / 20
    assert narrated


def test_saved_shot_is_narrated_once_at_the_save():
    lines = run(Narrator(MATCH["match"], use_ai=False))
    saved_shot_ids = {e["id"] for e in EVENTS if e["type"] == "shot" and e["outcome"] == "saved"}
    assert not any(l["event_id"] in saved_shot_ids for l in lines)


def test_model_cannot_cite_events_it_was_not_shown():
    fake = lambda system, user: {"text": "What a moment!", "evidence_ids": [1, 999999]}
    lines = run(Narrator(MATCH["match"], ask=fake))
    goal = next(l for l in lines if l["kind"] == "goal")
    assert goal["text"] == "What a moment!"
    assert 999999 not in goal["evidence_ids"] and 1 not in goal["evidence_ids"]
    assert goal["evidence_ids"] == [goal["event_id"]]


def test_falls_back_to_templates_if_the_model_fails():
    def broken(system, user):
        raise ConnectionError("Azure unreachable")

    lines = run(Narrator(MATCH["match"], ask=broken))
    assert lines and all(l["source"] == "template" for l in lines)
    assert any("GOAL!" in l["text"] for l in lines)


def test_stoppage_time_clock():
    assert clock({"period": 1, "minute": 45, "second": 34}) == "45+0:34"
    assert clock({"period": 2, "minute": 61, "second": 15}) == "61:15"


def test_save_is_described_from_the_attacking_side():
    save = next(e for e in EVENTS if e["type"] == "save" and e["player"] == "V. Amado")
    text = what_happened(save, EVENTS[: save["id"]], MATCH["match"])
    assert "Aurora United were attacking" in text
    assert "Saltmarsh Rovers goalkeeper" in text


def test_extra_time_is_not_called_stoppage_time():
    belmonte_goal = next(e for e in EVENTS if e["player"] == "R. Belmonte" and e.get("score"))
    assert phase(belmonte_goal) == "first period of extra time"
    assert phase({"period": 2, "minute": 92, "type": "pass"}) == "stoppage time at the end of the second half"


def test_wrong_score_falls_back_to_template():
    fake = lambda system, user: {"text": "Aurora United now lead 4-2!", "evidence_ids": []}
    lines = run(Narrator(MATCH["match"], ask=fake))
    own_goal = next(l for l in lines if l["kind"] == "goal" and l["score"] == {"home": 3, "away": 2})
    assert own_goal["source"] == "template"
    assert "4-2" not in own_goal["text"]
    assert "3-2" in own_goal["text"]


def test_scoreline_check():
    check_scorelines("Rovers trail 2-3 now", {"home": 3, "away": 2})   # either order is fine
    check_scorelines("an 11-pass move", {"home": 0, "away": 0})        # not a score
    try:
        check_scorelines("now 4-2", {"home": 3, "away": 2})
        assert False, "should have raised"
    except ValueError:
        pass


def test_event_ids_are_removed_from_text():
    assert clean("Belmonte heads it in! [1267, 1268]") == "Belmonte heads it in!"