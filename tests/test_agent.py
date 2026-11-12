"""Run: python -m pytest -q"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.agent import TriageAgent
from backend.app import app
from backend.preprocessing import contains_term, extract_entities
from backend.severity import score_severity


@pytest.fixture(scope="module")
def agent():
    return TriageAgent()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_bundled_tickets_category_and_severity(agent):
    df = pd.read_csv("data/maintenance_tickets.csv")
    for r in df.itertuples():
        res = agent.triage(r.description, r.asset_type, r.ticket_id, use_llm=False)
        assert res["category"]["id"] == r.target_category, r.ticket_id
        assert res["severity"]["severity"] == r.target_severity, r.ticket_id


def test_safety_ticket_is_critical_and_escalated(agent):
    res = agent.triage("Smoke and sparking seen in combiner box near block 4", "solar", use_llm=False)
    assert res["severity"]["severity"] == "critical"
    assert res["routing"]["priority"] == "P1"
    assert res["escalation"] and res["escalation"]["escalated"]


def test_routine_wording_downgrades():
    assert score_severity("Routine pitch encoder replacement scheduled", "pitch_system")["severity"] == "medium"


def test_word_boundary_matching():
    assert contains_term("trip", "inverter tripping")
    assert not contains_term("fall", "generation shortfall")


def test_entity_extraction():
    e = extract_entities("Gearbox vibration on WTG-12, Inverter 4 also tripping")
    ids = [a["id"] for a in e["asset_ids"]]
    assert "WTG-12" in ids and "Inverter 4" in ids


def test_trace_contains_all_tools(agent):
    res = agent.triage("SCADA data missing for block 3 for last two hours", "solar", use_llm=False)
    tools = [t["tool"] for t in res["trace"]]
    for t in ["extract_entities", "classify_ticket", "score_severity", "lookup_root_cause",
              "find_similar_tickets", "route_ticket", "summarize"]:
        assert t in tools


def test_ambiguous_ticket_is_flagged(agent):
    res = agent.triage("something wrong please check", "", use_llm=False)
    assert res["needs_human_review"]


def test_llm_failure_falls_back_to_template(agent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-key-for-test")
    res = agent.triage("Blade pitch error after high wind gust event", "wind", use_llm=True)
    assert res["summary"]["source"] == "template"
    assert res["summary"]["text"]


def test_api_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.post("/api/triage", json={"text": "Buchholz alarm on transformer T2", "asset_type": "grid"})
    assert r.status_code == 200 and r.json()["severity"]["severity"] == "critical"
    assert client.post("/api/triage", json={"text": "abc"}).status_code == 422
    assert client.post("/api/triage", json={"text": "valid text here", "asset_type": "nuclear"}).status_code == 422
    assert client.get("/").status_code == 200
    q = client.get("/api/queue").json()
    assert q[0]["severity"] == "critical"
