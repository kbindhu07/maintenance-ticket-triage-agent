"""Optional GenAI layer: turns the already-computed triage result into a short
operator-facing summary.

Design rules
- The LLM never decides category, severity or routing. Python computes those;
  the LLM only rewrites verified facts into readable text.
- No API key -> deterministic template summary (the app works fully offline).
- Any API error -> template summary; the error is reported in the response, never raised.
- The key is read from the ANTHROPIC_API_KEY environment variable only.
"""
import json
import os

LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = (
    "You write short triage summaries for a renewable-energy O&M control room. "
    "You are given a JSON triage result that has already been computed and verified. "
    "Write 2-3 plain sentences: what the issue is, how urgent it is and why, and the first action. "
    "Use only facts present in the JSON. Do not invent readings, causes, names or times. "
    "Call root causes 'probable', never confirmed. If human review is flagged, say so."
)


def template_summary(result: dict) -> str:
    cat = result["category"]["label"]
    sev = result["severity"]["severity"].upper()
    route = result["routing"]
    cause = result["root_cause"]["probable_causes"][0]["cause"] if result["root_cause"]["probable_causes"] else "unknown"
    assets = ", ".join(a["id"] for a in result["entities"]["asset_ids"]) or "the reported asset"
    text = (f"{sev} {cat} issue on {assets}; routed to {route['team']} as {route['priority']} "
            f"(respond within {route['sla_hours']} h). Probable cause: {cause.lower()}. "
            f"First action: {route['next_actions'][0]}.")
    if result.get("needs_human_review"):
        text += " Flagged for human review: " + "; ".join(result.get("review_reasons", [])) + "."
    return text


def _facts_for_llm(result: dict) -> dict:
    """Send only verified, structured facts - not the raw model internals."""
    return {
        "ticket_text": result["input"]["text"],
        "asset_type": result["input"]["asset_type"],
        "asset_ids": [a["id"] for a in result["entities"]["asset_ids"]],
        "category": result["category"]["label"],
        "classifier_confidence": result["category"]["confidence"],
        "severity": result["severity"]["severity"],
        "severity_reasons": result["severity"]["reasons"],
        "probable_causes": [c["cause"] for c in result["root_cause"]["probable_causes"][:2]],
        "team": result["routing"]["team"],
        "priority": result["routing"]["priority"],
        "sla_hours": result["routing"]["sla_hours"],
        "next_actions": result["routing"]["next_actions"][:3],
        "needs_human_review": result.get("needs_human_review", False),
        "review_reasons": result.get("review_reasons", []),
    }


def summarize(result: dict, use_llm: bool = True) -> dict:
    fallback = template_summary(result)
    if not use_llm or not os.getenv("ANTHROPIC_API_KEY"):
        return {"text": fallback, "source": "template", "note": "LLM disabled or ANTHROPIC_API_KEY not set"}
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        response = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(_facts_for_llm(result), indent=2)}],
        )
        if response.stop_reason == "refusal":
            return {"text": fallback, "source": "template", "note": "LLM declined; template used"}
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        # Output validation: must be non-empty, short, and must not contradict the computed severity.
        sev = result["severity"]["severity"]
        if not text or len(text) > 900:
            return {"text": fallback, "source": "template", "note": "LLM output failed validation"}
        if sev not in text.lower() and sev.upper() not in text:
            text = f"[{sev.upper()}] {text}"
        return {"text": text, "source": f"llm:{LLM_MODEL}", "note": "generated from verified triage facts"}
    except Exception as exc:  # network, auth, rate limit, SDK missing ...
        return {"text": fallback, "source": "template", "note": f"LLM unavailable ({type(exc).__name__}); template used"}
