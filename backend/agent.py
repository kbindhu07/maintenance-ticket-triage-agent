"""Triage agent: a deterministic, tool-calling workflow.

Honest framing: this is a rule/tool-based agent (as the brief asks), not an
autonomous LLM agent. A controller calls tools in sequence and makes explicit
decisions based on tool outputs (confidence gate, asset consistency check,
escalation). Every tool call and decision is recorded in `trace` so an operator
can see exactly why a ticket was routed where it was.

Tools
  1. extract_entities        - asset IDs, inferred asset type, durations, measurements
  2. classify_ticket         - ML category + confidence (TF-IDF + Logistic Regression)
  3. keyword_rules_check     - called only when ML confidence is low (second opinion)
  4. check_asset_consistency - does the category make sense for this asset type?
  5. score_severity          - auditable rule-based severity
  6. lookup_root_cause       - rank probable causes from the knowledge base using ticket evidence
  7. find_similar_tickets    - nearest historical tickets (TF-IDF cosine similarity)
  8. route_ticket            - team, priority, SLA, next actions
  9. create_escalation       - only for critical tickets (simulated on-call page)
 10. summarize               - optional LLM summary of the verified result (template fallback)
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import classifier, llm
from .knowledge_base import CATEGORIES, PRIORITY, SEVERITY_LEVELS, SLA_HOURS
from .preprocessing import clean_text, extract_entities, find_terms
from .severity import score_severity

CONFIDENCE_THRESHOLD = 0.45   # below this the agent asks the rules tool for a second opinion
REVIEW_THRESHOLD = 0.30       # below this a human must confirm the category
ML_SEVERITY_MIN_CONF = 0.50   # ML severity may raise the rule level only when reasonably confident


class TriageAgent:
    def __init__(self):
        models = classifier.load_or_train()
        self.model = models["category"]
        self.severity_model = models["severity"]
        self.history = classifier.load_training_data()
        self._sim_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._sim_matrix = self._sim_vec.fit_transform(self.history["description"].map(clean_text))

    # ---------------------------------------------------------------- tools
    def tool_extract_entities(self, text):
        return extract_entities(text)

    def tool_classify_ticket(self, text, asset_type):
        return classifier.predict(self.model, text, asset_type)

    def tool_keyword_rules_check(self, text, asset_type):
        return classifier.rules_classify(text, asset_type)

    def tool_check_asset_consistency(self, category, asset_type):
        allowed = CATEGORIES[category]["asset_types"]
        ok = (not asset_type) or asset_type in allowed
        return {"consistent": ok, "allowed_asset_types": allowed, "asset_type": asset_type}

    def tool_score_severity(self, text, category, asset_type=""):
        """Rules first; the ML severity model can only RAISE the level (conservative ensemble)."""
        result = score_severity(text, category)
        proba = self.severity_model.predict_proba([classifier.model_input(text, asset_type)])[0]
        classes = list(self.severity_model.classes_)
        ml_sev = classes[int(np.argmax(proba))]
        ml_conf = float(np.max(proba))
        result["ml_severity"] = {"severity": ml_sev, "confidence": round(ml_conf, 3)}
        ml_level = SEVERITY_LEVELS.index(ml_sev)
        if ml_level > result["level"] and ml_conf >= ML_SEVERITY_MIN_CONF:
            result["reasons"].append(
                f"ML severity model (trained on past tickets) says {ml_sev} with {ml_conf:.2f} confidence "
                f"-> raised from {result['severity']} (never under-triage)")
            result["level"], result["severity"] = ml_level, ml_sev
            result["raised_by_ml"] = True
        else:
            result["raised_by_ml"] = False
        return result

    def tool_lookup_root_cause(self, text, category):
        t = clean_text(text)
        ranked = []
        for prior_rank, rc in enumerate(CATEGORIES[category]["root_causes"]):
            hits = find_terms(rc["evidence"], t)
            ranked.append({"cause": rc["cause"], "matched_evidence": hits,
                           "_score": len(hits) - 0.01 * prior_rank})
        ranked.sort(key=lambda r: r["_score"], reverse=True)
        for r in ranked:
            r["support"] = "matched ticket evidence" if r["matched_evidence"] else "knowledge-base prior only"
            del r["_score"]
        return {"probable_causes": ranked[:3],
                "disclaimer": "Probable causes ranked by keyword evidence; confirm on site before closing."}

    def tool_find_similar_tickets(self, text, k=3):
        q = self._sim_vec.transform([clean_text(text)])
        sims = cosine_similarity(q, self._sim_matrix)[0]
        # Skip exact self-matches (similarity ~1.0) so the ticket is not shown as its own precedent.
        idx = [i for i in np.argsort(sims)[::-1] if 0 < sims[i] < 0.999][:k]
        return [{"ticket_id": str(self.history.iloc[i]["ticket_id"]),
                 "description": self.history.iloc[i]["description"],
                 "category": self.history.iloc[i]["target_category"],
                 "severity": self.history.iloc[i]["target_severity"],
                 "similarity": round(float(sims[i]), 3)} for i in idx]

    def tool_route_ticket(self, category, severity, asset_type):
        info = CATEGORIES[category]
        team = info["team"]
        actions = list(info["next_actions"])
        if severity == "critical" and category != "safety_hazard":
            actions.insert(0, "Notify shift in-charge now; consider stopping/isolating the asset until assessed")
        sla = SLA_HOURS[severity]
        due = datetime.now(timezone.utc) + timedelta(hours=sla)
        return {"team": team, "priority": PRIORITY[severity], "sla_hours": sla,
                "respond_by_utc": due.strftime("%Y-%m-%d %H:%M"), "next_actions": actions}

    def tool_create_escalation(self, category, severity, team):
        # Simulated: in production this would call the CMMS / paging API.
        return {"escalated": True, "channel": "on-call page (simulated)",
                "notify": [team, "Shift In-charge", "Site Manager"] + (["HSE Head"] if category == "safety_hazard" else [])}

    # ---------------------------------------------------------------- controller
    def triage(self, text: str, asset_type: str = "", ticket_id: str | None = None, use_llm: bool = True) -> dict:
        t0 = time.perf_counter()
        trace = []
        review = []
        asset_type = (asset_type or "").strip().lower()

        def log(step, tool, output, decision=""):
            trace.append({"step": len(trace) + 1, "tool": step if tool is None else tool,
                          "output": output, "decision": decision})

        entities = self.tool_extract_entities(text)
        if not asset_type and entities["inferred_asset_type"]:
            asset_type = entities["inferred_asset_type"]
            decision = f"asset_type not given -> inferred '{asset_type}' from ticket words"
        else:
            decision = "asset_type provided by user" if asset_type else "asset_type unknown"
        log(None, "extract_entities", {k: v for k, v in entities.items() if k != "clean_text"}, decision)

        pred = self.tool_classify_ticket(text, asset_type)
        category, confidence = pred["category"], pred["confidence"]
        method = "ml"
        log(None, "classify_ticket", pred,
            f"ML predicts {category} with confidence {confidence:.2f}")

        if confidence < CONFIDENCE_THRESHOLD:
            rules = self.tool_keyword_rules_check(text, asset_type)
            top_cats = [c["category"] for c in pred["top_k"]]
            if rules["category"] and rules["category"] in top_cats and rules["category"] != category:
                decision = (f"Low ML confidence; rules match '{rules['category']}' which is in ML top-3 "
                            f"-> switch category to {rules['category']}")
                category, method = rules["category"], "ml+rules"
            elif rules["category"] == category:
                decision = "Low ML confidence but rules agree -> keep category"
                method = "ml+rules"
            else:
                decision = "Low ML confidence and rules do not resolve it -> keep ML category, flag for review"
                review.append(f"low classifier confidence ({confidence:.2f})")
            log(None, "keyword_rules_check", rules, decision)
        if confidence < REVIEW_THRESHOLD and not any("confidence" in r for r in review):
            review.append(f"very low classifier confidence ({confidence:.2f})")

        consistency = self.tool_check_asset_consistency(category, asset_type)
        if not consistency["consistent"]:
            review.append(f"category '{category}' is unusual for asset type '{asset_type}'")
        log(None, "check_asset_consistency", consistency,
            "consistent" if consistency["consistent"] else "inconsistent -> flag for human review")

        sev = self.tool_score_severity(text, category, asset_type)
        log(None, "score_severity", sev, f"severity = {sev['severity']}")

        rc = self.tool_lookup_root_cause(text, category)
        log(None, "lookup_root_cause", rc, f"top probable cause: {rc['probable_causes'][0]['cause']}")

        similar = self.tool_find_similar_tickets(text)
        log(None, "find_similar_tickets", similar,
            f"{len(similar)} similar historical tickets retrieved as supporting evidence")

        route = self.tool_route_ticket(category, sev["severity"], asset_type)
        log(None, "route_ticket", route, f"route to {route['team']} as {route['priority']}")

        escalation = None
        if sev["severity"] == "critical":
            escalation = self.tool_create_escalation(category, sev["severity"], route["team"])
            log(None, "create_escalation", escalation, "critical -> immediate escalation")
        else:
            log(None, "create_escalation", {"escalated": False}, "not critical -> no page sent")

        result = {
            "ticket_id": ticket_id,
            "input": {"text": text, "asset_type": asset_type or "unknown"},
            "entities": {k: v for k, v in entities.items() if k != "clean_text"},
            "category": {"id": category, "label": CATEGORIES[category]["label"], "confidence": confidence,
                         "method": method, "top_k": pred["top_k"], "evidence_terms": pred["evidence_terms"]},
            "severity": sev,
            "root_cause": rc,
            "similar_tickets": similar,
            "routing": route,
            "escalation": escalation,
            "needs_human_review": bool(review),
            "review_reasons": review,
        }
        summary = llm.summarize(result, use_llm=use_llm)
        log(None, "summarize", {"source": summary["source"], "note": summary["note"]},
            "LLM rewrites verified facts only" if summary["source"].startswith("llm") else "template summary used")
        result["summary"] = summary
        result["trace"] = trace
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
