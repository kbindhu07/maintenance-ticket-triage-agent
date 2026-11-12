"""Rule-based severity scoring.

Severity is deliberately NOT a black-box model: it drives who gets woken up at
night, so every level must be traceable to a rule an O&M engineer can audit.

Score = category base level
        -> raised to 'critical' by safety terms (fire, smoke, injury ...)
        -> raised one level by category-specific escalations (e.g. gearbox + abnormal noise)
        -> raised one level by wide impact (entire plant, multiple turbines)
        -> raised to at least 'high' by operational-failure terms (trip, shutdown, outage)
        -> lowered one level by routine/minor wording (never below a safety floor)
"""
from .knowledge_base import (CATEGORIES, CATEGORY_ESCALATIONS, CRITICAL_TERMS, DOWNGRADE_TERMS,
                             HIGH_TERMS, SEVERITY_LEVELS, WIDE_IMPACT_TERMS)
from .preprocessing import clean_text, find_terms as _find


def score_severity(text: str, category: str) -> dict:
    t = clean_text(text)
    info = CATEGORIES.get(category, {"base_severity": "medium"})
    level = SEVERITY_LEVELS.index(info["base_severity"])
    reasons = [f"Base severity for '{category}' is {info['base_severity']}"]

    crit = _find(CRITICAL_TERMS, t)
    if crit:
        level = 3
        reasons.append(f"Safety/critical terms found: {', '.join(crit)} -> critical")

    esc = _find(CATEGORY_ESCALATIONS.get(category, []), t)
    if esc and level < 3:
        level += 1
        reasons.append(f"Category escalation for {category}: {', '.join(esc)} -> +1 level")

    wide = _find(WIDE_IMPACT_TERMS, t)
    if wide and level < 3:
        level += 1
        reasons.append(f"Wide impact: {', '.join(wide)} -> +1 level")

    high = _find(HIGH_TERMS, t)
    if high and level < 2 and category not in ("inspection_compliance",):
        level = 2
        reasons.append(f"Operational failure terms: {', '.join(high)} -> at least high")

    down = _find(DOWNGRADE_TERMS, t)
    if down and not crit and level > 0:
        level -= 1
        reasons.append(f"Routine/minor wording: {', '.join(down)} -> -1 level")

    severity = SEVERITY_LEVELS[level]
    return {"severity": severity, "level": level, "reasons": reasons}
