"""Generate a reproducible synthetic training set of maintenance tickets.

Why: the bundled dataset has only 6 tickets (one per category), which is not
enough to train or evaluate a classifier. This script creates ~400 labelled
tickets from domain templates with randomised assets, phrasing, abbreviations,
typos and cross-category "confuser" wording.

Labels are assigned by the template author (domain judgement), not by the
severity rules in backend/severity.py. Limitation: synthetic text is cleaner
and more regular than real field tickets, so metrics on the synthetic test split
are optimistic. The hand-written holdout (data/holdout_handwritten.csv) is the
more honest check.

Run:  python scripts/generate_dataset.py
"""
import csv
import random
from pathlib import Path

SEED = 42
PER_TEMPLATE = 5
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_tickets.csv"

rng = random.Random(SEED)

SLOTS = {
    "wtg": lambda: f"WTG-{rng.randint(1, 60):02d}",
    "inv": lambda: f"Inverter {rng.randint(1, 40)}",
    "blk": lambda: f"block {rng.randint(1, 12)}",
    "unit": lambda: f"Unit {rng.randint(1, 4)}",
    "trf": lambda: f"transformer T{rng.randint(1, 6)}",
    "gate": lambda: f"gate {rng.randint(1, 8)}",
    "hrs": lambda: rng.choice(["two hours", "3 hours", "the last 45 minutes", "since morning", "since 6 am", "overnight"]),
    "temp": lambda: f"{rng.randint(68, 92)} degC",
    "when": lambda: rng.choice(["after noon", "during peak irradiance", "around 1 pm", "in the afternoon", "today"]),
    "site": lambda: rng.choice(["Site A", "Kurnool", "Anantapur", "Jaisalmer", "Pinnapuram", "site"]),
}

# (template, asset_type, severity, category)
T = [
    # ---------------- inverter_thermal
    ("{inv} tripping repeatedly {when}, high temperature alarm visible", "solar", "high", "inverter_thermal"),
    ("{inv} in {blk} derating due to over temperature, IGBT temp {temp}", "solar", "high", "inverter_thermal"),
    ("Cooling fan of {inv} not running, cabinet temperature rising", "solar", "high", "inverter_thermal"),
    ("{inv} shows thermal warning {when} but still generating, minor", "solar", "medium", "inverter_thermal"),
    ("PCU heat sink very hot, {inv} tripped twice {when}", "solar", "high", "inverter_thermal"),
    ("inv overtemp fault on {inv}, air filter looks clogged with dust", "solar", "high", "inverter_thermal"),
    ("Smoke and burning smell from {inv} cabinet, isolated by technician", "solar", "critical", "safety_hazard"),
    ("All inverters in {blk} tripping on over temperature {when}", "solar", "critical", "inverter_thermal"),
    # ---------------- gearbox_vibration
    ("Gearbox vibration has increased for turbine {wtg} with abnormal noise", "wind", "critical", "gearbox_vibration"),
    ("CMS alert: high speed shaft bearing vibration above warning level on {wtg}", "wind", "high", "gearbox_vibration"),
    ("{wtg} gearbox oil temperature high and vibration trending up", "wind", "critical", "gearbox_vibration"),
    ("Metal particles found in gearbox oil filter of {wtg}", "wind", "critical", "gearbox_vibration"),
    ("Grinding sound from nacelle of {wtg}, drivetrain vib alarm", "wind", "critical", "gearbox_vibration"),
    ("Slight gearbox vibration on {wtg} within limits, monitor at next visit", "wind", "medium", "gearbox_vibration"),
    ("Main bearing vibration warning on {wtg} after restart", "wind", "high", "gearbox_vibration"),
    ("gbx vib high {wtg}, turbine stopped by controller", "wind", "high", "gearbox_vibration"),
    # ---------------- inspection_compliance
    ("Inspection checklist pending for spillway {gate} maintenance", "hydro", "medium", "inspection_compliance"),
    ("Quarterly PM checklist for {wtg} not closed in CMMS", "wind", "medium", "inspection_compliance"),
    ("Statutory certificate for crane at {site} expires next week, renewal pending", "hydro", "high", "inspection_compliance"),
    ("Routine visual inspection of {blk} module mounting due, no issues reported", "solar", "low", "inspection_compliance"),
    ("Fire extinguisher inspection overdue at {site} control room", "grid", "medium", "inspection_compliance"),
    ("Penstock inspection report for {unit} awaiting sign-off", "hydro", "medium", "inspection_compliance"),
    ("Audit observation: earthing pit test records missing for {blk}", "solar", "medium", "inspection_compliance"),
    ("Annual compliance audit scheduled, documents to be prepared when possible", "grid", "low", "inspection_compliance"),
    # ---------------- data_gap
    ("SCADA data missing for {blk} for last {hrs}", "solar", "medium", "data_gap"),
    ("No data from {wtg} on SCADA since {hrs}, turbine status unknown", "wind", "medium", "data_gap"),
    ("Data logger in {blk} not communicating, values frozen", "solar", "medium", "data_gap"),
    ("Comms lost to entire plant SCADA, no visibility of generation", "solar", "high", "data_gap"),
    ("Historian showing flatline for {inv} power tag {hrs}", "solar", "medium", "data_gap"),
    ("RTU at {unit} rebooting, intermittent data gap", "hydro", "medium", "data_gap"),
    ("Network switch down, modbus timeout for {wtg} and {wtg}", "wind", "medium", "data_gap"),
    ("Minor timestamp mismatch in daily report for {blk}, informational", "solar", "low", "data_gap"),
    # ---------------- pitch_system
    ("Blade pitch error after high wind gust event on {wtg}", "wind", "high", "pitch_system"),
    ("Pitch battery voltage low on blade 2 of {wtg}, cannot feather safely", "wind", "critical", "pitch_system"),
    ("Pitch motor overcurrent fault on {wtg}, turbine stopped", "wind", "high", "pitch_system"),
    ("Blade angle deviation between blade 1 and 3 on {wtg}, encoder suspected", "wind", "high", "pitch_system"),
    ("Hydraulic pitch accumulator pressure low on {wtg}", "wind", "critical", "pitch_system"),
    ("Pitch converter warning on {wtg} cleared after reset, monitor", "wind", "medium", "pitch_system"),
    ("{wtg} pitch system fault during storm, blades not feathering", "wind", "critical", "pitch_system"),
    ("Pitch encoder replacement scheduled for {wtg}, routine", "wind", "low", "pitch_system"),
    # ---------------- grid_curtailment
    ("Curtailment instruction not reflected in generation dashboard", "grid", "medium", "grid_curtailment"),
    ("SLDC backdown instruction received, PPC setpoint not applied at {site}", "grid", "high", "grid_curtailment"),
    ("Dispatch schedule revised, plant still exporting above limit", "grid", "high", "grid_curtailment"),
    ("Curtailment hours for yesterday not logged for compensation claim", "grid", "medium", "grid_curtailment"),
    ("Power plant controller setpoint mismatch with grid instruction at {site}", "solar", "medium", "grid_curtailment"),
    ("Wind farm curtailed by grid for {hrs}, need confirmation of instruction", "wind", "medium", "grid_curtailment"),
    ("Update curtailment report format, low priority", "grid", "low", "grid_curtailment"),
    # ---------------- transformer_fault
    ("Buchholz alarm on {trf} at pooling substation", "grid", "critical", "transformer_fault"),
    ("{trf} oil temperature high, cooling fans not starting", "grid", "high", "transformer_fault"),
    ("Breaker tripped on 33 kV feeder, protection relay shows earth fault", "grid", "high", "transformer_fault"),
    ("Oil leak observed near bushing of {trf}", "grid", "high", "transformer_fault"),
    ("DGA results for {trf} show rising acetylene, gas trend increasing", "grid", "critical", "transformer_fault"),
    ("Tap changer counter reading to be recorded, routine", "grid", "low", "transformer_fault"),
    ("Inverter duty transformer in {blk} winding temperature alarm", "solar", "high", "transformer_fault"),
    ("Switchgear panel humming noise at substation, relay healthy", "grid", "medium", "transformer_fault"),
    # ---------------- safety_hazard
    ("Fire reported near {inv} in {blk}, area evacuated", "solar", "critical", "safety_hazard"),
    ("Technician injured while climbing {wtg} ladder", "wind", "critical", "safety_hazard"),
    ("Sparking observed at cable trench near {blk} combiner box", "solar", "critical", "safety_hazard"),
    ("Unsafe condition: missing guard rail on {gate} walkway", "hydro", "critical", "safety_hazard"),
    ("Smoke seen from {trf} cooling radiator area", "grid", "critical", "safety_hazard"),
    ("Flood water entering powerhouse of {unit}", "hydro", "critical", "safety_hazard"),
    ("Electric shock reported by contractor at {site} switchyard", "grid", "critical", "safety_hazard"),
    ("Arc flash incident in LT panel at {site}, panel isolated", "solar", "critical", "safety_hazard"),
    # ---------------- module_soiling
    ("Heavy dust on modules in {blk}, PR dropped by 4%", "solar", "low", "module_soiling"),
    ("{blk} underperforming vs neighbours, soiling suspected", "solar", "low", "module_soiling"),
    ("Bird droppings on rows 10-14 of {blk}, cleaning needed", "solar", "low", "module_soiling"),
    ("Vegetation growth causing shading on {blk} strings", "solar", "low", "module_soiling"),
    ("Low generation from {blk} after dust storm, entire block affected", "solar", "medium", "module_soiling"),
    ("Cleaning cycle missed for {blk}, performance ratio dropping", "solar", "low", "module_soiling"),
    ("Hotspots seen in thermography on {blk} strings, possible soiling", "solar", "medium", "module_soiling"),
]

PREFIXES = ["", "", "", "Site reports: ", "Operator note - ", "Shift log: ", "URGENT: ", "FYI ", "Control room: "]
SUFFIXES = ["", "", "", " pls check", ". Please advise.", " - raised by night shift", ". Photos attached.", " (repeat ticket)"]
ABBREV = [("Inverter", "INV"), ("temperature", "temp"), ("vibration", "vib"), ("communication", "comms"),
          ("transformer", "trf"), ("turbine", "WTG"), ("hours", "hrs")]


def typo(text: str) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randint(1, len(w) - 3)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def render(template: str) -> str:
    out = template
    for key, fn in SLOTS.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", fn(), 1)
    out = rng.choice(PREFIXES) + out + rng.choice(SUFFIXES)
    if rng.random() < 0.3:
        a, b = rng.choice(ABBREV)
        out = out.replace(a, b)
    if rng.random() < 0.15:
        out = typo(out)
    if rng.random() < 0.2:
        out = out.lower()
    return out


def main():
    rows, seen = [], set()
    n = 0
    for template, asset, severity, category in T:
        made = 0
        attempts = 0
        while made < PER_TEMPLATE and attempts < 50:
            attempts += 1
            text = render(template)
            if text in seen:
                continue
            seen.add(text)
            n += 1
            made += 1
            rows.append({"ticket_id": f"S{n:04d}", "asset_type": asset, "description": text,
                         "target_severity": severity, "target_category": category,
                         "template_id": T.index((template, asset, severity, category))})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} tickets from {len(T)} templates -> {OUT}")


if __name__ == "__main__":
    main()
