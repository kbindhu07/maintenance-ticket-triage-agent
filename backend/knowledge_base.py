"""Domain knowledge used by the agent's tools.

This is a small, hand-curated O&M (operations & maintenance) knowledge base for
renewable-energy assets. In production it would come from a CMMS (e.g. SAP PM /
Maximo) failure-code catalogue and the site's escalation matrix. Keeping it as
plain Python data makes every routing decision auditable.
"""

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# Target response time (hours) per severity -> drives the SLA shown to operators.
SLA_HOURS = {"critical": 1, "high": 4, "medium": 24, "low": 72}

# Priority label shown in the queue.
PRIORITY = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}

CATEGORIES = {
    "inverter_thermal": {
        "label": "Inverter thermal / tripping",
        "asset_types": ["solar"],
        "base_severity": "high",
        "team": "Solar Electrical O&M",
        "keywords": ["inverter", "pcu", "igbt", "overtemperature", "over temperature",
                     "high temperature", "thermal", "cooling fan", "derating", "derate"],
        "root_causes": [
            {"cause": "Cooling fan failure or blocked air filter", "evidence": ["fan", "filter", "airflow", "dust"]},
            {"cause": "High ambient temperature at peak irradiance causing thermal derating/trip", "evidence": ["noon", "afternoon", "peak", "ambient", "summer", "heat"]},
            {"cause": "IGBT / power module degradation", "evidence": ["igbt", "module", "repeated", "repeatedly", "frequent"]},
            {"cause": "DC overvoltage or insulation fault on string side", "evidence": ["dc", "insulation", "ground fault", "string", "earth"]},
        ],
        "next_actions": [
            "Pull inverter fault log and temperature trend from SCADA for the last 72 h",
            "Inspect cooling fans, air filters and heat-sink cleanliness",
            "Check IGBT/module temperatures with thermal camera during peak load",
            "If trips continue, derate inverter and raise OEM service request",
        ],
    },
    "gearbox_vibration": {
        "label": "Gearbox / drivetrain vibration",
        "asset_types": ["wind"],
        "base_severity": "high",
        "team": "Wind Mechanical O&M",
        "keywords": ["gearbox", "vibration", "bearing", "drivetrain", "main shaft", "cms",
                     "noise", "rumbling", "grinding", "high speed shaft"],
        "root_causes": [
            {"cause": "Gear tooth or bearing wear / pitting", "evidence": ["bearing", "wear", "pitting", "noise", "grinding", "rumbling"]},
            {"cause": "Lubrication problem (low oil level, degraded oil, filter clogged)", "evidence": ["oil", "lubrication", "temperature", "filter", "pressure"]},
            {"cause": "Shaft misalignment or loose mounting", "evidence": ["alignment", "misalignment", "loose", "mounting", "coupling"]},
            {"cause": "Metal debris indicating advanced component damage", "evidence": ["metal", "particles", "debris", "chips"]},
        ],
        "next_actions": [
            "Review condition-monitoring (CMS) vibration spectra and trend vs baseline",
            "Take oil sample for particle count and check oil level/temperature",
            "Restrict turbine power or stop turbine if vibration exceeds alarm limit",
            "Schedule borescope inspection of gear stages and bearings",
        ],
    },
    "inspection_compliance": {
        "label": "Inspection / compliance",
        "asset_types": ["hydro", "solar", "wind", "grid"],
        "base_severity": "medium",
        "team": "HSE & Compliance",
        "keywords": ["inspection", "checklist", "audit", "compliance", "permit", "certificate",
                     "statutory", "overdue", "pending", "preventive maintenance", "pm schedule"],
        "root_causes": [
            {"cause": "Planned inspection not scheduled or resource unavailable", "evidence": ["pending", "not scheduled", "resource", "crew", "manpower"]},
            {"cause": "Documentation / checklist not closed in CMMS", "evidence": ["checklist", "document", "record", "sign", "closed"]},
            {"cause": "Statutory certificate or permit expiring", "evidence": ["certificate", "permit", "statutory", "expiry", "expire", "renewal"]},
        ],
        "next_actions": [
            "Confirm due date and regulatory consequence of the overdue item",
            "Assign inspector and book the work window in the CMMS",
            "Close the checklist with evidence (photos, readings) and get sign-off",
        ],
    },
    "data_gap": {
        "label": "SCADA / data gap",
        "asset_types": ["solar", "wind", "hydro", "grid"],
        "base_severity": "medium",
        "team": "SCADA & IT/OT Support",
        "keywords": ["scada", "data missing", "data gap", "no data", "communication", "comms",
                     "logger", "rtu", "frozen", "stale", "flatline", "network", "modbus", "ping"],
        "root_causes": [
            {"cause": "Communication link down (fibre/radio/switch) between site and SCADA", "evidence": ["communication", "comms", "network", "switch", "fibre", "fiber", "ping", "offline"]},
            {"cause": "Data logger / RTU failure or reboot", "evidence": ["logger", "rtu", "reboot", "power", "ups"]},
            {"cause": "Historian / tag mapping or time-sync issue", "evidence": ["tag", "historian", "timestamp", "time", "mapping", "frozen", "stale"]},
        ],
        "next_actions": [
            "Check RTU/data-logger and network switch status from the control room",
            "Verify whether the plant is generating (local HMI / meter) to rule out a real outage",
            "Backfill missing data from logger buffer once the link is restored",
        ],
    },
    "pitch_system": {
        "label": "Blade pitch system",
        "asset_types": ["wind"],
        "base_severity": "high",
        "team": "Wind Electrical O&M",
        "keywords": ["pitch", "blade angle", "pitch motor", "pitch battery", "accumulator",
                     "hydraulic pitch", "pitch encoder", "feathering"],
        "root_causes": [
            {"cause": "Pitch motor / drive fault", "evidence": ["motor", "drive", "converter", "overcurrent"]},
            {"cause": "Pitch battery or hydraulic accumulator low (emergency feathering not guaranteed)", "evidence": ["battery", "accumulator", "hydraulic", "pressure"]},
            {"cause": "Encoder / blade angle sensor mismatch", "evidence": ["encoder", "sensor", "angle", "mismatch", "deviation"]},
            {"cause": "Mechanical load after extreme wind event", "evidence": ["gust", "storm", "high wind", "cyclone", "extreme"]},
        ],
        "next_actions": [
            "Read pitch controller fault codes for all three blades",
            "Check pitch battery/accumulator health before restarting turbine",
            "Compare blade angles between blades to detect encoder mismatch",
            "Keep turbine stopped in safe position until fault is cleared",
        ],
    },
    "grid_curtailment": {
        "label": "Grid curtailment / dispatch",
        "asset_types": ["grid", "solar", "wind"],
        "base_severity": "medium",
        "team": "Grid & Dispatch Desk",
        "keywords": ["curtailment", "curtail", "dispatch", "sldc", "schedule", "setpoint",
                     "backdown", "back down", "grid instruction", "ppc", "power plant controller"],
        "root_causes": [
            {"cause": "Curtailment setpoint not propagated to power plant controller (PPC)", "evidence": ["setpoint", "ppc", "controller", "not reflected", "not applied"]},
            {"cause": "Dashboard / reporting not updated with dispatch instruction", "evidence": ["dashboard", "report", "display", "reflected"]},
            {"cause": "Grid-side constraint or SLDC backdown instruction", "evidence": ["sldc", "grid", "backdown", "frequency", "congestion"]},
        ],
        "next_actions": [
            "Confirm the active dispatch instruction with the grid desk / SLDC",
            "Verify PPC setpoint and actual export at the meter",
            "Log curtailment period for deemed-generation / compensation claim",
        ],
    },
    "transformer_fault": {
        "label": "Transformer / substation electrical",
        "asset_types": ["grid", "solar", "wind", "hydro"],
        "base_severity": "high",
        "team": "Substation & HV Maintenance",
        "keywords": ["transformer", "buchholz", "oil temperature", "winding", "breaker",
                     "switchgear", "substation", "relay", "busbar", "dga", "bushing", "tap changer"],
        "root_causes": [
            {"cause": "Insulation degradation / internal fault (gas generation)", "evidence": ["buchholz", "gas", "dga", "insulation", "winding"]},
            {"cause": "Overloading or cooling system failure", "evidence": ["oil temperature", "winding temperature", "cooling", "radiator", "overload", "fan"]},
            {"cause": "Protection relay / breaker mal-operation", "evidence": ["relay", "breaker", "trip", "tripped", "protection"]},
            {"cause": "Oil leak from gasket, valve or bushing", "evidence": ["leak", "oil", "gasket", "bushing", "seepage"]},
        ],
        "next_actions": [
            "Check protection relay event records and breaker status",
            "Take transformer oil sample for DGA if gas/Buchholz alarm present",
            "Inspect cooling (fans/radiators) and oil level; reduce load if overheating",
            "Follow LOTO / permit-to-work before any HV inspection",
        ],
    },
    "safety_hazard": {
        "label": "Safety hazard / incident",
        "asset_types": ["solar", "wind", "hydro", "grid"],
        "base_severity": "critical",
        "team": "HSE Emergency Response",
        "keywords": ["fire", "smoke", "injury", "injured", "electric shock", "arc flash",
                     "unsafe", "hazard", "burning", "sparking", "fall", "collapse", "flood", "evacuate"],
        "root_causes": [
            {"cause": "Electrical fault leading to arcing / fire", "evidence": ["fire", "smoke", "sparking", "arc", "burning", "cable"]},
            {"cause": "Unsafe working condition or missing safety barrier", "evidence": ["unsafe", "barrier", "guard", "ladder", "fall", "harness"]},
            {"cause": "Structural or civil failure (collapse, flooding, landslide)", "evidence": ["collapse", "crack", "flood", "landslide", "structure"]},
        ],
        "next_actions": [
            "Make the area safe: isolate equipment and keep people away",
            "Inform site in-charge and HSE emergency contact immediately",
            "Record incident in HSE system; preserve evidence for investigation",
        ],
    },
    "module_soiling": {
        "label": "PV module soiling / performance loss",
        "asset_types": ["solar"],
        "base_severity": "low",
        "team": "Solar Performance & Cleaning",
        "keywords": ["soiling", "dust", "cleaning", "dirty", "bird droppings", "performance ratio",
                     "pr drop", "underperforming", "low generation", "shading", "vegetation"],
        "root_causes": [
            {"cause": "Dust / soiling accumulation since last cleaning cycle", "evidence": ["dust", "soiling", "dirty", "cleaning", "droppings"]},
            {"cause": "Shading from vegetation or new structure", "evidence": ["shading", "vegetation", "grass", "tree", "shadow"]},
            {"cause": "Module degradation, hotspots or string underperformance", "evidence": ["hotspot", "string", "degradation", "crack", "underperforming"]},
        ],
        "next_actions": [
            "Compare block PR against neighbouring blocks to size the loss",
            "Schedule module cleaning / vegetation cutting for affected rows",
            "Run IV-curve or thermography on worst strings if loss persists after cleaning",
        ],
    },
}

# Generic severity modifiers applied by the severity tool.
CRITICAL_TERMS = ["fire", "smoke", "injury", "injured", "electric shock", "arc flash",
                  "explosion", "burning", "collapse", "evacuate", "sparking", "flood",
                  "metal particles", "metal debris", "buchholz", "overspeed"]
HIGH_TERMS = ["trip", "tripping", "tripped", "shutdown", "shut down", "stopped", "offline",
              "failure", "failed", "outage", "alarm", "repeatedly", "repeated", "leak",
              "abnormal", "not starting"]
DOWNGRADE_TERMS = ["routine", "scheduled", "cosmetic", "minor", "no impact", "low priority",
                   "when possible", "next visit", "informational", "slightly", "keep under observation",
                   "monitor at", "keep monitoring", "within limits", "no trip yet", "cleared after reset"]
WIDE_IMPACT_TERMS = ["entire plant", "whole plant", "all inverters", "all turbines",
                     "multiple turbines", "multiple inverters", "entire block", "full site",
                     "complete site", "several turbines", "plant output", "pooling station"]

# Category-specific escalations (domain rules an O&M engineer would apply).
CATEGORY_ESCALATIONS = {
    # Rising vibration with noise or heat on a gearbox is a precursor to catastrophic failure.
    # Iron/metal in oil = active component wear, regardless of how it is phrased.
    "gearbox_vibration": ["abnormal noise", "grinding", "increased", "rising", "metal", "temperature",
                          "debris", "iron", "sideband", "noise heard"],
    # A pitch fault without a working backup means the rotor cannot be feathered safely.
    "pitch_system": ["battery", "accumulator", "cannot feather", "not feathering", "not reaching",
                     "runaway", "overspeed", "cyclone", "storm"],
    "transformer_fault": ["buchholz", "gas", "fire", "smoke", "bushing"],
    # An expired statutory certificate means the equipment may not legally be used.
    "inspection_compliance": ["expired", "expires", "statutory"],
    # Exporting above an SLDC instruction is a grid-code violation (penalty risk).
    "grid_curtailment": ["still at", "above limit", "still exporting", "not applied"],
}
