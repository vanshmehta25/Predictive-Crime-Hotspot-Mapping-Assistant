#!/usr/bin/env python3
"""
================================================================
BOB Crime Intelligence Assistant - All-in-One Single-File Edition
================================================================

A predictive crime hotspot mapping assistant for a district police
station. Built on real precedents (Delhi Police ZIPNET as the data
layer that's missing a predictive layer; Bengaluru's 2023 predictive
policing pilot as proof this class of tool can move the needle on
property crime).

----------------------------------------------------------------
USAGE
----------------------------------------------------------------
  python app.py                       # Auto-init if needed, then start server
  python app.py --init                # Just generate data + predictions + charts + map (no server)
  python app.py --serve               # Just start server (auto-init if data missing)
  python app.py --regenerate          # Re-run full pipeline on existing data
  python app.py --help                # Show all CLI options

Once running, open http://localhost:3000 in your browser.

----------------------------------------------------------------
ROUTES
----------------------------------------------------------------
  /                     Dashboard (overview, top-5 cards, accuracy & bias summary, charts)
  /map                  Interactive Leaflet map (Google Maps tiles, no API key) with
                        severity-weighted heat layers + correlation overlay drawn
                        from the mock-data correlation analysis
  /brief                SHO patrol redeployment brief (HTML-rendered memo)
  /accuracy             Last-week prediction accuracy scorecard
  /bias                 Bias / fairness audit dashboard
  /upload               Browser form to inject Excel/CSV data
  /api/incidents        GET: JSON of all incidents  |  POST: upload Excel/CSV/JSON
  /api/predictions      JSON of all predictions
  /download             Download the project as a zip (if available)
  /regenerate           Re-run the pipeline (data -> predictions -> heatmaps -> Folium map)

----------------------------------------------------------------
TECH STACK
----------------------------------------------------------------
  Python 3.10+, Flask, Folium/Leaflet, Google Maps tiles (no API key),
  pandas, scikit-learn, matplotlib, openpyxl.

================================================================
"""

import os
import sys
import json
import random
import argparse
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================
# SECTION 1: CONSTANTS & PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "img")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# District definition (Connaught Place, New Delhi)
DISTRICT_CENTER = (28.6315, 77.2167)
DISTRICT_NAME = "PS Connaught Place (ZIPNET District 7)"
SPATIAL_RADIUS_DEG = 0.025  # ~2.7 km radius
GRID_N = 8

# Micro-zone grid bounds
LAT_MIN = DISTRICT_CENTER[0] - SPATIAL_RADIUS_DEG
LAT_MAX = DISTRICT_CENTER[0] + SPATIAL_RADIUS_DEG
LNG_MIN = DISTRICT_CENTER[1] - SPATIAL_RADIUS_DEG
LNG_MAX = DISTRICT_CENTER[1] + SPATIAL_RADIUS_DEG
LAT_STEP = (LAT_MAX - LAT_MIN) / GRID_N
LNG_STEP = (LNG_MAX - LNG_MIN) / GRID_N
ROW_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Crime type definitions: base_rate (probability), severity (1-5), peak_hour range
CRIME_TYPES = {
    "Theft":           {"base_rate": 0.32, "severity": 2, "peak_hour": (20, 23)},
    "Chain Snatching": {"base_rate": 0.14, "severity": 3, "peak_hour": (7, 10)},
    "Vehicle Theft":   {"base_rate": 0.18, "severity": 3, "peak_hour": (1, 5)},
    "Burglary":        {"base_rate": 0.12, "severity": 4, "peak_hour": (1, 5)},
    "Pickpocketing":   {"base_rate": 0.16, "severity": 2, "peak_hour": (12, 16)},
    "Assault":         {"base_rate": 0.05, "severity": 5, "peak_hour": (22, 2)},
    "Vandalism":       {"base_rate": 0.03, "severity": 2, "peak_hour": (23, 3)},
}

# Ground-truth hotspot seeds (where crime actually clusters in our mock data)
HOTSPOT_SEEDS = [
    {"zone": "D4", "lat": 28.6315, "lng": 77.2167, "intensity": 3.5, "crime_types": ["Theft", "Pickpocketing"]},
    {"zone": "B6", "lat": 28.6350, "lng": 77.2230, "intensity": 2.8, "crime_types": ["Chain Snatching", "Theft"]},
    {"zone": "F3", "lat": 28.6260, "lng": 77.2110, "intensity": 2.4, "crime_types": ["Vehicle Theft", "Burglary"]},
    {"zone": "E7", "lat": 28.6290, "lng": 77.2260, "intensity": 2.0, "crime_types": ["Assault", "Vandalism"]},
    {"zone": "C2", "lat": 28.6330, "lng": 77.2090, "intensity": 1.8, "crime_types": ["Pickpocketing", "Theft"]},
]

# Historical event calendar (used in mock data generation)
EVENT_CALENDAR = [
    {"date": "2025-03-14", "name": "Holi", "type": "festival", "crime_boost": 1.6,
     "crime_types": ["Assault", "Vandalism"], "zones_affected": ["D4", "E7"]},
    {"date": "2025-06-07", "name": "Eid al-Adha", "type": "festival", "crime_boost": 1.4,
     "crime_types": ["Theft", "Pickpocketing"], "zones_affected": ["B6"]},
    {"date": "2025-08-15", "name": "Independence Day", "type": "national_day", "crime_boost": 1.5,
     "crime_types": ["Pickpocketing", "Theft"], "zones_affected": ["D4", "C2"]},
]

# Forward-looking events for the prediction week (deterministic boosts)
UPCOMING_EVENTS = [
    {"date": "2025-09-01", "name": "Payday (month start)", "type": "payday",
     "crime_boost": 1.15, "zones_affected": "all"},
    {"date": "2025-09-05", "name": "Teachers Day / Local market day", "type": "event",
     "crime_boost": 1.20, "zones_affected": ["B6", "D4"]},
    {"date": "2025-09-07", "name": "Weekend gathering (Sat)", "type": "weekend",
     "crime_boost": 1.25, "zones_affected": ["D4", "E7"]},
]

# Prediction window
PREDICTION_WEEK_START = datetime(2025, 9, 1)
LAST_WEEK_START = datetime(2025, 8, 25)
DATA_START = datetime(2025, 3, 1)
DATA_END = datetime(2025, 8, 31)


def build_grid():
    """Build the 8x8 micro-zone grid (64 cells, A-H x 1-8)."""
    cells = []
    for ri, rl in enumerate(ROW_LABELS):
        for ci in range(1, GRID_N + 1):
            cells.append({
                "zone_id": f"{rl}{ci}",
                "lat_min": LAT_MIN + ri * LAT_STEP,
                "lat_max": LAT_MIN + (ri + 1) * LAT_STEP,
                "lng_min": LNG_MIN + (ci - 1) * LNG_STEP,
                "lng_max": LNG_MIN + ci * LNG_STEP,
                "lat_center": LAT_MIN + (ri + 0.5) * LAT_STEP,
                "lng_center": LNG_MIN + (ci - 0.5) * LNG_STEP,
            })
    return cells


GRID_CELLS = build_grid()


def zone_for_point(lat, lng):
    for z in GRID_CELLS:
        if z["lat_min"] <= lat < z["lat_max"] and z["lng_min"] <= lng < z["lng_max"]:
            return z["zone_id"]
    return None


# ============================================================
# SECTION 2: DATA LOADING (CSV or XLSX, auto-derives date/hour/weekday)
# ============================================================

def data_file_path():
    """Return path to whichever data file exists (xlsx preferred)."""
    xlsx = os.path.join(DATA_DIR, "crime_incidents.xlsx")
    csv = os.path.join(DATA_DIR, "crime_incidents.csv")
    if os.path.exists(xlsx):
        return xlsx
    return csv


def load_df():
    """Load incidents from xlsx (preferred) or csv. Auto-derive date/hour/weekday from timestamp."""
    path = data_file_path()
    if path.endswith(".xlsx"):
        df = pd.read_excel(path, parse_dates=["timestamp"])
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    # Auto-derive optional columns from timestamp
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date.astype(str)
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["timestamp"].dt.day_name()
    return df


def load_predictions():
    with open(os.path.join(DATA_DIR, "predictions.json")) as f:
        return json.load(f)


def load_zones():
    with open(os.path.join(DATA_DIR, "zones.json")) as f:
        return json.load(f)


def load_events():
    with open(os.path.join(DATA_DIR, "events.json")) as f:
        return json.load(f)


# ============================================================
# SECTION 2b: UPLOAD NORMALIZATION (any input format -> canonical schema)
# ============================================================

# Recognized aliases for upload columns (matched case/space-insensitively), so
# JSON lists, CSVs and Excel exports with different headers all work: e.g.
# lat/lon/lng, datetime/time/date, type/crime/category, sev/weight, zone/beat.
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "date", "occurred_at",
                  "occurred_on", "reported_at", "reported_on", "incident_date"],
    "latitude": ["latitude", "lat", "y", "y_coord", "y_coordinate"],
    "longitude": ["longitude", "lon", "lng", "long", "x", "x_coord", "x_coordinate"],
    "crime_type": ["crime_type", "type", "crime", "crime_name", "category", "offence", "offense"],
    "severity": ["severity", "sev", "weight", "priority", "gravity"],
    "reporting_mode": ["reporting_mode", "mode", "source", "reported_via", "reported_by", "channel"],
    "zone_id": ["zone_id", "zone", "grid_zone", "beat", "sector"],
    "incident_id": ["incident_id", "id", "ref", "case_no", "fir_no"],
}


def normalize_uploaded_df(df_new):
    """Tolerantly map any reasonable input format onto the canonical schema.

    Handles different column names (via COLUMN_ALIASES), inconsistent casing or
    spacing, string-encoded numbers, missing optional fields, and bad rows.
    Returns (df, notes). When the input cannot be salvaged, returns (None, [error]).
    """
    notes = []

    # --- 1. Column-name mapping ---
    ren = {}
    for col in df_new.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for std, aliases in COLUMN_ALIASES.items():
            if key in aliases:
                if std not in ren.values():  # first match wins, no duplicate targets
                    ren[col] = std
                break
    renamed = {c: r for c, r in ren.items() if c != r}
    if renamed:
        notes.append("Mapped columns: " + ", ".join(f"'{c}' -> '{r}'" for c, r in renamed.items()))
    df_new = df_new.rename(columns=ren)

    # --- 2. Timestamp (required for all temporal analysis) ---
    if "timestamp" in df_new.columns:
        # format="mixed" parses each row individually - uploads often mix
        # "2025-06-01 21:15:00" and "2025-06-02 08:00" styles in one column
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors="coerce", format="mixed")
    else:
        return None, ["No timestamp-like column found (tried: "
                      + ", ".join(COLUMN_ALIASES["timestamp"]) + ")."]
    n_bad_ts = int(df_new["timestamp"].isna().sum())
    if n_bad_ts:
        df_new = df_new[df_new["timestamp"].notna()]
        notes.append(f"Dropped {n_bad_ts} row(s) with unparseable timestamps.")

    # --- 3. Coordinates (required for the map) ---
    if "latitude" not in df_new.columns or "longitude" not in df_new.columns:
        return None, ["No latitude/longitude columns found (tried aliases like lat/lon/lng/x/y)."]
    df_new["latitude"] = pd.to_numeric(df_new["latitude"], errors="coerce")
    df_new["longitude"] = pd.to_numeric(df_new["longitude"], errors="coerce")
    n_bad_geo = int((df_new["latitude"].isna() | df_new["longitude"].isna()).sum())
    if n_bad_geo:
        df_new = df_new[df_new["latitude"].notna() & df_new["longitude"].notna()]
        notes.append(f"Dropped {n_bad_geo} row(s) with invalid coordinates.")
    if df_new.empty:
        return None, ["No usable rows remained after cleaning - nothing to analyze."]

    # --- 4. Zone assignment: derive from coordinates for missing/unknown zone ids ---
    valid_zones = {z["zone_id"] for z in GRID_CELLS}
    if "zone_id" in df_new.columns:
        needs_zone = ~df_new["zone_id"].astype(str).str.strip().isin(valid_zones)
    else:
        df_new["zone_id"] = None
        needs_zone = pd.Series(True, index=df_new.index)
    n_zoned = int(needs_zone.sum())
    if n_zoned:
        df_new.loc[needs_zone, "zone_id"] = [
            zone_for_point(a, b)
            for a, b in zip(df_new.loc[needs_zone, "latitude"], df_new.loc[needs_zone, "longitude"])
        ]
        before = len(df_new)
        df_new = df_new[df_new["zone_id"].notna()]
        dropped_outside = before - len(df_new)
        notes.append(f"Derived zone_id from coordinates for {n_zoned} row(s)"
                     + (f"; dropped {dropped_outside} row(s) outside the district grid."
                        if dropped_outside else "."))
    if df_new.empty:
        return None, ["All rows fall outside the district grid - nothing to analyze."]

    # --- 5. Categorical / numeric fields with sensible defaults ---
    if "crime_type" not in df_new.columns:
        df_new["crime_type"] = "Unknown"
        notes.append("No crime-type column found; used 'Unknown'.")
    else:
        n_missing_ct = int(df_new["crime_type"].isna().sum())
        if n_missing_ct:
            notes.append(f"Filled {n_missing_ct} missing crime type(s) with 'Unknown'.")
    df_new["crime_type"] = (df_new["crime_type"].astype(str).str.strip()
                            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "NaN": "Unknown"}))
    if "severity" in df_new.columns:
        df_new["severity"] = pd.to_numeric(df_new["severity"], errors="coerce").fillna(2).clip(1, 5).astype(int)
    else:
        df_new["severity"] = 2
        notes.append("No severity column found; defaulted to 2.")
    if "reporting_mode" not in df_new.columns:
        df_new["reporting_mode"] = "FIR"
        notes.append("No reporting-mode column found; defaulted to 'FIR'.")
    df_new["reporting_mode"] = df_new["reporting_mode"].astype(str).str.strip().replace({"nan": "FIR", "": "FIR", "None": "FIR"})
    if "incident_id" not in df_new.columns:
        df_new["incident_id"] = [f"INC-U{i:05d}" for i in range(1, len(df_new) + 1)]
        notes.append("Generated incident ids.")
    df_new["incident_id"] = df_new["incident_id"].astype(str)

    # --- 6. Derived temporal columns + canonical column order ---
    df_new["date"] = df_new["timestamp"].dt.date.astype(str)
    df_new["hour"] = df_new["timestamp"].dt.hour
    df_new["weekday"] = df_new["timestamp"].dt.day_name()
    df_new = df_new[["incident_id", "timestamp", "date", "hour", "weekday",
                     "latitude", "longitude", "zone_id", "crime_type", "severity",
                     "reporting_mode"]]
    return df_new, notes


# ============================================================
# SECTION 3: MOCK DATA GENERATOR (from generate_data.py)
# ============================================================

def _is_payday(d):
    return d.day in (1, 28, 29, 30, 31)


def _is_weekend_eve(d):
    return d.weekday() in (4, 5)  # Friday or Saturday


def generate_mock_data(seed=42):
    """Generate 6 months of mock crime incidents for the district."""
    random.seed(seed)
    np.random.seed(seed)

    incidents = []
    cur = DATA_START
    incident_id = 1
    while cur <= DATA_END:
        base_count = np.random.poisson(12)  # ~12 incidents/day average
        event_today = next((e for e in EVENT_CALENDAR if e["date"] == cur.strftime("%Y-%m-%d")), None)
        boost = 1.0
        if event_today:
            boost = event_today["crime_boost"]
        if _is_payday(cur):
            boost *= 1.15
        if _is_weekend_eve(cur):
            boost *= 1.25

        n_today = int(base_count * boost)
        for _ in range(n_today):
            # 60% of incidents cluster around hotspot seeds
            if random.random() < 0.6:
                hs = random.choice(HOTSPOT_SEEDS)
                lat = hs["lat"] + np.random.normal(0, 0.0015)
                lng = hs["lng"] + np.random.normal(0, 0.0015)
                crime_type = random.choice(hs["crime_types"])
            else:
                lat = np.random.uniform(LAT_MIN, LAT_MAX)
                lng = np.random.uniform(LNG_MIN, LNG_MAX)
                crime_type = random.choices(
                    list(CRIME_TYPES.keys()),
                    weights=[v["base_rate"] for v in CRIME_TYPES.values()],
                    k=1,
                )[0]

            # Pick hour based on crime type's peak window
            cpk = CRIME_TYPES[crime_type]
            ph = cpk["peak_hour"]
            if ph[0] < ph[1]:
                hour = int(np.clip(np.random.normal((ph[0] + ph[1]) / 2, 1.5), 0, 23))
            else:  # peak window wraps midnight
                h = np.random.normal(ph[0], 2.0)
                if h > 23:
                    h -= 24
                hour = int(np.clip(h, 0, 23))

            ts = cur.replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
            severity = min(5, cpk["severity"] + (1 if random.random() < 0.2 else 0))
            reporting = random.choices(
                ["FIR", "PCR Call", "Patrol Observed"],
                weights=[0.45, 0.35, 0.20],
            )[0]
            zone = zone_for_point(lat, lng)
            if zone is None:
                continue
            incidents.append({
                "incident_id": f"INC-{incident_id:05d}",
                "timestamp": ts,
                "date": ts.date().isoformat(),
                "hour": ts.hour,
                "weekday": ts.strftime("%A"),
                "latitude": round(lat, 6),
                "longitude": round(lng, 6),
                "zone_id": zone,
                "crime_type": crime_type,
                "severity": severity,
                "reporting_mode": reporting,
            })
            incident_id += 1
        cur += timedelta(days=1)
    return incidents


def save_data(df):
    """Save incidents DataFrame as CSV (replaces existing data)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "crime_incidents.csv"), index=False)
    # Remove any xlsx so CSV is preferred next load
    xlsx = os.path.join(DATA_DIR, "crime_incidents.xlsx")
    if os.path.exists(xlsx):
        os.remove(xlsx)


def save_meta():
    """Save zones.json and events.json (static metadata)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "zones.json"), "w") as f:
        json.dump(GRID_CELLS, f, indent=2)
    with open(os.path.join(DATA_DIR, "events.json"), "w") as f:
        json.dump(EVENT_CALENDAR, f, indent=2)


# ============================================================
# SECTION 4: PREDICTION ENGINE (from prediction_engine.py)
# ============================================================

def compute_zone_features(df, zones, prediction_week_start):
    """Per-zone spatial-temporal feature extraction + next-week prediction.

    Weighted ensemble (interpretable, no black-box ML):
        predicted(zone) = 0.45 * recent_4wk_avg
                       + 0.30 * full_6mo_baseline
                       + 0.15 * trend_adjusted_recent
                       + 0.10 * event_boost_factor
    """
    pred_start = prediction_week_start
    pred_end = pred_start + timedelta(days=7)
    last_4wk_start = pred_start - timedelta(days=28)
    prior_4wk_start = pred_start - timedelta(days=56)
    full_data_start = df["timestamp"].min()
    full_data_end = df["timestamp"].max()

    df_full = df.copy()
    df_recent = df[(df["timestamp"] >= last_4wk_start) & (df["timestamp"] < pred_start)]
    df_prior = df[(df["timestamp"] >= prior_4wk_start) & (df["timestamp"] < last_4wk_start)]

    results = []
    for z in zones:
        zid = z["zone_id"]
        inc_full = df_full[df_full["zone_id"] == zid]
        inc_recent = df_recent[df_recent["zone_id"] == zid]
        inc_prior = df_prior[df_prior["zone_id"] == zid]

        n_full = len(inc_full)
        n_recent = len(inc_recent)
        n_prior = len(inc_prior)

        weeks_in_full = max(1, (full_data_end - full_data_start).days / 7)
        avg_weekly_full = n_full / weeks_in_full
        avg_weekly_recent = n_recent / 4.0
        avg_weekly_prior = n_prior / 4.0
        trend_pct = ((avg_weekly_recent - avg_weekly_prior) / max(0.5, avg_weekly_prior)) * 100

        dow_profile = inc_full["weekday"].value_counts().to_dict() if n_full > 0 else {}
        hour_profile = inc_full["hour"].value_counts().to_dict() if n_full > 0 else {}
        if hour_profile:
            peak_hour = max(hour_profile, key=hour_profile.get)
            window_start = max(0, peak_hour - 1)
            window_end = min(23, peak_hour + 2)
        else:
            peak_hour = None
            window_start = window_end = None

        ct_breakdown = inc_full["crime_type"].value_counts().to_dict() if n_full > 0 else {}
        dominant_dow = max(dow_profile, key=dow_profile.get) if dow_profile else None

        # Event boost (deterministic, rule-based)
        event_boost = 1.0
        event_flags = []
        for ev in UPCOMING_EVENTS:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d")
            if pred_start <= ev_date < pred_end:
                if ev["zones_affected"] == "all" or zid in ev["zones_affected"]:
                    event_boost *= ev["crime_boost"]
                    event_flags.append(ev["name"])

        # Weighted ensemble
        recent_component = avg_weekly_recent * 0.45
        baseline_component = avg_weekly_full * 0.30
        trend_component = (avg_weekly_recent * (1 + np.clip(trend_pct / 100, -0.5, 0.5))) * 0.15
        event_component = (recent_component + baseline_component) * (event_boost - 1.0) * 0.10

        predicted_incidents = max(0, recent_component + baseline_component + trend_component + event_component)

        results.append({
            "zone_id": zid,
            "lat_center": z["lat_center"],
            "lng_center": z["lng_center"],
            "total_incidents_6mo": n_full,
            "incidents_last_4wk": n_recent,
            "incidents_prior_4wk": n_prior,
            "avg_weekly_full": round(avg_weekly_full, 2),
            "avg_weekly_recent": round(avg_weekly_recent, 2),
            "trend_pct": round(trend_pct, 1),
            "peak_hour": peak_hour,
            "peak_window": f"{window_start:02d}:00-{window_end:02d}:00" if peak_hour is not None else None,
            "dominant_dow": dominant_dow,
            "dominant_crime_types": list(ct_breakdown.items())[:3],
            "event_flags": event_flags,
            "event_boost": round(event_boost, 3),
            "predicted_incidents_next_week": round(predicted_incidents, 3),
        })

    # Normalize to probability mass
    total_pred = sum(r["predicted_incidents_next_week"] for r in results) or 1.0
    for r in results:
        r["probability"] = round(r["predicted_incidents_next_week"] / total_pred, 4)
    results.sort(key=lambda x: x["predicted_incidents_next_week"], reverse=True)
    return results


def generate_rationale(zone_feat, district_avg_weekly):
    """Plain-language rationale for a top-5 prediction (for SHO readability + audit trail)."""
    zid = zone_feat["zone_id"]
    parts = []
    parts.append(
        f"Zone {zid}: {zone_feat['incidents_last_4wk']} incidents in last 4 weeks "
        f"(avg {zone_feat['avg_weekly_recent']}/wk vs district avg {district_avg_weekly:.1f}/wk)."
    )
    tr = zone_feat["trend_pct"]
    if tr > 10:
        parts.append(f"Trend: +{tr:.0f}% vs prior 4-week window (rising).")
    elif tr < -10:
        parts.append(f"Trend: {tr:.0f}% vs prior 4-week window (declining).")
    else:
        parts.append(f"Trend: {tr:+.0f}% vs prior 4-week window (stable).")
    if zone_feat["peak_hour"] is not None:
        parts.append(f"Peak window: {zone_feat['peak_window']} on {zone_feat['dominant_dow']}s.")
    if zone_feat["dominant_crime_types"]:
        ct_str = ", ".join([f"{ct} ({n})" for ct, n in zone_feat["dominant_crime_types"]])
        parts.append(f"Dominant crime types: {ct_str}.")
    if zone_feat["event_flags"]:
        parts.append(f"Upcoming event boost: {' / '.join(zone_feat['event_flags'])} (x{zone_feat['event_boost']}).")
    else:
        parts.append("No upcoming event-linked spike forecast for this zone.")
    return " ".join(parts)


def generate_sho_brief(top5, district_avg_weekly, prediction_week_start):
    """Auto-generate the SHO patrol redeployment brief."""
    pred_end = prediction_week_start + timedelta(days=6)
    date_str = f"{prediction_week_start.strftime('%d %b %Y')} - {pred_end.strftime('%d %b %Y')}"
    brief = [
        "PATROL REDEPLOYMENT BRIEF - WEEKLY PREDICTIVE INTELLIGENCE",
        f"Station: {DISTRICT_NAME}",
        f"Prediction window: {date_str}",
        "Generated by: BOB Crime Intelligence Assistant (v1.0)",
        "=" * 70,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 70,
    ]
    top1 = top5[0]
    brief.append(
        f"Top risk concentration in Zone {top1['zone_id']} "
        f"(predicted {top1['predicted_incidents_next_week']:.1f} incidents, "
        f"{top1['probability']*100:.1f}% of district probability mass). "
        f"Recommended: shift 2 beat officers to Zone {top1['zone_id']} "
        f"during {top1['peak_window'] or 'evening'} window."
    )
    brief.append("")
    brief.append("TOP-5 AT-RISK MICRO-ZONES (NEXT 7 DAYS)")
    brief.append("-" * 70)
    for i, z in enumerate(top5, 1):
        brief.append(f"\n{i}. Zone {z['zone_id']} -- Predicted: {z['predicted_incidents_next_week']:.1f} incidents ({z['probability']*100:.1f}% mass)")
        brief.append(f"   Rationale: {generate_rationale(z, district_avg_weekly)}")
        brief.append(f"   Recommended action: {'HIGH' if i<=2 else 'MEDIUM'} confidence -- {'redeploy 2 officers' if i<=2 else 'awareness ping + CCTV check'}")
    brief.append("")
    brief.append("EVENT / SEASONAL FLAGS")
    brief.append("-" * 70)
    for ev in UPCOMING_EVENTS:
        ev_date = datetime.strptime(ev["date"], "%Y-%m-%d")
        if prediction_week_start <= ev_date < pred_end:
            brief.append(f"- {ev['date']} ({ev['name']}): expected x{ev['crime_boost']} boost in zones {ev['zones_affected']}")
    brief.append("")
    brief.append("DISCLAIMER")
    brief.append("-" * 70)
    brief.append("Advisory only. Final deployment decision rests with SHO. Predictions are")
    brief.append("spatial-temporal pattern extrapolations from past 6 months of reported")
    brief.append("incidents; under-reporting bias may exist. Audit last week's accuracy at")
    brief.append("/accuracy route before relying on this week's brief.")
    return "\n".join(brief)


def compute_last_week_accuracy(df, zones, last_week_start):
    """Scorecard: predicted vs actual incident counts per zone for last week."""
    actual_start = last_week_start
    actual_end = actual_start + timedelta(days=7)
    df_actual = df[(df["timestamp"] >= actual_start) & (df["timestamp"] < actual_end)]
    actual_by_zone = df_actual["zone_id"].value_counts().to_dict()

    df_pre = df[df["timestamp"] < actual_start].copy()
    pre_start_4wk = actual_start - timedelta(days=28)
    df_pre_4wk = df_pre[df_pre["timestamp"] >= pre_start_4wk]
    pred_by_zone = df_pre_4wk["zone_id"].value_counts().to_dict()
    total_actual = sum(actual_by_zone.values()) or 1
    total_pred = sum(pred_by_zone.values()) or 1
    scale = total_actual / total_pred

    pred_top5 = sorted(pred_by_zone.items(), key=lambda x: x[1], reverse=True)[:5]
    pred_top5_zones = [z for z, _ in pred_top5]
    actual_top5 = sorted(actual_by_zone.items(), key=lambda x: x[1], reverse=True)[:5]
    actual_top5_zones = [z for z, _ in actual_top5]
    overlap = set(pred_top5_zones) & set(actual_top5_zones)

    per_zone = []
    for z in zones:
        zid = z["zone_id"]
        p = pred_by_zone.get(zid, 0) * scale
        a = actual_by_zone.get(zid, 0)
        per_zone.append({"zone_id": zid, "predicted": round(p, 1), "actual": a, "abs_error": round(abs(p - a), 1)})

    mae = float(np.mean([p["abs_error"] for p in per_zone]))
    pred_ranks = pd.Series([pred_by_zone.get(z["zone_id"], 0) * scale for z in zones]).rank()
    actual_ranks = pd.Series([actual_by_zone.get(z["zone_id"], 0) for z in zones]).rank()
    rank_corr = pred_ranks.corr(actual_ranks)

    return {
        "prediction_window": f"{actual_start.strftime('%d %b')} - {(actual_end - timedelta(days=1)).strftime('%d %b')}",
        "predicted_top5": pred_top5_zones,
        "actual_top5": actual_top5_zones,
        "top5_overlap_count": len(overlap),
        "top5_overlap_pct": round(len(overlap) / 5 * 100, 1),
        "mae_per_zone": round(mae, 2),
        "rank_correlation": round(rank_corr, 3) if not np.isnan(rank_corr) else None,
        "per_zone": per_zone,
    }


def bias_audit(df, zones, top5):
    """Self-audit: does predicted hotspot concentration correlate with anything other than crime density?

    Uses three proxies (since real demographics aren't available in this prototype):
    (1) reporting-mode mix per top-5 zone vs district average (patrol-coverage bias check)
    (2) severity mix per top-5 zone (low-severity-heavy hotspots = suspicious)
    (3) geographic Gini coefficient (expected high for crime data)
    """
    top5_zids = [z["zone_id"] for z in top5]
    district_mode_pct = (df["reporting_mode"].value_counts(normalize=True) * 100).to_dict()
    per_zone_mode = {}
    for zid in top5_zids:
        sub = df[df["zone_id"] == zid]
        if len(sub) > 0:
            per_zone_mode[zid] = (sub["reporting_mode"].value_counts(normalize=True) * 100).to_dict()
        else:
            per_zone_mode[zid] = {}

    district_sev_pct = (df["severity"].value_counts(normalize=True).sort_index() * 100).to_dict()
    per_zone_sev = {}
    for zid in top5_zids:
        sub = df[df["zone_id"] == zid]
        if len(sub) > 0:
            per_zone_sev[zid] = (sub["severity"].value_counts(normalize=True).sort_index() * 100).to_dict()
        else:
            per_zone_sev[zid] = {}

    counts = df["zone_id"].value_counts().sort_values().values
    n = len(counts)
    cum = np.cumsum(counts)
    total = cum[-1] if len(cum) > 0 else 1
    gini = (2 * sum((i + 1) * counts[i] for i in range(n))) / (n * total) - (n + 1) / n

    patrol_pct_top5 = [per_zone_mode[zid].get("Patrol Observed", 0) for zid in top5_zids]
    patrol_pct_district = district_mode_pct.get("Patrol Observed", 0)
    patrol_bias_flag = any(p > patrol_pct_district * 1.5 for p in patrol_pct_top5)

    return {
        "district_reporting_mode_pct": district_mode_pct,
        "top5_reporting_mode_pct": per_zone_mode,
        "district_severity_pct": district_sev_pct,
        "top5_severity_pct": per_zone_sev,
        "geographic_gini": round(gini, 3),
        "patrol_coverage_bias_flag": patrol_bias_flag,
        "bias_warning": (
            "WARNING: One or more predicted hotspots show >50% higher Patrol-Observed "
            "reporting than district average. This may indicate the model is detecting "
            "patrol coverage patterns rather than true crime density. Recommend manual "
            "review before deployment."
            if patrol_bias_flag else
            "No major reporting-bias flag detected in top-5 zones. Standard audit applies."
        ),
    }


def compute_correlations(df):
    """Correlate the mock data to find co-occurring patterns.

    Three correlation analyses:
    (1) Zone-zone correlation matrix - which zones spike together (cluster)
    (2) Crime-crime correlation matrix - which crime types co-occur
    (3) Lag cascades - crime A today predicts crime B N days later (cascade detection)

    Returns a dict ready for JSON serialization + heatmap rendering.
    """
    # --- (1) Zone-zone correlation (daily incident counts per zone) ---
    daily_zone = df.groupby(["date", "zone_id"]).size().unstack(fill_value=0)
    # Only keep zones with at least 10 incidents for meaningful correlation
    active_zones = [z for z in daily_zone.columns if daily_zone[z].sum() >= 10]
    if len(active_zones) >= 2:
        zone_corr = daily_zone[active_zones].corr()
    else:
        zone_corr = pd.DataFrame()

    # Extract top zone-zone pairs (excluding diagonal)
    zone_pairs = []
    if not zone_corr.empty:
        for i, za in enumerate(zone_corr.index):
            for zb in zone_corr.columns[i + 1:]:
                c = zone_corr.loc[za, zb]
                if pd.notna(c) and abs(c) >= 0.20:  # lowered from 0.30 to surface more patterns in mock data
                    zone_pairs.append({
                        "zone_a": za, "zone_b": zb,
                        "correlation": round(float(c), 3),
                        "interpretation": f"Zone {za} and Zone {zb} spike on same days (r={c:.2f}) - shared offenders or shared venue",
                    })
    zone_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Adaptive fallback: sparse uploads (or short date ranges) may yield few pairs
    # above the threshold. Always surface the strongest remaining pairs (up to 10
    # total) so the /map "Zone correlation links" layer has something to draw
    # regardless of the input format or volume.
    if not zone_corr.empty and len(zone_pairs) < 10:
        seen = {(p["zone_a"], p["zone_b"]) for p in zone_pairs}
        extra = []
        for i, za in enumerate(zone_corr.index):
            for zb in zone_corr.columns[i + 1:]:
                if (za, zb) in seen:
                    continue
                c = zone_corr.loc[za, zb]
                if pd.notna(c):
                    extra.append({
                        "zone_a": za, "zone_b": zb,
                        "correlation": round(float(c), 3),
                        "interpretation": f"Zone {za} and Zone {zb} spike on same days (r={c:.2f}) - shared offenders or shared venue",
                    })
        extra.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        zone_pairs += extra[: 10 - len(zone_pairs)]
    zone_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # --- (2) Crime-crime correlation (daily counts per crime type) ---
    daily_ct = df.groupby(["date", "crime_type"]).size().unstack(fill_value=0)
    ct_corr = daily_ct.corr() if len(daily_ct.columns) >= 2 else pd.DataFrame()

    crime_pairs = []
    if not ct_corr.empty:
        for i, ca in enumerate(ct_corr.index):
            for cb in ct_corr.columns[i + 1:]:
                c = ct_corr.loc[ca, cb]
                if pd.notna(c) and abs(c) >= 0.20:
                    crime_pairs.append({
                        "crime_a": ca, "crime_b": cb,
                        "correlation": round(float(c), 3),
                        "interpretation": f"{ca} and {cb} co-occur (r={c:.2f})",
                    })
    crime_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Same adaptive fallback for crime-crime pairs (up to 10 total).
    if not ct_corr.empty and len(crime_pairs) < 10:
        seen = {(p["crime_a"], p["crime_b"]) for p in crime_pairs}
        extra = []
        for i, ca in enumerate(ct_corr.index):
            for cb in ct_corr.columns[i + 1:]:
                if (ca, cb) in seen:
                    continue
                c = ct_corr.loc[ca, cb]
                if pd.notna(c):
                    extra.append({
                        "crime_a": ca, "crime_b": cb,
                        "correlation": round(float(c), 3),
                        "interpretation": f"{ca} and {cb} co-occur (r={c:.2f})",
                    })
        extra.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        crime_pairs += extra[: 10 - len(crime_pairs)]
    crime_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # --- (3) Lag cascades: does crime A today predict crime B N days later? ---
    lags = [1, 2, 3, 7]

    def _collect_lag_cascades(min_r):
        out = []
        if ct_corr.empty:
            return out
        for crime_a in daily_ct.columns:
            for crime_b in daily_ct.columns:
                if crime_a == crime_b:
                    continue
                for lag in lags:
                    a_series = daily_ct[crime_a][:-lag]
                    b_series = daily_ct[crime_b].shift(-lag)[:-lag]
                    if len(a_series) < 10:
                        continue
                    corr = a_series.corr(b_series)
                    if pd.notna(corr) and abs(corr) >= min_r:
                        direction = "predicts" if corr > 0 else "suppresses"
                        out.append({
                            "crime_a": crime_a, "crime_b": crime_b,
                            "lag_days": lag,
                            "correlation": round(float(corr), 3),
                            "interpretation": f"{crime_a} today {direction} {crime_b} {lag} day(s) later (r={corr:.2f})",
                        })
        out.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return out

    lag_results = _collect_lag_cascades(0.15)
    if len(lag_results) < 5:  # sparse-upload fallback - relax the threshold
        lag_results = _collect_lag_cascades(0.08)[:10]
    lag_results.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "zone_zone_correlation": {
            "matrix": zone_corr.round(3).to_dict() if not zone_corr.empty else {},
            "active_zones": active_zones,
            "top_pairs": zone_pairs[:10],
        },
        "crime_crime_correlation": {
            "matrix": ct_corr.round(3).to_dict() if not ct_corr.empty else {},
            "top_pairs": crime_pairs[:10],
        },
        "lag_cascades": lag_results[:15],
    }


def run_prediction_engine():
    """Full prediction pipeline -> writes data/predictions.json."""
    df = load_df()
    zones = GRID_CELLS

    feats = compute_zone_features(df, zones, PREDICTION_WEEK_START)
    top5 = feats[:5]
    district_avg_weekly = float(np.mean([f["avg_weekly_full"] for f in feats]))
    rationale_texts = {z["zone_id"]: generate_rationale(z, district_avg_weekly) for z in top5}
    sho_brief = generate_sho_brief(top5, district_avg_weekly, PREDICTION_WEEK_START)
    accuracy = compute_last_week_accuracy(df, zones, LAST_WEEK_START)
    bias = bias_audit(df, zones, top5)
    correlations = compute_correlations(df)

    output = {
        "prediction_week_start": PREDICTION_WEEK_START.strftime("%Y-%m-%d"),
        "prediction_week_end": (PREDICTION_WEEK_START + timedelta(days=6)).strftime("%Y-%m-%d"),
        "district_name": DISTRICT_NAME,
        "district_center": list(DISTRICT_CENTER),
        "district_avg_weekly": round(district_avg_weekly, 2),
        "total_incidents_6mo": int(len(df)),
        "top5": top5,
        "rationale_texts": rationale_texts,
        "sho_brief": sho_brief,
        "upcoming_events": UPCOMING_EVENTS,
        "accuracy_scorecard": accuracy,
        "bias_audit": bias,
        "correlations": correlations,
        "all_zones": feats,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[prediction_engine] Saved predictions to {out_path}")
    print(f"[prediction_engine] Top-5 zones: {[z['zone_id'] for z in top5]}")
    print(f"[prediction_engine] District avg weekly incidents: {district_avg_weekly:.2f}")
    print(f"[prediction_engine] Last-week accuracy: top-5 overlap {accuracy['top5_overlap_count']}/5 ({accuracy['top5_overlap_pct']}%), MAE {accuracy['mae_per_zone']}")
    print(f"[prediction_engine] Bias flag: patrol-coverage bias = {bias['patrol_coverage_bias_flag']}, geographic Gini = {bias['geographic_gini']}")
    print(f"[prediction_engine] Correlations: {len(correlations['zone_zone_correlation']['top_pairs'])} zone pairs, "
          f"{len(correlations['crime_crime_correlation']['top_pairs'])} crime pairs, "
          f"{len(correlations['lag_cascades'])} lag cascades")
    return output


# ============================================================
# SECTION 5: STATIC PNG HEATMAP GENERATOR (from generate_heatmaps.py)
# ============================================================

def _setup_matplotlib():
    """Configure matplotlib for headless PNG output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    # Try to add Chinese font fallbacks (skips silently if missing)
    for fp in ['/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
               '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
            except Exception:
                pass
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def generate_static_heatmaps():
    """Generate 5 PNG heatmaps via matplotlib."""
    plt = _setup_matplotlib()
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_df()
    preds = load_predictions()
    zones = GRID_CELLS

    # --- (1) Spatial KDE heatmap (severity-weighted, smoothed) + top-5 markers ---
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sev_weights = pd.to_numeric(df["severity"], errors="coerce").fillna(2.0).clip(1, 5).values
    H, xedges, yedges = np.histogram2d(df["longitude"].values, df["latitude"].values,
                                       bins=90, weights=sev_weights)
    try:  # optional Gaussian smoothing when scipy is installed
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=1.8)
    except Exception:
        pass
    H = np.ma.masked_where(H <= 0.01, H)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H.T, extent=extent, origin="lower", cmap="YlOrRd", aspect="auto",
                   alpha=0.9, interpolation="bilinear")
    for z in zones:
        rect = plt.Rectangle((z["lng_min"], z["lat_min"]),
                              z["lng_max"]-z["lng_min"], z["lat_max"]-z["lat_min"],
                              fill=False, edgecolor="#1a3a5c", linewidth=0.4, alpha=0.5)
        ax.add_patch(rect)
        ax.text(z["lng_center"], z["lat_center"], z["zone_id"],
                fontsize=6, ha="center", va="center", color="#1a3a5c", alpha=0.7)
    for i, z in enumerate(preds["top5"], 1):
        ax.plot(z["lng_center"], z["lat_center"], "o", markersize=14,
                markerfacecolor="none", markeredgewidth=2,
                markeredgecolor="#0066ff" if i<=2 else "#666666")
        ax.annotate(f"#{i}", xy=(z["lng_center"], z["lat_center"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#0066ff")
    plt.colorbar(im, ax=ax, label="Severity-weighted incident density (6 months)", shrink=0.8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Severity-weighted Crime Density Heatmap -- {preds['district_name']}\n"
                 f"(Blue circles = top-5 predicted next-week hotspots)", fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, "spatial_heatmap.png"), dpi=140)
    plt.close()

    # --- (2) Zone choropleth (predicted incidents per zone, 8x8 grid) ---
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    zone_dict = {z["zone_id"]: z for z in zones}
    grid = np.zeros((8, 8))
    for z in preds["all_zones"]:
        zid = z["zone_id"]
        if zid in zone_dict:
            ri = "ABCDEFGH".index(zid[0])
            ci = int(zid[1:]) - 1
            grid[ri, ci] = z["predicted_incidents_next_week"]
    im = ax.imshow(grid, cmap="YlOrRd", origin="upper", aspect="equal")
    for ri in range(8):
        for ci in range(8):
            row_label = "ABCDEFGH"[ri]
            zid = f"{row_label}{ci+1}"
            val = grid[ri, ci]
            color = "white" if val > grid.max() * 0.6 else "black"
            ax.text(ci, ri, f"{zid}\n{val:.1f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="Predicted incidents next week", shrink=0.7)
    ax.set_xticks(range(8)); ax.set_xticklabels([f"{i+1}" for i in range(8)])
    ax.set_yticks(range(8)); ax.set_yticklabels(list("ABCDEFGH"))
    ax.set_title("Predicted Next-Week Incident Count by Micro-Zone", fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, "zone_choropleth.png"), dpi=140)
    plt.close()

    # --- (3) Temporal matrix (weekday x hour) ---
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    df["weekday_cat"] = pd.Categorical(df["timestamp"].dt.day_name(),
        categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], ordered=True)
    pivot = df.pivot_table(index="weekday_cat", columns="hour", values="incident_id",
                           aggfunc="count", fill_value=0, observed=False)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_yticks(range(7)); ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Incidents (6 mo total)", shrink=0.8)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Day of week")
    ax.set_title("Spatio-temporal Pattern: Incidents by Day-of-Week x Hour", fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, "temporal_matrix.png"), dpi=140)
    plt.close()

    # --- (4) Crime type pie ---
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    counts = df["crime_type"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90, textprops={"fontsize": 8})
    ax.set_title("Crime Type Distribution (6 months)", fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, "crime_type_pie.png"), dpi=140)
    plt.close()

    # --- (5) Accuracy chart (predicted vs actual per zone) ---
    acc = preds["accuracy_scorecard"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pz = sorted(acc["per_zone"], key=lambda x: x["actual"], reverse=True)[:15]
    zids = [p["zone_id"] for p in pz]
    actual = [p["actual"] for p in pz]
    predicted = [p["predicted"] for p in pz]
    x = np.arange(len(zids))
    w = 0.4
    ax.bar(x - w/2, actual, w, label="Actual", color="#1a3a5c")
    ax.bar(x + w/2, predicted, w, label="Predicted", color="#e07b39")
    ax.set_xticks(x); ax.set_xticklabels(zids, fontsize=8)
    ax.set_ylabel("Incident count")
    ax.set_title(f"Last-Week Prediction Accuracy ({acc['prediction_window']})\n"
                 f"Top-5 overlap: {acc['top5_overlap_count']}/5  |  MAE/zone: {acc['mae_per_zone']}  |  Rank-corr: {acc['rank_correlation']}",
                 fontsize=10)
    ax.legend()
    plt.savefig(os.path.join(IMG_DIR, "accuracy_chart.png"), dpi=140)
    plt.close()
    print("[heatmaps] Generated 5 PNG heatmaps + 3 correlation matrix PNGs in static/img/")


def generate_correlation_heatmaps():
    """Generate correlation matrix heatmaps: zone-zone, crime-crime, lag cascade bar chart."""
    plt = _setup_matplotlib()
    os.makedirs(IMG_DIR, exist_ok=True)
    preds = load_predictions()
    df = load_df()
    corr = preds.get("correlations", {})
    if not corr:
        print("[heatmaps] No correlation data found, skipping correlation PNGs")
        return

    # --- (6) Zone-zone correlation matrix heatmap ---
    zone_corr_data = corr.get("zone_zone_correlation", {})
    matrix = zone_corr_data.get("matrix", {})
    active_zones = zone_corr_data.get("active_zones", [])
    if matrix and len(active_zones) >= 2:
        # Rebuild matrix as DataFrame
        zc_df = pd.DataFrame(matrix).reindex(index=active_zones, columns=active_zones)
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        im = ax.imshow(zc_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(active_zones)))
        ax.set_xticklabels(active_zones, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(active_zones)))
        ax.set_yticklabels(active_zones, fontsize=8)
        # Annotate cells with correlation value
        for i in range(len(active_zones)):
            for j in range(len(active_zones)):
                v = zc_df.values[i, j]
                if pd.notna(v):
                    color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)
        plt.colorbar(im, ax=ax, label="Pearson r (daily incident counts)", shrink=0.7)
        ax.set_title("Zone-Zone Correlation Matrix\n(Red = spike together, Blue = inverse)", fontsize=11)
        plt.savefig(os.path.join(IMG_DIR, "zone_correlation_heatmap.png"), dpi=140)
        plt.close()
    else:
        print("[heatmaps] Not enough active zones for zone-zone correlation heatmap")

    # --- (7) Crime-crime correlation matrix heatmap ---
    ct_corr_data = corr.get("crime_crime_correlation", {})
    ct_matrix = ct_corr_data.get("matrix", {})
    if ct_matrix:
        # Rebuild - keys are crime types
        ct_df = pd.DataFrame(ct_matrix)
        # Reorder consistently
        ct_order = [c for c in df["crime_type"].unique() if c in ct_df.index]
        ct_df = ct_df.reindex(index=ct_order, columns=ct_order)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(ct_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ct_order)))
        ax.set_xticklabels(ct_order, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(ct_order)))
        ax.set_yticklabels(ct_order, fontsize=8)
        for i in range(len(ct_order)):
            for j in range(len(ct_order)):
                v = ct_df.values[i, j]
                if pd.notna(v):
                    color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, label="Pearson r (daily counts)", shrink=0.7)
        ax.set_title("Crime-Crime Correlation Matrix\n(Red = co-occur, Blue = inverse)", fontsize=11)
        plt.savefig(os.path.join(IMG_DIR, "crime_correlation_heatmap.png"), dpi=140)
        plt.close()
    else:
        print("[heatmaps] No crime-crime correlation data")

    # --- (8) Lag cascade horizontal bar chart (always generate, even if empty) ---
    lags = corr.get("lag_cascades", [])
    fig, ax = plt.subplots(figsize=(10, max(3, len(lags) * 0.4 + 2)), constrained_layout=True)
    if lags:
        labels = [f"{l['crime_a']} -> {l['crime_b']} (+{l['lag_days']}d)" for l in lags]
        values = [l["correlation"] for l in lags]
        colors = ["#d9534f" if v > 0 else "#5bc0de" for v in values]
        y = range(len(labels))
        ax.barh(y, values, color=colors)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Pearson r (daily count today vs count N days later)")
        ax.set_xlim(-1, 1)
        ax.set_title("Crime Lag Cascades\n(Does crime A today predict crime B N days later?)", fontsize=11)
        for i, v in enumerate(values):
            ax.text(v + (0.02 if v >= 0 else -0.02), i, f"r={v:.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No lag cascades found (|r| >= 0.15)\n\nThis means no crime type today\npredicts another crime type N days later\nwith statistical significance.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#888",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f9f9f9", edgecolor="#ccc"))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Crime Lag Cascades - No Significant Patterns Detected", fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, "lag_cascade_chart.png"), dpi=140)
    plt.close()
    print("[heatmaps] Correlation PNGs generated: zone_correlation_heatmap.png, crime_correlation_heatmap.png, lag_cascade_chart.png")


# ============================================================
# SECTION 6: FOLIUM INTERACTIVE MAP BUILDER (from build_folium_map.py)
# ============================================================

def build_folium_map():
    """Build the interactive Leaflet map with Google Maps tiles (no API key needed).

    Base layers use Google's public tile servers (mt0-mt3.google.com), which serve the
    same tiles you see on maps.google.com. No API key, no billing, no setup:
    - Google Streets / Satellite / Hybrid / Terrain (switchable, load-balanced)
    - Google Traffic (live): the same street map with Google's live congestion layer
      painted on each road (green = fast, orange = moderate, red = slow)
    - OpenStreetMap (backup in case Google's tile server is unreachable)

    Data overlays on top:
    - 4 toggleable severity-weighted heatmap layers (6-month, last-4-weeks, property
      crime, violent/other crime) with live radius/blur/intensity/gradient controls
    - Top-5 predicted hotspots; popups list correlated zones + linked crime patterns
    - Correlation overlay drawn from the mock-data correlation analysis: red lines
      connect zones whose daily incident counts spike together, dashed blue lines
      connect inversely-correlated zones, plus an avg-correlation shading layer
    - Micro-zone grid (64 cells) and recent 4-week incident cluster
    """
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    df = load_df()
    preds = load_predictions()
    zones = GRID_CELLS
    center = preds["district_center"]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)

    # ---- Base layers: Google Maps WITHOUT an API key (public mt{0-3}.google.com tiles) ----
    # "Google Traffic (live)" uses Google's traffic style modifier (lyrs=m@<style>,traffic)
    # which paints every road with live congestion colours - the same layer you see when
    # you enable the Traffic view on maps.google.com, still with no API key.
    base_layers = [
        ("Google Streets",   "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
         "&copy; Google", "0123"),
        ("Google Traffic (live)",
         "https://mt{s}.google.com/vt?lyrs=m@221097413,traffic&x={x}&y={y}&z={z}",
         "&copy; Google", "0123"),
        ("Google Satellite", "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
         "&copy; Google", "0123"),
        ("Google Hybrid",    "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
         "&copy; Google", "0123"),
        ("Google Terrain",   "https://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
         "&copy; Google", "0123"),
        ("OpenStreetMap (backup)", "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
         "&copy; OpenStreetMap contributors", "abc"),
    ]
    for i, (name, url, attr, subdoms) in enumerate(base_layers):
        folium.TileLayer(
            tiles=url, attr=attr, name=name, overlay=False, control=True,
            show=(name == "Google Traffic (live)"),  # live-traffic view by default
            subdomains=subdoms,
        ).add_to(m)

    # ---- Correlation analysis results (computed from the mock data) ----
    corr = preds.get("correlations", {}) or {}
    zc = corr.get("zone_zone_correlation", {}) or {}
    matrix = zc.get("matrix", {}) or {}
    zone_center = {z["zone_id"]: (z["lat_center"], z["lng_center"]) for z in zones}

    def correlated_zones_for(zid, top_n=3, min_r=0.25):
        """Top zones whose daily incident counts correlate with this zone."""
        partners = []
        for zb, inner in matrix.items():
            if zb == zid or not isinstance(inner, dict):
                continue
            v = inner.get(zid)
            if v is not None and pd.notna(v) and abs(float(v)) >= min_r:
                partners.append((zb, float(v)))
        partners.sort(key=lambda t: abs(t[1]), reverse=True)
        return partners[:top_n]

    crime_corr = corr.get("crime_crime_correlation", {}) or {}
    lag_list = corr.get("lag_cascades", []) or []

    def crime_notes_for(zone_feat, top_n=3):
        """Co-occurrence / lag-cascade notes involving this zone's dominant crime types."""
        dom = [ct for ct, _ in zone_feat.get("dominant_crime_types", [])]
        notes = []
        for l in lag_list:
            if l["crime_a"] in dom or l["crime_b"] in dom:
                notes.append(f"{l['crime_a']} today &rarr; {l['crime_b']} +{l['lag_days']}d "
                             f"(r={l['correlation']:+.2f})")
        for p in crime_corr.get("top_pairs", []):
            if p["crime_a"] in dom or p["crime_b"] in dom:
                notes.append(f"{p['crime_a']} &harr; {p['crime_b']} (r={p['correlation']:+.2f})")
        return notes[:top_n]

    # ---- Heat layers: severity-weighted [lat, lng, weight] points ----
    HEAT_GRADIENT = {
        0.15: "#313695", 0.30: "#4575b4", 0.45: "#74add1", 0.55: "#abd9e9",
        0.65: "#e0f3f8", 0.75: "#fee090", 0.85: "#fdae61", 0.95: "#f46d43",
        1.0: "#a50026",
    }

    def _heat_points(sub_df):
        """[lat, lng, weight] points; weight = severity/5 so severe clusters glow harder."""
        if len(sub_df) == 0:
            return []
        lat = pd.to_numeric(sub_df["latitude"], errors="coerce")
        lng = pd.to_numeric(sub_df["longitude"], errors="coerce")
        sev = pd.to_numeric(sub_df["severity"], errors="coerce").fillna(2.0).clip(1, 5)
        ok = lat.notna() & lng.notna()
        w = (sev[ok] / 5.0).astype(float)
        return [[float(a), float(b), float(c)] for a, b, c in zip(lat[ok], lng[ok], w)]

    pred_start = datetime.strptime(preds["prediction_week_start"], "%Y-%m-%d")
    recent = df[(df["timestamp"] >= pred_start - timedelta(days=28)) & (df["timestamp"] < pred_start)]
    property_mask = df["crime_type"].isin(
        ["Theft", "Chain Snatching", "Vehicle Theft", "Burglary", "Pickpocketing"])

    heat_defs = [
        ("Heat: all incidents (6 mo, severity-weighted)", _heat_points(df), True),
        ("Heat: last 4 weeks (current risk)", _heat_points(recent), False),
        ("Heat: property crime (theft family)", _heat_points(df[property_mask]), False),
        ("Heat: violent / other crime", _heat_points(df[~property_mask]), False),
    ]
    for name, pts, show in heat_defs:
        if not pts:
            continue
        group = folium.FeatureGroup(name=name, show=show)
        HeatMap(pts, radius=14, blur=20, min_opacity=0.25, max_zoom=16,
                gradient=HEAT_GRADIENT).add_to(group)
        group.add_to(m)

    # ---- Correlation overlay: lines between zones that spike together ----
    top_pairs = zc.get("top_pairs", [])
    if top_pairs:
        corr_group = folium.FeatureGroup(name="Zone correlation links (spike together)")
        for p in top_pairs:
            a, b = p["zone_a"], p["zone_b"]
            if a not in zone_center or b not in zone_center:
                continue
            r = float(p["correlation"])
            positive = r >= 0
            folium.PolyLine(
                locations=[zone_center[a], zone_center[b]],
                color="#d9534f" if positive else "#3b7dd8",
                weight=max(2, min(9, 1.5 + abs(r) * 9)),
                opacity=0.75,
                dash_array=None if positive else "6,8",
                tooltip=(f"Zone {a} <-> Zone {b}: r = {r:+.2f} "
                         + ("(spike together)" if positive else "(inverse)")),
                popup=folium.Popup(p.get("interpretation", ""), max_width=280),
            ).add_to(corr_group)
        corr_group.add_to(m)

    # ---- Correlation overlay: zones shaded by average correlation strength ----
    if matrix:
        avg_r = {}
        for za, inner in matrix.items():
            if not isinstance(inner, dict):
                continue
            vals = [abs(float(v)) for zb, v in inner.items()
                    if zb != za and v is not None and pd.notna(v)]
            if vals:
                avg_r[za] = sum(vals) / len(vals)
        if avg_r:
            max_avg = max(avg_r.values()) or 1.0
            strength_group = folium.FeatureGroup(
                name="Zones shaded by correlation strength (avg |r|)", show=False)
            for zid, ar in avg_r.items():
                zcell = next((zz for zz in zones if zz["zone_id"] == zid), None)
                if zcell is None:
                    continue
                folium.Rectangle(
                    bounds=[[zcell["lat_min"], zcell["lng_min"]],
                            [zcell["lat_max"], zcell["lng_max"]]],
                    color="#d9534f", weight=1.2, fill=True, fill_color="#d9534f",
                    fill_opacity=0.08 + 0.45 * (ar / max_avg),
                    tooltip=f"Zone {zid}: avg |r|={ar:.2f} with other active zones",
                ).add_to(strength_group)
            strength_group.add_to(m)

    # ---- Top-5 predicted hotspots with rich popups (incl. correlations) ----
    top5_group = folium.FeatureGroup(name="Top-5 predicted next-week hotspots")
    for i, z in enumerate(preds["top5"], 1):
        rank_color = "red" if i == 1 else ("orange" if i <= 2 else "blue")
        rationale = preds["rationale_texts"].get(z["zone_id"], "")
        partners = correlated_zones_for(z["zone_id"])
        corr_str = (", ".join(f"Zone {b} ({r:+.2f})" for b, r in partners)
                    if partners else "none detected")
        notes = crime_notes_for(z)
        notes_html = ""
        if notes:
            notes_html = (
                '<div style="font-size:11px; line-height:1.4; color:#7a1f1f; margin-top:6px;">'
                "<b>Linked crime patterns (correlation analysis):</b><br>"
                + "<br>".join("&bull; " + n for n in notes) + "</div>"
            )
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 260px;">
            <h4 style="margin:0 0 6px 0; color:#1a3a5c;">Zone {z['zone_id']} - Rank #{i}</h4>
            <table style="font-size:12px; border-collapse: collapse;">
                <tr><td><b>Predicted incidents (next wk):</b></td><td>{z['predicted_incidents_next_week']:.1f}</td></tr>
                <tr><td><b>Probability mass:</b></td><td>{z['probability']*100:.1f}%</td></tr>
                <tr><td><b>Incidents last 4 wks:</b></td><td>{z['incidents_last_4wk']}</td></tr>
                <tr><td><b>4-wk trend:</b></td><td>{z['trend_pct']:+.1f}%</td></tr>
                <tr><td><b>Peak window:</b></td><td>{z['peak_window'] or '-'}</td></tr>
                <tr><td><b>Dominant day:</b></td><td>{z['dominant_dow'] or '-'}</td></tr>
                <tr><td><b>Top crime types:</b></td><td>{', '.join(f'{c}({n})' for c,n in z['dominant_crime_types'])}</td></tr>
                <tr><td><b>Event flags:</b></td><td>{', '.join(z['event_flags']) if z['event_flags'] else '-'}</td></tr>
                <tr><td><b>Correlated zones:</b></td><td>{corr_str}</td></tr>
            </table>
            <hr style="margin:8px 0;">
            <div style="font-size:11px; line-height:1.4; color:#333;">{rationale}</div>
            {notes_html}
        </div>
        """
        folium.CircleMarker(
            location=[z["lat_center"], z["lng_center"]],
            radius=18 + (6-i)*2, color=rank_color, fill=True,
            fill_color=rank_color, fill_opacity=0.6, weight=3,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"#{i} Zone {z['zone_id']} - {z['predicted_incidents_next_week']:.1f} predicted",
        ).add_to(top5_group)
    top5_group.add_to(m)

    # ---- Micro-zone grid (rectangles + labels) ----
    grid_group = folium.FeatureGroup(name="Micro-zone grid (64 cells)")
    for z in zones:
        folium.Rectangle(
            bounds=[[z["lat_min"], z["lng_min"]], [z["lat_max"], z["lng_max"]]],
            color="#1a3a5c", weight=0.8, fill=False, opacity=0.4,
        ).add_to(grid_group)
        folium.map.Marker(
            location=[z["lat_center"], z["lng_center"]],
            icon=folium.DivIcon(icon_size=(20, 20), icon_anchor=(10, 10),
                html=f'<div style="font-size:8px; color:#1a3a5c; opacity:0.7; text-align:center;">{z["zone_id"]}</div>'),
        ).add_to(grid_group)
    grid_group.add_to(m)

    # ---- Recent incidents cluster (off by default - toggle for drill-down) ----
    cluster_group = folium.FeatureGroup(name="Recent 4-week incidents (clustered)", show=False)
    cluster = MarkerCluster().add_to(cluster_group)
    for _, row in recent.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 11px;">
            <b>{row['incident_id']}</b><br>
            Type: {row['crime_type']}<br>Severity: {row['severity']}/5<br>
            When: {row['timestamp']}<br>Zone: {row['zone_id']}<br>Reported via: {row['reporting_mode']}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]], radius=4,
            color="black", fill=True,
            fill_color="red" if row["severity"] >= 4 else ("orange" if row["severity"] == 3 else "gray"),
            fill_opacity=0.7, weight=0.6,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(cluster)
    cluster_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ---- Legend overlay (bottom-right, above attribution) ----
    legend_html = """
    <div style="position:fixed; z-index:9999; right:10px; bottom:45px;
                background:white; padding:10px 12px; border:1px solid #999;
                border-radius:8px; font-family:Arial; font-size:11px; color:#222;
                box-shadow:2px 2px 8px rgba(0,0,0,0.2); line-height:1.6;">
        <div style="font-weight:bold; margin-bottom:4px;">Legend</div>
        <div style="background:linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #fee090, #fdae61, #f46d43, #a50026); height:10px; width:180px; border:1px solid #ccc; border-radius:3px;"></div>
        <div style="display:flex; justify-content:space-between; width:180px; color:#666;">
            <span>low</span><span>density &times; severity</span><span>high</span>
        </div>
        <div style="margin-top:6px;">
            <span style="display:inline-block; width:24px; border-top:3px solid #d9534f; vertical-align:middle; margin-right:5px;"></span>
            zones spike together (r &gt; 0)
        </div>
        <div>
            <span style="display:inline-block; width:24px; border-top:3px dashed #3b7dd8; vertical-align:middle; margin-right:5px;"></span>
            inverse correlation (r &lt; 0)
        </div>
        <div>
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; border:3px solid #d9534f; vertical-align:middle; margin-right:5px;"></span>
            top-5 predicted hotspot
        </div>
        <div style="margin-top:6px; border-top:1px solid #ddd; padding-top:5px;">
            <span style="font-weight:bold;">Live traffic</span>
            <span style="color:#666;">(Google Traffic base layer)</span>
        </div>
        <div>
            <span style="display:inline-block; width:24px; height:4px; background:#22a45d; vertical-align:middle; margin-right:5px; border-radius:2px;"></span>fast
            &nbsp;<span style="display:inline-block; width:24px; height:4px; background:#f5a623; vertical-align:middle; margin-right:5px; border-radius:2px;"></span>moderate
        </div>
        <div style="margin-top:2px;">
            <span style="display:inline-block; width:24px; height:4px; background:#e0453a; vertical-align:middle; margin-right:5px; border-radius:2px;"></span>slow
            &nbsp;<span style="display:inline-block; width:24px; height:4px; background:#8b1a1a; vertical-align:middle; margin-right:5px; border-radius:2px;"></span>stopped
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---- Title overlay (top-left, clear of the zoom + layer controls) ----
    title_html = f"""
    <div style="position:fixed; z-index:9999; left:60px; top:10px;
                background:white; padding:10px 14px; border:2px solid #1a3a5c;
                border-radius:8px; font-family:Arial; font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.2); max-width:430px;">
        <div style="font-weight:bold; color:#1a3a5c; font-size:13px; margin-bottom:4px;">
            BOB Crime Intelligence - {preds['district_name']}
        </div>
        <div>Prediction window: <b>{preds['prediction_week_start']} &rarr; {preds['prediction_week_end']}</b></div>
        <div>Top-5 zones: {', '.join(z['zone_id'] for z in preds['top5'])}</div>
        <div style="font-size:10px; color:#666; margin-top:4px;">
            Switch Google base layers + overlays (top-right). Red/blue lines = zone
            correlations. Tune the heatmap with the panel at the bottom-left.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ---- Live heatmap controls (radius / blur / intensity / opacity / gradient) ----
    controls_js = """
<script>
(function () {
  function findLeafletMap() {
    var found = null;
    Object.keys(window).forEach(function (k) {
      try {
        if (window[k] && window.L && window[k] instanceof window.L.Map) { found = window[k]; }
      } catch (e) { /* ignore */ }
    });
    return found;
  }

  window.addEventListener('load', function () {
    try {
      var map = findLeafletMap();
      if (!map) { return; }

      var GRADIENTS = {
        'Cold -> Hot (default)': {0.15:'#313695',0.30:'#4575b4',0.45:'#74add1',0.55:'#abd9e9',0.65:'#e0f3f8',0.75:'#fee090',0.85:'#fdae61',0.95:'#f46d43',1.0:'#a50026'},
        'Classic heat': {0.2:'blue',0.4:'lime',0.6:'yellow',0.8:'orange',1.0:'red'},
        'Fire': {0.2:'#000004',0.4:'#57106e',0.6:'#bc3754',0.8:'#f98e09',1.0:'#fcffa4'},
        'Traffic light': {0.25:'#1a9641',0.5:'#ffffbf',0.75:'#fdae61',1.0:'#d7191c'}
      };

      function heatLayers() {
        var out = [];
        Object.keys(map._layers).forEach(function (id) {
          var l = map._layers[id];
          if (l && l._latlngs && l.options &&
              typeof l.options.radius === 'number' && typeof l.options.blur === 'number') {
            out.push(l);
          }
        });
        return out;
      }

      function refresh(l) {
        // Rebuild simpleheat's cached blurred-circle sprite + gradient, then redraw
        if (l._heat && typeof l._heat.radius === 'function') {
          l._heat.radius(l.options.radius, l.options.blur);
          l._heat.gradient(l.options.gradient);
          l._heat.max(l.options.max === undefined ? 1 : l.options.max);
        }
        if (typeof l._reset === 'function') { l._reset(); }
        else if (typeof l._redraw === 'function') { l._redraw(); }
        else if (typeof l.redraw === 'function') { l.redraw(); }
      }

      var panel = document.createElement('div');
      panel.style.cssText = 'position:fixed;z-index:9999;left:10px;bottom:80px;' +
        'background:rgba(255,255,255,0.96);padding:10px 12px;border:1px solid #999;' +
        'border-radius:8px;font-family:Arial;font-size:11px;color:#222;width:180px;' +
        'box-shadow:2px 2px 8px rgba(0,0,0,0.2);';
      panel.innerHTML =
        '<div style="font-weight:bold;margin-bottom:6px;">Heatmap controls</div>' +
        'Radius <span id="hc-rv">14</span><br>' +
        '<input id="hc-radius" type="range" min="5" max="45" value="14" style="width:100%;">' +
        'Blur <span id="hc-bv">20</span><br>' +
        '<input id="hc-blur" type="range" min="0" max="45" value="20" style="width:100%;">' +
        'Intensity <span id="hc-iv">1.0</span><br>' +
        '<input id="hc-intensity" type="range" min="0.3" max="2" step="0.1" value="1" style="width:100%;">' +
        'Opacity <span id="hc-ov">0.25</span><br>' +
        '<input id="hc-opacity" type="range" min="0" max="0.9" step="0.05" value="0.25" style="width:100%;">' +
        'Gradient<br>' +
        '<select id="hc-grad" style="width:100%;font-size:11px;">' +
        Object.keys(GRADIENTS).map(function (k) { return '<option>' + k + '</option>'; }).join('') +
        '</select>';
      document.body.appendChild(panel);

      document.getElementById('hc-radius').addEventListener('input', function () {
        var v = +this.value;
        document.getElementById('hc-rv').textContent = v;
        heatLayers().forEach(function (l) { l.options.radius = v; refresh(l); });
      });
      document.getElementById('hc-blur').addEventListener('input', function () {
        var v = +this.value;
        document.getElementById('hc-bv').textContent = v;
        heatLayers().forEach(function (l) { l.options.blur = v; refresh(l); });
      });
      document.getElementById('hc-intensity').addEventListener('input', function () {
        var v = +this.value;
        document.getElementById('hc-iv').textContent = v.toFixed(1);
        heatLayers().forEach(function (l) { l.options.max = v; refresh(l); });
      });
      document.getElementById('hc-opacity').addEventListener('input', function () {
        var v = +this.value;
        document.getElementById('hc-ov').textContent = v.toFixed(2);
        heatLayers().forEach(function (l) { l.options.minOpacity = v; refresh(l); });
      });
      document.getElementById('hc-grad').addEventListener('change', function () {
        var g = GRADIENTS[this.value];
        heatLayers().forEach(function (l) { l.options.gradient = g; refresh(l); });
      });
    } catch (e) {
      if (window.console) { console.warn('Heatmap controls unavailable:', e); }
    }
  });
})();
</script>
"""
    m.get_root().html.add_child(folium.Element(controls_js))

    os.makedirs(STATIC_DIR, exist_ok=True)
    out_path = os.path.join(STATIC_DIR, "heatmap.html")
    m.save(out_path)
    print(f"[folium] Saved interactive map to {out_path}")
    print(f"[folium] Google base layers: Streets / Traffic (live) / Satellite / Hybrid / Terrain (no API key) + OSM backup")
    print(f"[folium] Heat layers: {sum(1 for _, pts, _ in heat_defs if pts)} | "
          f"correlation links: {len([p for p in top_pairs if p['zone_a'] in zone_center and p['zone_b'] in zone_center])}")


# ============================================================
# SECTION 7: EXCEL TEMPLATE GENERATOR (from make_template.py)
# ============================================================

def make_excel_template():
    """Generate static/template_incidents.xlsx with 3 sample rows showing the schema."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    samples = [
        {"incident_id": "INC-SAMPLE-001", "timestamp": "2025-09-01 22:30:00", "date": "2025-09-01",
         "hour": 22, "weekday": "Monday", "latitude": 28.6315, "longitude": 77.2167,
         "zone_id": "D4", "crime_type": "Theft", "severity": 2, "reporting_mode": "FIR"},
        {"incident_id": "INC-SAMPLE-002", "timestamp": "2025-09-02 08:15:00", "date": "2025-09-02",
         "hour": 8, "weekday": "Tuesday", "latitude": 28.6350, "longitude": 77.2230,
         "zone_id": "B6", "crime_type": "Chain Snatching", "severity": 3, "reporting_mode": "PCR Call"},
        {"incident_id": "INC-SAMPLE-003", "timestamp": "2025-09-03 02:00:00", "date": "2025-09-03",
         "hour": 2, "weekday": "Wednesday", "latitude": 28.6260, "longitude": 77.2110,
         "zone_id": "F3", "crime_type": "Vehicle Theft", "severity": 3, "reporting_mode": "Patrol Observed"},
    ]
    df = pd.DataFrame(samples)
    out_path = os.path.join(STATIC_DIR, "template_incidents.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[template] Saved Excel template to {out_path}")


# ============================================================
# SECTION 8: PIPELINE ORCHESTRATOR
# ============================================================

def init_data_if_missing():
    """Auto-generate mock data + metadata if no data file exists yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(data_file_path()):
        print("[init] No data file found - generating 6 months of mock crime data...")
        incs = generate_mock_data()
        df = pd.DataFrame(incs)
        save_data(df)
        save_meta()
        print(f"[init] Generated {len(df)} incidents across {df['zone_id'].nunique()} zones")
    else:
        # Make sure metadata exists too
        if not os.path.exists(os.path.join(DATA_DIR, "zones.json")):
            save_meta()
            print("[init] Saved zones.json + events.json (metadata)")


def run_full_pipeline():
    """Run the full pipeline: predictions -> heatmaps -> Folium map.
    Assumes data file already exists (run init_data_if_missing first)."""
    run_prediction_engine()
    generate_static_heatmaps()
    generate_correlation_heatmaps()
    build_folium_map()
    make_excel_template()
    print("[pipeline] Full pipeline complete.")


def regenerate_all():
    """Drop existing data, regenerate mock data, run pipeline."""
    print("[regenerate] Regenerating mock data + running full pipeline...")
    incs = generate_mock_data()
    df = pd.DataFrame(incs)
    save_data(df)
    save_meta()
    print(f"[regenerate] Generated {len(df)} incidents across {df['zone_id'].nunique()} zones")
    run_full_pipeline()


# ============================================================
# SECTION 9: FLASK WEBAPP
# ============================================================

# Defer Flask import until needed (so --init works without Flask installed if user just wants data)
def create_app():
    """Build the Flask app with all routes."""
    from flask import (Flask, render_template_string, send_from_directory,
                       jsonify, request, redirect)

    app = Flask(__name__, static_folder=STATIC_DIR)
    app.secret_key = "bob-crime-intel-dev"

    # ---- Base HTML template ----
    BASE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - BOB Crime Intelligence</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f8; color:#222; }}
  header {{ background:#1a3a5c; color:white; padding:14px 28px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 4px rgba(0,0,0,0.15); }}
  header h1 {{ margin:0; font-size:20px; font-weight:600; }}
  header .sub {{ font-size:12px; opacity:0.85; margin-top:2px; }}
  nav {{ background:#22466a; padding:0 28px; }}
  nav a {{ color:#e6edf3; text-decoration:none; padding:12px 16px; display:inline-block; font-size:13px; border-bottom:3px solid transparent; }}
  nav a:hover, nav a.active {{ background:#1a3a5c; border-bottom-color:#e07b39; }}
  main {{ padding:24px 28px; max-width:1400px; margin:0 auto; }}
  .card {{ background:white; border-radius:8px; padding:18px 22px; margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .card h2 {{ margin:0 0 12px 0; color:#1a3a5c; font-size:17px; border-bottom:2px solid #e0e0e0; padding-bottom:8px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }}
  .stat {{ background:#1a3a5c; color:white; border-radius:8px; padding:16px 20px; }}
  .stat .label {{ font-size:11px; text-transform:uppercase; letter-spacing:0.5px; opacity:0.8; }}
  .stat .value {{ font-size:26px; font-weight:bold; margin-top:4px; }}
  .top5-item {{ border-left:4px solid #e07b39; padding:12px 14px; background:#f9f9f9; border-radius:4px; margin-bottom:10px; }}
  .top5-item.r1 {{ border-left-color:#d9534f; }}
  .top5-item.r2 {{ border-left-color:#e07b39; }}
  .top5-item.r3 {{ border-left-color:#f0ad4e; }}
  .top5-item.r4 {{ border-left-color:#5bc0de; }}
  .top5-item.r5 {{ border-left-color:#5cb85c; }}
  .top5-item .head {{ display:flex; justify-content:space-between; align-items:center; }}
  .top5-item .rank {{ font-weight:bold; color:#1a3a5c; }}
  .top5-item .pred {{ font-size:13px; color:#444; }}
  .rationale {{ font-size:12px; color:#555; margin-top:6px; line-height:1.5; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; margin-left:6px; }}
  .badge.risk-high {{ background:#f8d7da; color:#721c24; }}
  .badge.risk-med {{ background:#fff3cd; color:#856404; }}
  .badge.risk-low {{ background:#d4edda; color:#155724; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #e0e0e0; }}
  th {{ background:#1a3a5c; color:white; font-weight:600; }}
  tr:nth-child(even) {{ background:#f7f9fb; }}
  .brief {{ font-family: 'Courier New', monospace; font-size:13px; white-space:pre-wrap; line-height:1.5; background:#fffef0; padding:24px 28px; border:1px solid #d4d4aa; border-radius:4px; }}
  .warn {{ background:#fff3cd; border:1px solid #ffeaa7; color:#856404; padding:10px 14px; border-radius:4px; margin:10px 0; font-size:13px; }}
  .ok {{ background:#d4edda; border:1px solid #b8e0c2; color:#155724; padding:10px 14px; border-radius:4px; margin:10px 0; font-size:13px; }}
  img {{ max-width:100%; height:auto; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.1); }}
  footer {{ text-align:center; padding:14px; color:#888; font-size:11px; }}
  .map-container {{ border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.15); }}
  .map-container iframe {{ display:block; width:100%; height:720px; border:0; }}
  .two-col {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  form input[type=file] {{ padding:10px; border:1px solid #ccc; border-radius:4px; width:100%; margin:8px 0; }}
  form button {{ background:#1a3a5c; color:white; padding:10px 22px; border:0; border-radius:4px; font-size:14px; cursor:pointer; }}
  form button:hover {{ background:#22466a; }}
  pre.code {{ background:#1a1a2e; color:#e6e6e6; padding:14px; border-radius:6px; overflow-x:auto; font-size:12px; line-height:1.5; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>BOB - Crime Intelligence Assistant</h1>
    <div class="sub">{district} | Prediction window: {pred_window}</div>
  </div>
  <div style="text-align:right; font-size:11px;">
    <div>Generated: {generated}</div>
    <div>Engine v1.0 - Advisory only</div>
  </div>
</header>
<nav>
  <a href="/" class="{active_home}">Dashboard</a>
  <a href="/map" class="{active_map}">Heatmap</a>
  <a href="/brief" class="{active_brief}">SHO Brief</a>
  <a href="/accuracy" class="{active_acc}">Accuracy</a>
  <a href="/bias" class="{active_bias}">Bias Audit</a>
  <a href="/correlations" class="{active_corr}">Correlations</a>
  <a href="/upload" class="{active_upload}">Upload Data</a>
  <a href="/api/predictions">API: Predictions</a>
  <a href="/api/incidents">API: Incidents</a>
</nav>
<main>
{body}
</main>
<footer>
  BOB Crime Intelligence Assistant v1.0 - Single-file edition.
  Built on Folium/Leaflet + Google Maps tiles (no API key required).
  Predictions are advisory; final deployment decisions rest with the SHO.
</footer>
</body>
</html>"""

    def render(title, body, active=""):
        preds = load_predictions()
        df = load_df()
        return BASE.format(
            title=title,
            district=preds["district_name"],
            pred_window=f"{preds['prediction_week_start']} to {preds['prediction_week_end']}",
            generated=pd.Timestamp.now().strftime("%d %b %Y, %H:%M"),
            active_home="active" if active=="home" else "",
            active_map="active" if active=="map" else "",
            active_brief="active" if active=="brief" else "",
            active_acc="active" if active=="acc" else "",
            active_bias="active" if active=="bias" else "",
            active_corr="active" if active=="corr" else "",
            active_upload="active" if active=="upload" else "",
            body=body,
        )

    # ---- / Dashboard ----
    @app.route("/")
    def home():
        preds = load_predictions()
        df = load_df()
        total_inc = len(df)
        n_zones_active = df["zone_id"].nunique()
        n_crime_types = df["crime_type"].nunique()
        date_range = f"{df['timestamp'].min().strftime('%d %b %Y')} - {df['timestamp'].max().strftime('%d %b %Y')}"
        acc = preds["accuracy_scorecard"]
        bias = preds["bias_audit"]
        top5_html = ""
        for i, z in enumerate(preds["top5"], 1):
            risk_class = "risk-high" if i <= 2 else ("risk-med" if i <= 3 else "risk-low")
            risk_label = "HIGH" if i <= 2 else ("MEDIUM" if i <= 3 else "LOW")
            ct_str = ", ".join(f"{ct} ({n})" for ct, n in z["dominant_crime_types"][:3])
            top5_html += f"""
            <div class="top5-item r{i}">
                <div class="head">
                    <div>
                        <span class="rank">#{i} Zone {z['zone_id']}</span>
                        <span class="badge {risk_class}">{risk_label}</span>
                    </div>
                    <div class="pred">
                        Predicted: <b>{z['predicted_incidents_next_week']:.1f}</b> incidents
                        ({z['probability']*100:.1f}% mass)
                    </div>
                </div>
                <div class="rationale">
                    {preds['rationale_texts'].get(z['zone_id'], '')}
                    <br><span style="color:#888; font-size:11px;">Top crime types: {ct_str}</span>
                </div>
            </div>
            """
        body = f"""
        <div class="card">
            <h2>District Snapshot</h2>
            <div class="grid">
                <div class="stat"><div class="label">Total incidents (6 mo)</div><div class="value">{total_inc}</div></div>
                <div class="stat"><div class="label">Active micro-zones</div><div class="value">{n_zones_active} / 64</div></div>
                <div class="stat"><div class="label">Crime types tracked</div><div class="value">{n_crime_types}</div></div>
                <div class="stat"><div class="label">District avg weekly</div><div class="value">{preds['district_avg_weekly']:.1f}</div></div>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#666;">Data window: {date_range}</div>
        </div>
        <div class="two-col">
            <div class="card">
                <h2>Top-5 At-Risk Micro-Zones (Next 7 Days)</h2>
                {top5_html}
                <div style="margin-top:12px; font-size:11px;">
                    <a href="/brief" style="color:#1a3a5c; font-weight:bold;">View full SHO patrol redeployment brief -></a>
                </div>
            </div>
            <div>
                <div class="card">
                    <h2>Last-Week Prediction Accuracy</h2>
                    <div class="grid" style="grid-template-columns: 1fr 1fr;">
                        <div class="stat"><div class="label">Top-5 overlap</div><div class="value">{acc['top5_overlap_count']}/5</div></div>
                        <div class="stat"><div class="label">MAE per zone</div><div class="value">{acc['mae_per_zone']}</div></div>
                        <div class="stat"><div class="label">Rank correlation</div><div class="value">{acc['rank_correlation'] or 'N/A'}</div></div>
                        <div class="stat"><div class="label">Prediction window</div><div class="value" style="font-size:14px;">{acc['prediction_window']}</div></div>
                    </div>
                    <img src="/static/img/accuracy_chart.png" alt="accuracy chart" style="margin-top:12px;">
                </div>
                <div class="card">
                    <h2>Bias / Fairness Audit</h2>
                    <div class="warn" style="display:{'block' if bias['patrol_coverage_bias_flag'] else 'none'};">{bias['bias_warning']}</div>
                    <div class="ok" style="display:{'block' if not bias['patrol_coverage_bias_flag'] else 'none'};">{bias['bias_warning']}</div>
                    <table style="margin-top:8px;">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Geographic Gini coefficient</td><td>{bias['geographic_gini']}</td></tr>
                        <tr><td>District-wide Patrol-Observed reporting</td><td>{bias['district_reporting_mode_pct'].get('Patrol Observed', 0):.1f}%</td></tr>
                    </table>
                    <div style="margin-top:8px; font-size:12px;">
                        <a href="/bias" style="color:#1a3a5c; font-weight:bold;">View full bias audit -></a>
                    </div>
                </div>
            </div>
        </div>
        <div class="card">
            <h2>Spatial Crime Density Heatmap</h2>
            <img src="/static/img/spatial_heatmap.png" alt="spatial heatmap">
            <div style="margin-top:8px; font-size:12px; color:#666;">
                6-month severity-weighted crime density (KDE) overlaid with 8x8 micro-zone grid. Blue circles mark
                top-5 predicted next-week hotspots. For interactive pan/zoom/click-popup version, see
                <a href="/map" style="color:#1a3a5c; font-weight:bold;">/map</a>.
            </div>
        </div>
        <div class="two-col">
            <div class="card">
                <h2>Temporal Pattern Matrix</h2>
                <img src="/static/img/temporal_matrix.png" alt="temporal matrix">
            </div>
            <div class="card">
                <h2>Crime Type Distribution</h2>
                <img src="/static/img/crime_type_pie.png" alt="crime pie">
            </div>
        </div>
        """
        return render("Dashboard", body, active="home")

    # ---- /map Interactive Folium heatmap ----
    @app.route("/map")
    def map_view():
        body = """
        <div class="card">
            <h2>Interactive Crime Heatmap (Google Maps tiles - no API key)</h2>
            <div class="map-container" style="margin-top:12px;">
                <iframe src="/static/heatmap.html" title="Crime heatmap"></iframe>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#666; line-height:1.6;">
                <b>Base layers:</b> Google Streets / Satellite / Hybrid / Terrain plus
                <b>Google Traffic (live)</b> - the street map with Google's live congestion
                layer on every road (green = fast, orange = moderate, red = slow, dark red =
                stopped; see legend). Public tile servers, no API key; OpenStreetMap as
                fallback - switch top-right.
                <br><b>Heat layers:</b> severity-weighted 6-month heatmap, last-4-weeks current
                risk, property crime, and violent/other crime. Tune radius, blur, intensity,
                opacity and gradient live with the panel at the bottom-left of the map.
                <br><b>Correlation overlays (from mock-data analysis):</b> red lines connect
                zones whose daily incident counts spike together (r &gt; 0), dashed blue lines
                mark inverse correlations (r &lt; 0), and a toggle shades zones by average
                correlation strength. Top-5 hotspot popups list each zone's correlated zones
                and linked crime patterns.
                <br><b>Other layers:</b> top-5 predicted hotspots (click for rationale),
                micro-zone grid (64 cells), recent 4-week incident cluster.
            </div>
        </div>
        """
        return render("Interactive Heatmap", body, active="map")

    # ---- /brief SHO patrol redeployment brief ----
    @app.route("/brief")
    def brief():
        preds = load_predictions()
        body = f"""
        <div class="card">
            <h2>SHO Patrol Redeployment Brief</h2>
            <div style="font-size:12px; margin-bottom:10px; color:#666;">
                Auto-generated weekly advisory. Hand to the Station House Officer before the next shift briefing.
            </div>
            <div class="brief">{preds['sho_brief']}</div>
            <div style="margin-top:14px;">
                <a href="/static/img/zone_choropleth.png" target="_blank" style="font-size:12px;">
                    <img src="/static/img/zone_choropleth.png" alt="zone choropleth" style="max-height:400px;">
                </a>
                <div style="margin-top:6px; font-size:11px; color:#666;">Click image to enlarge. Predicted incidents per micro-zone for next week.</div>
            </div>
        </div>
        """
        return render("SHO Brief", body, active="brief")

    # ---- /accuracy Last-week accuracy scorecard ----
    @app.route("/accuracy")
    def accuracy():
        preds = load_predictions()
        acc = preds["accuracy_scorecard"]
        rows = ""
        for p in sorted(acc["per_zone"], key=lambda x: x["actual"], reverse=True)[:20]:
            err_color = "red" if p["abs_error"] >= 2 else "black"
            rows += f"""
            <tr>
                <td>{p['zone_id']}</td>
                <td>{p['predicted']:.1f}</td>
                <td>{p['actual']}</td>
                <td style="color:{err_color};">{p['abs_error']:.1f}</td>
            </tr>
            """
        pred_set = set(acc["predicted_top5"])
        actual_set = set(acc["actual_top5"])
        overlap_html = ""
        for z in acc["predicted_top5"]:
            color = "green" if z in actual_set else "orange"
            mark = "both" if z in actual_set else "predicted-only"
            overlap_html += f'<span class="badge" style="background:{color}; color:white;">{z} ({mark})</span> '
        for z in actual_set:
            if z not in pred_set:
                overlap_html += f'<span class="badge" style="background:#d9534f; color:white;">{z} (missed)</span> '
        body = f"""
        <div class="card">
            <h2>Last-Week Prediction Accuracy Scorecard</h2>
            <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="stat"><div class="label">Prediction window</div><div class="value" style="font-size:16px;">{acc['prediction_window']}</div></div>
                <div class="stat"><div class="label">Top-5 overlap</div><div class="value">{acc['top5_overlap_count']}/5 ({acc['top5_overlap_pct']}%)</div></div>
                <div class="stat"><div class="label">MAE per zone</div><div class="value">{acc['mae_per_zone']}</div></div>
                <div class="stat"><div class="label">Spearman rank corr</div><div class="value">{acc['rank_correlation'] or 'N/A'}</div></div>
            </div>
        </div>
        <div class="card">
            <h2>Top-5 Overlap Detail</h2>
            <div style="margin-bottom:10px;">{overlap_html}</div>
            <img src="/static/img/accuracy_chart.png" alt="accuracy chart" style="margin-top:12px;">
        </div>
        <div class="card">
            <h2>Per-Zone Predicted vs Actual (Top 20 by actual volume)</h2>
            <table>
                <thead><tr><th>Zone</th><th>Predicted</th><th>Actual</th><th>Abs Error</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        return render("Accuracy Scorecard", body, active="acc")

    # ---- /bias Bias / fairness audit ----
    @app.route("/bias")
    def bias():
        preds = load_predictions()
        bias = preds["bias_audit"]
        top5_zids = [z["zone_id"] for z in preds["top5"]]
        mode_rows = ""
        for zid in top5_zids:
            zmode = bias["top5_reporting_mode_pct"].get(zid, {})
            dmode = bias["district_reporting_mode_pct"]
            row = f"<tr><td>{zid}</td>"
            for mode in ["FIR", "PCR Call", "Patrol Observed"]:
                z_pct = zmode.get(mode, 0)
                d_pct = dmode.get(mode, 0)
                delta = z_pct - d_pct
                color = "red" if abs(delta) > 10 else "black"
                row += f'<td>{z_pct:.1f}%</td><td>{d_pct:.1f}%</td><td style="color:{color};">{delta:+.1f}</td>'
            mode_rows += row + "</tr>"
        sev_rows = ""
        for zid in top5_zids:
            zsev = bias["top5_severity_pct"].get(zid, {})
            row = f"<tr><td>{zid}</td>"
            for s in [1, 2, 3, 4, 5]:
                z_pct = zsev.get(s, zsev.get(str(s), 0))
                row += f"<td>{z_pct:.1f}%</td>"
            sev_rows += row + "</tr>"
        warn_class = "warn" if bias["patrol_coverage_bias_flag"] else "ok"
        body = f"""
        <div class="card">
            <h2>Bias / Fairness Self-Audit</h2>
            <div class="{warn_class}">{bias['bias_warning']}</div>
            <p style="font-size:13px; color:#444;">
                This audit checks whether the model's predicted hotspots may be tracking
                something other than true crime density. Three proxies are used because real
                demographic data isn't available in this prototype:
                <b>(1) reporting-mode mix</b> -- if a zone has far more "Patrol Observed"
                reports than district average, the model may be detecting patrol coverage,
                not crime. <b>(2) severity mix</b> -- if a hotspot is mostly low-severity
                reports, that's a flag. <b>(3) geographic Gini</b> -- a high Gini means
                crime is concentrated; that's expected for crime data, not necessarily a bias.
            </p>
        </div>
        <div class="card">
            <h2>Reporting Mode Mix - Top-5 vs District</h2>
            <table>
                <thead><tr>
                    <th>Zone</th>
                    <th>FIR (zone)</th><th>FIR (district)</th><th>delta</th>
                    <th>PCR (zone)</th><th>PCR (district)</th><th>delta</th>
                    <th>Patrol (zone)</th><th>Patrol (district)</th><th>delta</th>
                </tr></thead>
                <tbody>{mode_rows}</tbody>
            </table>
        </div>
        <div class="card">
            <h2>Severity Mix - Top-5 vs District (Severity 1-5)</h2>
            <table>
                <thead><tr><th>Zone</th><th>Sev 1</th><th>Sev 2</th><th>Sev 3</th><th>Sev 4</th><th>Sev 5</th></tr></thead>
                <tbody>{sev_rows}</tbody>
            </table>
        </div>
        <div class="card">
            <h2>Geographic Concentration</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                <tr><td>Geographic Gini coefficient</td><td>{bias['geographic_gini']}</td>
                    <td>{'High concentration (typical of crime data)' if bias['geographic_gini'] > 0.5 else 'Moderate concentration'}</td></tr>
            </table>
        </div>
        """
        return render("Bias Audit", body, active="bias")

    # ---- /correlations Crime correlation matrices and cascades ----
    @app.route("/correlations")
    def correlations():
        preds = load_predictions()
        corr = preds.get("correlations", {})
        zone_data = corr.get("zone_zone_correlation", {})
        crime_data = corr.get("crime_crime_correlation", {})
        lag_cascades = corr.get("lag_cascades", [])

        # Top zone pairs as table rows
        zone_pairs_rows = ""
        for p in zone_data.get("top_pairs", []):
            sign_color = "#d9534f" if p["correlation"] > 0 else "#5bc0de"
            zone_pairs_rows += f"""
            <tr>
                <td>{p['zone_a']}</td><td>{p['zone_b']}</td>
                <td style="color:{sign_color}; font-weight:bold;">{p['correlation']:+.3f}</td>
                <td>{p['interpretation']}</td>
            </tr>
            """

        # Top crime pairs as table rows
        crime_pairs_rows = ""
        for p in crime_data.get("top_pairs", []):
            sign_color = "#d9534f" if p["correlation"] > 0 else "#5bc0de"
            crime_pairs_rows += f"""
            <tr>
                <td>{p['crime_a']}</td><td>{p['crime_b']}</td>
                <td style="color:{sign_color}; font-weight:bold;">{p['correlation']:+.3f}</td>
                <td>{p['interpretation']}</td>
            </tr>
            """

        # Lag cascade rows
        lag_rows = ""
        for l in lag_cascades:
            sign_color = "#d9534f" if l["correlation"] > 0 else "#5bc0de"
            lag_rows += f"""
            <tr>
                <td>{l['crime_a']}</td><td>{l['crime_b']}</td>
                <td>+{l['lag_days']} day(s)</td>
                <td style="color:{sign_color}; font-weight:bold;">{l['correlation']:+.3f}</td>
                <td>{l['interpretation']}</td>
            </tr>
            """

        body = f"""
        <div class="card">
            <h2>Crime Data Correlation Analysis</h2>
            <p style="font-size:13px; color:#444;">
                This page correlates the mock incident data three ways to surface hidden patterns:
                <b>(1) Zone-Zone correlation</b> - which zones spike on the same days (shared offenders or shared venues);
                <b>(2) Crime-Crime correlation</b> - which crime types co-occur in time;
                <b>(3) Lag Cascades</b> - whether crime A today predicts crime B N days later (cascade / displacement effect).
                Red = positive correlation (co-occur), Blue = negative (inverse).
                Only pairs with |r| &gt;= 0.20 are shown (lag cascades: |r| &gt;= 0.15).
                Zone-Zone correlations are also drawn spatially on the interactive map - see the
                <a href="/map" style="color:#1a3a5c; font-weight:bold;">Heatmap</a> page's
                "Zone correlation links" layer.
            </p>
        </div>

        <div class="card">
            <h2>Zone-Zone Correlation Matrix Heatmap</h2>
            <img src="/static/img/zone_correlation_heatmap.png" alt="zone-zone correlation heatmap">
            <div style="margin-top:8px; font-size:12px; color:#666;">
                Pearson r of daily incident counts across zones. Diagonal = 1.0 (self).
                Red cells = two zones spike on the same days. Active zones only (>=10 incidents in 6 months).
            </div>
        </div>

        <div class="card">
            <h2>Top Zone-Zone Pairs (Spike Together)</h2>
            <table>
                <thead><tr><th>Zone A</th><th>Zone B</th><th>r</th><th>Interpretation</th></tr></thead>
                <tbody>{zone_pairs_rows if zone_pairs_rows else '<tr><td colspan=4 style="color:#888;">No pairs with |r| >= 0.20 found</td></tr>'}</tbody>
            </table>
        </div>

        <div class="card">
            <h2>Crime-Crime Correlation Matrix Heatmap</h2>
            <img src="/static/img/crime_correlation_heatmap.png" alt="crime-crime correlation heatmap">
            <div style="margin-top:8px; font-size:12px; color:#666;">
                Pearson r of daily counts per crime type. Red cells = two crime types spike on the same days
                (e.g., chain snatching + theft both peak on weekends).
            </div>
        </div>

        <div class="card">
            <h2>Top Crime-Crime Pairs (Co-Occur)</h2>
            <table>
                <thead><tr><th>Crime A</th><th>Crime B</th><th>r</th><th>Interpretation</th></tr></thead>
                <tbody>{crime_pairs_rows if crime_pairs_rows else '<tr><td colspan=4 style="color:#888;">No pairs with |r| >= 0.20 found</td></tr>'}</tbody>
            </table>
        </div>

        <div class="card">
            <h2>Lag Cascade Detection (Crime A Today -> Crime B N Days Later)</h2>
            <img src="/static/img/lag_cascade_chart.png" alt="lag cascade chart" style="margin-bottom:14px;">
            <table>
                <thead><tr><th>Crime A (today)</th><th>Crime B (lag)</th><th>Lag</th><th>r</th><th>Interpretation</th></tr></thead>
                <tbody>{lag_rows if lag_rows else '<tr><td colspan=5 style="color:#888;">No lag cascades with |r| >= 0.15 found</td></tr>'}</tbody>
            </table>
            <div style="margin-top:10px; font-size:12px; color:#666;">
                <b>Operational use:</b> If chain-snatching spikes today, expect vehicle theft to spike 2 days later - pre-deploy patrols to known vehicle-theft hotspots in advance.
                Red bars = positive cascade (A predicts B). Blue bars = suppressive (A predicts drop in B).
            </div>
        </div>
        """
        return render("Correlations", body, active="corr")

    # ---- /upload Browser form for data injection ----
    @app.route("/upload", methods=["GET"])
    def upload_form():
        body = """
        <div class="card">
            <h2>Inject Crime Data (Excel or CSV)</h2>
            <p style="font-size:13px; color:#444;">
                Upload an Excel (<code>.xlsx</code>) or CSV (<code>.csv</code>) file of crime incidents.
                The system replaces the current dataset, re-runs the prediction engine, regenerates
                all heatmaps, and rebuilds the interactive map automatically (~5-10 seconds).
            </p>
            <div class="warn">
                <b>Minimum required:</b> a timestamp column and latitude/longitude -
                everything else is optional and auto-filled. Common column aliases are
                recognized automatically (lat/lon/lng, datetime/time/date, type/crime/category,
                sev/weight, zone/beat/sector, case_no/fir_no), in JSON, CSV or Excel -
                casing and spacing don't matter.
                <br><b>Auto-derived:</b> date, hour, weekday (from timestamp); zone_id
                (from coordinates); incident ids. Severity defaults to 2, reporting mode
                to FIR when absent; rows with bad timestamps/coordinates or points outside
                the district grid are dropped and reported back to you.
                <br>Click <a href="/static/template_incidents.xlsx" style="color:#1a3a5c; font-weight:bold;">here</a> to download an Excel template (or see /api/incidents for the current schema).
            </div>
            <form method="POST" action="/api/incidents/upload" enctype="multipart/form-data" style="margin-top:14px;">
                <input type="file" name="file" accept=".xlsx,.csv" required>
                <button type="submit">Upload & Re-run Pipeline</button>
            </form>
        </div>
        <div class="card">
            <h2>API Injection (programmatic)</h2>
            <p style="font-size:13px; color:#444;">
                You can also POST data directly without the browser form. Two formats accepted:
            </p>
            <h3 style="color:#1a3a5c;">Option A - File upload (multipart)</h3>
            <pre class="code">curl -F "file=@my_incidents.xlsx" http://localhost:3000/api/incidents/upload</pre>
            <h3 style="color:#1a3a5c;">Option B - JSON body (list of incidents)</h3>
            <pre class="code">curl -X POST http://localhost:3000/api/incidents/upload \\
  -H "Content-Type: application/json" \\
  -d '[{"incident_id":"INC-X01","timestamp":"2025-09-01 22:30:00","latitude":28.6315,"longitude":77.2167,"zone_id":"D4","crime_type":"Theft","severity":2,"reporting_mode":"FIR"}]'</pre>
        </div>
        <div class="card">
            <h2>Regenerate Mock Data</h2>
            <p style="font-size:13px; color:#444;">
                Replace the current dataset with a fresh 6-month mock dataset (re-runs the full pipeline).
            </p>
            <form method="POST" action="/regenerate" style="margin-top:10px;">
                <button type="submit" style="background:#e07b39;">Regenerate Mock Data</button>
            </form>
        </div>
        """
        return render("Upload Data", body, active="upload")

    # ---- POST /api/incidents/upload - inject Excel/CSV/JSON ----
    @app.route("/api/incidents/upload", methods=["POST"])
    def api_upload():
        """Accept Excel/CSV file upload OR JSON body. Validate first, save, re-run pipeline."""
        try:
            if "file" in request.files:
                f = request.files["file"]
                filename = f.filename.lower()
                if filename.endswith(".xlsx"):
                    df_new = pd.read_excel(f)
                    save_ext = "xlsx"
                elif filename.endswith(".csv"):
                    df_new = pd.read_csv(f)
                    save_ext = "csv"
                else:
                    return jsonify({"error": "Unsupported file type. Use .xlsx or .csv"}), 400
                source = f"file:{filename}"
            elif request.is_json:
                data = request.get_json()
                if isinstance(data, dict) and "incidents" in data:
                    data = data["incidents"]
                if not isinstance(data, list):
                    return jsonify({"error": 'JSON body must be a list of incident dicts (or {"incidents": [...]})'}), 400
                df_new = pd.DataFrame(data)
                save_ext = "csv"
                source = "json-body"
            else:
                return jsonify({"error": "No file or JSON body received"}), 400

            # Normalize any input format onto the canonical schema (aliases, types,
            # missing fields, bad rows). Raises a 400 with a clear message if the
            # input has no usable timestamp + coordinates at all.
            df_new, notes = normalize_uploaded_df(df_new)
            if df_new is None:
                return jsonify({"error": "; ".join(notes)}), 400

            # Save (xlsx takes priority; remove the other)
            csv_path = os.path.join(DATA_DIR, "crime_incidents.csv")
            xlsx_path = os.path.join(DATA_DIR, "crime_incidents.xlsx")
            if save_ext == "xlsx":
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                df_new.to_excel(xlsx_path, index=False)
            else:
                if os.path.exists(xlsx_path):
                    os.remove(xlsx_path)
                df_new.to_csv(csv_path, index=False)

            # Re-run the full pipeline (blocks ~5-10s)
            run_full_pipeline()

            # Reload predictions for response
            preds = load_predictions()
            return jsonify({
                "status": "ok",
                "source": source,
                "rows_ingested": int(len(df_new)),
                "zones_active": int(df_new["zone_id"].nunique()),
                "date_range": [str(df_new["timestamp"].min()), str(df_new["timestamp"].max())],
                "top5_zones": [z["zone_id"] for z in preds["top5"]],
                "accuracy": {
                    "top5_overlap": preds["accuracy_scorecard"]["top5_overlap_count"],
                    "mae_per_zone": preds["accuracy_scorecard"]["mae_per_zone"],
                    "rank_correlation": preds["accuracy_scorecard"]["rank_correlation"],
                },
                "bias": {
                    "geographic_gini": preds["bias_audit"]["geographic_gini"],
                    "patrol_coverage_bias_flag": preds["bias_audit"]["patrol_coverage_bias_flag"],
                },
                "normalization_notes": notes,
                "correlations": {
                    "zone_pairs_on_map": len(preds.get("correlations", {})
                                             .get("zone_zone_correlation", {}).get("top_pairs", [])),
                    "crime_pairs": len(preds.get("correlations", {})
                                       .get("crime_crime_correlation", {}).get("top_pairs", [])),
                    "lag_cascades": len(preds.get("correlations", {}).get("lag_cascades", [])),
                },
                "message": ("Pipeline re-run complete. See updated predictions on the dashboard, "
                            "correlation lines on /map (Zone correlation links layer), and full "
                            "matrices on /correlations."),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- POST /regenerate - regenerate mock data + pipeline ----
    @app.route("/regenerate", methods=["POST"])
    def regenerate_route():
        try:
            regenerate_all()
            return redirect("/")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- API: predictions JSON ----
    @app.route("/api/predictions")
    def api_predictions():
        return jsonify(load_predictions())

    # ---- API: incidents JSON ----
    @app.route("/api/incidents", methods=["GET"])
    def api_incidents():
        df = load_df()
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(df.to_dict(orient="records"))

    # ---- Static files ----
    @app.route("/static/<path:path>")
    def static_files(path):
        return send_from_directory(STATIC_DIR, path)

    return app


# ============================================================
# SECTION 10: CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BOB Crime Intelligence Assistant - All-in-One",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                  # Auto-init + start server on http://localhost:3000
  python app.py --init           # Just generate data + predictions + charts + map (no server)
  python app.py --serve          # Just start server (auto-init if data missing)
  python app.py --regenerate     # Drop existing mock data, regenerate fresh, run pipeline
  python app.py --port 5000      # Start server on a different port
""",
    )
    parser.add_argument("--init", action="store_true",
                        help="Generate mock data + predictions + heatmaps + Folium map, then exit (no server)")
    parser.add_argument("--serve", action="store_true",
                        help="Just start the Flask server (auto-init if data missing)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate mock data + re-run full pipeline (drops existing data)")
    parser.add_argument("--port", type=int, default=3000,
                        help="Port to run the Flask server on (default: 3000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the Flask server to (default: 0.0.0.0)")
    args = parser.parse_args()

    # Handle --init (just generate everything and exit)
    if args.init:
        print("=" * 60)
        print("BOB Crime Intelligence - INIT MODE")
        print("=" * 60)
        init_data_if_missing()
        run_full_pipeline()
        print("\nInit complete. Run `python app.py` to start the server.")
        return

    # Handle --regenerate (drop and regenerate)
    if args.regenerate:
        print("=" * 60)
        print("BOB Crime Intelligence - REGENERATE MODE")
        print("=" * 60)
        regenerate_all()
        print("\nRegenerate complete. Run `python app.py` to start the server.")
        return

    # Default and --serve: start the server (auto-init if needed)
    print("=" * 60)
    print("BOB Crime Intelligence - SERVER MODE")
    print("=" * 60)
    init_data_if_missing()
    # If predictions.json or heatmaps missing, run pipeline once
    if (not os.path.exists(os.path.join(DATA_DIR, "predictions.json"))
        or not os.path.exists(os.path.join(IMG_DIR, "spatial_heatmap.png"))
        or not os.path.exists(os.path.join(STATIC_DIR, "heatmap.html"))):
        print("[server] Predictions or heatmaps missing - running full pipeline...")
        run_full_pipeline()

    # Import Flask (deferred until needed)
    try:
        from flask import Flask  # noqa
    except ImportError:
        print("\nERROR: Flask is not installed. Install dependencies:")
        print("  pip install flask folium pandas numpy scikit-learn matplotlib openpyxl")
        sys.exit(1)

    app = create_app()
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Open this URL in your browser to see the dashboard.")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
