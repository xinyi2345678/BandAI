# streamlit_app.py — No‑RAG version
# -----------------------------------------------------------
from __future__ import annotations
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
from io import BytesIO
import json
import re
import time
import typing as T
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Similarity threshold for “copy from precedent”
SIM_THRESHOLD_DEFAULT = 0.88


from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from AI_main import *


# -----------------------------------------------------------
# Validators (strict schema enforcement)
# -----------------------------------------------------------
class ResultSchema(BaseModel):
    feature_id: str | int
    feature_name: str = ""
    feature_description: str
    violation: Literal["Yes", "No", "Unclear"] = "Unclear"
    confidence_level: float | int = 0.0
    reason: str = ""
    regulations: List[str]

    @classmethod
    def normalize(cls, row: dict, allowed_regs: List[str]):
        try:
            inst = cls(**{
                "feature_id": row.get("feature_id"),
                "feature_name": row.get("feature_name", "") or "",
                "feature_description": row.get("feature_description", ""),
                "violation": sanitize_violation(row.get("violation")),
                "confidence_level": sanitize_confidence(row.get("confidence_level", 0.0)),
                "reason": str(row.get("reason", "")).strip(),
                "regulations": sanitize_regulations_dynamic(row.get("regulations", []), allowed_regs),
            })
            return inst.model_dump()
        except ValidationError as e:
            return {
                "feature_id": row.get("feature_id"),
                "feature_name": row.get("feature_name", ""),
                "feature_description": row.get("feature_description", ""),
                "violation": "Unclear",
                "confidence_level": 0.0,
                "reason": f"Validation error: {e}",
                "regulations": ["None"],
            }
def safe_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    t = raw.strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t, re.I)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None

def enforce_schema(data: dict, allowed_regs: list[str]) -> dict:
    """
    Force fields/types to match AI_1 schema, clamp confidence, and restrict regs.
    """
    norm = ResultSchema.normalize(data, allowed_regs=allowed_regs)
    # ensure these exist
    if "reason2" not in norm:
        norm["reason2"] = ""
    if "past_record" not in norm:
        norm["past_record"] = {}
    return norm


# -----------------------------------------------------------
# Utilities & Config
# -----------------------------------------------------------
# Best practice: put your service account JSON into st.secrets["gcp_service_account"].
# Fallback to inline dict if needed (NOT RECOMMENDED for public repos).



SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SERVICE_ACCOUNT_INFO_DICT = st.secrets["SERVICE_ACCOUNT_INFO"]
SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
ALLOWED_REGS_DEFAULT = [
    "EU Digital Services Act (DSA)",
    "California state law - Protecting Our Kids from Social Media Addiction Act",
    "Florida state law - Online Protections for Minors",
    "Utah Social Media Regulation Act",
    "US law on reporting child sexual abuse content to NCMEC -  Reporting requirements of providers",
    "None",
]

# -----------------------------------------------------------
# Detectors
# -----------------------------------------------------------
def static_code_like_scan(text: str) -> list[str]:
    signals = []
    if re.search(r"\b(EU|EEA|GDPR|DSA|California|Utah|Florida|minors|under\s*18)\b", text, re.I):
        signals.append("geo_or_minor_ref")
    return signals

def config_like_scan(text: str) -> list[str]:
    hits = []
    if re.search(r"enabled_countries|age_limit|curfew", text, re.I):
        hits.append("config_geo_rule")
    return hits

def runtime_like_scan(text: str) -> list[str]:
    hits = []
    if re.search(r"blocked|retention|audit log", text, re.I):
        hits.append("runtime_trace")
    return hits

def combined_detectors(text: str) -> list[str]:
    signals = []
    signals += static_code_like_scan(text)
    signals += config_like_scan(text)
    signals += runtime_like_scan(text)
    return signals

# -----------------------------------------------------------
# Auth & Sheets IO
# -----------------------------------------------------------
def _gc_from_dict():
    creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO_DICT, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False)
def load_existing_sheets(spreadsheet_id: str):
    gc = _gc_from_dict()
    sh = gc.open_by_key(spreadsheet_id)
    # DASHBOARD
    try:
        dash_ws = sh.worksheet("DASHBOARD")
        df_dash = pd.DataFrame(dash_ws.get_all_records())
    except gspread.WorksheetNotFound:
        df_dash = pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])
    # TIMELOGS
    try:
        logs_ws = sh.worksheet("TIMELOGS")
        df_logs = pd.DataFrame(logs_ws.get_all_records())
    except gspread.WorksheetNotFound:
        df_logs = pd.DataFrame(columns=[
            "ts","event","feature_id","feature_name","feature_description",
            "before_violation","before_confidence","before_regulations",
            "after_violation","after_confidence","after_regulations","note"
        ])
    return df_dash, df_logs

# -----------------------------------------------------------
# Helpers: IDs, logging, sanitation
# -----------------------------------------------------------
def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sanitize_violation(v: T.Any) -> str:
    v_str = str(v).strip().capitalize()
    return v_str if v_str in {"Yes", "No", "Unclear"} else "Unclear"

def sanitize_confidence(x: T.Any) -> T.Union[float, int]:
    try:
        val = float(x)
        if val == 2:
            return 2
        return max(0.0, min(val, 1.0))
    except Exception:
        return 0.0

def sanitize_regulations_dynamic(regs: T.Union[str, List[str]], allowed: List[str]) -> List[str]:
    if isinstance(regs, str):
        try:
            maybe = json.loads(regs)
            regs_list = maybe if isinstance(maybe, list) else [regs]
        except Exception:
            regs_list = [r.strip() for r in regs.split(",") if r.strip()]
    elif isinstance(regs, list):
        regs_list = regs
    else:
        regs_list = []
    out = [r for r in regs_list if r in allowed and r != "None"]
    out = sorted(set(out))
    return out if out else ["None"]

def log_event(event: str, feature: dict | None = None, before: dict | None = None, after: dict | None = None, note: str = ""):
    entry = {
        "ts": _now_iso(),
        "event": event,
        "feature_id": feature.get("feature_id") if feature else None,
        "feature_name": feature.get("feature_name") if feature else None,
        "feature_description": feature.get("feature_description") if feature else None,
        "before_violation": before.get("violation") if before else None,
        "before_confidence": before.get("confidence_level") if before else None,
        "before_regulations": ", ".join(before.get("regulations", [])) if before and isinstance(before.get("regulations"), list) else (before.get("regulations") if before else None),
        "after_violation": after.get("violation") if after else None,
        "after_confidence": after.get("confidence_level") if after else None,
        "after_regulations": ", ".join(after.get("regulations", [])) if after and isinstance(after.get("regulations"), list) else (after.get("regulations") if after else None),
        "note": note or "",
    }
    st.session_state.TIMELOGS.append(entry)



@st.cache_data(show_spinner=False)
def _model_name_cached(name: str) -> str:
    return name

def call_gemini(prompt: str, model_name: str) -> str:
    model = GenerativeModel(_model_name_cached(model_name))
    resp = model.generate_content(prompt)
    return resp.text or ""

# -----------------------------------------------------------
# Parsing uploads → canonical JSON/text blocks
# -----------------------------------------------------------

def parse_terminology_upload(upload) -> Tuple[pd.DataFrame, str]:
    """Return (df, json_text). Accept CSV/XLSX. JSON is a list[dict]."""
    if upload is None:
        return pd.DataFrame(), "[]"
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
        # Fill NaNs and convert to records
        df = df.fillna("")
        records = df.to_dict(orient="records")
        json_text = json.dumps(records, ensure_ascii=False, indent=2)
        return df, json_text
    except Exception as e:
        st.error(f"Terminology parse failed: {e}")
        return pd.DataFrame(), "[]"


def parse_regulations_txt(files: list) -> str:
    """Concatenate multiple TXT files with separators into one big text."""
    if not files:
        return ""
    parts = []
    for f in files:
        try:
            txt = f.getvalue().decode("utf-8", errors="replace")
        except Exception:
            txt = str(f.getvalue())
        parts.append(f"# File: {f.name}\n{txt.strip()}\n")
    return "\n\n".join(parts).strip()


def parse_features(
    file_upload, 
    text_input: str, 
    existing_dash: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    """
    Build feature table with assigned incremental feature_id based on existing DASHBOARD.
    Input can be CSV/XLSX (expects columns: feature_name, feature_description) OR
    a multiline text where each non-empty line is "Name: Description".
    Returns (df, combined_feature_json_text).
    """
    # Determine current max feature_id
    max_id = 0
    if not existing_dash.empty and "feature_id" in existing_dash.columns:
        # consider only numeric-like ids
        def _to_int(x):
            try:
                return int(re.findall(r"\d+", str(x))[0])
            except Exception:
                return None
        vals = [v for v in existing_dash["feature_id"].apply(_to_int).tolist() if v is not None]
        max_id = max(vals) if vals else 0

    # From file
    rows = []
    if file_upload is not None:
        try:
            if file_upload.name.lower().endswith(".csv"):
                df_file = pd.read_csv(file_upload)
            else:
                df_file = pd.read_excel(file_upload)
            # normalize columns
            cols = {c.lower().strip(): c for c in df_file.columns}
            fn_col = cols.get("feature_name")
            fd_col = cols.get("feature_description")
            if not fn_col or not fd_col:
                st.error("Uploaded Features file must have columns: feature_name, feature_description")
                return pd.DataFrame(), "[]"
            for _, r in df_file.iterrows():
                name = str(r[fn_col]).strip()
                desc = str(r[fd_col]).strip()
                if not (name or desc):
                    continue
                rows.append({"feature_name": name, "feature_description": desc})
        except Exception as e:
            st.error(f"Features parse failed: {e}")
            return pd.DataFrame(), "[]"

    # From text area
    for line in (text_input or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, desc = line.split(":", 1)
            rows.append({"feature_name": name.strip(), "feature_description": desc.strip()})
        else:
            # if no colon, treat whole line as name
            rows.append({"feature_name": line, "feature_description": ""})

    # Assign ids
    out_rows = []
    cur = max_id
    for r in rows:
        cur += 1
        out_rows.append({
            "feature_id": cur,
            "feature_name": r.get("feature_name", ""),
            "feature_description": r.get("feature_description", ""),
        })

    df = pd.DataFrame(out_rows)
    json_text = json.dumps(out_rows, ensure_ascii=False, indent=2)
    return df, json_text

# -----------------------------------------------------------
# Prompts (Single AI)
# -----------------------------------------------------------

def ai1_prompt(
    feature_row: dict,
    terminology_json_text: str,
    regulations_text: str,
    detector_signals: List[str],
    allowed_regs: List[str],
) -> str:
    allowed_str = ", ".join(sorted(set(allowed_regs)))

    schema = f"""
Return ONLY valid JSON exactly as:
{{
  "feature_id": "<string or number>",
  "feature_name": "<short title if available or derive from description>",
  "feature_description": "<original text>",
  "violation": "<Yes|No|Unclear>",
  "confidence_level": <number>,  // 0.0–1.0
  "reason": "<concise explanation>",
  "regulations": ["<values ONLY from: {allowed_str}>"]
}}
""".strip()

    det = f"[DETECTORS] {', '.join(detector_signals)}" if detector_signals else "[DETECTORS] none"

    return f"""
You are AI_1, a conservative geo-compliance detector.
Use only the provided Terminology JSON and Regulations text as context.

STRICT RULES FOR "violation":
- If the feature description or detectors mention any terminology, region, minors, or regulation that could relate to compliance → output "Yes".
- Output "No" only if you are highly confident the feature has NOTHING to do with compliance or regulations.
- Output "Unclear" only if the description is so vague that you cannot reasonably decide.

Use only these regulation labels in the output (exact match): {allowed_str}. If none apply, use ["None"].

[TERMINOLOGY JSON]
{terminology_json_text}

[REGULATIONS TEXT]
{regulations_text}

{det}

[FEATURE]
feature_id: {feature_row.get('feature_id')}
feature_name: {feature_row.get('feature_name', '')}
feature_description: {feature_row.get('feature_description')}

{schema}
""".strip()

# -----------------------------------------------------------
# Streamlit App — UI
# -----------------------------------------------------------
logo_icon = None
try:
    logo_icon = Image.open("Image/full_width.png")
except Exception:
    pass

st.set_page_config(
    page_title="Geo-Reg Compliance — No‑RAG",
    page_icon=logo_icon if logo_icon else "🛡️",
    layout="wide",
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Layout row: logo left, BandAI center
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if logo_icon is not None:
        st.image(logo_icon, width=200)  # bigger logo

with col2:
    
    st.markdown(
        """
        <div style='text-align:center;'>
            <div class='bandai-title'>
                <span class="letter">B</span>
                <span class="letter">a</span>
                <span class="letter">n</span>
                <span class="letter">d</span>
                <span class="letter">A</span>
                <span class="letter">I</span>
            </div>
            <div class="rocket">🚀</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:
    st.markdown("")
    


st.title("🛡️ Geo-Reg Compliance Checker — DUO AI POWERED BY RAG")

st.link_button("Click to open Looker Studio Visualisation by Google Analytics", url = "https://lookerstudio.google.com/reporting/e081e397-63bd-4000-9227-893fc9d86033")


# Session defaults
defaults = {
    "RESULTS_READY": False,
    "AI1_DF": None,
    "TIMELOGS": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load existing sheets for ID assignment
existing_dash, existing_logs = load_existing_sheets(SPREADSHEET_ID)
st.session_state.EXISTING_DASH = existing_dash
st.session_state.EXISTING_LOGS = existing_logs
# ---- Build/searchable precedent memory from DASHBOARD
def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make DF safe for st.dataframe/pyarrow by stringifying complex cols."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # stringify lists/dicts that break pyarrow
    if "regulations" in out.columns:
        out["regulations"] = out["regulations"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ("" if x is None else str(x))
        )
    if "past_record" in out.columns:
        out["past_record"] = out["past_record"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list))
            else ("" if x is None else str(x))
        )
    if "reason2" in out.columns:
        out["reason2"] = out["reason2"].apply(lambda x: "" if x is None else str(x))

    # keep numeric column truly numeric
    if "confidence_level" in out.columns:
        out["confidence_level"] = pd.to_numeric(out["confidence_level"], errors="coerce")

    return out

def _prep_text(name: str, desc: str) -> str:
    name = (name or "").strip()
    desc = (desc or "").strip()
    return f"{name}\n{desc}".strip()

def build_precedent_index(dashboard_df: pd.DataFrame):
    """
    Vectorize past rows for nearest-neighbor lookup by feature text.
    Stores into session:
      PRECEDENT_ROWS: list[dict] canonicalized past rows
      PRECEDENT_VEC:  fitted TfidfVectorizer
      PRECEDENT_MAT:  tf-idf matrix for past rows
    """
    rows = []
    if dashboard_df is not None and not dashboard_df.empty:
        # normalize columns we care about
        df = dashboard_df.copy()
        wanted = ["feature_id","feature_name","feature_description","violation","confidence_level","reason","regulations"]
        for c in wanted:
            if c not in df.columns:
                df[c] = "" if c not in ("confidence_level",) else 0.0

        for _, r in df.iterrows():
            regs = r.get("regulations")
            if isinstance(regs, str):
                regs_list = [x.strip() for x in regs.split(",") if x.strip()]
            elif isinstance(regs, list):
                regs_list = regs
            else:
                regs_list = []
            rows.append({
                "feature_id": str(r.get("feature_id", "")),
                "feature_name": str(r.get("feature_name", "")),
                "feature_description": str(r.get("feature_description", "")),
                "violation": sanitize_violation(r.get("violation", "")),
                "confidence_level": sanitize_confidence(r.get("confidence_level", 0.0)),
                "reason": str(r.get("reason", "")),
                "regulations": sanitize_regulations_dynamic(regs_list, ALLOWED_REGS_DEFAULT),
            })

    texts = [_prep_text(x["feature_name"], x["feature_description"]) for x in rows]
    if texts:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        mat = vec.fit_transform(texts)
        st.session_state.PRECEDENT_ROWS = rows
        st.session_state.PRECEDENT_VEC = vec
        st.session_state.PRECEDENT_MAT = mat
    else:
        st.session_state.PRECEDENT_ROWS = []
        st.session_state.PRECEDENT_VEC = None
        st.session_state.PRECEDENT_MAT = None

build_precedent_index(st.session_state.EXISTING_DASH)

def find_similar_precedents(feature_text: str, top_k: int = 3) -> list[tuple[float, dict]]:
    vec = st.session_state.get("PRECEDENT_VEC")
    mat = st.session_state.get("PRECEDENT_MAT")
    rows = st.session_state.get("PRECEDENT_ROWS", [])
    if vec is None or mat is None or not rows:
        return []
    q = vec.transform([feature_text])
    sims = cosine_similarity(q, mat)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(float(sims[i]), rows[i]) for i in idxs]

def ai2_from_precedent(ai1_json: dict, allowed_regs: list[str], sim_threshold: float = SIM_THRESHOLD_DEFAULT) -> dict:
    """
    If a close precedent is found, copy over violation/regulations (and reason)
    and add reason2 + past_record. Else, pass through ai1_json with reason2 note.
    """
    text = _prep_text(ai1_json.get("feature_name",""), ai1_json.get("feature_description",""))
    matches = find_similar_precedents(text, top_k=3)
    if matches:
        top_sim, top = matches[0]
        if top_sim >= sim_threshold:
            # copy over “decision” fields from precedent
            out = {**ai1_json}
            out["violation"] = top["violation"]
            out["regulations"] = top["regulations"]
            # Merge reasons
            out["reason2"] = f"Matched precedent (sim={top_sim:.2f}) → copied decision."
            out["past_record"] = top
            # Keep the higher of the two confidences (bounded by 1 unless 2 means human)
            conf_ai1 = float(sanitize_confidence(ai1_json.get("confidence_level", 0.0)))
            conf_prec = float(sanitize_confidence(top.get("confidence_level", 0.0)))
            out["confidence_level"] = 2 if (conf_prec == 2 or conf_ai1 == 2) else max(conf_ai1, conf_prec)
            return enforce_schema(out, allowed_regs)
    # No close precedent → keep AI_1, but add reason2 note
    out = {**ai1_json}
    out["reason2"] = "No sufficiently similar precedent. Kept AI_1."
    out["past_record"] = {}
    return enforce_schema(out, allowed_regs)

with st.sidebar:
    st.subheader("1) Upload Terminology")
    term_upload = st.file_uploader("Terminology (CSV/XLSX)", type=["csv", "xlsx"])


    st.subheader("2) Features Source")
    features_file = st.file_uploader("Features (CSV/XLSX with feature_name, feature_description)", type=["csv", "xlsx"])
    st.caption("Or paste lines below as: Name: Description")


st.markdown("---")

# Terminology
st.markdown("### ⚠️ Dangerous Actions")


if "clear_button" not in st.session_state:
    st.session_state["clear_button"] = False
if st.button("🧹 Clear DASHBOARD + TIMELOGS (keep headers)", type="secondary"):
    st.session_state["clear_button"] = True
    st.warning("This will ERASE all rows below the header in both sheets. This cannot be undone. Click confirm if you're sure.", icon="⚠️")

if st.session_state["clear_button"]:
    if st.button("✅ Confirm Clear", type="primary"):
        st.session_state["clear_button"] = False
        try:
            gc = _gc_from_dict()
            sh = gc.open_by_key(SPREADSHEET_ID)

            # Clear DASHBOARD rows from row 2 onwards
            try:
                ws_dash = sh.worksheet("DASHBOARD")
                ws_dash.batch_clear(["A2:Z"])   # adjust cols if wider
            except gspread.WorksheetNotFound:
                st.info("DASHBOARD sheet not found, skipped.")

            # Clear TIMELOGS rows from row 2 onwards
            try:
                ws_logs = sh.worksheet("TIMELOGS")
                ws_logs.batch_clear(["A2:Z"])   # adjust cols if wider
            except gspread.WorksheetNotFound:
                st.info("TIMELOGS sheet not found, skipped.")

            st.success("✅ Cleared all rows (row 2 and below) in DASHBOARD + TIMELOGS.")
        except Exception as e:
            st.error(f"Failed to clear: {e}")
        st.text("Please reload the whole session after Confirm Clear button")
term_df, terminology_json_text = parse_terminology_upload(term_upload)




regulations_text = "placeholder so it works"

# Features
st.header("B) Features input")
features_text = st.text_area("For quick checks, paste features in following format without brackets (Feature_name: Feature_description per line)")
features_df, combined_feature_json = parse_features(features_file, features_text, existing_dash)

if features_df.empty:
    st.info("No features detected yet. Upload a CSV/XLSX or paste lines above.")
else:
    st.success(f"Prepared {len(features_df)} feature(s) with incremental IDs.")
    st.dataframe(features_df, use_container_width=True)

st.download_button("⬇️ Download combined_feature.json", data=combined_feature_json, file_name="combined_feature.json", mime="application/json")

# Guard before run
can_run = bool(features_df.shape[0]) and bool(terminology_json_text) and bool(regulations_text)
if not can_run:
    st.warning("To run AI_1, please provide: Terminology CSV/XLSX, Regulations TXT, and at least one Feature.")

# Run AI_1
st.header("C) Run Compliance Detection (AI_1)")
run_btn = st.button("▶️ Run with AI_1", disabled=not can_run)

ALLOWED_REGS = ALLOWED_REGS_DEFAULT  # still provide a guardrail list

if run_btn:
    rows_out = []
    prog, total = st.progress(0), len(features_df)
    for i, row in features_df.iterrows():
        r = {
            "feature_id": row.get("feature_id"),
            "feature_name": row.get("feature_name", "") or "",
            "feature_description": str(row.get("feature_description", "")),
        }
        try:
            det_signals = combined_detectors(r["feature_description"])  # optional evidence
            raw = rag_answer(r,terminology_json_text)
            data = None
            if raw:
                raw_strip = raw.strip()
                # try parse JSON; fallback to fenced json
                try:
                    data = json.loads(raw_strip)
                except Exception:
                    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_strip, re.I)
                    if m:
                        try:
                            data = json.loads(m.group(1))
                        except Exception:
                            data = None
            if not data:
                raise ValueError("AI_1 returned invalid JSON")
            norm = ResultSchema.normalize(data, allowed_regs=ALLOWED_REGS)
        except Exception as e:
            norm = {
                "feature_id": r["feature_id"],
                "feature_name": r["feature_name"],
                "feature_description": r["feature_description"],
                "violation": "Unclear",
                "confidence_level": 0.0,
                "reason": f"AI_1 error: {e}",
                "regulations": ["None"],
            }
        rows_out.append(norm)
        # log prediction
        log_event(
            "prediction",
            feature={k: r[k] for k in ["feature_id","feature_name","feature_description"]},
            after={k: norm[k] for k in ["violation","confidence_level","regulations"]},
            note="AI_1 output recorded",
        )
        prog.progress(int((i + 1) / total * 100)); time.sleep(0.02)
    prog.empty()

    st.session_state.AI1_DF = pd.DataFrame(rows_out)
    st.session_state.RESULTS_READY = True
    st.success("AI_1 completed.")

# Results & Exports
if st.session_state.RESULTS_READY and st.session_state.AI1_DF is not None:
    st.subheader("Results — AI_1")
    st.dataframe(st.session_state.AI1_DF, use_container_width=True)

    # Download JSON/CSV
    results_json = st.session_state.AI1_DF.to_json(orient="records", force_ascii=False, indent=2)
    st.download_button("⬇️ Download results.json", data=results_json, file_name="results_ai1.json", mime="application/json")
    st.download_button("⬇️ Download results.csv", data=st.session_state.AI1_DF.to_csv(index=False), file_name="results_ai1.csv", mime="text/csv")
# -----------------------------------------------------------
# AI_2 pass: precedent-aware auto-correction
# -----------------------------------------------------------
if st.session_state.RESULTS_READY and st.session_state.AI1_DF is not None:

    sim_threshold = 0.5
    allowed = ALLOWED_REGS_DEFAULT

    # -------- Compute phase: only when button is clicked ----------
    if st.button("▶️ Run AI_2 on AI_1 results"):
        ai2_rows = []
        prog, total = st.progress(0), len(st.session_state.AI1_DF)
        for i, r in st.session_state.AI1_DF.iterrows():
            ai1 = r.to_dict()
            ai2 = rag_answer2(ai1, terminology_json_text, st.session_state.PRECEDENT_ROWS)
            ai2 = ai2.strip()
            try:
                ai2 = json.loads(ai2)
            except Exception:
                m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", ai2, re.I)
                if m:
                    try:
                        ai2 = json.loads(m.group(1))
                    except Exception as e:
                        ai2 = {
                            "feature_id": r["feature_id"],
                            "feature_name": r["feature_name"],
                            "feature_description": r["feature_description"],
                            "violation": "Unclear",
                            "confidence_level": 0.0,
                            "reason": f"AI_1 error: {e}",
                            "regulations": ["None"],
                            "reason2": f"AI_1 error: {e}",
                            "past_records": ["None"]
                            }
            log_event("prediction",
                      feature={"feature_id": ai2.get("feature_id"),
                               "feature_name": ai2.get("feature_name"),
                               "feature_description": ai2.get("feature_description")},
                      after={"violation": ai2.get("violation"),
                             "confidence_level": ai2.get("confidence_level"),
                             "regulations": ai2.get("regulations")},
                      note="AI_2 output recorded")
            ai2_rows.append(ai2)
            prog.progress(int((i + 1) / total * 100)); time.sleep(0.02)
        prog.empty()

        ai2_df = pd.DataFrame(ai2_rows)
        st.session_state.AI2_DF = ai2_df.copy()
        st.session_state.AI2_FULL_DF = ai2_df.copy()

        # make editable copy (merge reason+reason2; stringify regs)
        edit_df = ai2_df.copy()

        def _merge_reasons(a, b):
            a = str(a or "").strip()
            b = str(b or "").strip()
            return (a if a else "") if not b else (f"{a} | {b}" if a else b)

        if "reason2" in edit_df.columns:
            edit_df["reason"] = [_merge_reasons(a, b) for a, b in zip(edit_df.get("reason", ""), edit_df["reason2"])]
            edit_df = edit_df.drop(columns=[c for c in ("reason2", "past_record") if c in edit_df.columns])

        edit_df["regulations"] = edit_df["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        # reset sources for editor
        st.session_state["ai2_editor_source"] = edit_df.copy(deep=True)
        st.session_state.AI2_EDITED = edit_df.copy(deep=True)

        # keep UI expanded after rerun
        st.session_state.AI2_EXPANDED = True


# -------- Helpers (defined once, used on submit) ----------
def _save_ai2_corrections(edited_df: pd.DataFrame):
    if st.session_state.get("AI2_FULL_DF") is None:
        return
    allowed = ALLOWED_REGS_DEFAULT
    base_df = st.session_state.AI2_FULL_DF

    # Map by feature_id (string) for stability
    base_map = base_df.assign(_fid=base_df["feature_id"].astype(str)).set_index("_fid")

    ui_df = edited_df.copy()
    saved = 0
    prec_rows = st.session_state.get("PRECEDENT_ROWS", [])

    for _, ui_row in ui_df.iterrows():
        row_after_ui = ui_row.to_dict()
        try:
            conf_val = float(row_after_ui.get("confidence_level", 0))
        except Exception:
            conf_val = 0.0
        if conf_val != 2:
            continue

        fid = str(row_after_ui.get("feature_id"))
        if fid not in base_map.index:
            continue

        row_before = base_map.loc[fid].to_dict()

        regs_str = row_after_ui.get("regulations", "")
        regs_list = [r.strip() for r in str(regs_str).split(",") if r.strip()]
        row_after_ui["regulations"] = sanitize_regulations_dynamic(regs_list, allowed)

        cleaned = {
            "feature_id": row_after_ui.get("feature_id"),
            "feature_name": row_after_ui.get("feature_name"),
            "feature_description": row_after_ui.get("feature_description"),
            "violation": row_after_ui.get("violation"),
            "confidence_level": row_after_ui.get("confidence_level"),
            "reason": row_after_ui.get("reason"),
            "regulations": row_after_ui.get("regulations"),
        }
        final_norm = enforce_schema(cleaned, allowed)
        final_norm["confidence_level"] = 2  # mark human intervention

        # Update precedent memory rows
        prec_rows.append({
            "feature_id": str(final_norm["feature_id"]),
            "feature_name": final_norm["feature_name"],
            "feature_description": final_norm["feature_description"],
            "violation": final_norm["violation"],
            "confidence_level": final_norm["confidence_level"],
            "reason": final_norm["reason"],
            "regulations": final_norm["regulations"],
        })

        # Reflect changes back to AI2_FULL_DF by fid
        mask = st.session_state.AI2_FULL_DF["feature_id"].astype(str) == fid
        for col in ("violation", "confidence_level", "reason", "regulations"):
            st.session_state.AI2_FULL_DF.loc[mask, col] = [final_norm[col]]

        # Update editor source (string regs)
        mask_src = st.session_state["ai2_editor_source"]["feature_id"].astype(str) == fid
        st.session_state["ai2_editor_source"].loc[mask_src, "violation"] = final_norm["violation"]
        st.session_state["ai2_editor_source"].loc[mask_src, "confidence_level"] = final_norm["confidence_level"]
        st.session_state["ai2_editor_source"].loc[mask_src, "reason"] = final_norm["reason"]
        st.session_state["ai2_editor_source"].loc[mask_src, "regulations"] = ", ".join(final_norm["regulations"])

        # Timelog entry
        feature_ctx = {
            "feature_id": final_norm.get("feature_id"),
            "feature_name": final_norm.get("feature_name"),
            "feature_description": final_norm.get("feature_description"),
        }
        before_norm = enforce_schema({**row_before, "regulations": row_before.get("regulations", [])}, allowed)
        log_event("human_intervention", feature=feature_ctx, before=before_norm, after=final_norm, note="Reviewer set confidence_level=2")
        saved += 1

    # Persist precedents & rebuild index once
    st.session_state.PRECEDENT_ROWS = prec_rows
    build_precedent_index(pd.DataFrame(prec_rows))
    st.session_state.AI2_DF = st.session_state.AI2_FULL_DF.copy()

    # Update DASHBOARD snapshot
    dash_df = st.session_state.AI2_FULL_DF.copy()
    dash_df["feature_id"] = dash_df["feature_id"].astype(str)
    dash_df["regulations"] = dash_df["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    st.session_state.DASHBOARD_READY = dash_df[["feature_id","feature_name","feature_description","violation","reason","regulations"]]

    st.session_state.AI2_SAVED_TS = _now_iso()
    st.session_state.AI2_SAVED_COUNT = saved

# -------- Render phase: always show if data exists ----------
def _recompute_needs_review():
    df = st.session_state.get("AI2_DF")
    if df is None or df.empty:
        st.session_state.NEEDS_REVIEW = pd.DataFrame()
        return
    try:
        conf = df["confidence_level"].astype(float)
    except Exception:
        conf = pd.to_numeric(df["confidence_level"], errors="coerce").fillna(0.0)
    st.session_state.NEEDS_REVIEW = df[(df["violation"] == "Unclear") | (conf < 0.3)]


if st.session_state.get("AI2_DF") is not None and st.session_state.get("ai2_editor_source") is not None:
    # 1) Read-only results (always reflects latest saved state)
    with st.expander("AI_2 Results (read-only with precedent)", expanded=True):
        st.dataframe(_arrow_safe(st.session_state.AI2_DF), use_container_width=True)

    # 2) Editable summary — user sets confidence_level == 2 to mark HUMAN correction
    with st.expander("AI_2 Summary (editable)", expanded=st.session_state.get("AI_2_EXPANDED", True)):
        st.text("Edit rows; set confidence_level = 2 to mark a HUMAN correction. Only those rows will be saved to memory.")

        # Freeze the source shown in the editor; don't mutate on keystrokes
        src = st.session_state["ai2_editor_source"].copy(deep=True)

        # Use a form so edits don’t change session state until Save is pressed
        with st.form("ai2_edit_form", clear_on_submit=False):
            edited_df = st.data_editor(
                src,
                key="ai2_editor_table",
                use_container_width=True,
                num_rows="fixed",     # ⛔ no row add/remove
            )
            submitted = st.form_submit_button("💾 Save corrections to memory", type="primary")

        if submitted:
            # keep UI open after save
            st.session_state.AI_2_EXPANDED = True

            # ✅ apply changes only now
            _save_ai2_corrections(edited_df)

            # refresh Needs Review view immediately
            def _recompute_needs_review():
                df = st.session_state.get("AI2_DF")
                if df is None or df.empty:
                    st.session_state.NEEDS_REVIEW = pd.DataFrame()
                else:
                    conf = pd.to_numeric(df["confidence_level"], errors="coerce").fillna(0.0)
                    st.session_state.NEEDS_REVIEW = df[(df["violation"] == "Unclear") | (conf < 0.3)]
            _recompute_needs_review()

            st.toast("Saved corrections. Tables updated below.", icon="✅")

    # 3) Needs Review — always recompute so it updates immediately after save
    _recompute_needs_review()
    if st.session_state.NEEDS_REVIEW is not None and not st.session_state.NEEDS_REVIEW.empty:
        with st.expander("🚨 Needs Review", expanded=False):
            st.dataframe(st.session_state.NEEDS_REVIEW, use_container_width=True)
            st.download_button(
                "⬇️ Download Needs Review CSV",
                data=st.session_state.NEEDS_REVIEW.to_csv(index=False),
                file_name="needs_review.csv",
                mime="text/csv",
            )

    # 4) Dashboard view (final, to upload) — reflects latest saved corrections
    if st.session_state.get("AI2_FULL_DF") is not None:
        dash_df = st.session_state.AI2_FULL_DF.copy()
        dash_df["feature_id"] = dash_df["feature_id"].astype(str)
        dash_df["regulations"] = dash_df["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        keep = ["feature_id","feature_name","feature_description","violation","reason","regulations"]
        st.session_state.DASHBOARD_READY = dash_df[keep]
        with st.expander("Dashboard (final, to upload)", expanded=False):
            st.dataframe(st.session_state.DASHBOARD_READY, use_container_width=True)

# -----------------------------------------------------------
# Export & Google Sheets sync
# -----------------------------------------------------------
with st.expander("D) Export Excel & Update Google Sheets", expanded=False):

    def _norm_feature_id(df: pd.DataFrame) -> pd.DataFrame:
        if "feature_id" in df.columns:
            df = df.copy()
            df["feature_id"] = df["feature_id"].astype(str)
        return df

    def current_dashboard_from_session() -> pd.DataFrame:
        """
        Prefer AI_2 final (post-edit) table if present; else fall back to AI_1.
        """
        if st.session_state.get("DASHBOARD_READY") is not None:
            return st.session_state.DASHBOARD_READY.copy()
        if st.session_state.get("AI2_DF") is not None and len(st.session_state.AI2_DF) > 0:
            df_cur = st.session_state.AI2_DF.copy()
        elif st.session_state.get("AI1_DF") is not None and len(st.session_state.AI1_DF) > 0:
            df_cur = st.session_state.AI1_DF.copy()
        else:
            return pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])

        df_cur["feature_id"] = df_cur["feature_id"].astype(str)
        df_cur["regulations"] = df_cur["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        keep = ["feature_id","feature_name","feature_description","violation","reason","regulations"]
        return df_cur[keep]


    def merged_dashboard(existing: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """
        Duplicate-safe merge of DASHBOARD:
        - Coerce feature_id to str
        - Ensure required columns exist
        - Drop duplicate feature_id (keep last)
        - Update overlap; append new
        """
        required = ["feature_id","feature_name","feature_description","violation","reason","regulations"]

        def _prep(df: pd.DataFrame) -> pd.DataFrame:
            if df is None:
                return pd.DataFrame(columns=required)
            df = df.copy()
            # ensure cols
            for c in required:
                if c not in df.columns:
                    df[c] = ""
            # coerce id + select cols
            df["feature_id"] = df["feature_id"].astype(str)
            df = df[required]
            # normalize text-y columns (avoid NaNs that annoy Arrow/Sheets)
            for c in ["feature_name","feature_description","violation","reason","regulations"]:
                df[c] = df[c].fillna("").astype(str)
            # drop dups (keep last seen)
            df = df.drop_duplicates(subset=["feature_id"], keep="last")
            return df

        base = _prep(existing)
        cur  = _prep(current)

        # Set index
        base = base.set_index("feature_id")
        cur  = cur.set_index("feature_id")

        # Update only intersecting keys
        intersect = base.index.intersection(cur.index)
        if len(intersect) > 0:
            base.loc[intersect, :] = cur.loc[intersect, :]

        # Append new keys not in base
        new_only = cur.loc[cur.index.difference(base.index)]
        out = pd.concat([base, new_only], axis=0)

        return out.reset_index()


    def appended_timelogs(existing_logs: pd.DataFrame) -> pd.DataFrame:
        new_logs = pd.DataFrame(st.session_state.TIMELOGS or [])
        if existing_logs is None or existing_logs.empty:
            return new_logs
        if new_logs.empty:
            return existing_logs
        cols = [
            "ts","event","feature_id","feature_name","feature_description",
            "before_violation","before_confidence","before_regulations",
            "after_violation","after_confidence","after_regulations","note"
        ]
        for df in (existing_logs, new_logs):
            for c in cols:
                if c not in df.columns:
                    df[c] = None
        return pd.concat([existing_logs[cols], new_logs[cols]], ignore_index=True)

    def make_excel_bytes(dashboard_df: pd.DataFrame, timelogs_df: pd.DataFrame) -> bytes:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            dashboard_df.to_excel(writer, sheet_name="DASHBOARD", index=False)
            timelogs_df.to_excel(writer, sheet_name="TIMELOGS", index=False)
            for sheet_name, df in [("DASHBOARD", dashboard_df), ("TIMELOGS", timelogs_df)]:
                ws = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    max_len = max([len(str(col))] + [len(str(x)) for x in df[col].astype(str).values]) if not df.empty else len(col)
                    ws.set_column(idx, idx, min(max_len + 2, 60))
        bio.seek(0)
        return bio.getvalue()

    current_dash = current_dashboard_from_session()
    dashboard_df = merged_dashboard(st.session_state.EXISTING_DASH, current_dash)
    timelogs_df = appended_timelogs(st.session_state.EXISTING_LOGS)

    st.subheader("Preview — DASHBOARD (merged)")
    st.dataframe(dashboard_df, use_container_width=True)
    st.subheader("Preview — TIMELOGS (appended)")
    st.dataframe(timelogs_df.tail(50), use_container_width=True)

    def ensure_ws(sh, title: str):
        try:
            return sh.worksheet(title)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=title, rows=100, cols=26)

    def write_preserve_header(ws, df: pd.DataFrame):
        ws.resize(1)
        set_with_dataframe(ws, df, row=2, col=1, include_column_header=False, resize=False)

    if st.button("⬆️ Update Google Sheets (DASHBOARD + TIMELOGS)"):
        try:
            gc = _gc_from_dict()
            sh = gc.open_by_key(SPREADSHEET_ID)
            ws_dash = ensure_ws(sh, "DASHBOARD")
            write_preserve_header(ws_dash, dashboard_df)
            ws_logs = ensure_ws(sh, "TIMELOGS")
            write_preserve_header(ws_logs, timelogs_df)
            st.session_state.EXISTING_DASH = dashboard_df.copy()
            st.session_state.EXISTING_LOGS = timelogs_df.copy()
            st.success("Google Sheets updated ✔ (headers preserved).")
        except Exception as e:
            st.error(f"Update failed: {e}")

    excel_bytes = make_excel_bytes(dashboard_df, timelogs_df)
    st.download_button(
        "⬇️ Download Excel (DASHBOARD + TIMELOGS)",
        data=excel_bytes,
        file_name="geo_reg_audit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
