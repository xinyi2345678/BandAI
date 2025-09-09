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
    norm = _coerce_violation_regs(norm, allowed_regs)
    return norm

# -----------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SERVICE_ACCOUNT_INFO_DICT = st.secrets["SERVICE_ACCOUNT_INFO"]
SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]

# -----------------------------------------------------------
# Auth & Sheets IO
# -----------------------------------------------------------
def _gc_from_dict():
    creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO_DICT, scopes=SCOPES)
    return gspread.authorize(creds)

def _coerce_violation_regs(norm: dict, _allowed_unused: list[str]) -> dict:
    """
    Simplified: DON'T flip violation/regs. Just normalize types/shapes.
    """
    regs = norm.get("regulations", [])
    if isinstance(regs, str):
        regs = sanitize_regulations_dynamic(regs, [])
    elif not isinstance(regs, list):
        regs = []
    norm["violation"] = sanitize_violation(norm.get("violation", "Unclear"))
    norm["regulations"] = regs
    return norm

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
    if isinstance(v, bool):
        return "Yes" if v else "No"
    v_str = str(v).strip().lower()
    if v_str in {"yes", "y", "true", "1"}:
        return "Yes"
    if v_str in {"no", "n", "false", "0"}:
        return "No"
    if v_str in {"unclear", "unknown", "maybe"}:
        return "Unclear"
    return "Unclear"

def sanitize_confidence(x: T.Any) -> T.Union[float, int]:
    try:
        val = float(x)
        if val == 2:
            return 2
        return max(0.0, min(val, 1.0))
    except Exception:
        return 0.0

def sanitize_regulations_dynamic(regs: T.Union[str, List[str]], _allowed_unused: List[str]) -> List[str]:
    """
    Permissive: accept any comma-separated text (no whitelist).
    - If input is a JSON list -> keep items (stringified & trimmed)
    - If string -> split by commas
    - Empty -> return []
    """
    if isinstance(regs, list):
        return [str(x).strip() for x in regs if str(x).strip()]
    if regs is None:
        return []
    s = str(regs).strip()
    if not s:
        return []
    # try JSON list first
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return [str(x).strip() for x in j if str(x).strip()]
    except Exception:
        pass
    # fallback: CSV
    return [t.strip() for t in s.split(",") if t.strip()]


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

def _parse_regs_text_any(v: T.Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    # Try JSON list
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return [str(x).strip() for x in j if str(x).strip()]
    except Exception:
        pass
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_features(
    file_upload, 
    text_input: str, 
    existing_dash: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    """
    Build a feature table, *reusing* feature_id if:
      - the uploaded file includes a 'feature_id' column, OR
      - we find an exact match (name+desc) in existing_dash.
    Otherwise, assign new incremental IDs after the current max.
    Returns (df, combined_feature_json_text).
    """
    # Map existing (name, desc) -> feature_id  (normalized lower/stripped)
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    existing_map = {}
    max_id = 0
    if existing_dash is not None and not existing_dash.empty:
        # current max id from existing
        def _to_int(x):
            try:
                return int(re.findall(r"\d+", str(x))[0])
            except Exception:
                return None
        for _, r in existing_dash.iterrows():
            fid = r.get("feature_id")
            n, d = _norm(str(r.get("feature_name",""))), _norm(str(r.get("feature_description","")))
            if n or d:
                existing_map[(n, d)] = str(fid) if fid is not None else None
        ints = [v for v in existing_dash["feature_id"].apply(_to_int).tolist() if v is not None]
        max_id = max(ints) if ints else 0

    # Collect incoming rows (optionally with feature_id)
    incoming_rows = []
    if file_upload is not None:
        try:
            if file_upload.name.lower().endswith(".csv"):
                df_file = pd.read_csv(file_upload)
            else:
                df_file = pd.read_excel(file_upload)
            df_file = df_file.fillna("")
            # normalize column names
            cols = {c.lower().strip(): c for c in df_file.columns}
            fn_col = cols.get("feature_name")
            fd_col = cols.get("feature_description")
            fid_col = cols.get("feature_id")  # optional

            if not fn_col or not fd_col:
                st.error("Uploaded Features file must have columns: feature_name, feature_description (feature_id optional)")
                return pd.DataFrame(), "[]"

            for _, r in df_file.iterrows():
                name = str(r[fn_col]).strip()
                desc = str(r[fd_col]).strip()
                if not (name or desc):
                    continue
                fid_val = str(r[fid_col]).strip() if fid_col else ""
                incoming_rows.append({"feature_id": fid_val, "feature_name": name, "feature_description": desc})
        except Exception as e:
            st.error(f"Features parse failed: {e}")
            return pd.DataFrame(), "[]"

    # Also support pasted text (one per line: Name: Description)
    for line in (text_input or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, desc = line.split(":", 1)
            incoming_rows.append({"feature_id": "", "feature_name": name.strip(), "feature_description": desc.strip()})
        else:
            incoming_rows.append({"feature_id": "", "feature_name": line, "feature_description": ""})

    # Assign IDs:
    # - Prefer provided feature_id (if non-empty)
    # - Else reuse by exact (name,desc) match
    # - Else assign fresh incremental
    out_rows = []
    seen_new_keys = {}  # within-this-upload de-dupe: (name,desc)->assigned id
    cur = max_id
    for r in incoming_rows:
        name = r.get("feature_name", "")
        desc = r.get("feature_description", "")
        fid  = str(r.get("feature_id", "")).strip()

        if fid:  # explicit incoming id wins
            out_rows.append({"feature_id": fid, "feature_name": name, "feature_description": desc})
            continue

        key = (_norm(name), _norm(desc))
        # 1) Try existing dashboard match
        if key in existing_map and existing_map[key]:
            out_rows.append({"feature_id": existing_map[key], "feature_name": name, "feature_description": desc})
            continue

        # 2) Try within-upload dedupe
        if key in seen_new_keys:
            out_rows.append({"feature_id": seen_new_keys[key], "feature_name": name, "feature_description": desc})
            continue

        # 3) Fresh id
        cur += 1
        new_id = str(cur)
        seen_new_keys[key] = new_id
        out_rows.append({"feature_id": new_id, "feature_name": name, "feature_description": desc})

    df = pd.DataFrame(out_rows)
    json_text = json.dumps(out_rows, ensure_ascii=False, indent=2)
    return df, json_text

# -----------------------------------------------------------
# Streamlit App — UI
# -----------------------------------------------------------
logo_icon = None
try:
    logo_icon = Image.open("Image/full_width.png")
except Exception:
    pass

st.set_page_config(
    page_title="Geo-Reg Compliance — BandAI",
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
    
st.markdown(
    """
    <div class="hero">
      <h1 class="hero-title">
        <span class="emoji">🛡️</span>
        <span class="grad">Geo-Reg Compliance Checker</span>
        <span class="tag">DUO AI • Powered by RAG</span>
      </h1>
      <div class="hero-actions">
        <a class="link-btn primary" href="https://lookerstudio.google.com/reporting/e081e397-63bd-4000-9227-893fc9d86033" target="_blank" rel="noopener">
          <svg class="ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3ZM5 5h6v2H7v10h10v-4h2v6H5V5Z"/>
          </svg>
          Open Looker Studio
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

    # Handle both singular and plural just in case
    for col in ("past_record", "past_records"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False, indent=2)
                if isinstance(x, (dict, list))
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

def build_precedent_index(dashboard_df: pd.DataFrame, allowed_regs: list[str]):
    """
    Vectorize past rows for nearest-neighbor lookup by feature text.
    Stores into session:
      PRECEDENT_ROWS: list[dict] canonicalized past rows
      PRECEDENT_VEC:  fitted TfidfVectorizer
      PRECEDENT_MAT:  tf-idf matrix for past rows

    Default confidence when missing/blank = 1.0 (strong stance).
    """
    rows = []
    if dashboard_df is not None and not dashboard_df.empty:
        df = dashboard_df.copy()

        wanted = ["feature_id", "feature_name", "feature_description",
                  "violation", "confidence_level", "reason", "regulations"]
        for c in wanted:
            if c not in df.columns:
                if c == "confidence_level":
                    df[c] = 1.0  # <-- default to 1.0 if column absent
                else:
                    df[c] = ""

        # If the column exists but has blanks/NaN/strings, coerce & fill with 1.0
        df["confidence_level"] = pd.to_numeric(df["confidence_level"], errors="coerce").fillna(1.0)

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
                "confidence_level": sanitize_confidence(r.get("confidence_level", 1.0)), 
                "reason": str(r.get("reason", "")),
                "regulations": sanitize_regulations_dynamic(regs_list, allowed_regs),
            })

    texts = [f"{x['feature_name'].strip()}\n{x['feature_description'].strip()}" for x in rows]
    if texts:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(texts)
        st.session_state.PRECEDENT_ROWS = rows
        st.session_state.PRECEDENT_VEC = vec
        st.session_state.PRECEDENT_MAT = mat
    else:
        st.session_state.PRECEDENT_ROWS = []
        st.session_state.PRECEDENT_VEC = None
        st.session_state.PRECEDENT_MAT = None

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
    out["reason2"] = "No sufficiently similar precedent. Kept Tik AI."
    out["past_record"] = {}
    return enforce_schema(out, allowed_regs)

# ================== SIDEBAR (COMPACT & SAFE) ==================
with st.sidebar:
    st.markdown('<div class="sidebar-wrap">', unsafe_allow_html=True)

    # ---------- 1) Upload Terminology ----------
    st.markdown('<div class="sb-card"><h4>Upload Terminology</h4>', unsafe_allow_html=True)
    with st.form("form_term_upload", clear_on_submit=False):
        term_upload = st.file_uploader("Terminology", type=["csv", "xlsx"], key="term_upload")
        c1, c2 = st.columns(2)
        submitted_term = c1.form_submit_button("📥 Load", use_container_width=True)
        clear_term     = c2.form_submit_button("🧹 Clear", use_container_width=True)

    if submitted_term:
        try:
            term_df, terminology_json_text = parse_terminology_upload(term_upload)
            st.session_state["TERMINOLOGY_DF"] = term_df
            st.session_state["TERMINOLOGY_JSON_TEXT"] = terminology_json_text
            if term_df is not None and not term_df.empty:
                st.success(f"Loaded {len(term_df)} terminology rows.")
                with st.expander("Preview (first 10)"):
                    st.dataframe(term_df.head(10), use_container_width=True)
            else:
                st.info("Terminology file is empty or could not be parsed.")
        except Exception as e:
            st.error(f"Failed to parse terminology: {e}")

    if clear_term:
        st.session_state.pop("TERMINOLOGY_DF", None)
        st.session_state.pop("TERMINOLOGY_JSON_TEXT", None)
        st.toast("Cleared terminology.", icon="🧹")

    if st.session_state.get("TERMINOLOGY_DF") is not None:
        st.caption(f"✓ Terminology ready · {len(st.session_state['TERMINOLOGY_DF'])} rows")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 2) Features Source ----------
    st.markdown('<div class="sb-card"><h4>Features Source</h4>', unsafe_allow_html=True)
    with st.form("form_features_upload", clear_on_submit=False):
        features_file = st.file_uploader(
            "Features (feature_name, feature_description)",
            type=["csv", "xlsx"], key="features_upload"
        )
        c1, c2 = st.columns(2)
        submitted_feat = c1.form_submit_button("📥 Load", use_container_width=True)
        clear_feat     = c2.form_submit_button("🧹 Clear",  use_container_width=True)

    if submitted_feat:
        st.session_state["FEATURES_FILE"] = features_file
        st.success("Features Loaded")

    if clear_feat:
        st.session_state.pop("FEATURES_FILE", None)
        st.toast("Cleared features.", icon="🧹")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 3) Regulations (.txt) — Optional ----------
    st.markdown('<div class="sb-card"><h4>Regulations (.txt) — Optional</h4>', unsafe_allow_html=True)
    with st.form("form_regs_upload", clear_on_submit=False):
        regs_uploads = st.file_uploader(
            "Upload TXT file(s)", type=["txt"], accept_multiple_files=True, key="regs_txt_uploader"
        )
        c1, c2 = st.columns(2)
        submitted_regs = c1.form_submit_button("💾 Save & Refresh RAG", use_container_width=True)
        clear_regs     = c2.form_submit_button("🧹 Clear list",          use_container_width=True)

    if submitted_regs:
        saved_names = []
        try:
            os.makedirs("Regulations", exist_ok=True)
            if regs_uploads:
                for f in regs_uploads:
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f.name.strip()) or "regulations.txt"
                    path = os.path.join("Regulations", safe_name)
                    content = f.getvalue()
                    if isinstance(content, str):  # ensure bytes
                        content = content.encode("utf-8", errors="replace")
                    with open(path, "wb") as out:
                        out.write(content)
                    saved_names.append(safe_name)

            loaded = reload_regulations("Regulations")
            st.session_state.LOADED_REG_FILES = loaded
            st.success(f"RAG refreshed with {len(loaded)} regulation file(s).")
            if saved_names:
                st.caption("Saved: " + ", ".join(saved_names))
        except Exception as e:
            st.error(f"Save/refresh failed: {e}")

    if clear_regs:
        # chỉ clear danh sách đã load trong session (không xóa file trên disk)
        st.session_state.pop("LOADED_REG_FILES", None)
        st.toast("Cleared loaded list (files on disk kept).", icon="🧹")

    loaded_regs = st.session_state.get("LOADED_REG_FILES") or []
    if loaded_regs:
        with st.expander(f"📚 Loaded regulations ({len(loaded_regs)})"):
            for fn in loaded_regs:
                st.markdown(f"• {fn}")
    else:
        st.caption("No regulation files loaded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /.sidebar-wrap
# ================== END SIDEBAR ==================


# --- Clear sheets with two-step confirm ---
if "clear_button_armed" not in st.session_state:
    st.session_state.clear_button_armed = False

# Step 1: arm the clear
if st.button("🧹 Clear Dashboard & Timelogs", type="secondary", key="btn_clear_arm"):
    st.session_state.clear_button_armed = True

# Step 2: show confirm UI only while armed
if st.session_state.clear_button_armed:
    st.warning(
        "This will ERASE all rows below the header in both sheets. "
        "This cannot be undone. Click confirm if you're sure.",
        icon="⚠️"
    )
    c1, c2 = st.columns(2)

    with c1:
        if st.button("✅ Confirm Clear", type="primary", key="btn_clear_confirm"):
            try:
                gc = _gc_from_dict()
                sh = gc.open_by_key(SPREADSHEET_ID)

                for sheet_name in ("DASHBOARD", "TIMELOGS"):
                    try:
                        ws = sh.worksheet(sheet_name)
                        ws.batch_clear(["A2:Z"])
                    except gspread.WorksheetNotFound:
                        st.info(f"{sheet_name} sheet not found, skipped.")

                # Reset local session state
                st.session_state.TIMELOGS = []
                st.session_state.AI1_DF = None
                st.session_state.AI2_DF = None
                st.session_state.AI2_FULL_DF = None
                st.session_state.DASHBOARD_READY = pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])
                st.session_state.EXISTING_DASH = pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])
                st.session_state.EXISTING_LOGS = pd.DataFrame(columns=[
                    "ts","event","feature_id","feature_name","feature_description",
                    "before_violation","before_confidence","before_regulations",
                    "after_violation","after_confidence","after_regulations","note"
                ])
                st.cache_data.clear()
                st.success("✅ Cleared DASHBOARD + TIMELOGS.")
            except Exception as e:
                st.error(f"Failed to clear: {e}")
            finally:
                # hide confirm UI after confirm
                st.session_state.clear_button_armed = False

    with c2:
        if st.button("✖️ Cancel", type="secondary", key="btn_clear_cancel"):
            st.info("Clear cancelled.")
            # hide confirm UI after cancel
            st.session_state.clear_button_armed = False

st.markdown("### 📚 Regulations")

# Lấy danh sách file
loaded_regs = st.session_state.get("LOADED_REG_FILES")
if loaded_regs is None:
    try:
        from AI_main import loaded_reg_files
        loaded_regs = loaded_reg_files
    except Exception:
        loaded_regs = []

# UI hiển thị
if not loaded_regs:
    st.warning("⚠️ No regulation files are currently loaded.")
else:
    st.success(f"✅ {len(loaded_regs)} regulation file(s) loaded.")
    with st.expander("📂 Show files", expanded=False):
        st.markdown(
            "<ul class='reg-list'>" +
            "".join([f"<li>{fn}</li>" for fn in loaded_regs]) +
            "</ul>",
            unsafe_allow_html=True
        )


# ---- Allowed regulations come from the Regulations/ folder + "None" ----
def _allowed_regs_from_loaded(loaded_files: list[str]) -> list[str]:
    names = []
    for fn in (loaded_files or []):
        base = os.path.splitext(os.path.basename(fn))[0]  # drop ".txt"
        if base:
            names.append(base.strip())
    names = sorted(set(n for n in names if n))
    # Always allow "None"
    return (names + ["None"]) if names else ["None"]

ALLOWED_REGS = _allowed_regs_from_loaded(loaded_regs)
build_precedent_index(st.session_state.EXISTING_DASH, ALLOWED_REGS)

# ---- Pull current inputs from session (avoid NameError on rerun) ----
# ---- Pull current inputs from session (avoid NameError on rerun) ----
features_file = st.session_state.get("FEATURES_FILE", None)
terminology_json_text = st.session_state.get("TERMINOLOGY_JSON_TEXT", "")

# ---- Build/keep a local "known features" pool for ID reuse across submits ----
# Seed with what's already on the sheet (EXISTING_DASH), then we’ll add newly staged features.
if "FEATURES_KNOWN_DF" not in st.session_state:
    base_pool = st.session_state.get("EXISTING_DASH", pd.DataFrame())
    if not base_pool.empty:
        pool = base_pool[["feature_id","feature_name","feature_description"]].copy()
        pool["feature_id"] = pool["feature_id"].astype(str)
    else:
        pool = pd.DataFrame(columns=["feature_id","feature_name","feature_description"])
    st.session_state.FEATURES_KNOWN_DF = pool

# -----------------------------
# Features input (TEXT) + Submit
# -----------------------------
st.header("Features input")
with st.form("features_text_form", clear_on_submit=False):
    features_text = st.text_area(
        "For quick checks, paste features in following format without brackets (Feature_name: Feature_description per line)",
        value=st.session_state.get("FEATURES_PASTE", ""),
        height=120,
    )
    submit_text = st.form_submit_button("📥 Submit text features")

if submit_text:
    # Parse ONLY the text just submitted, reusing IDs against the known pool
    df_from_text, _json_ignored = parse_features(
        file_upload=None,
        text_input=features_text,
        existing_dash=st.session_state.FEATURES_KNOWN_DF
    )
    # Stash raw text + preview DF
    st.session_state.FEATURES_PASTE = features_text
    st.session_state.FEAT_TEXT_DF = df_from_text

    # Expand the known pool so subsequent builds also reuse IDs
    pool = pd.concat(
        [st.session_state.FEATURES_KNOWN_DF, df_from_text[["feature_id","feature_name","feature_description"]]],
        ignore_index=True
    )
    pool["feature_id"] = pool["feature_id"].astype(str)
    # De-dupe by (name, desc)
    pool = pool.drop_duplicates(subset=["feature_name","feature_description"], keep="last")
    st.session_state.FEATURES_KNOWN_DF = pool

    st.success(f"Staged {len(df_from_text)} feature(s) from text.")

# ---------------------------------------------
# Combine text + loaded file, reuse IDs correctly
# ---------------------------------------------
if st.button("📦 Upload all features"):
    # Use last submitted text from session (don’t parse unsaved textarea)
    text_src = st.session_state.get("FEATURES_PASTE", "")

    # Known pool already includes EXISTING_DASH + any staged text so far
    combined_existing = st.session_state.FEATURES_KNOWN_DF

    # Build from BOTH sources in one go so within-upload de-dup also applies
    df_all, _json_from_parse = parse_features(
        file_upload=features_file,
        text_input=text_src,
        existing_dash=combined_existing
    )

    # De-dup the OUTPUT for display (same name+desc appearing in both text & file)
    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["feature_name","feature_description"], keep="last").reset_index(drop=True)

    st.session_state.FEATURES_DF = df_all
    st.session_state.COMBINED_FEATURE_JSON = json.dumps(
        df_all.to_dict(orient="records"),
        ensure_ascii=False,
        indent=2
    )

    # Grow the known pool with the final combined set
    pool = pd.concat(
        [combined_existing, df_all[["feature_id","feature_name","feature_description"]]],
        ignore_index=True
    )
    pool["feature_id"] = pool["feature_id"].astype(str)
    pool = pool.drop_duplicates(subset=["feature_name","feature_description"], keep="last")
    st.session_state.FEATURES_KNOWN_DF = pool

    st.success(f"Built {len(df_all)} feature(s) from text + file with ID reuse.")

# ---- Show the currently staged/combined features ----
features_df = st.session_state.get("FEATURES_DF", pd.DataFrame())
combined_feature_json = st.session_state.get("COMBINED_FEATURE_JSON", "[]")

if features_df.empty:
    st.info("No features detected yet. Submit text features and/or load a file, then click “Upload all features”.")
else:
    st.dataframe(features_df, use_container_width=True)
    st.download_button(
        "⬇️ Download combined_feature.json",
        data=combined_feature_json,
        file_name="combined_feature.json",
        mime="application/json",
    )

# ---- Guard before run (no undefined variables; no 'documents') ----
can_run = (features_df.shape[0] > 0) and bool(terminology_json_text)
if not can_run:
    st.warning("To run Tik AI, please provide: Terminology CSV/XLSX, and at least one Feature.")


# Run AI_1
st.header("Run Compliance Detection (Tik AI)")
run_btn = st.button("▶️ Run with Tik AI", disabled=not can_run)

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
                raise ValueError("Tik AI returned invalid JSON")
            norm = ResultSchema.normalize(data, allowed_regs=ALLOWED_REGS)
        except Exception as e:
            norm = {
                "feature_id": r["feature_id"],
                "feature_name": r["feature_name"],
                "feature_description": r["feature_description"],
                "violation": "Unclear",
                "confidence_level": 0.0,
                "reason": f"Tik AI error: {e}",
                "regulations": ["None"],
            }
        rows_out.append(norm)
        # log prediction
        log_event(
            "prediction",
            feature={k: r[k] for k in ["feature_id","feature_name","feature_description"]},
            after={k: norm[k] for k in ["violation","confidence_level","regulations"]},
            note="Tik AI output recorded",
        )
        prog.progress(int((i + 1) / total * 100)); time.sleep(0.02)
    prog.empty()

    st.session_state.AI1_DF = pd.DataFrame(rows_out)
    st.session_state.RESULTS_READY = True
    st.success("Tik AI completed.")

# Results & Exports
if st.session_state.RESULTS_READY and st.session_state.AI1_DF is not None:
    st.subheader("Results — Tik AI")
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
    allowed = ALLOWED_REGS

    # -------- Compute phase: only when button is clicked ----------
    if st.button("▶️ Run Tok AI on Tik AI results"):
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

            # --- Normalize Tok keys/shapes so downstream UI is happy ---
            # Prefer singular 'past_record'
            if isinstance(ai2, dict):
                if "past_records" in ai2 and "past_record" not in ai2:
                    ai2["past_record"] = ai2.pop("past_records")
                # Ensure past_record is JSON-serializable (dict or list); else make it {}
                if not isinstance(ai2.get("past_record"), (dict, list)):
                    ai2["past_record"] = ai2.get("past_record") or {}
                # Normalize regulations to list[str]
                regs = ai2.get("regulations", [])
                if isinstance(regs, str):
                    try:
                        j = json.loads(regs)
                        regs = j if isinstance(j, list) else [t.strip() for t in regs.split(",") if t.strip()]
                    except Exception:
                        regs = [t.strip() for t in regs.split(",") if t.strip()]
                ai2["regulations"] = [str(x).strip() for x in regs if str(x).strip()]

                        
            log_event("prediction",
                      feature={"feature_id": ai2.get("feature_id"),
                               "feature_name": ai2.get("feature_name"),
                               "feature_description": ai2.get("feature_description")},
                      after={"violation": ai2.get("violation"),
                             "confidence_level": ai2.get("confidence_level"),
                             "regulations": ai2.get("regulations")},
                      note="Tik AI output recorded")
            ai2_rows.append(ai2)
            prog.progress(int((i + 1) / total * 100)); time.sleep(0.02)
        prog.empty()

        ai2_df = pd.DataFrame(ai2_rows)
        st.session_state.AI2_DF = ai2_df.copy()
        st.session_state.AI2_FULL_DF = ai2_df.copy()

        # make editable copy (merge reason+reason2; stringify regs)
        edit_df = ai2_df.copy()

        # --- Normalize columns before editor ---
        # violation: only three
        valid_v_choices = ["Yes", "No", "Unclear"]
        edit_df["violation"] = edit_df["violation"].apply(lambda v: v if v in valid_v_choices else "Unclear")

        # confidence_level: coerce to number
        edit_df["confidence_level"] = pd.to_numeric(edit_df.get("confidence_level", 0), errors="coerce").fillna(0.0)

        # regulations: ensure list[str] in-memory (editor will show as comma string, we’ll parse on save)
        def _to_list(v):
            if isinstance(v, list):
                return v
            s = "" if v is None else str(v)
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x) for x in j]
            except Exception:
                pass
            return [t.strip() for t in s.split(",") if t.strip()]

        edit_df["regulations"] = edit_df["regulations"].apply(_to_list)

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
def _save_ai2_corrections_simple(edited_df: pd.DataFrame):
    """
    Save ANY row where confidence_level == 2 into AI2_FULL_DF and mirror into AI2_DF.
    - No regulation whitelist, no violation-regs coupling.
    - Multiple rows update correctly.
    """
    if st.session_state.get("AI2_FULL_DF") is None:
        # initialize from AI2_DF or AI1_DF
        if st.session_state.get("AI2_DF") is not None and not st.session_state.AI2_DF.empty:
            st.session_state.AI2_FULL_DF = st.session_state.AI2_DF.copy()
        elif st.session_state.get("AI1_DF") is not None and not st.session_state.AI1_DF.empty:
            st.session_state.AI2_FULL_DF = st.session_state.AI1_DF.copy()
        else:
            st.error("No results to save."); return

    base_df = st.session_state.AI2_FULL_DF.copy()
    base_df["feature_id"] = base_df["feature_id"].astype(str)

    saved = 0
    bad = []

    for _, row in edited_df.iterrows():
        fid = str(row.get("feature_id"))
        try:
            conf = sanitize_confidence(row.get("confidence_level", 0))
            if conf != 2:
                continue  # only save rows marked as human-reviewed

            if fid not in base_df["feature_id"].astype(str).values:
                bad.append((fid, "Unknown feature_id")); continue

            # Build final normalized record
            v = sanitize_violation(row.get("violation", "Unclear"))
            reason = str(row.get("reason", "") or "")
            regs_list = _parse_regs_text_any(row.get("regulations_text", ""))

            mask = (base_df["feature_id"].astype(str) == fid)
            idx = base_df.index[mask]
            if len(idx) == 0:
                bad.append((fid, "Unknown feature_id")); continue

            # Assign scalar/series so multi-row updates are safe
            base_df.loc[idx, "violation"] = v
            base_df.loc[idx, "confidence_level"] = 2
            base_df.loc[idx, "reason"] = reason
            base_df.loc[idx, "regulations"] = pd.Series([regs_list] * len(idx), index=idx)

            # timelog (optional)
            before = {}  # you can fetch old row if needed
            after = {
                "feature_id": fid,
                "violation": v,
                "confidence_level": 2,
                "reason": reason,
                "regulations": regs_list,
            }
            log_event("human_intervention_simple",
                      feature={"feature_id": fid,
                               "feature_name": row.get("feature_name"),
                               "feature_description": row.get("feature_description")},
                      before=before, after=after,
                      note="Human override saved (simple mode)")
            saved += 1
        except Exception as e:
            bad.append((fid, f"Unexpected error: {e}"))

    # Commit back to session
    st.session_state.AI2_FULL_DF = base_df.copy()
    st.session_state.AI2_DF = base_df.copy()  # mirror

    # Build dashboard-ready table
    dash_df = base_df.copy()
    dash_df["feature_id"] = dash_df["feature_id"].astype(str)
    dash_df["regulations"] = dash_df["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    st.session_state.DASHBOARD_READY = dash_df[["feature_id","feature_name","feature_description","violation","reason","regulations"]]

    if bad:
        st.error("Some rows were not saved:\n" + "\n".join(f"• {a}: {b}" for a, b in bad))
    if saved:
        st.success(f"Saved {saved} row(s) (confidence_level = 2).")

# -------- Editable table & save ----------
has_ai2 = st.session_state.get("AI2_DF") is not None and not st.session_state.AI2_DF.empty
has_ai1 = st.session_state.get("AI1_DF") is not None and not st.session_state.AI1_DF.empty

if has_ai2 or has_ai1:
    # Prefer Tok output when available; otherwise fall back to Tik output
    base_df = st.session_state.AI2_DF.copy() if has_ai2 else st.session_state.AI1_DF.copy()
    suite_name = "Tok" if has_ai2 else "Tik"

    # Ensure columns exist
    for c in ["feature_id","feature_name","feature_description","violation","confidence_level","reason","regulations"]:
        if c not in base_df.columns:
            base_df[c] = "" if c not in ("confidence_level","regulations") else (0.0 if c == "confidence_level" else [])

    # Normalize dtypes for editing
    base_df["feature_id"] = base_df["feature_id"].astype(str)
    base_df["violation"] = base_df["violation"].apply(sanitize_violation)
    base_df["confidence_level"] = pd.to_numeric(base_df.get("confidence_level", 0), errors="coerce").fillna(0.0)

    # ---- READ-ONLY FIRST ----
    with st.expander(f"{suite_name} AI Results (read-only)", expanded=True):
        st.dataframe(_arrow_safe(base_df), use_container_width=True)

    # Helpful hint if Tok hasn't been run yet
    if not has_ai2:
        st.info("Tok AI hasn’t been run yet. Click **▶️ Run Tok AI on Tik AI results** above to generate Tok output.")

    # Prepare editable view
    def _regs_to_text(x):
        if isinstance(x, list):
            return ", ".join([str(t) for t in x])
        return "" if x is None else str(x)

    edit_df = base_df.copy()
    edit_df["regulations_text"] = edit_df["regulations"].apply(_regs_to_text)

    # Only expose the main editable columns
    edit_df = edit_df[[
        "feature_id", "feature_name", "feature_description",
        "violation", "confidence_level", "reason", "regulations_text"
    ]]

    # ---- EDITABLE SECOND ----
    with st.expander(f"{suite_name} AI Summary (editable)", expanded=False):
        st.caption("Edit freely. Set confidence_level = 2 for rows you want to SAVE as human-reviewed.")
        with st.form("ai2_edit_form_simple", clear_on_submit=False):
            edited_df = st.data_editor(
                edit_df,
                key="ai2_editor_table_simple",
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "violation": st.column_config.SelectboxColumn(
                        "violation", options=["Yes","No","Unclear"]
                    ),
                    "confidence_level": st.column_config.NumberColumn(
                        "confidence_level", min_value=0.0, max_value=2.0, step=0.1
                    ),
                    "regulations_text": st.column_config.TextColumn(
                        "regulations (comma-separated)",
                        help="Type anything, e.g., Florida HB 3, Utah SMRA, COPPA Section 312.5"
                    ),
                },
            )
            submitted = st.form_submit_button("💾 Save rows where confidence_level = 2", type="primary")

        if submitted:
            st.session_state["ai2_editor_source"] = edited_df.copy(deep=True)
            _save_ai2_corrections_simple(edited_df)
            st.toast("Saved. Dashboard & tables updated.", icon="✅")

    # Needs Review (based on whichever table we're showing)
    def _recompute_needs_review():
        df = st.session_state.get("AI2_DF") if has_ai2 else st.session_state.get("AI1_DF")
        if df is None or df.empty:
            st.session_state.NEEDS_REVIEW = pd.DataFrame()
            return
        try:
            conf = df["confidence_level"].astype(float)
        except Exception:
            conf = pd.to_numeric(df["confidence_level"], errors="coerce").fillna(0.0)
        st.session_state.NEEDS_REVIEW = df[(df["violation"] == "Unclear") | (conf < 0.3)]

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

    # Dashboard preview (will be refreshed by save helper)
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
with st.expander("Export Excel & Update Google Sheets", expanded=False):

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
