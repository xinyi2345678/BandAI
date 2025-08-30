# streamlit_app.py
# -----------------------------------------------------------
# Geo-Reg Compliance Checker — Two-AI + Human-in-the-Loop
# - Fixed Vertex (project/region) + Gemini Flash (AI_1, AI_2)
# - RAG: Terminology + Regulations text -> chunks -> embeddings
# - Allowed regulations are parsed LIVE from the Regulations text (left of ':') + "None"
# - AI_1 produces JSON; AI_2 verifies/corrects using human-edited precedents
# - Human can edit AI_2 and save to memory; future similar items auto-correct
# - Results persist in session to avoid Streamlit rerun wipes
# -----------------------------------------------------------
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
from datetime import datetime
from io import BytesIO
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

import json
import re
import time
import typing as T
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

# ---- Fixed Vertex AI config ----
PROJECT_ID = "creative-apac"
LOCATION = "us-central1"
AI1_MODEL_NAME = "gemini-2.5-flash"
AI2_MODEL_NAME = "gemini-2.5-flash"

# ---- Vertex AI (Gemini + Embeddings) ----
try:
    from google.cloud import aiplatform
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_LIBS = True
except Exception:
    VERTEX_LIBS = False

# -----------------------------------------------------------
# Seed Knowledge (editable in UI)
# -----------------------------------------------------------
TERMINOLOGY_TEXT = """
NR: Not recommended; used to mark a setting that should not be overridden without parental consent.
PF: Personalized Feed; recommendation system that may require special handling for minors in some regions.
GH: Geo-handler; module that routes features based on user region.
CDS: Compliance Detection System; generates audit logs and evidence when rules trigger.
DRT: Data Retention Threshold; maximum duration for which logs can be stored.
LCP: Local Compliance Policy; region-specific policy parameters.
Redline: Flag for legal review.
Softblock: A user-level limitation applied silently without notifications.
Spanner: Name for rule engine used to isolate experiment cohorts.
ShadowMode: Non-user-visible deployment to collect analytics only.
T5: Tier-5 sensitivity content for escalation.
ASL: Age-sensitive logic; classification of user age to control features.
Glow: Compliance-flagging status for geo-based alerts.
NSP: Non-shareable policy; content should not be shared externally.
Jellybean: Parental control framework codename.
EchoTrace: Log tracing mode to verify compliance routing.
BB: Baseline Behavior; baseline used by anomaly detection.
Snowcap: Child safety policy framework codename.
FR: Feature rollout status tracking.
IMT: Internal monitoring trigger.
"""

REGULATIONS_TEXT = """
EU Digital Services Act (DSA): Platforms must ensure transparency and region-specific enforcement for content visibility, removal, and auditability in the EU/EEA.
California SB976: For users under 18 in California, certain personalization features (like PF) may be disabled by default unless parental opt-in is provided.
Florida minors law: Requires additional protections for minors, including stricter notification policies and parental safeguards.
Utah Social Media Regulation Act: Imposes curfew-based restrictions and parental controls for users under 18 located in Utah.
US federal reporting to NCMEC: Providers must report identified child sexual abuse material; triggers real-time scanning with audit logs.
General data retention requirements: Enforce DRT thresholds, automatic deletion, and region-aware retention audits via CDS.
"""
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
    def normalize(cls, row: dict, allowed_regs: T.List[str]):
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
            # fallback: unclear row
            return {
                "feature_id": row.get("feature_id"),
                "feature_name": row.get("feature_name", ""),
                "feature_description": row.get("feature_description", ""),
                "violation": "Unclear",
                "confidence_level": 0.0,
                "reason": f"Validation error: {e}",
                "regulations": ["None"],
            }
# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
SERVICE_ACCOUNT_INFO = {
  "type": "service_account",
  "project_id": "my-project-1573355121867",
  "private_key_id": "452992ce470b86f641b46a29a1c0f377d28ec2e3",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCv+cd0j9QprEfh\n5lIA7mz1GTB+P+FagEGmAcnV16T7ERfiGFz95i55BEiDtM7sVuB3nXOsVKncp6OW\nrcunxzDohdf+jrZZrqUAVdYXMuWevBgp4OjA23KAKqwh8lGjEh/xRPRMfYCHB8ck\n1USxc+s6Fgf5MGciaC1oNgusKXim885HSNT0BYrA+ygnm6ySiDpyHTlQNLKN9zyH\negPj0U1kI7iUicxBrE/DUfeCQcsrPqxvGvniF6bGVH8q1nJ49V5qgErkolJTvTLF\nbVcIZwszyAAWAgaFCrHead25uzIc/VPdk/AJnv9AjJh6nnigPAqmUwX5gaz+RSXG\nm8VWI+WRAgMBAAECggEADq9yacupbc9c4dEP0ZhUEymmmniQALaQk+8VBgGUmOei\npbl5BC65/Nsixov3kzFGUvEasrxPfxSl0hiflNkqf5MQ1QNZnBcXGbWwE7hJgYNK\npG8KmXKOxfuZqQ0Q60IDVKX3ma0FBW+8mpoqHQFylUK6qFzV1IEbXfWZqIS+9vag\n24dZrfm65EVqC5J6Wb3uy/WtPl09KyzngRhLGi4n1BT6bOswzqLRiZq+SrBJ9yPg\n+ZwnVHgysyegU6n25ZEa/8q2u62U3GXomtfn/IHjjzpWDtgPZvnPWvLWGjDIqpyG\nOC9yhmT7V9sm5lyAxK1BKAbbudEERszy4K4lCfDgiQKBgQDl+4YBvMmz1IIuCp/6\n9VXmPux3CSPerW3tK/eA98/p1Uz1BoLsw5paIwQ+RrY0FEP5JNPEwRi1NJU5Q3T2\n63nb4UZO6V3SfmoQPtciVAecDuXh9xsRPvMdebCIkLq4Kqdry/s928Mch0DpE+oq\ndv+j37yVf2L4GXSGOyJh+DvmyQKBgQDD4izk+dZZFesrHEAvBsDgLFuSs3Nzdkc+\nVLzpucYkF6sb6P5BbeFdSt2O8c6NyLqmTSgctPAS99CUzZTpCp5Hi4EYtZKtz8bt\nMwoWbHgxTJHf/24I+/YbpTj57f/kYDh72Buwz8GschDCWh+Q7jExXzeEnSr+ygS0\n4iH4KohEiQKBgFRwOs1cgTnzZjB9WiuL9BPrOmqiAnd5eYjAwciqM74IwI6d62f+\nkSdS//XVhIQuhJ5u9QmiU+4D9l3l9IXMAxvF5EiIyhfErjB0wgwqifi0R5blYRy9\n3gkOatBZQxTnJD0h0Yburv5EcoKg+zLIKigCt3y0HqQ0xGGcSI1r1KJxAoGAQ6y2\nef1e8rRB5UkDW7vnkwuAL7TT5EYu4vf/tHg8XmfW8/ORNCW0QLkGxsX/6Lg61A3A\nF/rjHoqDg4VrNwA2It2tok3I+UfZoEWL7KdY9x9PHqZu66exJWf1wVNanxonKZJG\nLtX4QY2/AIaGdVn1oOsWkTiDjDdbXOrrdYOsRJECgYBrGGSMHtc0mNI2Mj4NxL/T\n5HeOOItg3wBt9EpcN3gET2xPGqmG5sMYbABRMwAh2CwTtheJ9UpZ85wCNim18b2g\nPlt9SPpCT5nUkmRCCzbwYuLTtN8/dfdWV6ohrW3mgWzO8GNDxQArrJrMcpwiOsJV\nd+Bk04Z+eDU13yteDWjODw==\n-----END PRIVATE KEY-----\n",
  "client_email": "streamlit-sheets-writer@my-project-1573355121867.iam.gserviceaccount.com",
  "client_id": "110323219118800487585",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/streamlit-sheets-writer%40my-project-1573355121867.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def _gc_from_dict():
    """Authorize gspread using the inline service-account JSON."""
    creds = Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO, scopes=SCOPES
    )
    return gspread.authorize(creds)

def load_existing_sheets(spreadsheet_id: str):
    """Fetch DASHBOARD and TIMELOGS into DataFrames (empty if tabs missing)."""
    gc = _gc_from_dict()
    sh = gc.open_by_key(spreadsheet_id)

    # Try DASHBOARD
    try:
        dash_ws = sh.worksheet("DASHBOARD")
        df_dash = pd.DataFrame(dash_ws.get_all_records())
    except gspread.WorksheetNotFound:
        df_dash = pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])

    # Try TIMELOGS
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

def _as_str(x):  # keeps None as ""
    return "" if pd.isna(x) else str(x)


def _now_iso():
    # Always store in UTC for audit-friendliness
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def log_event(event: str, feature: dict | None = None, before: dict | None = None, after: dict | None = None, note: str = ""):
    """
    Append a structured event into session TIMELOGS.
    - event: "prediction" | "human_intervention" | "memory_clear" | "memory_import" | etc.
    - feature: minimal feature context (id/name/desc) to help find it in logs
    - before/after: dicts with only the relevant fields (kept small for readability)
    """
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

def static_code_like_scan(text: str) -> list[str]:
    """Very simple heuristic: flag if text mentions region or under-18 restrictions."""
    signals = []
    if re.search(r"\b(EU|EEA|GDPR|DSA|California|Utah|Florida|minors|under\s*18)\b", text, re.I):
        signals.append("geo_or_minor_ref")
    return signals # need to revise later, as this is hard coded
def config_like_scan(text: str) -> list[str]:
    """Stub: flag config-style geo rules."""
    hits = []
    if re.search(r"enabled_countries|age_limit|curfew", text, re.I):
        hits.append("config_geo_rule")
    return hits
def runtime_like_scan(text: str) -> list[str]:
    """Stub: flag runtime-like behaviors in text."""
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


def init_vertex() -> None:
    if not VERTEX_LIBS:
        st.error("Please install `google-cloud-aiplatform` and `vertexai`.")
        st.stop()
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    except Exception:
        pass

def chunk_text(text: str, max_chars: int = 600) -> T.List[str]:
    text = (text or "").strip().replace("\r", "")
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + (1 if buf else 0) + len(p) <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out

def build_kb_embeddings(passages: T.List[str]) -> np.ndarray:
    use_vertex = st.session_state.get("VERTEX_OK", False)
    if use_vertex:
        try:
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            embs = model.get_embeddings(passages)
            return np.array([e.values for e in embs], dtype=float)
        except Exception as e:
            st.warning(f"Vertex embedding failed; using TF-IDF. {e}")
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=4096)
    X = tfidf.fit_transform(passages).astype(float)
    st.session_state["TFIDF_VECTORIZER"] = tfidf
    return X.toarray()

def embed_query_vector(query: str, kb_texts: T.List[str]) -> np.ndarray:
    use_vertex = st.session_state.get("VERTEX_OK", False)
    if use_vertex:
        try:
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            v = model.get_embeddings([query])[0].values
            return np.array(v, dtype=float).reshape(1, -1)
        except Exception:
            pass
    tfidf: T.Optional[TfidfVectorizer] = st.session_state.get("TFIDF_VECTORIZER")
    if tfidf is None:
        tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=4096)
        _ = tfidf.fit_transform(kb_texts + [query])
        st.session_state["TFIDF_VECTORIZER"] = tfidf
    return tfidf.transform([query]).astype(float).toarray()

def retrieve(query: str, kb_texts: T.List[str], kb_vecs: np.ndarray, top_k: int = 3) -> T.List[str]:
    if kb_vecs is None or not kb_texts:
        return []
    sims = cosine_similarity(embed_query_vector(query, kb_texts), kb_vecs)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [kb_texts[i] for i in idxs]

def call_gemini(prompt: str, model_name: str) -> str:
    model = GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return resp.text or ""

def safe_json(s: str) -> T.Optional[dict]:
    if not s: return None
    s_strip = s.strip()
    try: return json.loads(s_strip)
    except Exception: pass
    s_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", s_strip, flags=re.I|re.M).strip()
    first, last = s_clean.find("{"), s_clean.rfind("}")
    if first == -1 or last == -1 or last <= first: return None
    for end in range(last, first, -1):
        try: return json.loads(s_clean[first:end+1])
        except Exception: continue
    return None

# -----------------------------------------------------------
# Allowed regulations (dynamic)
# -----------------------------------------------------------
def sanitize_violation(v: T.Any) -> str:
    v_str = str(v).strip().capitalize()
    return v_str if v_str in {"Yes", "No", "Unclear"} else "Unclear"

def sanitize_confidence(x: T.Any) -> T.Union[float, int]:
    try:
        val = float(x)
        if val == 2: return 2
        return max(0.0, min(val, 1.0))
    except Exception:
        return 0.0

def sanitize_regulations_dynamic(regs: T.Union[str, T.List[str]], allowed: T.List[str]) -> T.List[str]:
    if isinstance(regs, str):
        # accept JSON list or comma-separated text
        try:
            maybe = json.loads(regs)
            regs_list = maybe if isinstance(maybe, list) else [regs]
        except Exception:
            regs_list = [r.strip() for r in regs.split(",") if r.strip()]
    elif isinstance(regs, list):
        regs_list = regs
    else:
        regs_list = []
    # keep only exact labels from allowed, drop others
    out = [r for r in regs_list if r in allowed and r != "None"]
    out = sorted(set(out))
    return out if out else ["None"]

def enforce_schema(row: dict, allowed_regs: T.List[str]) -> dict:
    return ResultSchema.normalize(row, allowed_regs)

# -----------------------------------------------------------
# Two-AI prompts
# -----------------------------------------------------------
def ai1_prompt(feature_row: dict, evidence: T.List[str], allowed_regs: T.List[str]) -> str:
    allowed_str = ", ".join(allowed_regs)
    evid = "\n\n".join(
        [f"[EVIDENCE {i+1}]\n{e}" for i, e in enumerate(evidence)]
    ) if evidence else "(no evidence found)"
    schema = f"""
Return ONLY valid JSON exactly as:
{{
  "feature_id": "<string or number>",
  "feature_name": "<short title if available or derive from description>",
  "feature_description": "<original text>",
  "violation": "<Yes|No|Unclear>",
  "confidence_level": <number>,  // 0.0–1.0; human edits may later set 2
  "reason": "<concise explanation>",
  "regulations": ["<values ONLY from: {allowed_str}>"]
}}
""".strip()

    return f"""
You are AI_1, a compliance detector. Analyze the feature conservatively.

**STRICT RULES FOR "violation":**
- If the feature description or evidence mentions ANY terminology, region, minors, or regulation that could relate to compliance → always output "Yes".
- Output "No" only if you are highly confident the feature has NOTHING to do with compliance or regulations.
- Output "Unclear" only if the description is so vague that you cannot reasonably decide.

"regulations" MUST be chosen ONLY from this list: {allowed_str}. 
If none apply, use ["None"].

[FEATURE]
feature_id: {feature_row.get('feature_id')}
feature_name: {feature_row.get('feature_name', '')}
feature_description: {feature_row.get('feature_description')}

[EVIDENCE]
{evid}

{schema}
""".strip()

def ai2_prompt(ai1_json: dict, precedents: T.List[dict], allowed_regs: T.List[str]) -> str:
    allowed_str = ", ".join(allowed_regs)
    prec = "\n\n".join(
        [f"[PRECEDENT {i+1}] (intervened={p.get('intervened', False)})\n{json.dumps(p['output_json'], ensure_ascii=False)}"
         for i, p in enumerate(precedents)]
    ) if precedents else "(no similar precedents)"
    schema = f"""
Return ONLY valid JSON with the same fields as AI_1, using regulations ONLY from: {allowed_str}.
""".strip()
    return f"""
You are AI_2, a verifier/corrector. Align with highly similar human-edited precedents when appropriate.
Keep schema identical to AI_1 and use ONLY allowed regulations. Use ["None"] if none apply.

[AI_1 JSON]
{json.dumps(ai1_json, ensure_ascii=False)}

[PRECEDENTS]
{prec}

{schema}
""".strip()

# -----------------------------------------------------------
# AI calls
# -----------------------------------------------------------
def ai1_infer(row: dict, evidence: T.List[str], allowed_regs: T.List[str]) -> dict:
    raw = call_gemini(ai1_prompt(row, evidence, allowed_regs), model_name=AI1_MODEL_NAME)
    data = safe_json(raw)
    if not data:
        raise ValueError("AI_1 returned invalid JSON")
    return enforce_schema(data, allowed_regs)

def ai2_review(ai1_json: dict, feature_text: str, sim_threshold: float, allowed_regs: T.List[str]) -> dict:
    sims = memory_find_similar(feature_text, top_k=3)
    if sims:
        # if extremely similar precedent exists, copy it directly
        top_score, top_prec = sims[0]
        if top_score >= 0.90 and top_prec.get("intervened"):
            copied = {**ai1_json}
            copied.update(top_prec["output_json"])
            copied["reason"] = (copied.get("reason") or "") + " (applied from high-sim precedent)"
            return enforce_schema(copied, allowed_regs)

    strong = [p for (s, p) in sims if s >= sim_threshold and p.get("intervened")]
    precedents = strong if strong else [p for (_, p) in sims]
    # TODO: later replace call_gemini(...) with PROVIDER.generate(...) once AWS provider is ready
    raw = call_gemini(ai2_prompt(ai1_json, precedents, allowed_regs), model_name=AI2_MODEL_NAME)

    data = safe_json(raw)
    if not data:
        return ai1_json
    return enforce_schema(data, allowed_regs)


# -----------------------------------------------------------
# Corrections Memory
# -----------------------------------------------------------
def embed_texts(texts: T.List[str]) -> np.ndarray:
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embs = model.get_embeddings(texts)
    return np.array([e.values for e in embs], dtype=float)

def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0:1]

def memory_find_similar(query_text: str, top_k: int = 3) -> T.List[T.Tuple[float, dict]]:
    mem = st.session_state.get("CORR_MEM", [])
    M = st.session_state.get("CORR_EMBS", None)
    if not mem or M is None or M.shape[0] == 0:
        return []
    sims = cosine_similarity(embed_one(query_text), M)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(float(sims[i]), mem[i]) for i in idxs]

def memory_add(feature_text: str, final_json: dict):
    emb = embed_one(feature_text)
    rec = {"text": feature_text, "output_json": final_json, "intervened": True}
    st.session_state.CORR_MEM.append(rec)
    if st.session_state.CORR_EMBS is None or st.session_state.CORR_EMBS.shape[0] == 0:
        st.session_state.CORR_EMBS = emb
    else:
        st.session_state.CORR_EMBS = np.vstack([st.session_state.CORR_EMBS, emb])

# -----------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------

logo_icon = Image.open("Image/tiktok_logo.png")

# Page config with TikTok logo instead of shield
st.set_page_config(
    page_title="Geo-Reg Compliance — Two-AI (HIL)",
    page_icon=logo_icon,   # 👈 use your uploaded TikTok logo here
    layout="wide"
)
try:
    logo = Image.open("Image/full_width.png")
except Exception:
    logo = None

    
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Layout row: logo left, BandAI center
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if logo is not None:
        st.image(logo, width=200)  # bigger logo

with col2:
    st.markdown("""
    <div class="bandai-title">
        <span>B</span><span>a</span><span>n</span><span>d</span><span>A</span><span>I</span>
        <span class="rocket">🚀</span>
    </div>
    """,
    unsafe_allow_html=True)

with col3:
    st.markdown("")
    


st.title("🛡️ Geo-Reg Compliance Checker — Two-AI + Human-in-the-Loop")
st.caption(f"Project: {PROJECT_ID} • Region: {LOCATION} • Models: {AI1_MODEL_NAME}/{AI2_MODEL_NAME} • Gemini-only (no rules)")



# Session defaults
defaults = {
    "KB_TEXTS": [], "KB_VECS": None, "VERTEX_OK": False, "TFIDF_VECTORIZER": None,
    "TOP_K": 3, "SIM_THRESHOLD": 0.82, "CORR_MEM": [], "CORR_EMBS": None,
    "RESULTS_READY": False, "AI1_DF": None, "AI2_DF": None, "AI2_DF_STRING": None, "AI2_EDITED": None,
    "ALLOWED_REGS": None, "TIMELOGS": [],
}

for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.subheader("Knowledge Base (editable)")
    term_text = st.text_area("Terminology", TERMINOLOGY_TEXT, height=180)
    

    if st.button("🔁 Rebuild KB / Allowed regs"):
        # Build KB (RAG)
        kb_texts = chunk_text(term_text, 600)
        st.session_state.KB_TEXTS = kb_texts
        st.session_state.KB_VECS = build_kb_embeddings(kb_texts)
        # Parse allowed regs dynamically from the Regulations text
        
        st.success(f"KB rebuilt ({len(kb_texts)} chunks). Allowed regs: {len(st.session_state.ALLOWED_REGS)}")

    st.markdown("---")
    st.subheader("Retrieval & Similarity")
    st.session_state.TOP_K = st.slider("RAG: Top-K evidence", 1, 6, st.session_state.TOP_K)
    st.session_state.SIM_THRESHOLD = st.slider("AI_2 Similarity threshold", 0.50, 0.95, st.session_state.SIM_THRESHOLD, step=0.01)

    st.markdown("---")
    st.subheader("Corrections Memory")
    if st.session_state.CORR_MEM:
        st.caption(f"{len(st.session_state.CORR_MEM)} precedent(s) saved")
        if st.button("🧹 Clear memory"):
            st.session_state.CORR_MEM, st.session_state.CORR_EMBS = [], None
            st.success("Cleared corrections memory.")
        st.download_button(
            "⬇️ Export memory (JSONL)",
            data="\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.CORR_MEM),
            file_name="corrections_memory.jsonl",
            mime="application/jsonl"
        )
    else:
        st.caption("No precedents yet.")
    mem_upload = st.file_uploader("Import memory (JSONL)", type=["jsonl"])
    if mem_upload is not None:
        try:
            lines = [json.loads(l) for l in mem_upload.getvalue().decode("utf-8").splitlines() if l.strip()]
            texts = [r.get("text", "") for r in lines]
            st.session_state.CORR_MEM.extend(lines)
            if texts:
                embs = embed_texts(texts)
                st.session_state.CORR_EMBS = embs if st.session_state.CORR_EMBS is None else np.vstack([st.session_state.CORR_EMBS, embs])
            st.success(f"Imported {len(lines)} precedent(s).")
        except Exception as e:
            st.error(f"Import failed: {e}")

# Auto-init Vertex & KB on first load
if not st.session_state.VERTEX_OK:
    try:
        init_vertex()
        st.session_state.VERTEX_OK = True
        if not st.session_state.KB_TEXTS:
            kb_texts = chunk_text(TERMINOLOGY_TEXT, 600) + chunk_text(REGULATIONS_TEXT, 600)
            st.session_state.KB_TEXTS = kb_texts
            st.session_state.KB_VECS = build_kb_embeddings(kb_texts)
        if not st.session_state.ALLOWED_REGS:
            st.session_state.ALLOWED_REGS = parse_allowed_regulations(REGULATIONS_TEXT)
        st.success("Vertex initialized & KB ready.")
    except Exception as e:
        st.error(f"Vertex init failed: {e}")
        
SPREADSHEET_ID = "1sHmOYgL-J6AsYRg-4Bhap9ARxRfWMUEYyT3CEVT98i4"
existing_dash, existing_logs = load_existing_sheets(SPREADSHEET_ID)
st.session_state.EXISTING_DASH = existing_dash
st.session_state.EXISTING_LOGS = existing_logs


# 1) Upload CSV
with st.expander("1) Upload Feature CSV", expanded=True):
    st.write("CSV format (preferred): `feature_id, feature_name, feature_description` (extra columns ignored).")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file is None:
        st.info("No CSV uploaded. Using a small sample.")
        df = pd.DataFrame([
            {"feature_id": 1, "feature_name": "Utah Curfew Blocker", "feature_description": "Curfew login blocker with ASL and GH for Utah minors"},
            {"feature_id": 2, "feature_name": "EU Visibility Lock", "feature_description": "Content visibility lock with NSP and CDS for EU users per DSA"},
            {"feature_id": 3, "feature_name": "CA PF Default Off", "feature_description": "PF default toggle with NR enforcement for California teens"},
            {"feature_id": 4, "feature_name": "EU Video Replies Trial", "feature_description": "Trial run of video replies in EU (feature rollout only, no compliance)"},
            {"feature_id": 5, "feature_name": "Creators Leaderboard", "feature_description": "Leaderboard system for weekly creators"},
            {"feature_id": 6, "feature_name": "Unified Retention", "feature_description": "Unified retention control via DRT & CDS (automatic log deletion)"},
        ])
    else:
        df = pd.read_csv(csv_file)

    # Normalize columns
    required = ["feature_id", "feature_description"]
    if not all(c in df.columns for c in required):
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "feature_id", cols[1]: "feature_description"})
            if "feature_name" not in df.columns:
                df["feature_name"] = ""
        else:
            st.error("CSV must contain at least two columns, e.g., feature_id, feature_description")
            st.stop()
    elif "feature_name" not in df.columns:
        df["feature_name"] = ""

    st.dataframe(df[["feature_id", "feature_name", "feature_description"]].head(20), use_container_width=True)

# 2) Convert CSV -> JSON (preview)
with st.expander("2) Convert CSV → JSON", expanded=False):
    st.code(json.dumps(df[["feature_id", "feature_name", "feature_description"]].to_dict(orient="records")[:5], ensure_ascii=False, indent=2))

# Guard
if not st.session_state.VERTEX_OK:
    st.warning("This app requires Vertex AI (service account auth).")

# Core inference helpers
def predict_one(row: dict) -> T.Tuple[dict, dict]:
    kb_texts = st.session_state.KB_TEXTS
    kb_vecs = st.session_state.KB_VECS
    allowed = st.session_state.ALLOWED_REGS or ["None"]
    evidence = retrieve(row["feature_description"], kb_texts, kb_vecs, top_k=st.session_state.TOP_K) if kb_vecs is not None else []
    
    detector_signals = combined_detectors(row["feature_description"]) # add non-LLM detector evidence
    if detector_signals:
        evidence.append(f"[NON-LLM DETECTORS] Found: {', '.join(detector_signals)}")

    ai1 = ai1_infer(row, evidence, allowed_regs=allowed)
    ai2 = ai2_review(ai1, row["feature_description"], st.session_state.SIM_THRESHOLD, allowed_regs=allowed)
    ai1["source"], ai2["source"] = "AI_1", "AI_2"
    return ai1, ai2


# 3) Run
with st.expander("3) Run Compliance Detection (AI_1 → AI_2)", expanded=True):
    run_btn = st.button("▶️ Run with AI_1 → AI_2", disabled=not st.session_state.VERTEX_OK)

    if run_btn:
        ai1_rows, ai2_rows = [], []
        prog, total = st.progress(0), len(df)
        for i, row in df.iterrows():
            r = {
                "feature_id": row.get("feature_id"),
                "feature_name": row.get("feature_name", "") or "",
                "feature_description": str(row["feature_description"]),
            }
            try:
                ai1, ai2 = predict_one(r)
                log = {
                    "trace_id": f"{time.time_ns()}",
                    "feature_id": row["feature_id"],
                    "artifact": row["feature_description"],
                    "ai1_raw": ai1,
                    "ai2_raw": ai2,
                    "ts": _now_iso(),
                }
                if "AUDIT_LOGS" not in st.session_state:
                    st.session_state["AUDIT_LOGS"] = []
                st.session_state["AUDIT_LOGS"].append(log)

            except Exception as e:
                ai1 = {
                    "feature_id": r["feature_id"], "feature_name": r["feature_name"], "feature_description": r["feature_description"],
                    "violation": "Unclear", "confidence_level": 0.0, "reason": f"LLM error (AI_1): {e}", "regulations": ["None"], "source": "AI_1",
                }
                ai2 = {**ai1, "source": "AI_2"}
            ai1_rows.append(ai1); ai2_rows.append(ai2)
            prog.progress(int((i + 1) / total * 100)); time.sleep(0.02)
        prog.empty()

        # Persist tables (stringify regs for editing)
        ai1_df = pd.DataFrame(ai1_rows)
        ai2_df = pd.DataFrame(ai2_rows)
        ai2_df_str = ai2_df.copy()
        ai2_df_str["regulations"] = ai2_df_str["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        st.session_state.AI1_DF = ai1_df
        st.session_state.AI2_DF = ai2_df
        st.session_state.AI2_DF_STRING = ai2_df_str
        st.session_state.AI2_EDITED = ai2_df_str.copy()
        st.session_state.RESULTS_READY = True
        for idx, row in ai2_df.iterrows():
            feature_ctx = {
                "feature_id": row.get("feature_id"),
                "feature_name": row.get("feature_name"),
                "feature_description": row.get("feature_description"),
            }
            after = {
                "violation": row.get("violation"),
                "confidence_level": row.get("confidence_level"),
                "regulations": row.get("regulations"),
            }
            log_event("prediction", feature=feature_ctx, after=after, note="AI_2 output recorded")

        # Collect unclear / low-confidence rows for human review
        needs_review = ai2_df[
            (ai2_df["violation"] == "Unclear") | (ai2_df["confidence_level"].astype(float) < 0.3)
        ]
        st.session_state.NEEDS_REVIEW = needs_review

        if not needs_review.empty:
            st.warning(f"{len(needs_review)} feature(s) need human review (Unclear or low confidence).")

        st.success("AI_1 → AI_2 completed. You can edit below.")

# 4 & 5) View AI_1, Edit AI_2
if st.session_state.RESULTS_READY and st.session_state.AI1_DF is not None and st.session_state.AI2_DF_STRING is not None:
    with st.expander("4) AI_1 Drafts (read-only)", expanded=False):
        st.dataframe(st.session_state.AI1_DF, use_container_width=True)
        
    with st.expander("5) AI_2 Results (editable)", expanded=True):
        st.caption(
            "Edit rows you want to correct. To mark a row as HUMAN-INTERVENED, set confidence_level = 2. "
            "Only rows with confidence_level == 2 will be saved to memory."
        )
        if st.session_state.AI2_EDITED is not None:
            working_df = st.session_state.AI2_EDITED
        else:
            working_df = st.session_state.AI2_DF_STRING.copy()

        with st.form("ai2_edit_form", clear_on_submit=False):
            edited_df = st.data_editor(working_df, key="ai2_editor", use_container_width=True, num_rows="dynamic")
            st.session_state.AI2_EDITED = edited_df
            submitted = st.form_submit_button("💾 Save corrections to memory")

        if "NEEDS_REVIEW" in st.session_state and not st.session_state.NEEDS_REVIEW.empty:
            st.subheader("🚨 Needs Review")
            st.dataframe(st.session_state.NEEDS_REVIEW, use_container_width=True)
            st.download_button("⬇️ Download Needs Review CSV",
                data=st.session_state.NEEDS_REVIEW.to_csv(index=False),
                file_name="needs_review.csv",
                mime="text/csv")

        if submitted:
            base_df = st.session_state.AI2_DF_STRING
            allowed = st.session_state.ALLOWED_REGS or ["None"]
            saved = 0

            for idx in range(len(edited_df)):
                row_after_ui = edited_df.iloc[idx].to_dict()

                # Only accept as intervention if reviewer explicitly sets confidence_level=2
                try:
                    conf_val = float(row_after_ui.get("confidence_level", 0))
                except Exception:
                    conf_val = 0.0
                if conf_val != 2:
                    continue  # skip rows not flagged by human

                # BEFORE values (from last AI_2_STRING snapshot)
                row_before_ui = base_df.iloc[idx].to_dict()

                # Normalize regulations from edited text
                regs_str = row_after_ui.get("regulations", "")
                regs_list = [r.strip() for r in str(regs_str).split(",") if str(r).strip()]
                row_after_ui["regulations"] = sanitize_regulations_dynamic(regs_list, allowed)

                # Build canonical AFTER dict (list fields, numeric confidence, etc.)
                after_norm = enforce_schema(row_after_ui, allowed)
                after_norm["intervened"] = True

                # Save to memory (used by AI_2 as precedent)
                memory_add(after_norm["feature_description"], after_norm)
                saved += 1

                # Reflect normalized values back into session tables
                regs_str_norm = ", ".join(after_norm["regulations"])
                for target in ("AI2_DF_STRING", "AI2_EDITED"):
                    st.session_state[target].at[idx, "regulations"] = regs_str_norm
                    st.session_state[target].at[idx, "violation"] = after_norm["violation"]
                    st.session_state[target].at[idx, "confidence_level"] = after_norm["confidence_level"]
                    st.session_state[target].at[idx, "reason"] = after_norm["reason"]

                st.session_state.AI2_DF.at[idx, "regulations"] = after_norm["regulations"]
                for k in ("violation", "confidence_level", "reason"):
                    st.session_state.AI2_DF.at[idx, k] = after_norm[k]

                # 🔹 NEW: log the human intervention with before/after snapshot
                feature_ctx = {
                    "feature_id": row_after_ui.get("feature_id"),
                    "feature_name": row_after_ui.get("feature_name"),
                    "feature_description": row_after_ui.get("feature_description"),
                }
                before_norm = enforce_schema({
                    "feature_id": row_before_ui.get("feature_id"),
                    "feature_name": row_before_ui.get("feature_name"),
                    "feature_description": row_before_ui.get("feature_description"),
                    "violation": row_before_ui.get("violation"),
                    "confidence_level": row_before_ui.get("confidence_level"),
                    "reason": row_before_ui.get("reason"),
                    "regulations": [r.strip() for r in str(row_before_ui.get("regulations", "")).split(",") if r.strip()],
                }, allowed_regs=allowed)
                log_event("human_intervention", feature=feature_ctx, before=before_norm, after=after_norm, note="Reviewer set confidence_level=2")

            st.success(f"Saved {saved} human correction(s). Future similar features will be auto-corrected by AI_2.")

    
    # 6) Export
# 6) Export & Google Sheets sync
with st.expander("7) Export Excel (Dashboard + Timelogs)", expanded=False):

    def _norm_feature_id(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature_id to string to avoid mismatched merges."""
        if "feature_id" in df.columns:
            df = df.copy()
            df["feature_id"] = df["feature_id"].astype(str)
        return df

    def current_dashboard_from_session() -> pd.DataFrame:
        """Take current AI_2 results and normalize to dashboard format."""
        if st.session_state.AI2_DF is None or len(st.session_state.AI2_DF) == 0:
            return pd.DataFrame(columns=["feature_id","feature_name","feature_description","violation","reason","regulations"])
        df_cur = st.session_state.AI2_DF.copy()
        df_cur["feature_id"] = df_cur["feature_id"].astype(str)
        df_cur["regulations"] = df_cur["regulations"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        keep = ["feature_id","feature_name","feature_description","violation","reason","regulations"]
        return df_cur[keep]

    def merged_dashboard(existing: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """Overwrite existing rows by feature_id, append new rows."""
        base = _norm_feature_id(existing)
        cur = _norm_feature_id(current)

        all_cols = ["feature_id","feature_name","feature_description","violation","reason","regulations"]
        for df in (base, cur):
            for c in all_cols:
                if c not in df.columns:
                    df[c] = "" if c != "feature_id" else df.get("feature_id", "")
        base, cur = base[all_cols], cur[all_cols]

        base = base.set_index("feature_id")
        cur = cur.set_index("feature_id")
        base.update(cur)
        new_only = cur.loc[~cur.index.isin(base.index)]
        out = pd.concat([base, new_only], axis=0).reset_index()
        return out

    def appended_timelogs(existing_logs: pd.DataFrame) -> pd.DataFrame:
        """Append new timelogs to existing ones (history preserved)."""
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

    # --- Build merged dataframes
    current_dash = current_dashboard_from_session()
    dashboard_df = merged_dashboard(st.session_state.EXISTING_DASH, current_dash)
    timelogs_df = appended_timelogs(st.session_state.EXISTING_LOGS)

    # Preview
    st.subheader("Preview — DASHBOARD (merged)")
    st.dataframe(dashboard_df, use_container_width=True)
    st.subheader("Preview — TIMELOGS (appended)")
    st.dataframe(timelogs_df.tail(50), use_container_width=True)

    # Google Sheets helpers
    def ensure_ws(sh, title: str):
        try:
            return sh.worksheet(title)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=title, rows=100, cols=26)

    def write_preserve_header(ws, df: pd.DataFrame):
        ws.resize(1)  # keep row-1 headers
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

# # Later, Abstract LLM provider (prep for AWS)

