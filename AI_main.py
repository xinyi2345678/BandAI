import faiss
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
from dotenv import load_dotenv
import json

load_dotenv()
# Insert your API key here
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key= openai_api_key)

# Load and index regulations
documents = []
index: faiss.IndexFlatL2 | None = None
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def reload_regulations(folder: str = "Regulations") -> list[str]:
    """
    (Re)load all .txt files from `folder`, rebuild embeddings and FAISS index.
    Returns: list of loaded filenames (sorted).
    """
    global documents, index

    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    txt_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
    txt_paths.sort()

    new_docs: list[str] = []
    for p in txt_paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            new_docs.append(f.read())

    # Empty folder → keep empty state
    if not new_docs:
        documents = []
        index = None
        return []

    # Embed & build FAISS
    doc_embeddings = embedder.encode(new_docs, convert_to_numpy=True)
    dim = int(doc_embeddings.shape[1])
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.asarray(doc_embeddings, dtype=np.float32))

    documents = new_docs
    index = faiss_index
    return [os.path.basename(p) for p in txt_paths]

# Call once at import
loaded_reg_files = reload_regulations("Regulations")
def _allowed_reg_names() -> list[str]:
    # Take basenames from loaded_reg_files, drop ".txt"
    names = []
    for fn in (loaded_reg_files or []):
        base = os.path.splitext(os.path.basename(fn))[0]
        if base:
            names.append(base.strip())
    names = sorted(set(n for n in names if n))
    # Always allow "None"
    return (names + ["None"]) if names else ["None"]

def rag_answer(row_dict, term_text, top_k=3):
    feature_id = row_dict.get('feature_id')
    feature_name = row_dict.get('feature_name', '')
    feature_description = row_dict.get('feature_description')
    query = (feature_name or "").strip() + "\n" + (feature_description or "").strip()

    # Optional RAG retrieval
    context = ""
    if index is not None and documents:
        query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        k = min(top_k, len(documents))
        D, I = index.search(query_embedding, k)
        retrieved_docs = [documents[i] for i in I[0] if 0 <= i < len(documents)]
        context = "\n".join(retrieved_docs)

    allowed = _allowed_reg_names()
    allowed_lines = "\n".join(f"- \"{a}\"" for a in allowed)

    prompt = f"""
You are a compliance assistant. Decide if the feature violates listed regulations.

Hard rules:
- "violation" MUST be either "Yes" or "No".
- Reason must be a carefully thought and concise explanation of why you think it is a violation or not.
- If Yes, you must return the relevant regulation(s) that the feature violates. However, if you think it's a violation but you unable to find at least 1 relevant regulation, return no, but state why you think it violates.
- If No, you must talk about which regulation is relevant and explain why the feature does not violate the regulations.
- You are ONLY allowed to return regulation names from the ALLOWED LIST below. Do NOT invent new names.
- If you cannot cite at least one allowed regulation, set"regulations": ["None"].

ALLOWED REGULATION NAMES:
{allowed_lines}

Return ONLY valid JSON exactly as:
{{
  "feature_id": <original id>,
  "feature_name": "<short title if available or derive from description>",
  "feature_description": "<original text>",
  "violation": "<Yes|No>",
  "reason": "<concise explanation>",
  "confidence_level": <number between 0 and 1 inclusive>,
  "regulations": [list of allowed regulation names OR ["None"]]
}}

Terminologies in json:
{term_text}

Regulations (may be empty if none provided):
{context}
and "None".

feature_id:
{feature_id}

feature_name:
{feature_name}

feature_description:
{feature_description}
"""
    response = client.chat.completions.create(
        model="ft:gpt-4.1-mini-2025-04-14:personal:bandai:CAEAHbEe",
        messages=[
            {"role": "system", "content": "You are a compliance assistant. Determine Yes/No and cite only ALLOWED regulation names."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content



def rag_answer2(response1, term_text, past_records):
    if len(past_records) > 100:
        past_texts = [json.dumps(pr, ensure_ascii=False) for pr in past_records]
        emb = embedder.encode(past_texts, convert_to_numpy=True)
        dim = int(emb.shape[1])
        local_idx = faiss.IndexFlatL2(dim)
        local_idx.add(np.asarray(emb, dtype=np.float32))

        q = f"{response1.get('feature_name','')}\n{response1.get('feature_description','')}".strip()
        q_emb = embedder.encode([q], convert_to_numpy=True)
        k = min(50, len(past_records))
        D, I = local_idx.search(np.asarray(q_emb, dtype=np.float32), k)
        retrieved_docs = [past_records[i] for i in I[0]]   # <-- use retrieved_docs
    else:
        retrieved_docs = past_records
    prompt = f"""
    You are a compliance assistant. Your job is to compare past records to see if the current decision made on whether a new feature violates a regulation. You are to either support or reject the decision made based on what happened in the past. Disregard the original confidence level when making your decisions.
    If no past records are given, just agree to whatever input you are given.
    Current decision:
    {response1}
    
    Terminologies:
    {term_text}
    Past records:
    {json.dumps(retrieved_docs)}
    Return ONLY valid JSON exactly as:
    {{
        "feature_id": original id of the feature,
        "feature_name": "<original text>",
        "feature_description": "<original text>",
        "violation": new decision for you to decide whether it is a violation or not. return Yes or No,
        "reason": "original reason for violation or non-violation",
        "reason2": reason for why you support or reject the current reason for violation or non-violation
        "confidence_level": original confidence_level. you do not need to give new confidence level for this output
        "regulations": original list of regulation(s) being violated, whether you support or reject it
        "past_records" : list of past records that are relevant to your support or reject argument containing the full json text.
        
    }}
    """
    response = client.chat.completions.create(
    model="ft:gpt-4.1-mini-2025-04-14:personal:bandai:CAEAHbEe",
    messages=[
        {"role": "system", "content": "You are a compliance assistant. Your job is to compare past records to see if the current decision made on whether a new feature violates a regulation."},
        {"role": "user", "content": prompt}
    ]
    )

    return response.choices[0].message.content
