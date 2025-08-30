import faiss
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
load_dotenv()
# Insert your API key here
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key= openai_api_key)

txt_folder = "Regulations"  # Change to your folder name
txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith(".txt")]

documents = []
for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as f:
        doc_text = f.read()
        documents.append(doc_text)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def rag_answer(row_dict, term_text, top_k=3):
    feature_id = row_dict.get('feature_id')
    feature_name = row_dict.get('feature_name', '')
    feature_description = row_dict.get('feature_description')
    query = feature_name + "\n" + feature_description
    # Embed the query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    # Retrieve top_k documents
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    # Concatenate context
    context = "\n".join(retrieved_docs)
    
    # Build prompt
    prompt = f"""
    You are a compliance assistant. Your job is to check if a software feature violates any of the regulations listed above. 

    If you think it violates, you must return the relevant regulation(s) that the feature violates. However, if you are unable to find atleast 1 relevant regulation, return no, but state why you think it violates even though you can't find the regulation that is being violated.

    The list of regulations that you might or might not be given is:
    "EU Digital Services Act (DSA)",
    "California state law - Protecting Our Kids from Social Media Addiction Act",
    "Florida state law - Online Protections for Minors",
    "Utah Social Media Regulation Act",
    "US law on reporting child sexual abuse content to NCMEC -  Reporting requirements of providers",
    "None"

    You must give the exact name(s) stated in the list of regulations when returning the relevant regulation(s) that the feature violates.
    If there is no violations, you should be returning "None" for regulations.
    Return ONLY valid JSON exactly as:
    {{
        "feature_id": original id of the feature,
        "feature_name": "<short title if available or derive from description>",
        "feature_description": "<original text>",
        "violation": "<Yes|No|Unclear>",
        "reason": "<concise explanation>",
        "confidence_level": your confidence level between 0 and 1 (inclusive)
        "regulations": list of regulation(s) being violated
    }}
    Terminologies in json:
    {term_text}
    
    Regulations:
    {context}
    
    feature_id:
    {feature_id}
    
    feature_name:
    {feature_name}
    
    feature_description:
    {feature_description}
    
    Answer:
    Return ONLY valid JSON exactly as:
    {{
        "feature_id": original id of the feature,
        "feature_name": "<short title if available or derive from description>",
        "feature_description": "<original text>",
        "violation": "<Yes|No>",
        "reason": "<concise explanation>",
        "confidence_level": your confidence level between 0 and 1 (inclusive)
        "regulations": list of regulation(s) being violated
    }}
    """

    response = client.chat.completions.create(
    model="ft:gpt-4.1-mini-2025-04-14:personal:bandai:CAEAHbEe",
    messages=[
        {"role": "system", "content": "You are a compliance assistant. Your job is to check if a new feature violates any of the regulations listed."},
        {"role": "user", "content": prompt}
    ]
    )

    return response.choices[0].message.content


def rag_answer2(response1,term_text, past_records):
    if len(past_records)>100:
        text_documents = [json.loads(past_records) for item in json_text_list]
        embeddings = embedder.encode(text_documents, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        feature_name = response1.get("feature_name")
        feature_description = response1.get("feature_description")
        query = f"{feature_name}\n{feature_description}"
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(np.array(query_embedding), 50)
        retrieved_docs = [documents[i] for i in I[0]]  # Full JSON dicts
    else:
        retrieved_docs = past_records
    prompt = f"""
    You are a compliance assistant. Your job is to compare past records to see if the current decision made on whether a new feature violates a regulation. You are to either support or reject the decision made based on what happened in the past. Disregard the original confidence level when making your decisions.
    Current decision:
    {response1}
    
    Terminologies:
    {term_text}
    Past records:
    {retrieved_docs}
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
