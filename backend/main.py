import os
import io
import json
import re
import uuid
import fitz # PyMuPDF
import requests
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---------------------- Init ----------------------
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------- Embeddings & FAISS ----------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
doc_chunks = {}

# ---------------------- Groq API ----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"} if GROQ_API_KEY else {}

def call_groq(messages, model="llama-3.1-8b-instant"):
    print(f"[DEV] Calling Groq API with model={model} ...")
    if not GROQ_API_KEY:
        print("[DEV] ERROR: GROQ_API_KEY not set.")
        # Return a mock response for development without an API key
        return {"mock": True, "content": "Groq not configured"}
    try:
        resp = requests.post(
            GROQ_URL, headers=HEADERS, json={"model": model, "messages": messages}
        )
        resp.raise_for_status()
        result = resp.json()
        if "choices" not in result:
            print("[DEV] Groq API failed:", result)
            return {"error": "Groq API failed", "details": result}
        print("[DEV] Groq API call success.")
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"[DEV] Groq API Request failed: {e}")
        return {"error": "Groq API request failed", "details": str(e)}

def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def clean_json_response(answer: str):
    # This is the key change. We use a more robust regex to find the JSON block.
    json_match = re.search(r'\{.*\}', answer.strip(), re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # Fallback in case regex fails
    cleaned = re.sub(r"^```json", "", answer.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()

# ---------------------- Data Models ----------------------
class Flashcard(BaseModel):
    question: str
    answer: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str

class UploadResponse(BaseModel):
    summary: List[str]
    flashcards: List[Flashcard]
    quiz: List[QuizQuestion]

class ChatResponse(BaseModel):
    tutor_reply: str

class EssayRequest(BaseModel):
    essay: str

class AnalysisResult(BaseModel):
    claim: str
    evaluation: str

class ArgumentCheckerResponse(BaseModel):
    results: List[AnalysisResult]

# ---------------------- Endpoints ----------------------
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    print("[DEV] /upload called.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save the uploaded file.")

    # extract text via PyMuPDF
    doc_text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            doc_text += page.get_text()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process file. Ensure it is a valid PDF.")

    # Generate content using Groq
    prompt = f"""
    Based on the following document text, generate a JSON object with a brief summary (3-4 bullet points), 5 flashcards (each with a 'question' and 'answer'), and 10 multiple-choice quiz questions (each with a 'question', 'options' array, and the correct 'answer'). The output MUST be a valid JSON object and nothing else.

    Document Text:
    {doc_text[:4000]} # truncate to avoid token limits
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates JSON."},
        {"role": "user", "content": prompt}
    ]

    raw_response = call_groq(messages, model="llama-3.1-8b-instant")
    if isinstance(raw_response, dict) and raw_response.get("error"):
        raise HTTPException(status_code=500, detail="AI content generation failed.")

    try:
        data = json.loads(clean_json_response(raw_response))
    except json.JSONDecodeError as e:
        print(f"[DEV] ERROR: Invalid JSON from Groq. Raw response: {raw_response}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON response from AI. Details: {e}")
    
    # Embed and index text
    doc_text = doc_text.replace('\n', ' ').replace('\r', '')
    chunks = chunk_text(doc_text)
    for chunk in chunks:
        emb = embedding_model.encode([chunk])
        index.add(np.array(emb, dtype=np.float32))
        chunk_id = str(uuid.uuid4())
        doc_chunks[chunk_id] = {"doc": file.filename, "text": chunk}
    
    print(f"[DEV] Upload complete. {len(chunks)} chunks added.")
    return data

@app.post("/tutor", response_model=ChatResponse)
async def tutor(user_prompt: str = Form(...)):
    print("[DEV] /tutor called. Prompt:", user_prompt[:80], "...")
    if not doc_chunks:
        return {"tutor_reply": "No documents uploaded yet. Please upload a file in the 'Upload' tab."}

    q_emb = embedding_model.encode([user_prompt])
    D, I = index.search(np.array(q_emb, dtype=np.float32), 3)

    context = ""
    for i in I[0]:
        if i < len(doc_chunks):
            chunk_key = list(doc_chunks.keys())[i]
            context += doc_chunks[chunk_key]["text"] + "\n"

    messages = [
        {"role": "system", "content": "You are a helpful tutor who answers question in a simple way so that people of any age can understand. Answer the user's question concisely using the provided context.Dont mention as it is not in the document"},
        {"role": "user", "content": f"Question: {user_prompt}\n\nContext:\n{context}"}
    ]
    reply = call_groq(messages)
    if isinstance(reply, dict) and reply.get("error"):
        raise HTTPException(status_code=500, detail="AI tutor response failed.")

    print("[DEV] Tutor reply generated.")
    return {"tutor_reply": reply}

@app.post("/argument-checker", response_model=ArgumentCheckerResponse)
async def argument_checker(req: EssayRequest):
    print("[DEV] /argument-checker called.")
    if not doc_chunks:
        raise HTTPException(status_code=400, detail="No documents uploaded yet to check against. Please upload a file in the 'Upload' tab.")

    # Split essay into sentences/claims
    claims = [c.strip() for c in re.split(r'(?<=[.!?])\s+', req.essay) if c.strip()]
    results = []

    for claim in claims:
        claim_vec = embedding_model.encode([claim])
        D, I = index.search(np.array(claim_vec, dtype=np.float32), 3)
        retrieved = [doc_chunks[list(doc_chunks.keys())[i]]["text"] for i in I[0] if i < len(doc_chunks)]

        messages = [
            {"role": "system", "content": "You are an argument checker. Classify the user's 'Claim' based on the 'Reference' text provided. Output only the classification and a brief explanation. Classification should be one of: 'SUPPORTED', 'UNSUPPORTED', or 'CONTRADICTED'. If the reference is not helpful, classify as 'UNSUPPORTED'. Do not add any extra text or conversational phrases."},
            {"role": "user", "content": f"""
Claim: "{claim}"
Reference: {" ".join(retrieved)}
Classification & Explanation:
"""}
        ]
        
        eval_result = call_groq(messages, model="llama-3.1-8b-instant")
        if isinstance(eval_result, dict) and eval_result.get("error"):
            results.append({"claim": claim, "evaluation": "Analysis failed."})
        else:
            results.append({"claim": claim, "evaluation": eval_result.strip()})
        print(f"[DEV] Claim checked: {claim[:50]}...")

    print("[DEV] Argument checking complete.")
    return {"results": results}


