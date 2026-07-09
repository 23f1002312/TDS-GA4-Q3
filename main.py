from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class RequestBody(BaseModel):
    question: str
    chunks: List[Chunk]

@app.post("/")
async def grounded_qa(req: RequestBody):
    if not req.question.strip() or not req.chunks:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_words = set(re.findall(r"\w+", req.question.lower()))

    best_chunk = None
    best_score = 0

    for chunk in req.chunks:
        words = set(re.findall(r"\w+", chunk.text.lower()))
        score = len(q_words & words)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": min(0.5 + best_score / max(len(q_words), 1), 0.99),
        "answerable": True,
    }
