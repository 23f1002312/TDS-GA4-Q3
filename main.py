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


class Query(BaseModel):
    question: str
    chunks: List[Chunk]


STOPWORDS = {
    "what","when","where","which","who","why","how","is","are","was","were",
    "the","a","an","of","to","in","on","for","and","or","did","does","do",
    "be","been","being","from","with","by","at","as","that","this","it"
}


def keywords(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/")
async def grounded_qa(req: Query):

    if not req.question.strip() or len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_words = set(keywords(req.question))

    best_chunk = None
    best_score = 0
    best_sentence = None

    for chunk in req.chunks:

        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)

        for sentence in sentences:

            s_words = set(keywords(sentence))
            score = len(q_words & s_words)

            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_sentence = sentence.strip()

    if best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    confidence = min(
        0.55 + 0.45 * best_score / max(len(q_words), 1),
        0.99,
    )

    return {
        "answer": best_sentence,
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True,
    }
