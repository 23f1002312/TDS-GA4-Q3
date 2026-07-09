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


STOPWORDS = {
    "the","a","an","is","are","was","were","of","to","in","on","for",
    "and","or","with","by","at","from","as","that","this","it",
    "be","been","being","do","does","did",
    "what","which","who","when","where","why","how"
}


def tokenize(text):
    return [
        w for w in re.findall(r"[A-Za-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def expected_answer_present(question, sentence):
    q = question.lower()
    s = sentence

    if "what year" in q or "when" in q:
        return re.search(r"\b(19|20)\d{2}\b", s) is not None

    if "how many" in q or "how much" in q:
        return re.search(r"\b\d+(\.\d+)?\b", s) is not None

    if q.startswith("who"):
        # require at least one capitalized word
        return re.search(r"\b[A-Z][a-z]+\b", s) is not None

    return True


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/")
async def grounded(req: RequestBody):

    if not req.question.strip() or len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_words = set(tokenize(req.question))

    best_chunk = None
    best_sentence = None
    best_score = 0

    for chunk in req.chunks:

        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)

        for sentence in sentences:

            s_words = set(tokenize(sentence))
            overlap = len(q_words & s_words)

            if overlap < 2:
                continue

            if not expected_answer_present(req.question, sentence):
                continue

            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
                best_sentence = sentence.strip()

    if best_chunk is None:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    confidence = min(
        0.55 + 0.08 * best_score,
        0.99,
    )

    return {
        "answer": best_sentence,
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True,
    }
