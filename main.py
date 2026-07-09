from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI()

# Allow CORS
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


class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]


STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "of","to","in","on","for","from","by","with","as","at",
    "and","or","this","that","these","those","it",
    "what","which","who","when","where","why","how",
    "do","does","did","can","could","would","should",
    "has","have","had"
}


def tokenize(text):
    return [
        w for w in re.findall(r"[A-Za-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def sentence_score(question_words, sentence):
    sentence_words = set(tokenize(sentence))
    return len(question_words & sentence_words)


def expected_answer_present(question, sentence):
    q = question.lower()
    s = sentence

    if "what year" in q or q.startswith("when"):
        return re.search(r"\b(18|19|20)\d{2}\b", s)

    if q.startswith("who"):
        # at least one proper noun
        return re.search(r"\b[A-Z][a-zA-Z]+\b", s)

    if "how many" in q or "how much" in q:
        return re.search(r"\b\d+(\.\d+)?\b", s)

    return True


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/grounded-answer")
async def grounded_answer(req: QARequest):

    if not req.question.strip():
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    if len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    question_words = set(tokenize(req.question))

    best_sentence = None
    best_chunk = None
    best_score = 0

    for chunk in req.chunks:

        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)

        for sentence in sentences:

            score = sentence_score(question_words, sentence)

            if score < 2:
                continue

            if not expected_answer_present(req.question, sentence):
                continue

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
                best_chunk = chunk

    if best_sentence is None:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    confidence = min(0.60 + best_score * 0.08, 0.99)

    return {
        "answer": best_sentence,
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True,
    }
