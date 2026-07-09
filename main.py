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


class RequestBody(BaseModel):
    question: str
    chunks: List[Chunk]


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "at", "from", "as", "that",
    "this", "it", "be", "been", "being",
    "what", "which", "who", "when", "where",
    "why", "how", "do", "does", "did"
}


def tokenize(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [
        w
        for w in words
        if w not in STOPWORDS and len(w) > 2
    ]


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/")
async def grounded_qa(req: RequestBody):

    # Invalid input
    if not req.question.strip() or not req.chunks:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_words = set(tokenize(req.question))

    # -------------------------
    # Score each chunk
    # -------------------------
    chunk_scores = []

    for chunk in req.chunks:
        score = len(set(tokenize(chunk.text)) & q_words)
        chunk_scores.append((score, chunk))

    max_score = max(score for score, _ in chunk_scores)

    # Nothing matched
    if max_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    # Keep only best chunk(s)
    best_chunks = [
        chunk
        for score, chunk in chunk_scores
        if score == max_score
    ]

    answer_sentences = []
    citations = []

    for chunk in best_chunks:

        citations.append(chunk.chunk_id)

        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)

        best_sentence = ""
        best_score = -1

        for sentence in sentences:

            score = len(set(tokenize(sentence)) & q_words)

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

        if best_sentence:
            answer_sentences.append(best_sentence)

    # Remove duplicates while preserving order
    seen = set()
    final_answer = []

    for sentence in answer_sentences:
        if sentence not in seen:
            seen.add(sentence)
            final_answer.append(sentence)

    confidence = min(
        0.60 + 0.08 * max_score,
        0.99
    )

    return {
        "answer": " ".join(final_answer),
        "citations": citations,
        "confidence": round(confidence, 2),
        "answerable": True,
    }
