"""Lightweight OpenAI-compatible embedding server for local HuggingFace models.

Serves bge-small-en-v1.5 (or any sentence-transformers model) on port 6100.
MIRIX's hugging-face embedding type calls POST /v1/embeddings with OpenAI-style payload.

Usage:
    python evals/embedding_server.py --model models/bge-small-en-v1.5 --port 6100
"""

import argparse
import time
from typing import List, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="Local Embedding Server")

tokenizer = None
model = None
model_name = ""


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = ""
    user: Optional[str] = ""


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def encode_texts(texts: List[str]) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS pooling for BGE
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def create_embedding(request: Request):
    body = await request.json()
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = [str(t) for t in raw_input]
    else:
        texts = [str(raw_input)]

    vectors = encode_texts(texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)

    data = [
        {"object": "embedding", "embedding": vec.tolist(), "index": i}
        for i, vec in enumerate(vectors)
    ]

    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name}


def main():
    global tokenizer, model, model_name

    parser = argparse.ArgumentParser(description="Local embedding server")
    parser.add_argument("--model", type=str, default="models/bge-small-en-v1.5")
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    print(f"Model loaded: {model_name}, dim={model.config.hidden_size}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
