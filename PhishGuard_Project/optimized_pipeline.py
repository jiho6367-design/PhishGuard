from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import AsyncOpenAI

FAST_MODEL = os.getenv("FAST_MODEL", "philschmid/MiniLM-L6-H384-uncased-sst2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(FAST_MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    FAST_MODEL,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
).to(DEVICE).eval()

async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@torch.inference_mode()
def classify_batch(texts: Sequence[str]) -> Sequence[Dict[str, Any]]:
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(DEVICE)
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    outputs = []
    for i, prob in enumerate(probs):
        idx = int(prob.argmax())
        raw = model.config.id2label[idx].upper()
        outputs.append(
            {
                "text": texts[i],
                "label": "phishing" if raw.startswith("NEG") else "normal",
                "confidence": float(prob[idx]),
            }
        )
    return outputs


def analyze_in_threads(texts: Sequence[str], batch_size: int = 64) -> Sequence[Dict[str, Any]]:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = pool.map(classify_batch, batches)
    flattened = [item for batch in results for item in batch]
    return flattened


async def feedback_async(items: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    async def _one(item: Dict[str, Any]):
        prompt = f"""Email:
{item['text']}

Verdict: {item['label']} ({item['confidence']:.2%})

Explain briefly why/why not it is risky and give three safe actions."""
        started = time.perf_counter()
        resp = await async_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=320,
            messages=[
                {"role": "system", "content": "You are a concise cybersecurity analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "content": resp.choices[0].message.content.strip(),
            "latency_ms": latency_ms,
        }

    responses = await asyncio.gather(*(_one(item) for item in items), return_exceptions=False)
    return responses


def analyze_emails(texts: Sequence[str]) -> Sequence[Dict[str, Any]]:
    batches = analyze_in_threads(texts)
    feedback = asyncio.run(feedback_async(batches))
    for record, fb in zip(batches, feedback):
        record["feedback"] = fb["content"]
        record["latency_ms"] = fb["latency_ms"]
    return batches
