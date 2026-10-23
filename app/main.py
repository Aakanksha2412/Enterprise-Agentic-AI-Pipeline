import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import redis
from dotenv import load_dotenv

from app.agent_graph import ai_brain

load_dotenv()

app = FastAPI(title="Enterprise Agentic RAG API with Redis Caching")

# Initialize Redis client
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_cache_key(question: str) -> str:
    """Generates a normalized cache key for incoming queries."""
    normalized = question.strip().lower()
    return f"cache:query:{normalized}"


async def event_generator(question: str):
    cache_key = get_cache_key(question)

    # 1. Check Redis Cache
    try:
        cached_response = redis_client.get(cache_key)
        if cached_response:
            yield f'data: {json.dumps({"status": "cache_hit", "msg": "Serving response from Redis Cache..."})}\n\n'
            await asyncio.sleep(0.05)

            tokens = cached_response.split(" ")
            for token in tokens:
                yield f'data: {json.dumps({"status": "generating", "token": token + " "})}\n\n'
                await asyncio.sleep(0.02)

            yield "data: [DONE]\n\n"
            return
    except Exception as e:
        # Fallthrough on Redis failure to maintain system availability
        print(f"Redis Cache Warning: {e}")

    # 2. Cache Miss: Execute Intent Routing & Retrieval
    yield f'data: {json.dumps({"status": "routing", "msg": "Analyzing query intent..."})}\n\n'
    await asyncio.sleep(0.1)

    yield f'data: {json.dumps({"status": "searching", "msg": "Retrieving context from Qdrant & Cohere..."})}\n\n'
    
    # 3. Invoke LangGraph State Machine
    result = ai_brain.invoke({"question": question})
    final_answer = result.get("final_answer", "No response generated.")

    # 4. Save Response to Redis (TTL: 1 Hour / 3600 seconds)
    try:
        redis_client.setex(name=cache_key, time=3600, value=final_answer)
    except Exception as e:
        print(f"Redis Save Warning: {e}")

    # 5. Stream Tokens to Client
    tokens = final_answer.split(" ")
    for token in tokens:
        yield f'data: {json.dumps({"status": "generating", "token": token + " "})}\n\n'
        await asyncio.sleep(0.05)

    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_endpoint(question: str):
    return StreamingResponse(event_generator(question), media_type="text/event-stream")