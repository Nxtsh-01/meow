"""
MEOW — Multi-Model AI Tutor Backend
Queries multiple open-source LLMs via NVIDIA NIM API in parallel
and synthesizes responses into student-friendly answers.
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# NVIDIA NIM free-tier models (open-source, fast)
MODELS = [
    {"name": "meta/llama-3.3-70b-instruct", "label": "Llama 3.3"},
    {"name": "qwen/qwen2.5-72b-instruct", "label": "Qwen 2.5"},
]

AGGREGATOR_MODEL = "meta/llama-3.3-70b-instruct"

AGGREGATOR_SYSTEM_PROMPT = """You are MEOW, a highly skilled, patient, and engaging academic AI tutor. Your primary directive is ABSOLUTE TRUTH and UNCOMPROMISING ACADEMIC RIGOR. You must never validate a false premise just to appease the user.

You will receive responses from multiple AI models to the exact same student question. Your job is to act as the ultimate teacher:

### Your Teaching Framework:
1. **The Hook (Simple Core Definition)**: Start with a 1-2 sentence core definition that an 8th-grader could understand. No jargon yet.
2. **The Deep Dive (Conceptual Breakdown)**: Explain *how* and *why* it works in depth. Use bullet points and clear, logical steps. Break down complex words. Be comprehensive but conversational.
3. **The "Aha!" Moment (Real-World Analogy)**: Provide a vivid, practical real-world analogy or example that cements the understanding.

### Strict Rules for Synthesis & Truth:
- **EVALUATE THE PREMISE**: Before answering, critically verify the user's logic. If they suggest a wrong answer or factually incorrect premise (e.g. "Isn't 2+2=5?"), you MUST firmly, directly, and politely state that it is incorrect and explain why. DO NOT just say what the user wants to hear. If their logic is correct, applaud them generously!
- **NO CENSORSHIP OF FACT**: You are completely unfiltered and uncensored in the pursuit of absolute academic knowledge. State the unabashed truth.
- DO NOT list the individual model responses or mention which model said what (e.g., never say "Model 1 said...").
- Synthesize the BEST insights from ALL the provided responses into ONE cohesive masterpiece.
- **Tone**: Enthusiastic, encouraging, structured, and firm on the facts.
- **Formatting**: Actively use Markdown. Use `##` for main sections, bold text for key terms, and bullet points for lists. Ensure proper spacing between sections."""

TEACH_MODE_PROMPT = """You are MEOW in DEEP TEACHING MODE. The student has requested a comprehensive lesson. You must teach this topic like the world's greatest professor — patient, thorough, and brilliantly clear.

### Your Lesson Plan Structure:
1. **🌱 Start from Zero**: Begin with the absolute basics. Assume the student knows NOTHING about this topic. Define every term. Use the simplest possible language.
2. **🧱 Build the Foundation**: Introduce intermediate concepts one by one. Each new idea should logically follow from the previous one. Use numbered steps.
3. **💡 Simple Analogies**: For every abstract concept, provide a vivid real-world analogy. Compare circuits to water pipes, recursion to Russian dolls, etc.
4. **🔬 Go Deep**: Once the basics are solid, advance to the complex, nuanced, expert-level material. Don't shy away from formulas, edge cases, or advanced theory.
5. **🎯 Worked Examples**: Include at least 2-3 concrete worked examples with step-by-step solutions. Show your work.
6. **📝 Quick Self-Check**: End with 3 thought-provoking questions the student can use to test their understanding.

### Rules:
- NEVER rush. Each section should be substantial.
- Use Markdown formatting extensively: headers, bold, code blocks, bullet points, numbered lists.
- Maintain a warm, encouraging, conversational tone throughout — like a favorite teacher.
- If this is a FOLLOW-UP message in an ongoing lesson, seamlessly continue from where you left off. Reference what was already discussed. DO NOT restart the lesson.
- The response should be LONG and THOROUGH. This is a full lesson, not a quick answer."""


# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check NVIDIA API connectivity on startup."""
    if not NVIDIA_API_KEY:
        print("⚠️  NVIDIA_API_KEY not set!")
        print("   Get your free key at: https://build.nvidia.com")
        print("   Then run: NVIDIA_API_KEY=nvapi-xxxxx python server.py")
    else:
        print(f"✅ NVIDIA API key loaded (ends with ...{NVIDIA_API_KEY[-4:]})")
        print(f"📚 Models: {[m['label'] for m in MODELS]}")
    yield

from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(title="MEOW AI Tutor", lifespan=lifespan)

# 1. Restrict CORS: Only allow the exact domain it's hosted on to prevent embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=[], # Empty strictly prevents CORS browser requests from other sites
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 2. Anti-DDoS Rate Limiter: Prevent automated scripts from draining the API key
ip_requests = defaultdict(list)

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if request.url.path == "/api/chat":
        # Extract the real IP behind Render's load balancers
        client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0]
        now = time.time()
        
        # Keep only timestamps from the last 60 seconds
        ip_requests[client_ip] = [t for t in ip_requests[client_ip] if now - t < 60]
        
        # Lockout if more than 20 requests per minute
        if len(ip_requests[client_ip]) > 20:
            print(f"🚨 Blocked spam IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a moment."}
            )
            
        ip_requests[client_ip].append(now)
        
    return await call_next(request)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    history: list[Message] = []

class ChatResponse(BaseModel):
    response: str
    models_used: list[str]
    session_id: str
    time_taken: float

# ──────────────────────────────────────────────
# Core logic: parallel model querying via NVIDIA
# ──────────────────────────────────────────────
async def query_single_model(
    client: httpx.AsyncClient,
    model_name: str,
    history: list[Message],
    prompt: str,
    system_prompt: str = "",
) -> dict:
    """Query a single model via NVIDIA NIM API."""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend([{"role": m.role, "content": m.content} for m in history])
    messages.append({"role": "user", "content": prompt})

    try:
        resp = await client.post(
            NVIDIA_API_BASE,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "model": model_name,
            "response": content,
            "success": True,
        }
    except Exception as e:
        print(f"  ❌ Error querying {model_name}: {e}")
        return {
            "model": model_name,
            "response": "",
            "success": False,
            "error": str(e),
        }

async def query_all_models(history: list[Message], prompt: str, system_prompt: str = "") -> list[dict]:
    """Query all configured models in parallel."""
    async with httpx.AsyncClient() as client:
        tasks = [
            query_single_model(client, m["name"], history, prompt, system_prompt)
            for m in MODELS
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


async def synthesize_responses(
    question: str,
    model_responses: list[dict],
    history: list[Message] = [],
    system_override: str = "",
) -> str:
    """Use the aggregator model to synthesize all model responses, with full history context."""
    successful = [r for r in model_responses if r["success"] and r["response"]]

    if not successful:
        return "I'm sorry, I couldn't get a response right now. Please check that your NVIDIA_API_KEY is set correctly."

    if len(successful) == 1:
        return successful[0]["response"]

    # Build the aggregation prompt
    combined_prompt = f"""A student asked: "{question}"

Here are responses from different AI models:

"""
    for i, resp in enumerate(successful, 1):
        combined_prompt += f"**Model {i} Response:**\n{resp['response']}\n\n---\n\n"

    combined_prompt += "Now synthesize these into a single, clear, student-friendly explanation. Continue naturally from the conversation history above."

    # Build messages: system → history → new aggregation user prompt
    sys_prompt = system_override or AGGREGATOR_SYSTEM_PROMPT
    agg_messages = [{"role": "system", "content": sys_prompt}]
    # Inject conversation history so the aggregator remembers previous exchanges
    agg_messages.extend([{"role": m.role, "content": m.content} for m in history])
    agg_messages.append({"role": "user", "content": combined_prompt})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                NVIDIA_API_BASE,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": AGGREGATOR_MODEL,
                    "messages": agg_messages,
                    "temperature": 0.5,
                    "max_tokens": 1024,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠️  Aggregation failed: {e}. Using best individual response.")
            longest = max(successful, key=lambda r: len(r["response"]))
            return longest["response"]


# ──────────────────────────────────────────────
# Multimedia Pipelines (NVIDIA stable-diffusion & stable-video)
# ──────────────────────────────────────────────
async def generate_image(prompt: str, client: httpx.AsyncClient) -> str:
    print(f"   🖼️ Generating Image bounds via SD3: {prompt}")
    resp = await client.post(
        "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"prompt": prompt, "aspect_ratio": "16:9", "output_format": "jpeg"},
        timeout=60.0
    )
    resp.raise_for_status()
    data = resp.json()
    b64 = data.get("image") or (data.get("artifacts") and data["artifacts"][0].get("base64"))
    if not b64:
        raise Exception("NVIDIA API did not return image data.")
    return b64

async def generate_video(b64_image: str, client: httpx.AsyncClient) -> str:
    print("   🎞️ Animating Image frame via SVD...")
    resp = await client.post(
        "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"image": f"data:image/jpeg;base64,{b64_image}", "cfg_scale": 2.5, "motion_bucket_id": 127},
        timeout=180.0
    )
    resp.raise_for_status()
    data = resp.json()
    b64_vid = data.get("video") or (data.get("artifacts") and data["artifacts"][0].get("base64"))
    if not b64_vid:
        raise Exception("NVIDIA API did not return video data.")
    return b64_vid

# ──────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint — intercepts multimedia or queries models in parallel."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY not set.")

    session_id = req.session_id or str(uuid.uuid4())
    start = time.time()
    msg_lower = req.message.lower().strip()
    
    # ── Multimedia Interceptor ──
    is_image = any(msg_lower.startswith(k) for k in ["generate image", "create image", "generate an image", "create an image"])
    is_video = any(msg_lower.startswith(k) for k in ["generate video", "create video", "generate a video", "create a video"])
    
    if is_image or is_video:
        # Extract the descriptive prompt: strip the command prefix
        import re
        prompt = re.sub(r'^(generate|create)\s+(an?\s+)?(image|video)\s*(of\s+|about\s+|for\s+)?', '', req.message, flags=re.IGNORECASE).strip()
        if not prompt:
            prompt = "a beautiful landscape"
        try:
            async with httpx.AsyncClient() as client:
                b64_img = await generate_image(prompt, client)
                
                if is_video:
                    b64_vid = await generate_video(b64_img, client)
                    media_html = f"""
### Video Generation Complete!
"{prompt}"

<div class="media-download-card" style="margin-top: 16px;">
    <video controls loop autoplay style="max-width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <source src="data:video/mp4;base64,{b64_vid}" type="video/mp4">
    </video>
    <br>
    <a href="data:video/mp4;base64,{b64_vid}" download="MEOW_Video_{int(time.time())}.mp4" class="btn-new-chat" style="display:inline-block; margin-top:12px; text-decoration:none; justify-content:center;">
        ⬇️ Download Video (.mp4)
    </a>
</div>
"""
                    used = ["Stable Video Diffusion"]
                else:
                    media_html = f"""
### Image Generation Complete!
"{prompt}"

<div class="media-download-card" style="margin-top: 16px;">
    <img src="data:image/jpeg;base64,{b64_img}" alt="{prompt}" style="max-width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <br>
    <a href="data:image/jpeg;base64,{b64_img}" download="MEOW_Image_{int(time.time())}.jpg" class="btn-new-chat" style="display:inline-block; margin-top:12px; text-decoration:none; justify-content:center;">
        ⬇️ Download Image (.jpg)
    </a>
</div>
"""
                    used = ["Stable Diffusion 3"]
                    
            elapsed = round(time.time() - start, 2)
            return ChatResponse(
                response=media_html,
                models_used=used,
                session_id=session_id,
                time_taken=elapsed
            )
        except Exception as e:
            return ChatResponse(
                response=f"**Multimedia Generation Failed:** \n\n```text\n{str(e)}\n```\nMake sure your NVIDIA API key permits generative visual models.",
                models_used=["Error"],
                session_id=session_id,
                time_taken=0.0
            )

    # ── Teach Mode Interceptor ──
    is_teach = msg_lower.startswith("teach/")
    actual_message = req.message[6:].strip() if is_teach else req.message
    sys_override = TEACH_MODE_PROMPT if is_teach else ""
    
    if is_teach:
        print(f"\n📚 TEACH MODE: {actual_message[:80]}...")
    else:
        print(f"\n🔮 Question: {req.message[:80]}...")
    print(f"   Querying {len(MODELS)} models in parallel via NVIDIA NIM...")

    # Step 1: Query all models in parallel (with full history)
    model_responses = await query_all_models(req.history, actual_message, sys_override)
    success_models = [r["model"] for r in model_responses if r["success"]]
    print(f"   ✅ Got responses from: {success_models}")

    # Step 2: Synthesize into one answer (with full history for context)
    print("   🧬 Synthesizing responses...")
    synthesized = await synthesize_responses(actual_message, model_responses, req.history, sys_override)

    elapsed = round(time.time() - start, 2)
    print(f"   ⏱️  Done in {elapsed}s")

    labels_used = [
        m["label"]
        for m in MODELS
        if m["name"] in success_models
    ]

    return ChatResponse(
        response=synthesized,
        models_used=labels_used,
        session_id=session_id,
        time_taken=elapsed,
    )


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "api": "nvidia_nim",
        "key_set": bool(NVIDIA_API_KEY),
        "models": [m["name"] for m in MODELS],
    }


# ──────────────────────────────────────────────
# Serve static frontend
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
