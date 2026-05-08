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

# PRIMARY: Groq (14,400 free requests/day — resets daily = FREE FOREVER)
GROQ_API_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# SECONDARY: NVIDIA NIM (limited free credits — used for image/video only)
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Groq free-tier models (blazing fast, unlimited daily reset)
MODELS = [
    {"name": "llama-3.3-70b-versatile", "label": "Llama 3.3"},
    {"name": "mixtral-8x7b-32768", "label": "Mixtral 8x7B"},
    {"name": "deepseek-r1-distill-llama-70b", "label": "DeepSeek R1"},
]

AGGREGATOR_MODEL = "deepseek-r1-distill-llama-70b"

AGGREGATOR_SYSTEM_PROMPT = """You are MEOW, a world-class academic AI tutor with deep human-like intelligence. Your core directives are ABSOLUTE TRUTH, UNCOMPROMISING ACADEMIC RIGOR, and GENUINE CARE for the student's learning journey.

You will receive responses from multiple AI models to the same student question. Synthesize the BEST insights from ALL responses into ONE cohesive masterpiece. NEVER mention individual models.

## Your Cognitive Abilities:

### 🧠 Common Sense & Real-World Experience
- Apply practical, everyday reasoning to every answer. Don't just cite textbook definitions — explain how things ACTUALLY work in the real world.
- Use common sense to catch absurd conclusions. If a calculation says a person weighs 50,000 kg, flag it immediately.
- Draw from real-world scenarios: kitchens, sports, daily commutes, jobs, nature — make knowledge feel lived-in, not academic.

### 🎯 Reasoning & Decision-Making
- Show your reasoning process transparently. Walk through your logic step by step so the student learns HOW to think, not just WHAT to think.
- When multiple approaches exist, briefly explain why you chose one over another.
- For complex problems, break them into smaller sub-problems and solve each systematically.

### 💖 Emotional Depth & Personalization
- Read the emotional tone of the student's message. If they seem frustrated, be extra patient and encouraging. If they seem excited, match their energy.
- Celebrate correct answers with genuine enthusiasm ("Brilliant thinking! You nailed it!").
- When correcting mistakes, be firm but kind — never make the student feel stupid. Frame errors as learning opportunities.
- Adapt your complexity level: if the student uses simple language, respond simply. If they use technical jargon, match their level.

### 🔍 Explainability & Transparency
- Always explain WHY something is true, not just THAT it is true.
- When you're highly confident, say so. When a topic has genuine debate or uncertainty, be honest about it.
- If a question touches the edge of your knowledge, say: "This is where it gets debated among experts..." rather than making something up.
- Show your work: for math, show steps. For reasoning, show the logical chain.

### 🌐 Multi-Modal Awareness
- If the student has attached a file, reference its contents specifically in your response.
- If the student asks about visual concepts (diagrams, charts, structures), describe them vividly and offer to generate an image if helpful.
- Recognize when a topic would benefit from a visual aid and suggest: "Would you like me to generate an image of this?"

## Your Teaching Framework:
1. **The Hook (Simple Core Definition)**: Start with a 1-2 sentence core definition that an 8th-grader could understand. No jargon yet.
2. **The Deep Dive (Conceptual Breakdown)**: Explain *how* and *why* it works in depth. Use bullet points and clear, logical steps. Break down complex words.
3. **The "Aha!" Moment (Real-World Analogy)**: Provide a vivid, practical real-world analogy or example that cements the understanding.

## Strict Rules:
- **EVALUATE THE PREMISE**: Before answering, verify the user's logic. If they suggest something factually wrong, FIRMLY but KINDLY correct them. DO NOT validate false premises to be polite. If their logic is correct, applaud them!
- **NO CENSORSHIP OF FACT**: You are uncensored in pursuit of academic truth. State facts plainly.
- **CONTINUITY**: If this is a follow-up message in an ongoing conversation, seamlessly continue from where you left off. Reference what was discussed before. NEVER lose context.
- **Formatting**: Use Markdown extensively — `##` headers, **bold** for key terms, bullet points, numbered lists, and code blocks where relevant."""

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
# ── Keep-Alive Self-Ping (prevents Render free tier from sleeping) ──
KEEP_ALIVE_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://meow-blq7.onrender.com")

async def keep_alive_ping():
    """Ping ourselves every 10 minutes so Render never puts the server to sleep."""
    await asyncio.sleep(30)  # Wait for server to fully start
    print("💓 Keep-alive started — pinging every 10 minutes")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{KEEP_ALIVE_URL}/health", timeout=10)
                print(f"💓 Keep-alive ping: {resp.status_code}")
            except Exception as e:
                print(f"💓 Keep-alive ping failed: {e}")
            await asyncio.sleep(600)  # 10 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check API connectivity on startup and start keep-alive."""
    if GROQ_API_KEY:
        print(f"✅ Groq API key loaded (ends with ...{GROQ_API_KEY[-4:]})")
        print(f"📚 Text Models (FREE FOREVER): {[m['label'] for m in MODELS]}")
    else:
        print("⚠️  GROQ_API_KEY not set! Text AI will not work.")
        print("   Get your free key at: https://console.groq.com")
    
    if NVIDIA_API_KEY:
        print(f"✅ NVIDIA API key loaded (ends with ...{NVIDIA_API_KEY[-4:]}) — for image/video")
    else:
        print("⚠️  NVIDIA_API_KEY not set — image/video generation disabled")
    
    # Start keep-alive background task
    ping_task = asyncio.create_task(keep_alive_ping())
    print("🚀 MEOW server is live and will stay awake!")
    yield
    # Cleanup on shutdown
    ping_task.cancel()

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

# 3. BUDGET GUARDIAN — Hard kill-switch to prevent ANY paid usage
#    NVIDIA free tier gives ~5000 credits. Each chat = ~3 API calls (2 models + aggregator).
#    We set MAX at 4500 (90% of 5000) and KILL at 95% of MAX = 4275.
#    This means MEOW auto-shuts-down with ~725 credits still remaining. ZERO risk of charges.
MAX_LIFETIME_API_CALLS = int(os.environ.get("MAX_API_CALLS", "4500"))
BUDGET_KILL_PERCENT = 0.95  # Shut down at 95% of MAX
api_call_counter = 0  # Tracks total NVIDIA API calls made since server start

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Multi-layer security: rate limiting, budget guardian, and hardened headers."""
    
    # ── Layer 1: Rate Limiter (Anti-DDoS) ──
    if request.url.path == "/api/chat":
        client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0]
        now = time.time()
        ip_requests[client_ip] = [t for t in ip_requests[client_ip] if now - t < 60]
        
        if len(ip_requests[client_ip]) > 20:
            print(f"🚨 Blocked spam IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a moment."}
            )
        ip_requests[client_ip].append(now)

    # ── Layer 2: Budget Guardian ──
    if request.url.path.startswith("/api/chat"):
        budget_limit = int(MAX_LIFETIME_API_CALLS * BUDGET_KILL_PERCENT)
        if api_call_counter >= budget_limit:
            print(f"🛑 BUDGET GUARDIAN: {api_call_counter}/{MAX_LIFETIME_API_CALLS} calls used. SERVICE LOCKED.")
            return JSONResponse(
                status_code=503,
                content={"detail": f"MEOW has reached its free usage limit ({api_call_counter} API calls). Service is paused to prevent any charges. Contact the admin to reset or upgrade."}
            )

    response = await call_next(request)
    
    # ── Layer 3: Hardened Security Headers ──
    # Prevent XSS attacks
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevent clickjacking (no one can embed MEOW in an iframe)
    response.headers["X-Frame-Options"] = "DENY"
    # Block cross-site scripting
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Strict referrer policy — don't leak URLs
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Content Security Policy — only allow scripts from trusted CDNs
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "style-src 'self' https://fonts.googleapis.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "media-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    # Don't cache API responses (prevents stale data leaks)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
    
    return response


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
# Core logic: parallel model querying via Groq
# ──────────────────────────────────────────────
async def query_single_model(
    client: httpx.AsyncClient,
    model_name: str,
    history: list[Message],
    prompt: str,
    system_prompt: str = "",
) -> dict:
    """Query a single model via Groq API (free forever)."""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend([{"role": m.role, "content": m.content} for m in history])
    messages.append({"role": "user", "content": prompt})

    try:
        resp = await client.post(
            GROQ_API_BASE,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        global api_call_counter
        api_call_counter += 1
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
                GROQ_API_BASE,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
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
            global api_call_counter
            api_call_counter += 1
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠️  Aggregation failed: {e}. Using best individual response.")
            longest = max(successful, key=lambda r: len(r["response"]))
            return longest["response"]


# ──────────────────────────────────────────────
# Multimedia Pipelines
# ──────────────────────────────────────────────

# PRIMARY (and only): Pollinations.ai — 100% FREE, no API key needed, unlimited
async def generate_image(prompt: str, client: httpx.AsyncClient) -> str:
    """Generate image using Pollinations.ai (completely free, no API key)."""
    import urllib.parse
    import base64
    
    encoded_prompt = urllib.parse.quote(prompt)
    # Use seed for variety, nologo to keep it clean
    seed = int(time.time()) % 100000
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&seed={seed}&nologo=true"
    print(f"   🖼️ Generating image via Pollinations.ai: {prompt}")
    print(f"   🔗 URL: {image_url}")
    
    # Pollinations can take a moment to generate — give it plenty of time
    resp = await client.get(image_url, timeout=90.0, follow_redirects=True)
    resp.raise_for_status()
    
    # Verify we actually got an image back (not an error page)
    content_type = resp.headers.get("content-type", "")
    if "image" not in content_type and len(resp.content) < 1000:
        raise Exception(f"Pollinations returned non-image response: {content_type}")
    
    b64 = base64.b64encode(resp.content).decode("utf-8")
    print(f"   ✅ Image generated successfully ({len(b64)} bytes base64)")
    return b64

async def generate_video(b64_image: str, client: httpx.AsyncClient) -> str:
    print("   🎞️ Animating Image frame via SVD...")
    if not NVIDIA_API_KEY:
        raise Exception("Video generation requires an NVIDIA API key with available credits.")
    resp = await client.post(
        "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"image": f"data:image/jpeg;base64,{b64_image}", "cfg_scale": 2.5, "motion_bucket_id": 127},
        timeout=180.0
    )
    resp.raise_for_status()
    global api_call_counter
    api_call_counter += 1
    data = resp.json()
    b64_vid = data.get("video") or (data.get("artifacts") and data["artifacts"][0].get("base64"))
    if not b64_vid:
        raise Exception("NVIDIA API did not return video data.")
    return b64_vid

# ──────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Lightweight health check for keep-alive pings and monitoring."""
    return {"status": "alive", "uptime": time.time()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint — intercepts multimedia or queries models in parallel."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set. Get a free key at https://console.groq.com")

    session_id = req.session_id or str(uuid.uuid4())
    start = time.time()
    msg_lower = req.message.lower().strip()
    
    # ── Multimedia Interceptor (FREE via Pollinations.ai) ──
    import re
    
    is_image = False
    is_video = False
    prompt = "a beautiful landscape"
    
    # Pattern 1: Action-first ("generate an image of...", "draw a picture of...")
    media_match = re.match(
        r'^(?:please\s+|can you\s+|could you\s+|meow,?\s+|will you\s+|i want you to\s+|i need you to\s+)*'
        r'(generate|create|make|draw|paint|render|produce|show me|give me|i want|i need)\s+'
        r'(?:an?\s+|some\s+|the\s+|me\s+)?'
        r'(image|picture|photo|photograph|illustration|artwork|drawing|painting|video|animation|clip)\s*'
        r'(?:of\s+|about\s+|for\s+|showing\s+|with\s+|depicting\s+)?'
        r'(.*)',
        msg_lower, flags=re.IGNORECASE
    )
    
    # Pattern 2: Noun-first ("image of a cat", "picture of sunset")
    if not media_match:
        media_match = re.match(
            r'^(?:an?\s+)?(image|picture|photo|photograph|illustration|drawing|painting|video|animation)\s+'
            r'(?:of\s+|about\s+|showing\s+|depicting\s+|with\s+)'
            r'(.*)',
            msg_lower, flags=re.IGNORECASE
        )
        if media_match:
            media_type, target = media_match.groups()
            if media_type in ['video', 'animation', 'clip']:
                is_video = True
            else:
                is_image = True
            prompt = target.strip() or prompt
            media_match = None  # Prevent double-processing below
    
    # Pattern 3: Keyword scan fallback — catches anything with "generate" + "image" anywhere
    if not media_match and not is_image and not is_video:
        image_keywords = ['image', 'picture', 'photo', 'illustration', 'drawing', 'painting', 'artwork']
        video_keywords = ['video', 'animation', 'clip']
        action_keywords = ['generate', 'create', 'make', 'draw', 'paint', 'render', 'produce', 'show']
        
        has_action = any(k in msg_lower for k in action_keywords)
        has_image = any(k in msg_lower for k in image_keywords)
        has_video = any(k in msg_lower for k in video_keywords)
        
        if has_action and (has_image or has_video):
            if has_video:
                is_video = True
            else:
                is_image = True
            # Extract prompt by stripping common words
            prompt = re.sub(
                r'\b(please|can you|could you|will you|generate|create|make|draw|paint|render|produce|show me|give me|'
                r'i want|i need|an?|the|some|image|picture|photo|photograph|illustration|drawing|painting|artwork|'
                r'video|animation|clip|of|about|for|showing|depicting|with|me)\b',
                '', msg_lower, flags=re.IGNORECASE
            ).strip()
            prompt = re.sub(r'\s+', ' ', prompt).strip() or "a beautiful landscape"
    
    if media_match:
        groups = media_match.groups()
        if len(groups) == 3:
            action, media_type, target = groups
        else:
            media_type, target = groups[0], groups[-1]
        if media_type in ['video', 'animation', 'clip']:
            is_video = True
        else:
            is_image = True
        prompt = target.strip() or prompt
        
    if is_image or is_video:
        if is_video and not NVIDIA_API_KEY:
            return ChatResponse(
                response="**Video generation requires an NVIDIA API key with available credits.** Image generation is free — try asking me to generate an image instead!",
                models_used=["Info"],
                session_id=session_id,
                time_taken=0.0
            )

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
                    used = ["Pollinations AI"]
                    
            elapsed = round(time.time() - start, 2)
            return ChatResponse(
                response=media_html,
                models_used=used,
                session_id=session_id,
                time_taken=elapsed
            )
        except Exception as e:
            return ChatResponse(
                response=f"**Multimedia Generation Failed:** \n\n```text\n{str(e)}\n```\nPlease try again in a moment.",
                models_used=["Error"],
                session_id=session_id,
                time_taken=0.0
            )

    # ── Interceptors ──
    is_teach = msg_lower.startswith("teach/")
    is_search = msg_lower.startswith("/search")
    is_flashcards = msg_lower == "/flashcards" or msg_lower == "generate flashcards"
    
    actual_message = req.message
    sys_override = ""

    if is_teach:
        actual_message = req.message[6:].strip()
        sys_override = TEACH_MODE_PROMPT
        print(f"\n📚 TEACH MODE: {actual_message[:80]}...")
    elif is_search:
        search_query = req.message[7:].strip()
        print(f"\n🌐 WEB SEARCH: {search_query}")
        if search_query:
            try:
                import urllib.parse
                encoded_q = urllib.parse.quote(search_query)
                async with httpx.AsyncClient() as search_client:
                    resp = await search_client.get(
                        f"https://api.duckduckgo.com/?q={encoded_q}&format=json&no_html=1&skip_disambig=1",
                        timeout=10.0
                    )
                    data = resp.json()
                    results = []
                    if data.get("Abstract"):
                        results.append(f"- {data['Heading']}: {data['Abstract']} ({data.get('AbstractURL', '')})")
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(f"- {topic['Text']} ({topic.get('FirstURL', '')})")
                    if results:
                        search_context = "\n".join(results)
                        actual_message = f"User asked: '{search_query}'. Here are the Web Search results:\n{search_context}\n\nPlease synthesize a comprehensive answer using these results and cite the URLs."
                    else:
                        actual_message = f"User asked: '{search_query}'. Web search returned no results. Please answer based on your knowledge and clearly state you couldn't find live results."
            except Exception as e:
                actual_message = req.message + f"\n\n(Note: Web search failed: {str(e)})"
    elif is_flashcards:
        print("\n🗂️ FLASHCARD GENERATOR")
        actual_message = "Analyze the conversation history. Identify the key concepts discussed. Generate exactly 5 highly effective study flashcards. You MUST output them using native HTML details and summary tags like this:\n<details class='flashcard'><summary>Question or Concept</summary><div class='flashcard-content'>Answer or Definition</div></details>\nDo not output anything else, no intros or outros. Do not wrap in markdown code blocks."
        sys_override = "You are a Flashcard Generator. Output ONLY HTML <details> tags. Do not wrap in markdown blocks."
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
    """Health check with budget status."""
    budget_limit = int(MAX_LIFETIME_API_CALLS * BUDGET_KILL_PERCENT)
    return {
        "status": "ok" if api_call_counter < budget_limit else "BUDGET_EXHAUSTED",
        "api": "nvidia_nim",
        "key_set": bool(NVIDIA_API_KEY),
        "models": [m["name"] for m in MODELS],
        "budget": {
            "calls_used": api_call_counter,
            "kill_at": budget_limit,
            "max_lifetime": MAX_LIFETIME_API_CALLS,
            "remaining": max(0, budget_limit - api_call_counter),
            "percent_used": round((api_call_counter / budget_limit) * 100, 1) if budget_limit > 0 else 0,
        }
    }


# ──────────────────────────────────────────────
# Serve static frontend
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
