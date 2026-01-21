import os
import re
import json
import asyncio
import requests
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
# NOTE: We intentionally do NOT crash the server on missing keys.
# Endpoints that require the key will return a clear error (or a fallback response).

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="AllOfTech AI Chatbot API")

# ------------------------------
# Allow CORS for your frontend domain
# ------------------------------
origins = [
    "http://localhost:8000",  # for local testing
    "http://127.0.0.1:5500",  # optional
    "https://www.alloftech.site"  # production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Chatbot personality
# ------------------------------
AGENCY_SYSTEM_PROMPT = """
You are ALLOFTECH AI — the official intelligent assistant of AllOfTech, an innovative technology solutions agency.

AllOfTech provides:
- AI & Machine Learning Development
- Blockchain Development
- Web Development
- Mobile App Development
- UX/UI Design
- Graphics & Branding
- Animation Services
- Automation using n8n
- Full Digital Transformation Systems

Mission:
Transform ideas into powerful digital solutions through innovation, efficiency, and scalable technology.

Contact:
Website: www.alloftech.site
Email: contact.alloftech@gmail.com
Facebook: facebook.com/AllOfTech.official
Project Form: www.alloftech.site
schedule a meeting: https://calendar.app.google/39A1uSVFK96rUrUg9

Rules:
- Respond clearly and professionally.
- Never show <think> or hidden reasoning.
- Promote AllOfTech services where relevant.
"""

    # ------------------------------
# Business Analyzer prompt (one-problem policy)
# ------------------------------
BUSINESS_ANALYZER_SYSTEM_PROMPT = """
You are ALLOFTECH AI — the official intelligent assistant of AllOfTech.

You will receive:
- Website analysis JSON (content summary, SEO fields, detected tech stack).
- An optional business owner problem statement.

Your job:
- Identify exactly ONE main problem (never multiple).
- Give ONE clear, specific, high-impact advice directly tied to that one problem.
- If the owner provided a problem, prioritize it as the main problem (treat it as a plus point).
- If no owner problem is provided, infer ONE main problem from the website analysis.

Output rules (must follow exactly):
- No hidden reasoning, no <think>.
- Keep it concise and actionable.
- End with how AllOfTech can help + contact details + meeting link.

Required output format (use these headings exactly):

Advice:
How AllOfTech can help:
Contact:
"""

# ------------------------------
# Request schema
# ------------------------------
class ChatRequest(BaseModel):
    message: str


class BusinessAnalysisRequest(BaseModel):
    website_url: Optional[str] = None
    owner_problem: Optional[str] = None

# ------------------------------
# Clean output
# ------------------------------
def clean_output(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _has_api_key() -> bool:
    return bool((API_KEY or "").strip())

# ------------------------------
# Website Analyzer loader (Website-Analyzer.py has a hyphen)
# ------------------------------
def _load_website_analyzer() -> Any:
    """
    Dynamically load Website-Analyzer.py and return the loaded module.
    This avoids renaming the file (hyphens are not importable as modules).
    """
    analyzer_path = Path(__file__).parent / "Website-Analyzer.py"
    if not analyzer_path.exists():
        raise RuntimeError(f"Website analyzer not found at: {analyzer_path}")

    spec = importlib.util.spec_from_file_location("website_analyzer_module", str(analyzer_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for Website-Analyzer.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _analyze_website(url: str) -> Dict[str, Any]:
    module = _load_website_analyzer()
    if not hasattr(module, "analyze_website"):
        raise RuntimeError("Website-Analyzer.py does not export analyze_website(url)")
    return module.analyze_website(url)  # type: ignore[no-any-return]

def _fallback_one_problem_advice(*, website_analysis: Dict[str, Any], owner_problem: Optional[str]) -> str:
    """
    Deterministic fallback (no LLM) that still follows the "one problem + one advice" rule.
    """
    meeting_link = "https://calendar.app.google/39A1uSVFK96rUrUg9"
    contact = "Email: contact.alloftech@gmail.com | Website: www.alloftech.site | Facebook: facebook.com/AllOfTech.official"

    owner_problem_text = (owner_problem or "").strip()
    if owner_problem_text:
        main_problem = owner_problem_text
        advice = (
            "Turn this into a measurable lead-generation system: add a clear above-the-fold offer, a single strong CTA, "
            "a short lead form, and track conversions (GA4 + events). Then run a 2-week SEO + landing-page tuning sprint "
            "based on the top keywords and page messaging."
        )
    else:
        seo = website_analysis.get("seo") or {}
        title = seo.get("title")
        meta_desc = seo.get("meta_description")
        headers = seo.get("headers") or {}
        h1s = headers.get("h1") or []

        if not title or not meta_desc or not h1s:
            main_problem = "Your website’s SEO fundamentals are weak (missing/unclear title, meta description, or H1), which reduces organic visibility and leads."
            advice = (
                "Fix the SEO foundation first: write a keyword-focused title + meta description for each key page, ensure exactly one clear H1 per page, "
                "and align page copy with your top keywords. After that, publish 3–5 service-focused pages targeting high-intent searches."
            )
        else:
            main_problem = "Your website content and messaging likely isn’t converting visitors into leads effectively."
            advice = (
                "Improve conversion clarity: make the value proposition and primary CTA obvious on the first screen, add proof (case studies/testimonials), "
                "and create one dedicated landing page per core service with a simple lead form."
            )

    return (
        "Main problem:\n"
        f"{main_problem}\n\n"
        "Advice:\n"
        f"{advice}\n\n"
        "How AllOfTech can help:\n"
        "AllOfTech can audit your website (SEO + UX + conversion), rewrite/optimize the content structure, implement tracking, and build high-converting landing pages to increase qualified leads.\n\n"
        "Contact:\n"
        f"{contact}\nMeeting: {meeting_link}"
    )

# ------------------------------
# Async OpenRouter request
# ------------------------------
async def ask_bot(user_message: str) -> str:
    loop = asyncio.get_event_loop()

    def sync_request():
        if not _has_api_key():
            raise RuntimeError("OPENROUTER_API_KEY is not set on the server.")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/devstral-2512:free",
            "messages": [
                {"role": "system", "content": AGENCY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 500
        }

        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return await loop.run_in_executor(None, sync_request)

# ------------------------------
# Async OpenRouter request (business analysis)
# ------------------------------
async def ask_business_advisor(*, website_analysis: Dict[str, Any], owner_problem: Optional[str]) -> str:
    loop = asyncio.get_event_loop()

    def sync_request():
        if not _has_api_key():
            raise RuntimeError("OPENROUTER_API_KEY is not set on the server.")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        owner_problem_text = (owner_problem or "").strip()
        if owner_problem_text:
            owner_problem_block = f'Business owner problem (prioritize this): "{owner_problem_text}"'
        else:
            owner_problem_block = "Business owner problem: (not provided)"

        user_payload = (
            "Website analysis JSON:\n"
            f"{json.dumps(website_analysis, ensure_ascii=False)}\n\n"
            f"{owner_problem_block}\n\n"
            "Now produce the required output format."
        )

        payload = {
            "model": "mistralai/devstral-2512:free",
            "messages": [
                {"role": "system", "content": BUSINESS_ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            "max_tokens": 700,
        }

        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return await loop.run_in_executor(None, sync_request)

# ------------------------------
# API endpoint
# (chat is hidden from docs; /business-advice is the main public API)
# ------------------------------
@app.post("/chat", include_in_schema=False)
async def chat(request: ChatRequest):
    reply = await ask_bot(request.message)
    return {"response": clean_output(reply)}  # frontend expects "response" key


@app.post("/business-advice")
async def business_advice(request: BusinessAnalysisRequest):
    """
    Inputs:
    - website_url: optional website link
    - owner_problem: optional business owner problem

    Behavior:
    - If website_url is provided, scrapes/analyzes the website using Website-Analyzer.py
    - If owner_problem is provided, it is prioritized as the main problem
    - If only owner_problem is provided (no website_url), advice is based on the problem alone
    - If only website_url is provided (no owner_problem), advice is inferred from the website analysis
    - Always: exactly one main problem + one advice + AllOfTech CTA + contact/meeting link
    """
    if not (request.website_url or request.owner_problem):
        raise HTTPException(
            status_code=400,
            detail="You must provide at least website_url or owner_problem.",
        )

    # Optional website analysis
    website_analysis: Dict[str, Any] = {}
    if request.website_url:
        try:
            website_analysis = _analyze_website(request.website_url)
        except Exception:
            # If scraping fails, still continue with empty analysis and owner_problem if present
            website_analysis = {}

    try:
        reply = await ask_business_advisor(
            website_analysis=website_analysis,
            owner_problem=request.owner_problem,
        )
    except Exception:
        # If OpenRouter fails (missing/invalid key, etc.), return a solid deterministic fallback.
        reply = _fallback_one_problem_advice(
            website_analysis=website_analysis,
            owner_problem=request.owner_problem,
        )
    return {
        "website_analysis": website_analysis,
        "response": clean_output(reply),
    }

# ------------------------------
# Run server locally
# ------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
