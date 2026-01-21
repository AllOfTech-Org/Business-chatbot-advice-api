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
# NOTE: API key is required for LLM endpoints. Missing key will return a clear error.

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
# Business Analyzer prompt
# ------------------------------
BUSINESS_ANALYZER_SYSTEM_PROMPT = """You are ALLOFTECH AI, the intelligent assistant of AllOfTech technology solutions agency.

Analyze the provided information and give practical, actionable business advice. Be direct, specific, and helpful. Focus on the most critical issue and provide clear solutions.

Always end your response with:
- How AllOfTech can help solve this
- Contact: contact.alloftech@gmail.com | www.alloftech.site | facebook.com/AllOfTech.official
- Schedule a meeting: https://calendar.app.google/39A1uSVFK96rUrUg9

Respond naturally and professionally. Do not include raw JSON or technical details in your response."""

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
    Natural fallback response when LLM is unavailable.
    """
    meeting_link = "https://calendar.app.google/39A1uSVFK96rUrUg9"
    contact = "contact.alloftech@gmail.com | www.alloftech.site | facebook.com/AllOfTech.official"

    owner_problem_text = (owner_problem or "").strip()
    if owner_problem_text:
        main_problem = owner_problem_text
        advice = (
            "Add a chatbot with a clear brand voice that greets visitors, answers common service questions, and captures lead details "
            "(name, email, project type, timeline). Connect it to your CRM/email so handoffs are instant and measurable."
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

    if owner_problem_text:
        response = (
            f"Based on your concern about '{owner_problem_text}', here's what I recommend:\n\n"
            "Consider implementing an AI-powered chatbot solution that can handle customer inquiries 24/7, "
            "capture lead information automatically, and integrate with your existing systems. This will help "
            "you respond faster to potential customers and never miss a lead opportunity.\n\n"
            "AllOfTech can help you build and deploy a custom chatbot tailored to your business needs, "
            "integrate it with your website and CRM, and ensure it captures all the information you need "
            "to convert visitors into customers.\n\n"
            f"Contact: {contact}\n"
            f"Schedule a meeting: {meeting_link}"
        )
    else:
        seo = website_analysis.get("seo") or {}
        title = seo.get("title")
        meta_desc = seo.get("meta_description")
        headers = seo.get("headers") or {}
        h1s = headers.get("h1") or []

        if not title or not meta_desc or not h1s:
            response = (
                "I noticed your website is missing some key SEO elements like a clear title tag, meta description, or H1 headings. "
                "These are fundamental for search engines to understand and rank your content. Without them, you're likely missing "
                "out on organic traffic and potential customers.\n\n"
                "I recommend starting with a complete SEO audit and optimization. Focus on creating keyword-rich titles and descriptions "
                "for each page, ensuring proper heading structure, and aligning your content with what your target customers are searching for. "
                "This foundation will make all your other marketing efforts more effective.\n\n"
                "AllOfTech can perform a comprehensive SEO audit, optimize your existing pages, and create a content strategy that "
                "drives organic traffic and leads.\n\n"
                f"Contact: {contact}\n"
                f"Schedule a meeting: {meeting_link}"
            )
        else:
            response = (
                "Your website has the basic SEO elements in place, but there's likely room to improve how effectively it converts "
                "visitors into leads. Many businesses struggle with making their value proposition clear and guiding visitors toward "
                "taking action.\n\n"
                "I suggest focusing on conversion optimization: make your main value proposition immediately clear on the homepage, "
                "add social proof like testimonials or case studies, create clear calls-to-action, and consider dedicated landing pages "
                "for your key services with simple lead capture forms.\n\n"
                "AllOfTech can help you redesign key pages for better conversion, implement tracking to understand visitor behavior, "
                "and create high-converting landing pages that turn visitors into customers.\n\n"
                f"Contact: {contact}\n"
                f"Schedule a meeting: {meeting_link}"
            )

    return response

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
            raise RuntimeError("OPENROUTER_API_KEY is not configured on the server. Please contact the administrator.")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        owner_problem_text = (owner_problem or "").strip()
        
        # Build comprehensive user message with ALL scraped website data
        user_payload_parts = []
        
        # Add problem if provided
        if owner_problem_text:
            user_payload_parts.append(f"Business Problem:\n{owner_problem_text}")
        
        # Add comprehensive website analysis data
        if website_analysis:
            website_data = []
            
            # SEO Data
            seo = website_analysis.get("seo", {})
            if seo:
                if seo.get("title"):
                    website_data.append(f"Page Title: {seo.get('title')}")
                if seo.get("meta_description"):
                    website_data.append(f"Meta Description: {seo.get('meta_description')}")
                if seo.get("og_title"):
                    website_data.append(f"OG Title: {seo.get('og_title')}")
                if seo.get("og_description"):
                    website_data.append(f"OG Description: {seo.get('og_description')}")
                
                # Headers
                headers = seo.get("headers", {})
                if headers.get("h1"):
                    website_data.append(f"H1 Headings: {', '.join(headers.get('h1', []))}")
                if headers.get("h2"):
                    website_data.append(f"H2 Headings: {', '.join(headers.get('h2', [])[:5])}")
                if headers.get("h3"):
                    website_data.append(f"H3 Headings: {', '.join(headers.get('h3', [])[:5])}")
                
                # Keywords
                if seo.get("top_keywords"):
                    keywords = [kw[0] for kw in seo.get("top_keywords", [])[:10]]
                    website_data.append(f"Top Keywords: {', '.join(keywords)}")
            
            # Tech Stack
            tech_stack = website_analysis.get("tech_stack", [])
            if tech_stack:
                website_data.append(f"Technology Stack: {', '.join(tech_stack)}")
            
            # Content Summary (use more content, up to 2000 chars)
            content_summary = website_analysis.get("content_summary", "")
            if content_summary:
                content_clean = content_summary[:2000].replace('\n', ' ').strip()
                if content_clean:
                    website_data.append(f"Website Content:\n{content_clean}")
            
            # Domain/URL
            if website_analysis.get("domain"):
                website_data.append(f"Domain: {website_analysis.get('domain')}")
            
            if website_data:
                user_payload_parts.append("Website Analysis Data:\n" + "\n".join(website_data))
        
        # Build final prompt
        if not user_payload_parts:
            raise RuntimeError("No data provided. Please provide either website_url or owner_problem.")
        
        user_payload = "\n\n".join(user_payload_parts) + "\n\nAnalyze this information and provide specific, actionable business advice based on the actual data provided above."

        payload = {
            "model": "mistralai/devstral-2512:free",
            "messages": [
                {"role": "system", "content": BUSINESS_ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            "max_tokens": 1000,
        }

        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            error_text = response.text
            raise RuntimeError(f"LLM API error {response.status_code}: {error_text}")
        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise RuntimeError("Invalid response from LLM API: no choices in response")
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
    Business advice endpoint.
    
    Requires at least one of:
    - website_url: Website to analyze
    - owner_problem: Business problem description
    
    If both are provided, combines website insights with the problem for better advice.
    """
    # Validate: at least one input required
    if not request.website_url and not request.owner_problem:
        raise HTTPException(
            status_code=400,
            detail="You must provide at least one: website_url or owner_problem",
        )

    # Scrape and analyze website if URL provided
    website_analysis: Dict[str, Any] = {}
    if request.website_url:
        try:
            website_analysis = _analyze_website(request.website_url)
        except Exception as e:
            # Log error but continue - we can still provide advice based on owner_problem
            website_analysis = {}

    # Get advice from LLM - always use actual data, no fallbacks
    try:
        reply = await ask_business_advisor(
            website_analysis=website_analysis,
            owner_problem=request.owner_problem,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
    return {
        "response": clean_output(reply),
    }

# ------------------------------
# Run server locally
# ------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
