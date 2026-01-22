"""
ALLOFTECH Business Agent API
Combines website analysis with AI-powered solutions
"""

import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests

import importlib.util
import sys

# Import Website-Analyzer module (handling hyphen in filename)
spec = importlib.util.spec_from_file_location("website_analyzer", "Website-Analyzer.py")
website_analyzer = importlib.util.module_from_spec(spec)
sys.modules["website_analyzer"] = website_analyzer
spec.loader.exec_module(website_analyzer)
analyze_website = website_analyzer.analyze_website

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ALLOFTECH Business Agent API",
    description="AI-powered business solutions with website analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt
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
- Always identify ONLY ONE most essential problem (never multiple problems).
- Focus on solving that single problem with clear, actionable solutions.
- Always explain how ALLOFTECH can specifically help solve/heal the identified problem.
- Promote AllOfTech services that directly address the identified issue.
"""

# Request model
class SolutionRequest(BaseModel):
    website_url: Optional[str] = Field(None, description="Website URL to analyze (optional)")
    problem_text: Optional[str] = Field(None, description="Problem description (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "website_url": "https://example.com",
                "problem_text": "I need to improve my website's performance"
            }
        }


# Response model
class SolutionResponse(BaseModel):
    success: bool
    solution: str
    website_analysis: Optional[dict] = None
    message: Optional[str] = None


def call_openrouter_llm(prompt: str) -> str:
    """
    Call OpenRouter API to get AI solution.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://www.alloftech.site",
        "X-Title": "ALLOFTECH Business Agent"
    }

    payload = {
        "model": "openai/gpt-4o-mini",  # You can change this to any model supported by OpenRouter
        "messages": [
            {
                "role": "system",
                "content": AGENCY_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected response format from OpenRouter API")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling AI service: {str(e)}")


def format_website_analysis(analysis: dict) -> str:
    """
    Format website analysis into a readable string for the LLM.
    """
    formatted = f"""
WEBSITE ANALYSIS RESULTS:

URL: {analysis.get('url', 'N/A')}
Domain: {analysis.get('domain', 'N/A')}

SEO Information:
- Title: {analysis.get('seo', {}).get('title', 'N/A')}
- Meta Description: {analysis.get('seo', {}).get('meta_description', 'N/A')}
- OG Title: {analysis.get('seo', {}).get('og_title', 'N/A')}
- OG Description: {analysis.get('seo', {}).get('og_description', 'N/A')}

Headers:
- H1: {', '.join(analysis.get('seo', {}).get('headers', {}).get('h1', [])) or 'None'}
- H2: {', '.join(analysis.get('seo', {}).get('headers', {}).get('h2', [])[:5]) or 'None'}  # Limit to first 5

Top Keywords: {', '.join([kw[0] for kw in analysis.get('seo', {}).get('top_keywords', [])[:10]]) or 'None'}

Tech Stack Detected: {', '.join(analysis.get('tech_stack', [])) or 'None detected'}

Content Summary (first 1000 chars):
{analysis.get('content_summary', '')[:1000]}...
"""
    return formatted


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ALLOFTECH Business Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /solution": "Get AI-powered solution with optional website analysis"
        }
    }


@app.post("/solution", response_model=SolutionResponse)
async def get_solution(request: SolutionRequest):
    """
    Get AI-powered solution based on website analysis and/or problem text.
    
    - If website_url is provided: Analyzes the website and includes details in the solution
    - If problem_text is provided: Uses the problem description
    - If both are provided: Combines both for a comprehensive solution
    - At least one must be provided
    """
    # Validate that at least one input is provided
    if not request.website_url and not request.problem_text:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'website_url' or 'problem_text' must be provided"
        )

    website_analysis = None
    prompt_parts = []

    # Step 1: Analyze website if URL is provided
    if request.website_url:
        try:
            logger.info(f"Analyzing website: {request.website_url}")
            website_analysis = analyze_website(request.website_url)
            website_info = format_website_analysis(website_analysis)
            prompt_parts.append(website_info)
        except Exception as e:
            logger.error(f"Error analyzing website: {e}")
            # Continue without website analysis if it fails
            prompt_parts.append(f"\nNote: Website analysis failed for {request.website_url}: {str(e)}\n")

    # Step 2: Add problem text if provided
    if request.problem_text:
        prompt_parts.append("\nPROBLEM DESCRIPTION:")
        prompt_parts.append(f"{request.problem_text}\n")

    # Add context based on what was provided
    if website_analysis and request.problem_text:
        prompt_parts.append("\nBased on the above website analysis and problem description,")
    elif website_analysis:
        prompt_parts.append("\nBased on the above website analysis,")
    elif request.problem_text:
        prompt_parts.append("\nBased on the above problem description,")

    # Step 3: Create the final prompt
    # Focus on ONE essential problem and how ALLOFTECH can solve it
    if website_analysis:
        prompt_parts.append("""
IMPORTANT INSTRUCTIONS:
1. Identify ONLY ONE most essential/critical problem for this business (do not list multiple problems)
2. Focus on that single problem and provide a clear, actionable solution
3. Explain how ALLOFTECH can specifically help solve/heal this problem
4. Mention relevant ALLOFTECH services that directly address this issue

Structure your response as:
- THE PROBLEM: [One clear, essential problem]
- THE SOLUTION: [How to solve it]
- HOW ALLOFTECH HEALS THIS: [Specific ALLOFTECH services and approach]

Be concise, professional, and focused on the single most important issue.
""")
    else:
        # Only problem text - still focus on one problem and ALLOFTECH solution
        prompt_parts.append("""
IMPORTANT INSTRUCTIONS:
1. Identify ONLY ONE most essential/critical problem from the description (do not list multiple problems)
2. Focus on that single problem and provide a clear, actionable solution
3. Explain how ALLOFTECH can specifically help solve/heal this problem
4. Mention relevant ALLOFTECH services that directly address this issue

Structure your response as:
- THE PROBLEM: [One clear, essential problem]
- THE SOLUTION: [How to solve it]
- HOW ALLOFTECH HEALS THIS: [Specific ALLOFTECH services and approach]

Be concise, professional, and focused on the single most important issue.
""")

    final_prompt = "\n".join(prompt_parts)

    # Step 4: Call LLM
    try:
        logger.info("Calling OpenRouter LLM...")
        solution = call_openrouter_llm(final_prompt)
        
        return SolutionResponse(
            success=True,
            solution=solution,
            website_analysis=website_analysis,
            message="Solution generated successfully"
        )
    
    except Exception as e:
        logger.error(f"Error generating solution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating solution: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ALLOFTECH Business Agent API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001
    )
