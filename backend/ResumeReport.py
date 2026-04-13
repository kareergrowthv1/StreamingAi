import json
import logging
import os
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import config
from ai_config_loader import get_ai_config

logger = logging.getLogger(config.APP_NAME)
router = APIRouter(prefix="/resume-report", tags=["Resume Report"])

class ResumeReportRequest(BaseModel):
    resumeText: str = Field(..., description="Full resume text extracted from file")
    reportLevel: str = Field("standard", description="Report depth: complete, standard, min")

def _get_openai_client(cfg: dict):
    try:
        import openai
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="openai package not installed.")

async def _generate_report(resume_text: str, level: str) -> dict:
    cfg = await get_ai_config()
    client = _get_openai_client(cfg)
    
    # Define analysis depth based on subscription plan
    depth_instructions = {
        "complete": """
ANALYSIS DEPTH: COMPLETE (Premium Plan)
- Analyze ALL content sections: About, Experience, Skills, Tools, Projects, Education
- Provide 3-5 specific mistakes per section
- Provide 3-5 specific improvement suggestions per section  
- Include detailed scoring per section
- Fix counts should reflect all issues found
""",
        "standard": """
ANALYSIS DEPTH: STANDARD (Standard Plan)
- Analyze main sections: About, Experience, Skills, Tools, Projects, Education
- Provide 2-3 specific mistakes per section
- Provide 2-3 specific improvement suggestions per section
- Focus on the most impactful issues only
""",
        "min": """
ANALYSIS DEPTH: BASIC (Basic Plan)
- Analyze sections: About, Experience, Skills, Tools, Projects, Education
- Provide 1-2 key mistakes per section
- Provide 1-2 key improvement suggestions per section
- Basic feedback only
"""
    }
    depth_note = depth_instructions.get(level, depth_instructions["standard"])

    system_message = f"""You are a Senior Talent Acquisition Specialist and Resume Expert. 
    Analyze the provided resume text and generate a COMPREHENSIVE structured JSON report.

    {depth_note}

    The report MUST include these exact keys:
    {{
        "candidate_info": {{
            "name": "Extract from resume header — REQUIRED",
            "job_title": "Extract target/current role — REQUIRED",
            "location": "City, Country or empty string",
            "email": "Candidate's email — REQUIRED",
            "phone": "Candidate's phone number if present",
            "social_links": {{ "github": "", "linkedin": "", "website": "", "portfolio": "" }}
        }},
        "overallScore": 85,
        "summary": "COMMERHENSIVE EXECUTIVE SUMMARY: Provide a detailed 3-4 sentence analysis that synthesizes findings from ALL sections (Experience, Skills, Education, etc.). It should highlight the biggest strengths and the most critical areas for improvement across the entire resume.",
        "recommended_summary": "AI-written 3-4 sentence professional summary/profile-intro the candidate should use at the top of their resume",
        "overall_recommendations": [
            "Highly strategic recommendation 1 for overall career impact",
            "Highly strategic recommendation 2 for overall resume effectiveness"
        ],
        "fix_counts": {{ "urgent": 2, "critical": 1, "optional": 3 }},
        "original_resume": {{
            "about": "Copy of the candidate's own About/Summary from their resume, or empty string",
            "experience": [
                {{ "company": "Company Name", "role": "Job Title", "duration": "Jan 2022 - Present", "location": "City", "description": "Key achievements and responsibilities" }}
            ],
            "skills": ["Skill 1", "Skill 2", "Skill 3"],
            "tools": ["Tool 1", "Tool 2"],
            "projects": [
                {{ "title": "Project name", "description": "What it does and tech used" }}
            ],
            "education": [
                {{ "school": "University Name", "degree": "Degree Name", "duration": "2018 - 2022" }}
            ]
        }},
        "sections": [
            {{
                "section_name": "About",
                "score": 70,
                "analysis": "Qualitative assessment of this section",
                "mistakes": ["Specific mistake 1", "Specific mistake 2"],
                "improvements": ["Specific improvement suggestion 1"],
                "urgent_count": 1,
                "critical_count": 0
            }}
        ]
    }}

    RULES:
    - candidate_info.name and candidate_info.job_title are REQUIRED — look at the very top of the resume
    - original_resume MUST contain the actual content from the resume, not placeholders
    - fix_counts MUST reflect the actual number of urgent_count, critical_count, and improvements across all sections
    - sections should cover the sections appropriate for the analysis depth above
    - Use these standard section_name values for consistency: "About", "Experience", "Skills", "Tools", "Projects", "Education"
    - Return ONLY the JSON object, no markdown, no extra text
    """

    try:
        response = client.chat.completions.create(
            model=cfg.get("model", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"RESUME TEXT TO ANALYZE:\n\n{resume_text}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        raw_json = response.choices[0].message.content.strip()
        logger.info(f"[_generate_report] Raw AI response length: {len(raw_json)}")
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error(f"[_generate_report] JSON parse error: {e}")
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"[_generate_report] AI error: {e}")
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

@router.post("/analyze")
async def analyze_resume_report(request: ResumeReportRequest = Body(...)):
    """Analyze resume and return structured JSON — no strict Pydantic validation on response."""
    if len(request.resumeText) < 100:
        raise HTTPException(status_code=400, detail="Resume text too short for analysis.")
    
    data = await _generate_report(request.resumeText, request.reportLevel)
    
    # Log key fields for debugging
    logger.info(f"[analyze_resume_report] name={data.get('candidate_info', {}).get('name', 'MISSING')}, "
                f"score={data.get('overallScore', 'MISSING')}, "
                f"sections={len(data.get('sections', []))}, "
                f"skills={len(data.get('original_resume', {}).get('skills', []))}")

    # Ensure required keys exist with safe defaults so backend never crashes
    data.setdefault("candidate_info", {})
    data["candidate_info"].setdefault("name", "")
    data["candidate_info"].setdefault("job_title", "")
    data["candidate_info"].setdefault("location", "")
    data["candidate_info"].setdefault("email", "")
    data["candidate_info"].setdefault("phone", "")
    data["candidate_info"].setdefault("social_links", {})
    data.setdefault("overallScore", 0)
    data.setdefault("summary", "")
    data.setdefault("recommended_summary", "")
    data.setdefault("overall_recommendations", [])
    data.setdefault("fix_counts", {"urgent": 0, "critical": 0, "optional": 0})
    data.setdefault("original_resume", {"about": "", "experience": [], "skills": [], "tools": [], "projects": [], "education": []})
    data.setdefault("sections", [])

    return JSONResponse(content=data)
