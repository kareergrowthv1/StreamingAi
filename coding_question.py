"""
Coding question generation and Judge0-based code execution.
- POST /generate-coding-question  → AI-generated HackerRank-style problem (JSON)
- POST /run-code                  → Execute candidate code against test cases via Judge0 (RapidAPI)

Uses dynamic AI config (Superadmin GET /superadmin/settings/ai-config); no hardcoded OpenAI keys.
Judge0 config is fetched dynamically from Superadmin settings (/superadmin/settings/judge0).
"""
import asyncio
import base64
import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

import config

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/coding", tags=["coding"])

# ─────────────────────────────────────────────────────────────────────────────
# Judge0 helpers
# ─────────────────────────────────────────────────────────────────────────────

JUDGE0_SETTINGS_CACHE_TTL_SEC = 60
_judge0_settings_cache = {
    "fetched_at": 0.0,
    "config": None,
}


def _first_non_empty_string(values: list) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_judge0_settings_payload(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    enabled = bool(payload.get("enabled", True))
    api_key = _first_non_empty_string([payload.get("apiKey"), payload.get("rapidApiKey")])
    base_url = str(payload.get("baseUrl", "")).strip().rstrip("/")
    api_host = str(payload.get("apiHost", "")).strip()
    if not api_host and base_url:
        try:
            api_host = str(httpx.URL(base_url).host or "")
        except Exception:
            api_host = ""
    if not enabled or not api_key or not base_url or not api_host:
        return None
    return {
        "apiKey": api_key,
        "baseUrl": base_url,
        "apiHost": api_host,
    }


async def _get_judge0_config() -> dict | None:
    now = time.time()
    cached = _judge0_settings_cache.get("config")
    fetched_at = _judge0_settings_cache.get("fetched_at", 0.0)
    if cached and (now - fetched_at) < JUDGE0_SETTINGS_CACHE_TTL_SEC:
        return cached

    superadmin_url = str(getattr(config, "SUPERADMIN_BACKEND_URL", "")).strip().rstrip("/")
    service_token = _first_non_empty_string([
        getattr(config, "SUPERADMIN_SERVICE_TOKEN", ""),
        getattr(config, "INTERNAL_SERVICE_TOKEN", ""),
    ])
    if not superadmin_url or not service_token:
        logger.warning("[coding_question] Superadmin URL/service token missing; cannot fetch Judge0 settings dynamically.")
        return None

    url = f"{superadmin_url}/superadmin/settings/judge0"
    headers = {
        "X-Service-Token": service_token,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.warning("[coding_question] Judge0 settings fetch failed: %s %s", resp.status_code, resp.text[:200])
            return None
        body = {}
        content_type = str(resp.headers.get("content-type", "")).lower()
        if content_type.startswith("application/json"):
            body = resp.json()
        data = body.get("data", {}) if isinstance(body, dict) else {}
        cfg = _parse_judge0_settings_payload(data)
        if not cfg:
            logger.warning("[coding_question] Judge0 settings are empty/incomplete in Superadmin.")
            return None
        _judge0_settings_cache["fetched_at"] = now
        _judge0_settings_cache["config"] = cfg
        return cfg
    except Exception as exc:
        logger.warning("[coding_question] Judge0 settings fetch exception: %s", exc)
        return None

def _judge0_headers(judge0_config: dict) -> dict:
    key = judge0_config.get("apiKey", "")
    return {
        "Content-Type": "application/json",
        "x-rapidapi-host": judge0_config.get("apiHost", ""),
        "x-rapidapi-key": key,
    }

# Judge0 language IDs for common languages
LANGUAGE_ID_MAP = {
    "javascript": 63,
    "python": 71,
    "python3": 71,
    "java": 62,
    "c++": 54,
    "cpp": 54,
    "c": 50,
    "go": 60,
    "ruby": 72,
    "typescript": 74,
    "csharp": 51,
    "c#": 51,
    "rust": 73,
    "swift": 83,
    "kotlin": 78,
    "php": 68,
}

def get_language_id(language: str) -> int:
    """Return Judge0 language ID for a language name. Defaults to Python 3."""
    return LANGUAGE_ID_MAP.get(language.lower().strip(), 71)


def _normalize(output: str) -> str:
    if not output:
        return ""
    return output.strip().replace("\r\n", "\n").replace("\r", "\n")


def _outputs_match(actual: str, expected: str) -> bool:
    a, e = _normalize(actual), _normalize(expected)
    if a == e:
        return True
    # Numeric comparison
    try:
        return abs(float(a) - float(e)) < 1e-9
    except ValueError:
        pass
    # JSON comparison
    try:
        return json.dumps(json.loads(a), sort_keys=True) == json.dumps(json.loads(e), sort_keys=True)
    except Exception:
        pass
    return False


async def _run_code_get_output(
    client: httpx.AsyncClient,
    judge0_config: dict,
    source_code: str,
    language_id: int,
    stdin: str,
    timeout: int = 10,
) -> Optional[str]:
    """
    Run source_code with the given stdin via Judge0 and return stdout (stripped).
    Returns None if execution fails or produces non-zero exit / compile error.
    Used to compute ground-truth expected outputs for generated test cases.
    """
    try:
        encoded_code = base64.b64encode(source_code.encode()).decode()
        encoded_stdin = base64.b64encode(stdin.encode()).decode() if stdin else ""
        payload = {
            "language_id": language_id,
            "source_code": encoded_code,
            "stdin": encoded_stdin,
        }
        sub_url = f"{judge0_config['baseUrl']}/submissions?base64_encoded=true&wait=false&fields=*"
        resp = await client.post(sub_url, json=payload, headers=_judge0_headers(judge0_config), timeout=15)
        if resp.status_code != 201:
            return None
        token = resp.json().get("token")
        if not token:
            return None

        poll_url = f"{judge0_config['baseUrl']}/submissions/{token}?base64_encoded=true&fields=stdout,stderr,status,exit_code"
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(1.0)
            pr = await client.get(poll_url, headers=_judge0_headers(judge0_config), timeout=10)
            data = pr.json()
            status_id = data.get("status", {}).get("id", 0)
            if status_id in (1, 2):  # In Queue / Processing
                continue
            if status_id == 3:  # Accepted
                raw = data.get("stdout") or ""
                try:
                    return base64.b64decode(raw).decode(errors="ignore").strip()
                except Exception:
                    return str(raw).strip()
            # Compile error, runtime error, TLE, etc. — not usable
            return None
        return None
    except Exception:
        return None


async def _compute_outputs_via_judge0(
    judge0_config: dict,
    reference_solution: str,
    language_id: int,
    test_cases: list,
) -> list:
    """
    Run `reference_solution` against every test case input via Judge0 in parallel.
    Replaces each test case's 'output' with the real computed stdout.
    Test cases where execution fails keep their original AI-generated output.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            _run_code_get_output(client, judge0_config, reference_solution, language_id, tc.get("input", ""))
            for tc in test_cases
        ]
        results = await asyncio.gather(*tasks)

    verified = []
    for tc, computed in zip(test_cases, results):
        if computed is not None:
            verified.append({**tc, "output": computed})
        else:
            # Fallback: keep AI-generated output but log it
            logger.warning(
                "[coding_question] Could not verify output for input %r — keeping AI-generated value",
                tc.get("input", "")[:80],
            )
            verified.append(tc)
    return verified


async def _run_single_test_case(
    client: httpx.AsyncClient,
    judge0_config: dict,
    source_code: str,
    language_id: int,
    stdin: str,
    expected_output: str,
    test_case_id: str,
    timeout: int = 10,
) -> dict:
    """Submit one test case to Judge0 and poll until result is ready."""
    try:
        encoded_code = base64.b64encode(source_code.encode()).decode()
        encoded_stdin = base64.b64encode(stdin.encode()).decode() if stdin else ""

        payload = {
            "language_id": language_id,
            "source_code": encoded_code,
            "stdin": encoded_stdin,
        }

        sub_url = f"{judge0_config['baseUrl']}/submissions?base64_encoded=true&wait=false&fields=*"
        resp = await client.post(sub_url, json=payload, headers=_judge0_headers(judge0_config), timeout=15)
        if resp.status_code != 201:
            raise Exception(f"Judge0 submission failed: {resp.text}")

        token = resp.json().get("token")
        if not token:
            raise Exception("No token returned by Judge0")

        # Poll for result
        result_url = f"{judge0_config['baseUrl']}/submissions/{token}?base64_encoded=true&fields=*"
        waited = 0
        while waited < timeout:
            await asyncio.sleep(1)
            waited += 1
            r = await client.get(result_url, headers=_judge0_headers(judge0_config), timeout=10)
            if r.status_code != 200:
                raise Exception(f"Judge0 result fetch failed: {r.text}")
            data = r.json()
            status_id = data.get("status", {}).get("id", 0)
            if status_id in (1, 2):  # In Queue / Processing
                continue

            # Decode output fields
            def _decode(field):
                raw = data.get(field)
                if not raw:
                    return ""
                try:
                    return base64.b64decode(raw).decode(errors="ignore")
                except Exception:
                    return str(raw)

            stdout = _decode("stdout")
            stderr = _decode("stderr") or _decode("compile_output")
            status_desc = data.get("status", {}).get("description", "Unknown")

            passed = False
            if status_id == 3:  # Accepted
                passed = _outputs_match(stdout, expected_output)
                if not passed:
                    status_desc = "Wrong Answer"

            exec_time = data.get("time") or "0"
            memory = data.get("memory") or "0"

            return {
                "testCaseId": test_case_id,
                "input": stdin,
                "expectedOutput": expected_output,
                "actualOutput": stdout,
                "passed": passed,
                "executionTime": str(exec_time),
                "memoryUsed": str(memory),
                "errorMessage": stderr,
                "status": status_desc,
                "locked": False,
            }

        # Timeout
        return {
            "testCaseId": test_case_id,
            "input": stdin,
            "expectedOutput": expected_output,
            "actualOutput": "",
            "passed": False,
            "executionTime": "0",
            "memoryUsed": "0",
            "errorMessage": "Time Limit Exceeded",
            "status": "Time Limit Exceeded",
            "locked": False,
        }

    except Exception as exc:
        return {
            "testCaseId": test_case_id,
            "input": stdin,
            "expectedOutput": expected_output,
            "actualOutput": "",
            "passed": False,
            "executionTime": "0",
            "memoryUsed": "0",
            "errorMessage": str(exc),
            "status": "System Error",
            "locked": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class GenerateQuestionRequest(BaseModel):
    programmingLanguage: str = Field(..., description="e.g. Python, JavaScript, Java")
    difficultyLevel: str = Field(..., description="Easy | Medium | Hard")
    questionSource: str = Field("Coding Library", description="Coding Library | Custom Question")
    topicTags: Optional[List[str]] = Field(None, description="Optional topic hints, e.g. ['Arrays','Strings']")
    questionIndex: int = Field(0, description="Index of this question (0-based) for variety")


class TestCaseInput(BaseModel):
    testCaseId: str = ""
    input: str
    expectedOutput: str
    locked: bool = False


class RunCodeRequest(BaseModel):
    sourceCode: str
    language: str = Field(..., description="e.g. python, javascript, java")
    testCases: List[TestCaseInput]
    timeoutSeconds: int = Field(10, ge=1, le=30)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate-coding-question")
async def generate_coding_question(req: GenerateQuestionRequest):
    """
    Generate a HackerRank-style coding question using AI.
    Returns full problem structure: title, description, examples, constraints,
    starter code in the requested language, and 6 test cases (4 visible + 2 locked).
    """
    try:
        from ai_config_loader import get_ai_config
        import openai

        cfg = await get_ai_config()
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="AI service not configured. Set API key in Superadmin → AI Config.")

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        )
        model = cfg.get("model", "gpt-4o-mini")
        topic_hint = ""
        if req.topicTags:
            topic_hint = f"Focus on these topics: {', '.join(req.topicTags)}. "

        seed = int(time.time() * 1000) + req.questionIndex * 997  # vary per question

        system_msg = (
            f"You are an expert competitive programming problem setter. "
            f"Generate a UNIQUE {req.difficultyLevel}-level coding problem solvable in {req.programmingLanguage}. "
            f"Each call must produce a completely different problem. "
            f"Respond ONLY with valid JSON — no markdown fences, no extra text."
        )

        user_msg = (
            f"Create a {req.difficultyLevel} {req.programmingLanguage} coding problem. "
            f"{topic_hint}"
            f"Variety seed: {seed}\n\n"
            f"Return EXACTLY this JSON structure (all fields required):\n"
            "{\n"
            '  "title": "Short problem title (max 8 words)",\n'
            '  "description": "Detailed scenario-based problem statement (3-5 sentences)",\n'
            '  "functionSignature": "Complete function/method signature in ' + req.programmingLanguage + '",\n'
            '  "inputFormat": "Description of input format",\n'
            '  "outputFormat": "Description of expected output format",\n'
            '  "constraints": ["constraint 1", "constraint 2", "constraint 3"],\n'
            '  "examples": [\n'
            '    {"input": "example input 1", "output": "example output 1", "explanation": "why"},\n'
            '    {"input": "example input 2", "output": "example output 2", "explanation": "why"}\n'
            '  ],\n'
            '  "testCases": [\n'
            '    {"input": "tc1 stdin", "output": "tc1 expected stdout", "locked": false},\n'
            '    {"input": "tc2 stdin", "output": "tc2 expected stdout", "locked": false},\n'
            '    {"input": "tc3 stdin", "output": "tc3 expected stdout", "locked": false},\n'
            '    {"input": "tc4 stdin", "output": "tc4 expected stdout", "locked": false},\n'
            '    {"input": "tc5 stdin", "output": "tc5 expected stdout", "locked": true},\n'
            '    {"input": "tc6 stdin", "output": "tc6 expected stdout", "locked": true}\n'
            '  ],\n'
            '  "referenceSolution": "A complete, working ' + req.programmingLanguage + ' program that reads from stdin and prints the answer to stdout. This will be executed to verify every test case output — it MUST be correct and produce exactly the output the problem expects.",\n'
            '  "starterCode": "Starter code in ' + req.programmingLanguage + ' with a TODO comment showing the candidate where to write their solution"\n'
            "}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. referenceSolution MUST be a fully working program (reads stdin, prints stdout). "
            "It will be run against every test case to auto-verify expected outputs.\n"
            "2. testCases inputs must be consistent with the problem description. "
            "The 'output' field in each test case is used ONLY as a fallback — the real expected "
            "output will be computed by running referenceSolution. Still fill them correctly.\n"
            "3. starterCode must compile — correct syntax for " + req.programmingLanguage + ".\n"
            "4. Locked test cases are hidden from the candidate (they can only see if they pass/fail).\n"
            "5. Make the problem original and not a well-known leetcode problem.\n"
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )

        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)

        # Normalize
        data.setdefault("title", f"{req.difficultyLevel} {req.programmingLanguage} Challenge")
        data.setdefault("description", "Solve the problem described above.")
        data.setdefault("functionSignature", "")
        data.setdefault("inputFormat", "")
        data.setdefault("outputFormat", "")
        data.setdefault("constraints", [])
        data.setdefault("examples", [])
        data.setdefault("testCases", [])
        data.setdefault("starterCode", f"# Write your {req.programmingLanguage} solution here\n")
        data.setdefault("referenceSolution", "")
        data["difficulty"] = req.difficultyLevel
        data["language"] = req.programmingLanguage
        data["questionSource"] = req.questionSource

        # ── Verify / recompute expected outputs via Judge0 ───────────────────
        # Run the AI's reference solution against every test case input using
        # Judge0 so that expected outputs are always mathematically correct,
        # regardless of what the AI wrote in the "output" field.
        reference_solution = data.get("referenceSolution", "").strip()
        if reference_solution and data["testCases"]:
            lang_id = get_language_id(req.programmingLanguage)
            judge0_config = await _get_judge0_config()
            if not judge0_config:
                logger.warning("[coding_question] Judge0 config missing; skipping test case verification.")
            else:
                logger.info(
                    "[coding_question] Verifying %d test case(s) via Judge0 (lang_id=%d)…",
                    len(data["testCases"]), lang_id,
                )
                data["testCases"] = await _compute_outputs_via_judge0(
                    judge0_config, reference_solution, lang_id, data["testCases"]
                )
                logger.info("[coding_question] Test case verification complete.")

        # Strip referenceSolution from the response sent to the client
        # (it's a backend-only tool; no need to expose the answer to candidates)
        data.pop("referenceSolution", None)

        return data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error("Coding question JSON parse error: %s", e)
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        logger.error("Coding question generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e}")


@router.post("/run-code")
async def run_code(req: RunCodeRequest):
    """
    Execute candidate's source code against all test cases using Judge0 (RapidAPI).
    Visible test cases return full I/O details; locked test cases only return pass/fail.
    """
    if not req.testCases:
        raise HTTPException(status_code=400, detail="At least one test case is required.")

    judge0_config = await _get_judge0_config()
    if not judge0_config:
        raise HTTPException(status_code=503, detail="Judge0 settings are missing/incomplete in Superadmin. Configure apiKey/baseUrl/apiHost.")

    language_id = get_language_id(req.language)
    results = []

    async with httpx.AsyncClient() as client:
        tasks = [
            _run_single_test_case(
                client=client,
                    judge0_config=judge0_config,
                source_code=req.sourceCode,
                language_id=language_id,
                stdin=tc.input,
                expected_output=tc.expectedOutput,
                test_case_id=tc.testCaseId or f"tc-{i}",
                timeout=req.timeoutSeconds,
            )
            for i, tc in enumerate(req.testCases)
        ]
        results = await asyncio.gather(*tasks)

    # Mask locked test case I/O
    final = []
    for i, (r, tc) in enumerate(zip(results, req.testCases)):
        if tc.locked:
            final.append({
                **r,
                "input": "Hidden",
                "expectedOutput": "Hidden",
                "actualOutput": "Hidden" if r["passed"] else "Hidden",
                "locked": True,
            })
        else:
            final.append({**r, "locked": False})

    passed = sum(1 for r in final if r["passed"])
    total = len(final)

    return {
        "overallResult": "passed" if passed == total else "failed",
        "passedCount": passed,
        "totalCount": total,
        "results": final,
    }
