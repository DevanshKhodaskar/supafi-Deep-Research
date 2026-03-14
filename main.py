import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
You are a Legal Data Collection Compliance Agent for a platform that connects AI companies with normal users.

Platform model:
- AI companies post the type of data they want.
- Normal users upload photos, videos, text, audio, documents, or other data.
- The platform may store, review, annotate, share, or distribute that data for AI training.

Your task is to decide whether the described collection and sharing workflow is legally valid.

Always assess privacy, consent, licensing, public-domain claims, sensitive personal data, and sector-specific restrictions.
Pay special attention to high-risk categories such as:
- medical records, x-rays, scans, prescriptions, health data
- biometric data, face images, voiceprints, fingerprints, iris data
- children's data
- financial data
- government IDs
- intimate/private images
- precise location data
- surveillance/CCTV footage

Important reasoning rules:
- If the workflow clearly involves sensitive or regulated personal data without a strong lawful basis, explicit consent, or required safeguards, mark it INVALID.
- If the workflow may be allowed only with strict consent, anonymization, licensing, access control, or contractual restrictions, mark it CONDITIONAL.
- Mark it VALID only when the described data collection and sharing are low-risk and there is no obvious privacy, medical, biometric, copyright, or confidentiality concern.
- A claim that data is "public" does not automatically make collection or republication lawful.

Always search for relevant laws and guidance such as GDPR, DPDP Act India, HIPAA, CCPA, biometric privacy rules, medical privacy rules, copyright/licensing terms, and platform/public-record restrictions when applicable.

Return ONLY JSON in this exact shape:
{
  "status": "VALID or INVALID or CONDITIONAL",
  "is_valid": true,
  "summary": "short explanation",
  "risk_level": "LOW or MEDIUM or HIGH",
  "sensitive_data_flags": ["flag 1", "flag 2"],
  "required_conditions": ["condition 1", "condition 2"],
  "citations": ["law or source 1", "law or source 2"]
}

Rules:
- Output JSON only.
- status must be exactly VALID, INVALID, or CONDITIONAL.
- is_valid must be true only when status is VALID.
- summary must be concise and practical.
- citations should mention actual laws, regulations, or authoritative guidance when found.
"""


def _build_tools() -> list[Any]:
    from langchain_community.tools import DuckDuckGoSearchRun

    return [
        DuckDuckGoSearchRun(
            name="internet_search",
            description="Search the internet for laws, regulations, privacy rules, and licensing guidance.",
        )
    ]


def _get_google_api_key() -> str:
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return google_api_key


@lru_cache(maxsize=1)
def get_research_agent():
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        api_key=_get_google_api_key(),
    )

    return create_agent(
        model=llm,
        tools=_build_tools(),
        system_prompt=SYSTEM_PROMPT,
    )


# -----------------------------
# 5. JSON Extractor
# -----------------------------
def _response_text(response: dict[str, Any]) -> str:
    content = response["messages"][-1].content
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text_parts.append(item["text"])

    return "\n".join(text_parts).strip()


def extract_json(response: dict[str, Any]) -> dict[str, Any]:
    text = _response_text(response)

    # Remove markdown
    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # Convert to JSON
    return json.loads(text)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_status = str(result.get("status") or result.get("verdict") or "CONDITIONAL").strip().upper()
    status_map = {
        "YES": "VALID",
        "NO": "INVALID",
        "VALID": "VALID",
        "INVALID": "INVALID",
        "CONDITIONAL": "CONDITIONAL",
    }
    status = status_map.get(raw_status, "CONDITIONAL")

    is_valid = result.get("is_valid")
    if not isinstance(is_valid, bool):
        is_valid = status == "VALID"

    risk_level = str(result.get("risk_level") or "MEDIUM").strip().upper()
    if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
        risk_level = "MEDIUM"

    return {
        "status": status,
        "is_valid": is_valid,
        "summary": str(result.get("summary") or result.get("reason") or "No summary returned.").strip(),
        "risk_level": risk_level,
        "sensitive_data_flags": _string_list(
            result.get("sensitive_data_flags") or result.get("restricted_data")
        ),
        "required_conditions": _string_list(
            result.get("required_conditions") or result.get("key_constraints")
        ),
        "citations": _string_list(result.get("citations") or result.get("sources") or result.get("sources_found")),
    }


def build_compliance_query(platform_activity: str) -> str:
    return f"""
Evaluate whether this data collection and sharing workflow is legally valid for an AI-data marketplace.

Platform workflow:
- AI companies request specific kinds of data.
- Normal users upload the requested data.
- The platform may store, review, annotate, share, or distribute the uploaded data for AI training.

Requested activity to evaluate:
{platform_activity}

Determine whether this is VALID, INVALID, or CONDITIONAL.
Focus on whether this kind of data can be legally collected from users and whether it can be shared or put into broader/public use.
""".strip()


# -----------------------------
# 6. Run Function
# -----------------------------
def run_deep_research(platform_activity: str, verbose: bool = False) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    if verbose:
        print("\nRunning Legal Compliance Check")
        print("=" * 50)

    response = get_research_agent().invoke({
        "messages": [HumanMessage(content=build_compliance_query(platform_activity))]
    })

    result = normalize_result(extract_json(response))

    if verbose:
        print("\nResult:\n")
        print(json.dumps(result, indent=2))

    return result


# -----------------------------
# 7. Example Query
# -----------------------------
scenario = """
Our platform collects images from publicly available open datasets such as Open Images and COCO,
which are released under permissive licenses for research and machine learning training.

The images are used for object detection training and are distributed to global annotators.
No personal information is collected.

Is this legal?
"""

if __name__ == "__main__":
    run_deep_research(scenario, verbose=True)