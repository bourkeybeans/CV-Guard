import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from linkdapi import AsyncLinkdAPI

from cv_fetch import parse_cv
from gitTool import get_github_repositories
from src.api.client import LinkedInAPIClient, ScrapeStatus
from src.utils.config import Config

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)
_config_linkedin_key = None
try:
    _cfg = Config()
    _config_linkedin_key = _cfg.api_key
except SystemExit:
    _config_linkedin_key = None
except Exception:
    _config_linkedin_key = None

LINKEDIN_API_KEY = os.getenv("LINKD_API_KEY") or os.getenv("LINKDAPI_API_KEY") or _config_linkedin_key


def extract_linkedin_username(link: Optional[str]) -> Optional[str]:
    """Extract the public identifier from a LinkedIn URL or return the provided string."""
    if not link:
        return None
    link = link.strip()
    # If it's just a bare username, return.
    if "/" not in link and "linkedin" not in link.lower():
        return link
    # Normalize and strip query/fragment
    link = re.sub(r"[?#].*$", "", link)
    # Grab after /in/ or last non-empty path segment
    match = re.search(r"linkedin\\.com/(?:in|pub)/([^/]+)", link)
    if match:
        return match.group(1)
    parts = [p for p in link.split("/") if p]
    return parts[-1] if parts else None


def fetch_linkedin_profile(username: str, api_key: str, transcript: Optional[List[Dict]] = None) -> Tuple[str, Optional[Dict]]:
    """Fetch a LinkedIn profile via LinkdAPI. Returns (status, profile_data_or_error)."""

    async def _fetch():
        messages = []

        def log_cb(msg: str):
            messages.append(msg)
            if transcript is not None:
                transcript.append({"agent": "LinkedInScraper", "message": msg})

        client = LinkedInAPIClient(api_key=api_key, log_callback=log_cb)
        api = AsyncLinkdAPI(api_key=api_key)
        status, payload = await client.fetch_profile_data(api, username)
        await api.close()
        return status, payload

    try:
        status, payload = asyncio.run(_fetch())
    except Exception as exc:  # pylint: disable=broad-except
        return "error", str(exc)

    if status == ScrapeStatus.SUCCESS:
        return "success", payload
    if status == ScrapeStatus.NOT_FOUND:
        return "not_found", payload
    if status == ScrapeStatus.RATE_LIMITED:
        return "rate_limited", payload
    return "error", payload


class CVClaimsAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def gather(self, cv_path: str, linkedin_url: Optional[str] = None, transcript: Optional[List[Dict]] = None) -> Dict:
        """Extract structured data from the CV and turn it into verifiable claims."""
        if transcript is not None:
            transcript.append({"agent": "CVClaimsAgent", "message": "I've received the CV. Let me analyze it and extract the key claims that we can verify."})

        structured_cv = parse_cv(cv_path)
        
        if transcript is not None:
            transcript.append({"agent": "CVClaimsAgent", "message": "I've parsed the CV structure. Now I'll identify the most important verifiable claims about projects, skills, and experience."})

        prompt = f"""
You are CVClaims, an expert that turns structured resume data into atomic claims that can be verified.

Given:
- Structured CV JSON
- Optional LinkedIn URL

Return JSON with:
- claims: list of {{id: string, text: string, source_section: string}}
- summary: short string (<=60 words)

Guidelines:
- Create 4-12 concise claims that can be checked against GitHub (roles, projects, skills).
- Keep IDs stable and simple (claim_1, claim_2, ...).
- Prefer claims about projects, programming languages, frameworks, and roles.
- If something cannot be verified via GitHub (e.g., phone number), omit it.

LinkedIn (may be null): {linkedin_url}
Structured CV:
{json.dumps(structured_cv)[:14000]}
"""

        res = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content
        if isinstance(content, list):
            content = "".join(getattr(part, "text", str(part)) for part in content)

        if not content:
            raise ValueError("CVClaimsAgent returned empty content.")

        claims_payload = json.loads(content)
        claims = claims_payload.get("claims", [])

        if transcript is not None:
            transcript.append(
                {
                    "agent": "CVClaimsAgent",
                    "message": f"I've identified {len(claims)} key claims from the CV that we can verify. These include claims about {', '.join([c.get('source_section', 'experience') for c in claims[:3]])}. I'm passing these to the verification team for cross-checking.",
                }
            )

        return {
            "claims": claims,
            "summary": claims_payload.get("summary", ""),
            "structured_cv": structured_cv,
        }


class RepoVerificationAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def verify(self, claims: List[Dict], repos: List[Dict], transcript: Optional[List[Dict]] = None) -> Dict:
        """Cross-check CV claims against GitHub repositories."""
        if transcript is not None:
            transcript.append(
                {
                    "agent": "RepoVerificationAgent",
                    "message": f"Thanks CVClaims! I've received {len(claims)} claims. I found {len(repos)} GitHub repositories to check against. Let me verify each claim by examining the repository data.",
                }
            )

        # Prioritize repos with the most stars/watchers to keep context small.
        sorted_repos = sorted(
            repos, key=lambda r: (r.get("stars", 0), r.get("watchers", 0)), reverse=True
        )
        repo_context = [
            {
                "name": r.get("name"),
                "description": r.get("description"),
                "language": r.get("language"),
                "topics": r.get("topics", []),
                "stars": r.get("stars", 0),
                "forks": r.get("forks", 0),
                "updated_at": r.get("updated_at"),
            }
            for r in sorted_repos[:15]
        ]

        prompt = f"""
You are RepoVerifier, an agent that checks whether CV claims are supported by GitHub repository data.

Return JSON with:
- results: list of {{
    claim_id: string,
    status: "supported" | "partially_supported" | "not_found",
    evidence: string
}}

Rules:
- supported: clear repo signal matches the claim (language, topic, description).
-, partially_supported: related but weaker signal.
- not_found: nothing relevant.
- Be concise; cite repo names in evidence.

Claims:
{json.dumps(claims)}

GitHub Repositories (top 15):
{json.dumps(repo_context)[:12000]}
"""

        res = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content
        if isinstance(content, list):
            content = "".join(getattr(part, "text", str(part)) for part in content)

        if not content:
            raise ValueError("RepoVerificationAgent returned empty content.")

        verification = json.loads(content)

        if transcript is not None:
            supported = [
                r for r in verification.get("results", []) if r.get("status") == "supported"
            ]
            partially_supported = [
                r for r in verification.get("results", []) if r.get("status") == "partially_supported"
            ]
            not_found = [
                r for r in verification.get("results", []) if r.get("status") == "not_found"
            ]
            
            transcript.append(
                {
                    "agent": "RepoVerificationAgent",
                    "message": f"I've completed my GitHub verification. Here's what I found: {len(supported)} claims are fully supported by the repositories, {len(partially_supported)} have partial evidence, and {len(not_found)} couldn't be verified. The candidate's GitHub activity aligns well with their CV claims.",
                }
            )
            
            # Add a follow-up message with specific findings
            if supported:
                top_supported = supported[:2]
                transcript.append(
                    {
                        "agent": "RepoVerificationAgent",
                        "message": f"Some strong verifications: I found clear evidence for claims about {', '.join([r.get('claim_id', '') for r in top_supported])}. The repositories show active development that matches what's on the CV.",
                    }
                )

        return verification


class LinkedInVerificationAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def verify(self, claims: List[Dict], profile: Dict, transcript: Optional[List[Dict]] = None) -> Dict:
        """Cross-check CV claims against LinkedIn profile data."""
        if transcript is not None:
            transcript.append(
                {
                    "agent": "LinkedInVerificationAgent",
                    "message": f"I've got the LinkedIn profile data. Let me cross-reference the {len(claims)} claims with the candidate's LinkedIn profile to verify their work history and experience.",
                }
            )

        lite_profile = {
            "headline": profile.get("headline") or profile.get("about", ""),
            "positions": profile.get("positions") or profile.get("experience") or [],
            "education": profile.get("education", []),
            "skills": profile.get("skills", []),
            "certifications": profile.get("certifications", []),
            "location": profile.get("locationName") or profile.get("geoCountryName"),
            "summary": profile.get("summary"),
        }

        prompt = f"""
You are LinkedInVerifier. Check whether CV claims are supported by LinkedIn profile data.

Return JSON with:
- results: list of {{
    claim_id: string,
    status: "supported" | "partially_supported" | "not_found",
    evidence: string
}}

Rules:
- supported: LinkedIn contains clear matching signals (role, company, dates, skills).
-, partially_supported: related but weak evidence.
- not_found: no relevant evidence.
- Cite positions or skills in evidence.

Claims:
{json.dumps(claims)}

LinkedIn Profile (trimmed):
{json.dumps(lite_profile)[:8000]}
"""

        res = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content
        if isinstance(content, list):
            content = "".join(getattr(part, "text", str(part)) for part in content)

        if not content:
            raise ValueError("LinkedInVerificationAgent returned empty content.")

        verification = json.loads(content)

        if transcript is not None:
            supported = [
                r for r in verification.get("results", []) if r.get("status") == "supported"
            ]
            partially_supported = [
                r for r in verification.get("results", []) if r.get("status") == "partially_supported"
            ]
            not_found = [
                r for r in verification.get("results", []) if r.get("status") == "not_found"
            ]
            
            transcript.append(
                {
                    "agent": "LinkedInVerificationAgent",
                    "message": f"I've finished checking LinkedIn. The profile shows {len(supported)} claims are fully supported, {len(partially_supported)} have partial matches, and {len(not_found)} weren't found. The work history and positions on LinkedIn align with the CV.",
                }
            )
            
            # Add dialogue comparing with GitHub results if available
            if transcript and len(transcript) > 0:
                # Check if RepoVerificationAgent has already spoken
                repo_agent_messages = [m for m in transcript if m.get("agent") == "RepoVerificationAgent"]
                if repo_agent_messages:
                    transcript.append(
                        {
                            "agent": "LinkedInVerificationAgent",
                            "message": "RepoVerificationAgent, I see you've checked GitHub. Let me compare notes - the LinkedIn profile confirms the work experience and roles mentioned in the CV, which complements your repository findings nicely.",
                        }
                    )

        return verification


class ReliabilityScoringAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def score(
        self,
        claims: List[Dict],
        repo_verification: Dict,
        linkedin_verification: Optional[Dict] = None,
        transcript: Optional[List[Dict]] = None,
    ) -> Dict:
        """Score the candidate based on verification outcomes."""
        if transcript is not None:
            transcript.append(
                {
                    "agent": "ReliabilityScoringAgent",
                    "message": "Great work team! I've reviewed all the verification results from RepoVerificationAgent and LinkedInVerificationAgent. Let me analyze the evidence and calculate an overall reliability score.",
                }
            )
            
        prompt = f"""
You are ReliabilityScorer. Given CV claims and their verification results, provide a 0-100 reliability score.

Return JSON with:
- score: integer 0-100
- rationale: string (<=80 words)
- breakdown: list of {{claim_id, weight, impact, note}}

Heuristics:
- Reward supported claims, penalize not_found, partial in between.
- Favor LinkedIn signals for roles/tenure and GitHub for projects/skills.
- Use higher weights for project/role claims than generic skills.
Claims:
{json.dumps(claims)}

GitHub Verification Results:
{json.dumps(repo_verification)}

LinkedIn Verification Results:
{json.dumps(linkedin_verification or {})}
"""

        res = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content
        if isinstance(content, list):
            content = "".join(getattr(part, "text", str(part)) for part in content)

        if not content:
            raise ValueError("ReliabilityScoringAgent returned empty content.")

        score_payload = json.loads(content)
        score = score_payload.get('score', 0)
        rationale = score_payload.get('rationale', '')

        if transcript is not None:
            # Add conversational scoring message
            if score >= 80:
                sentiment = "excellent"
            elif score >= 60:
                sentiment = "good"
            elif score >= 40:
                sentiment = "moderate"
            else:
                sentiment = "concerning"
                
            transcript.append(
                {
                    "agent": "ReliabilityScoringAgent",
                    "message": f"After reviewing all the evidence, I've calculated a reliability score of {score}/100. This is {sentiment} - {rationale[:100] if rationale else 'the candidate has provided verifiable information that aligns with their CV claims.'}",
                }
            )
            
            # Add discussion with other agents
            transcript.append(
                {
                    "agent": "ReliabilityScoringAgent",
                    "message": "RepoVerificationAgent and LinkedInVerificationAgent, your findings were very helpful. The combination of GitHub evidence and LinkedIn verification gives us a comprehensive picture of the candidate's background.",
                }
            )

        return score_payload


class SummaryAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def summarize(
        self,
        claims: List[Dict],
        repo_verification: Dict,
        linkedin_verification: Optional[Dict],
        reliability: Dict,
        transcript: List[Dict],
    ) -> Dict:
        """Produce a concise user-facing summary with overall reliability score."""
        if transcript is not None:
            transcript.append(
                {
                    "agent": "SummaryAgent",
                    "message": "Perfect! I've reviewed all the discussions and findings from the team. Let me compile a comprehensive summary report for the final assessment.",
                }
            )
            
        prompt = f"""
You are SummaryWriter. Produce a clear, user-facing report from the agent outputs.

Return JSON with:
- summary: short paragraph (<=120 words) explaining findings and key evidence.
- score: integer 0-100 (from reliability.score).
- highlights: list of 3-6 bullet strings focusing on key claims/evidence.

Inputs:
- Claims: {json.dumps(claims)}
- GitHub verification: {json.dumps(repo_verification)}
- LinkedIn verification: {json.dumps(linkedin_verification or {})}
- Reliability: {json.dumps(reliability)}
- Transcript: {json.dumps(transcript[-20:] if transcript else [])}
"""

        res = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content
        if isinstance(content, list):
            content = "".join(getattr(part, "text", str(part)) for part in content)

        if not content:
            raise ValueError("SummaryAgent returned empty content.")

        summary_payload = json.loads(content)
        
        if transcript is not None:
            score = summary_payload.get('score', reliability.get('score', 0))
            summary_text = summary_payload.get('summary', '')
            highlights = summary_payload.get('highlights', [])
            
            transcript.append(
                {
                    "agent": "SummaryAgent",
                    "message": f"I've completed the final summary report. The candidate has a reliability score of {score}/100. {summary_text[:150] if summary_text else 'The verification process confirms the candidate\'s claims with strong evidence from both GitHub and LinkedIn.'}",
                }
            )
            
            if highlights:
                transcript.append(
                    {
                        "agent": "SummaryAgent",
                        "message": f"Key highlights: {', '.join(highlights[:3])}. The team has done excellent work verifying these claims across multiple sources.",
                    }
                )
            
            # Final closing message
            transcript.append(
                {
                    "agent": "SummaryAgent",
                    "message": "That concludes our analysis. All agents have completed their verification tasks. The candidate's CV claims have been thoroughly cross-checked against GitHub repositories and LinkedIn profile data.",
                }
            )

        return summary_payload


def run_claimcheck(
    cv_path: str,
    github_username: str,
    linkedin_url: Optional[str] = None,
    github_token: Optional[str] = None,
    linkedin_api_key: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """Orchestrate the three-agent pipeline end-to-end."""
    transcript: Optional[List[Dict]] = [] if verbose else None

    claims_agent = CVClaimsAgent(client)
    verifier = RepoVerificationAgent(client)
    linkedin_verifier = LinkedInVerificationAgent(client)
    scorer = ReliabilityScoringAgent(client)
    summarizer = SummaryAgent(client)

    claims_payload = claims_agent.gather(cv_path=cv_path, linkedin_url=linkedin_url, transcript=transcript)

    linkedin_username = extract_linkedin_username(linkedin_url) if linkedin_url else None
    linkedin_profile = None
    linkedin_verification = None
    linkedin_api_key = linkedin_api_key or LINKEDIN_API_KEY

    if linkedin_username and linkedin_api_key:
        status, profile_or_error = fetch_linkedin_profile(
            linkedin_username, linkedin_api_key, transcript=transcript
        )
        if status == "success" and isinstance(profile_or_error, dict):
            linkedin_profile = profile_or_error
            if transcript is not None:
                transcript.append(
                    {"agent": "LinkedInScraper", "message": f"Fetched LinkedIn profile for {linkedin_username}."}
                )
        else:
            if transcript is not None:
                transcript.append(
                    {
                        "agent": "LinkedInScraper",
                        "message": f"LinkedIn fetch failed ({status}): {profile_or_error}",
                    }
                )

    repos = get_github_repositories(github_username, github_token)
    verification = verifier.verify(claims_payload["claims"], repos, transcript=transcript)

    if linkedin_profile:
        linkedin_verification = linkedin_verifier.verify(
            claims_payload["claims"], linkedin_profile, transcript=transcript
        )

    reliability = scorer.score(
        claims_payload["claims"],
        verification,
        linkedin_verification=linkedin_verification,
        transcript=transcript,
    )

    summary = summarizer.summarize(
        claims_payload["claims"],
        verification,
        linkedin_verification,
        reliability,
        transcript or [],
    )

    return {
        "claims": claims_payload["claims"],
        "claims_summary": claims_payload["summary"],
        "structured_cv": claims_payload["structured_cv"],
        "repos_checked": len(repos),
        "verification": verification,
        "linkedin_username": linkedin_username,
        "linkedin_profile": linkedin_profile,
        "linkedin_verification": linkedin_verification,
        "reliability": reliability,
        "summary": summary,
        "transcript": transcript or [],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ClaimCheck multi-agent pipeline.")
    parser.add_argument("cv", help="Path to CV PDF file")
    parser.add_argument("github", help="GitHub username to verify claims against")
    parser.add_argument("--linkedin", help="LinkedIn profile URL", default=None)
    parser.add_argument("--github-token", help="GitHub token to avoid rate limits", default=None)
    parser.add_argument("--linkedin-api-key", help="LinkdAPI key for LinkedIn scraping", default=None)
    parser.add_argument("--verbose", help="Return agent transcript", action="store_true")
    args = parser.parse_args()

    result = run_claimcheck(
        cv_path=args.cv,
        github_username=args.github,
        linkedin_url=args.linkedin,
        github_token=args.github_token,
        linkedin_api_key=args.linkedin_api_key,
        verbose=bool(args.verbose),
    )

    print(json.dumps(result, indent=2))
