import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from cv_fetch import parse_cv
from gitTool import get_github_repositories

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


class CVClaimsAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def gather(self, cv_path: str, linkedin_url: Optional[str] = None, transcript: Optional[List[Dict]] = None) -> Dict:
        """Extract structured data from the CV and turn it into verifiable claims."""
        if transcript is not None:
            transcript.append({"agent": "CVClaimsAgent", "message": f"Parsing CV at {cv_path}."})

        structured_cv = parse_cv(cv_path)

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
                    "message": f"Generated {len(claims)} verifiable claims; summary: {claims_payload.get('summary', '')[:120]}",
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
                    "message": f"Checking {len(claims)} claims against {len(repos)} GitHub repos.",
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
            transcript.append(
                {
                    "agent": "RepoVerificationAgent",
                    "message": f"Verification complete: {len(supported)} supported, {len(verification.get('results', [])) - len(supported)} other outcomes.",
                }
            )

        return verification


class ReliabilityScoringAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def score(self, claims: List[Dict], verification: Dict, transcript: Optional[List[Dict]] = None) -> Dict:
        """Score the candidate based on verification outcomes."""
        prompt = f"""
You are ReliabilityScorer. Given CV claims and their verification results, provide a 0-100 reliability score.

Return JSON with:
- score: integer 0-100
- rationale: string (<=80 words)
- breakdown: list of {{claim_id, weight, impact, note}}

Heuristics:
- Reward supported claims, penalize not_found, partial in between.
- Use higher weights for project/role claims than generic skills.
Claims:
{json.dumps(claims)}
Verification Results:
{json.dumps(verification)}
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

        if transcript is not None:
            transcript.append(
                {
                    "agent": "ReliabilityScoringAgent",
                    "message": f"Assigned score {score_payload.get('score')} with rationale: {score_payload.get('rationale', '')[:120]}",
                }
            )

        return score_payload


def run_claimcheck(
    cv_path: str,
    github_username: str,
    linkedin_url: Optional[str] = None,
    github_token: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """Orchestrate the three-agent pipeline end-to-end."""
    transcript: Optional[List[Dict]] = [] if verbose else None

    claims_agent = CVClaimsAgent(client)
    verifier = RepoVerificationAgent(client)
    scorer = ReliabilityScoringAgent(client)

    claims_payload = claims_agent.gather(cv_path=cv_path, linkedin_url=linkedin_url, transcript=transcript)
    repos = get_github_repositories(github_username, github_token)
    verification = verifier.verify(claims_payload["claims"], repos, transcript=transcript)
    reliability = scorer.score(claims_payload["claims"], verification, transcript=transcript)

    return {
        "claims": claims_payload["claims"],
        "claims_summary": claims_payload["summary"],
        "structured_cv": claims_payload["structured_cv"],
        "repos_checked": len(repos),
        "verification": verification,
        "reliability": reliability,
        "transcript": transcript or [],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ClaimCheck multi-agent pipeline.")
    parser.add_argument("cv", help="Path to CV PDF file")
    parser.add_argument("github", help="GitHub username to verify claims against")
    parser.add_argument("--linkedin", help="LinkedIn profile URL", default=None)
    parser.add_argument("--github-token", help="GitHub token to avoid rate limits", default=None)
    parser.add_argument("--verbose", help="Return agent transcript", action="store_true")
    args = parser.parse_args()

    result = run_claimcheck(
        cv_path=args.cv,
        github_username=args.github,
        linkedin_url=args.linkedin,
        github_token=args.github_token,
        verbose=bool(args.verbose),
    )

    print(json.dumps(result, indent=2))
