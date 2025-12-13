# CV Guard (EpimindsHack)

## **An innovative solution to integrity within the Hiring Process**

Multi-agent CV verification: parse a CV, generate verifiable claims, cross-check against GitHub and LinkedIn, score reliability, and present a live discussion + summary.

## What’s inside
- `agent_workflow.py`: pipeline of OpenAI agents (CVClaims, RepoVerification, LinkedInVerification, ReliabilityScoring, Summary).
- `cv_fetch.py`: CLI CV parser that extracts structured JSON from a PDF.
- `gitTool.py`: GitHub repo fetcher (supports PAT for higher rate limits).
- `app.py` + `templates/`: Flask web UI with streaming SSE updates and an agent discussion view.
- `agent_main.py` + `src/`: LinkedIn scraping helpers using LinkdAPI (optional).

### Agents and tools
- **CVClaimsAgent**: Uses OpenAI to turn structured CV JSON (from `cv_fetch.py`/pdfplumber) into atomic, verifiable claims and a short summary. No external calls beyond the LLM + parsed CV data.
- **RepoVerificationAgent**: Fetches GitHub repo metadata via `gitTool.get_github_repositories` (GitHub REST API; optional PAT), then prompts OpenAI to compare claims against repo fields (name/desc/language/topics/stars/etc.). No repo checkout; metadata only.
- **LinkedInVerificationAgent**: If a LinkdAPI key is present, fetches profile data via `src/api/client.py` (`LinkedInAPIClient` + `linkdapi.AsyncLinkdAPI` with retries), then prompts OpenAI to compare claims against positions/education/skills/headline. Skips if no key/URL.
- **ReliabilityScoringAgent**: Prompts OpenAI to fuse GitHub + LinkedIn verification results into a single 0–100 reliability score with rationale and weighted breakdown. Consumes previous outputs; no external API calls.
- **SummaryAgent**: Prompts OpenAI to produce a user-facing report (<=120 words, highlights, score) using claims, verification outputs, the reliability score, and recent transcript messages.

## Setup
1) Python 3.12+ recommended.  
2) Create `.env` with at least:
```
OPENAI_API_KEY=...
GITHUB_TOKEN=...          # optional, helps avoid rate limits
LINKD_API_KEY=...         # or LINKDAPI_API_KEY for LinkedIn scraping
```
3) Install deps:
```
pip install -r requirements.txt
```
4) For LinkedIn scraping, also set `config.ini` with your LinkdAPI key (already checked in for local use).

## CLI usage
Run the full multi-agent pipeline from the command line:
```
python agent_workflow.py path/to/cv.pdf <github_username> \
  --linkedin https://linkedin.com/in/example \
  --github-token <optional_token> \
  --linkedin-api-key <optional_linkd_key> \
  --verbose   # include agent transcript
```




## Web app
Start the Flask server:
```
python app.py
```
