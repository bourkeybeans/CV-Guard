# ClaimCheck (EpimindsHack)

Multi-agent CV verification: parse a CV, generate verifiable claims, cross-check against GitHub and LinkedIn, score reliability, and present a live discussion + summary.

## What’s inside
- `agent_workflow.py`: pipeline of OpenAI agents (CVClaims, RepoVerification, LinkedInVerification, ReliabilityScoring, Summary).
- `cv_fetch.py`: CLI CV parser that extracts structured JSON from a PDF.
- `gitTool.py`: GitHub repo fetcher (supports PAT for higher rate limits).
- `app.py` + `templates/`: Flask web UI with streaming SSE updates and an agent discussion view.
- `agent_main.py` + `src/`: LinkedIn scraping helpers using LinkdAPI (optional).

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

Quick CV parse only:
```
python cv_fetch.py path/to/cv.pdf
```

GitHub repo fetch:
```
python gitTool.py <github_username> --token <optional_pat> --stdout --pretty
```

LinkedIn agent (optional, CLI demo):
```
python agent_main.py
```

## Web app
Start the Flask server:
```
python app.py
```
- Upload a CV (PDF), LinkedIn URL, and GitHub username at `/`.
- The app streams agent events via `/stream` and shows a live discussion plus final summary.
- Uploaded PDFs go to `uploads/` (gitignored).

## How it works (pipeline)
1) CV parsed with pdfplumber → structured JSON.  
2) CVClaimsAgent turns structured CV data into atomic, verifiable claims.  
3) RepoVerificationAgent checks claims against GitHub repos.  
4) LinkedInVerificationAgent (if key provided) checks claims against LinkedIn profile data.  
5) ReliabilityScoringAgent combines evidence into a 0–100 score.  
6) SummaryAgent produces a user-facing report with highlights and the score.  
7) SSE events stream these steps to the UI; a transcript can be returned with `--verbose`.

## Notes
- Keep keys out of commits; `.env` is gitignored.
- LinkdAPI key can come from `.env` (`LINKD_API_KEY`/`LINKDAPI_API_KEY`) or `config.ini`.
- GitHub PAT recommended for heavier use to avoid rate limits.
