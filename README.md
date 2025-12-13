# epimindshack

Simple tools to parse CV PDFs with OpenAI and a Flask endpoint to accept uploads.

## Setup
- Python 3.12+ recommended.
- Create a `.env` with `OPENAI_API_KEY=<your_key>`.
- Install deps: `pip install -r requirements.txt`.

## CLI CV parser
- Run `python cv_fetch.py` to extract structured JSON from `testcv.pdf` (or point to another path).

## Three-agent claim verification
`agent_workflow.py` orchestrates three OpenAI-powered agents:
- CVClaimsAgent: extracts verifiable claims from the parsed CV.
- RepoVerificationAgent: cross-checks claims against GitHub repositories.
- LinkedInVerificationAgent: cross-checks claims against LinkedIn profile data (requires `LINKD_API_KEY`/`LINKDAPI_API_KEY` or `--linkedin-api-key`).
- ReliabilityScoringAgent: assigns a 0-100 reliability score with rationale.
- SummaryAgent: produces a user-facing report with highlights and the final score.

Run end-to-end from CLI:
```
python agent_workflow.py path/to/cv.pdf <github_username> --linkedin https://linkedin.com/in/example --github-token <optional_token> --linkedin-api-key <optional_linkd_key>
```
- Add `--verbose` to return an agent transcript of the reasoning steps.

## Flask upload endpoint
- Start with `python app.py` and POST a PDF + LinkedIn/GitHub fields.
- The form now streams live events from `/stream` so you can watch the agent discussion in real time (check "Verbose agent transcript" for full dialogue).
- Optional GitHub token field helps avoid rate limits; LinkedIn API key is read from `config.ini` or env (`LINKD_API_KEY`/`LINKDAPI_API_KEY`).
- Uploaded files are saved under `uploads/` (ignored by git).
