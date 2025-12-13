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
- ReliabilityScoringAgent: assigns a 0-100 reliability score with rationale.

Run end-to-end from CLI:
```
python agent_workflow.py path/to/cv.pdf <github_username> --linkedin https://linkedin.com/in/example --github-token <optional_token>
```
- Add `--verbose` to return an agent transcript of the reasoning steps.

## Flask upload endpoint
- Start with `python app.py` and POST a PDF + LinkedIn/GitHub fields to `/`.
- The endpoint runs the three-agent pipeline and returns JSON (enable the "Verbose agent transcript" checkbox to see the discussion).
- Uploaded files are saved under `uploads/` (ignored by git).
