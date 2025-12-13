# epimindshack

Simple tools to parse CV PDFs with OpenAI and a Flask endpoint to accept uploads.

## Setup
- Python 3.12+ recommended.
- Create a `.env` with `OPENAI_API_KEY=<your_key>`.
- Install deps: `pip install -r requirements.txt`.

## CLI CV parser
- Run `python cv_fetch.py` to extract structured JSON from `testcv.pdf` (or point to another path).

## Flask upload endpoint
- Start with `python app.py` and POST a PDF + optional LinkedIn/GitHub fields to `/`.
- Uploaded files are saved under `uploads/` (ignored by git).

