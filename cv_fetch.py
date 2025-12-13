import json
import os

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n\n"
    return text

def parse_cv(pdf_path: str) -> dict:
    raw_text = extract_text(pdf_path)

    prompt = f"""
Extract structured resume information from the text below.
Return valid JSON with the following keys:

- name: string or null
- contact: {{"email":string|null,"phone":string|null}} or null
- education: list of strings
- work_experience: list of {{
    role: string,
    company: string,
    start_date: string|null,
    end_date: string|null,
    description: string
}}
- projects: list of {{
    title: string,
    description: string
}}
- skills: list of strings

Only output valid JSON.

CV TEXT:
---
{raw_text[:15000]}
---
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    content = res.choices[0].message.content

    # content may be a plain string or a list of text parts depending on SDK version
    if isinstance(content, list):
        content = "".join(getattr(part, "text", str(part)) for part in content)

    if not content:
        raise ValueError("OpenAI response content was empty.")

    return json.loads(content)


if __name__ == "__main__":
    data = parse_cv("testcv.pdf")
    print(json.dumps(data, indent=2))
