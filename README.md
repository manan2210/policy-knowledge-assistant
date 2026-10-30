# Policy Knowledge Assistant

An evidence-first policy search proof of concept demonstrating retrieval, cited excerpts, and optional LLM summarization. Uses **fictional example policies**, not real company guidance. No API key is needed for local search.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Local Markdown policy files are split into sections.
2. TF-IDF word and bigram features rank excerpts by cosine similarity.
3. The interface displays citations and abstains if evidence is weak.
4. Optional LLM synthesis requires an `OPENAI_API_KEY` environment variable; it sends the question and retrieved fictional excerpts to the API, which may incur usage charges.

Not a production RAG service: there is no authentication, permissions, data freshness monitoring, validated evaluation benchmark, or guarantee that generated summaries are correct. Do not enter confidential records.
