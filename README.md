# 🎓 Assistant ENSAM Casablanca

A chatbot that answers questions about ENSAM Casablanca (programs, admissions, departments, labs, news) **using only the school's website**, with clickable sources for every answer.

Built with Retrieval-Augmented Generation (RAG): hybrid search over the scraped site, then answers from **Claude Sonnet 5.5** through the Anthropic API.

## Architecture

```
ensam-casa.ma ──► scraper.py ──► data/pages.jsonl ──► build_index.py ──► FAISS (embeddings) + BM25 (keywords)
                                                                                   │
user question ──► follow-up? rewrite with Claude Haiku 5.5 ──► hybrid search ──► top 8 chunks
                                                                                   │
                                   relevance gate ── nothing relevant ──► "information not found" (no API call)
                                                                                   │
                                   Claude Sonnet 5.5 (streaming) ──► answer with [n] citations ──► Gradio UI
```

| Step | File | What it does |
|---|---|---|
| 1. Scraping | `scraper.py` | Crawls the site (respects `robots.txt`), normalizes URLs, skips noisy pages, removes repeated menus and footers, de-duplicates by content, extracts PDF text |
| 2. Indexing | `build_index.py` | 250-word chunks with 50-word overlap, each prefixed with its page title. Multilingual embeddings in FAISS |
| 3. RAG | `rag.py` | Hybrid search (embeddings + BM25 fused with Reciprocal Rank Fusion), relevance gate, follow-up rewriting, grounded answer with citations |
| 4. Interface | `app.py` | Gradio chat with streaming, examples and source links |
| 5. Evaluation | `evaluate.py` | Retrieval hit rate, citation accuracy, out-of-scope refusals, latency → `eval/results.md` |
| Demo | `notebooks/ensam_chatbot_demo.ipynb` | Step-by-step walkthrough for the presentation (Colab-ready) |
| Settings | `config.py` | All tunable parameters in one place |

## Key design choices

- **Claude API instead of a local model.** The first version ran Qwen 2.5 1.5B on Colab: slow (tens of seconds per answer) and prone to repeating itself. Claude streams an answer in a few seconds, handles French, English and Arabic well, and needs no GPU.
- **Two models.** Sonnet 5.5 writes the answers. Haiku 5.5, which is much cheaper and faster, only rewrites follow-up questions ("et les frais ?") into standalone search queries.
- **Hybrid search.** Embeddings find passages with the same meaning; BM25 finds exact words such as acronyms (EPEI, PPFI) that embeddings handle poorly.
- **Grounding.** Sources are passed as numbered `<source>` blocks; the system prompt forbids outside knowledge and requires `[n]` citations. Only the sources Claude actually cites are shown.
- **Relevance gate.** If no chunk is similar enough, the bot refuses without calling the API, which costs nothing and prevents made-up answers.
- **Safety fallback.** Requests enable the API's server-side `fallbacks: "default"`, so a rare safety-classifier decline is retried automatically on a fallback model.

## Set up on a new computer (Windows cmd)

The scraped pages and the index (`data/`) are in the repo, so no scraping or indexing is needed.

```bat
git clone <repo-url>
cd ensam-chatbot
py -m venv .venv
.venv\Scriptsctivate.bat
pip install -r requirements.txt
copy .env.example .env
rem open .env and put your API key in it
python app.py
```

To refresh the data later: `python scraper.py` then `python build_index.py`.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your key from https://console.anthropic.com in .env

python scraper.py           # ~5 min, writes data/pages.jsonl
python build_index.py       # ~1 min, writes data/index/
python rag.py "Quelles sont les filières proposées ?"   # quick test in the terminal
python app.py               # open http://localhost:7860
python evaluate.py          # writes eval/results.md
```

### Free option: OpenRouter models

No Anthropic credits? Use a free model through [OpenRouter](https://openrouter.ai) instead. Create a free key at https://openrouter.ai/keys, then in `.env`:

```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
# optional, comma-separated, tried in order:
# OPENROUTER_MODELS=nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free
```

Free models are shared and often rate-limited upstream (HTTP 429), so the bot sends a fallback list (OpenRouter's `models` parameter) and, on a 429, retries starting from the next model after 2 s. Only if every model is busy does it show the "modèle gratuit saturé" message. Any id ending in `:free` works (list: https://openrouter.ai/models?max_price=0). Free models come and go, so update `OPENROUTER_MODELS` if one disappears. `OPENROUTER_MODEL` still works and puts that model first.

Free models are rate-limited (about 20 requests per minute, and a daily cap of roughly 50 requests on an account with no credits, 1,000 after a one-time $10 top-up). When the limit is hit, the bot replies with a French "réessayez dans une minute" message instead of crashing. Answer quality and citation discipline are usually below Claude's, and `evaluate.py` can burn through the daily cap quickly. Set `LLM_PROVIDER=anthropic` (the default) to switch back to Claude.

On Colab: upload the `ensam-chatbot` folder (or the zip), add `ANTHROPIC_API_KEY` in 🔑 Secrets, and open `notebooks/ensam_chatbot_demo.ipynb`.

## Deploy (permanent public link)

1. Create a free **Gradio** Space on https://huggingface.co/spaces.
2. Upload the project files **including `data/`** (after running steps 1 and 2 locally, so the Space doesn't re-scrape).
3. In the Space settings, add a secret `ANTHROPIC_API_KEY`.
4. The Space runs `app.py` automatically.

Never commit `.env` or put your key in the code.

## Cost

Each answer sends about 3,000 input tokens of context and returns about 300 tokens. With Sonnet 5.5 at $2 / $10 per million tokens, that's about **$0.01 per question**, so a whole presentation demo costs a few cents. Set a monthly spending limit in the Anthropic console.

## Tuning

All in `config.py`:
- `MIN_SCORE` / `MIN_KEYWORD_COVERAGE`: raise them if the bot answers off-topic questions, lower them if it refuses valid ones.
- `TOP_K`, `CHUNK_WORDS`: more or larger chunks give more context but cost more per answer.
- `SKIP_URL_PATTERNS`: add any other noisy page patterns you notice in `data/pages.jsonl`.
- `eval/questions.jsonl`: add real questions students ask, with the page that should answer them, then run `python evaluate.py`.

## Limitations and next steps

- The knowledge is a snapshot: re-run `scraper.py` and `build_index.py` when the site changes (this could be a weekly scheduled job).
- Scanned PDFs (images) have no extractable text; they would need OCR.
- Possible extensions: a feedback 👍/👎 button, a WhatsApp or Telegram front end, Arabic/Darija evaluation questions.

---
*Student project. Not an official ENSAM Casablanca service.*
