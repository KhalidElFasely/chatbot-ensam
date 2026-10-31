"""Central configuration for the ENSAM Casablanca chatbot."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PAGES_FILE = DATA_DIR / "pages.jsonl"   # output of scraper.py
INDEX_DIR = DATA_DIR / "index"          # output of build_index.py

# --- Scraping ---------------------------------------------------------------
START_URL = os.getenv("START_URL", "https://www.ensam-casa.ma/")
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.5"))  # seconds between requests (be polite)
USER_AGENT = "ENSAM-student-chatbot/1.0 (school project)"

# Pages that add noise to the index (login forms, legal pages, duplicate archive views...)
SKIP_URL_PATTERNS = [
    r"/login",
    r"/register",
    r"/termes-et-conditions",
    r"/politique-de-confidentialite",
    r"[?&]mode=(image|texte)",    # same page rendered 3 ways
    r"/verification/nouveau",
    r"[?&]mois=",                 # monthly archive = same article list repeated
]

# --- Chunking / retrieval -----------------------------------------------------
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # French + English + Arabic
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
TOP_K = 8               # chunks sent to Claude
MIN_SCORE = 0.25        # below this cosine similarity we refuse to answer
MIN_KEYWORD_COVERAGE = 0.6  # ...unless a chunk contains 60% of the question's words

# --- Claude -------------------------------------------------------------------
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")   # writes the answer
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "claude-haiku-5-5")  # turns follow-ups into standalone questions
MAX_HISTORY_TURNS = 6   # previous user/assistant pairs kept for context

# --- LLM provider -------------------------------------------------------------
# "anthropic" (Claude, paid) or "openrouter" (free models, rate-limited)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Free models are often rate-limited upstream, so we keep a fallback list (tried in order).
# OPENROUTER_MODELS="a:free,b:free" overrides it; OPENROUTER_MODEL=x:free puts one model first.
OPENROUTER_MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free",
).split(",") if m.strip()]
if os.getenv("OPENROUTER_MODEL"):
    OPENROUTER_MODELS = [os.getenv("OPENROUTER_MODEL").strip()] + [
        m for m in OPENROUTER_MODELS if m != os.getenv("OPENROUTER_MODEL").strip()]
OPENROUTER_MODEL = OPENROUTER_MODELS[0]  # used for answers and rewrites
