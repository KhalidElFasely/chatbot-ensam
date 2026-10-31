"""Step 3 - Retrieval-Augmented Generation with Claude.

    question ─┬─> (follow-up?) Claude Haiku rewrites it as a standalone question
              ├─> hybrid search: embeddings (meaning) + BM25 (exact words, acronyms)
              ├─> relevance gate: nothing relevant -> polite refusal, no API call
              └─> Claude Sonnet answers ONLY from the numbered sources, citing [n]

Quick test from the terminal:  python rag.py "Quelles sont les filières proposées ?"
"""
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field

import anthropic
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config

NO_ANSWER = ("Je n'ai pas trouvé cette information sur le site de l'ENSAM Casablanca. "
             "Vous pouvez contacter directement l'administration de l'école.")

RATE_LIMITED = ("Le modèle gratuit est momentanément saturé (limite de requêtes OpenRouter atteinte). "
                "Merci de réessayer dans une minute.")
API_ERROR = "Le service de génération de réponses est indisponible pour le moment. Merci de réessayer plus tard."

SYSTEM_PROMPT = """Tu es l'assistant virtuel de l'ENSAM Casablanca (École Nationale Supérieure d'Arts et Métiers), un projet étudiant non officiel.
Tu réponds aux questions des étudiants, des candidats et des visiteurs sur l'école.

Règles :
- Réponds UNIQUEMENT à partir des sources fournies dans <sources>. N'utilise jamais tes connaissances générales sur l'école, même si tu penses les connaître.
- Après chaque information, cite la source entre crochets, par exemple [2]. Ne cite que des numéros de sources qui existent.
- Si les sources ne contiennent pas la réponse, dis-le clairement en une phrase et propose de contacter l'administration. N'invente rien (dates, chiffres, noms, frais).
- Si la question n'a aucun rapport avec l'ENSAM, explique poliment que tu ne réponds qu'aux questions sur l'école.
- Réponds dans la langue de la question (français, anglais ou darija/arabe).
- Sois concis et structuré : listes à puces pour les énumérations, pas de répétitions."""

REWRITE_PROMPT = """Rewrite the user's last message as a single standalone search question, \
using the conversation to resolve pronouns and implicit references (e.g. "et les frais ?" -> \
"Quels sont les frais d'inscription de la filière Génie Industriel à l'ENSAM ?"). \
Keep the user's language. Output only the rewritten question, nothing else."""

STOPWORDS = set("""le la les un une des du de d l et ou a au aux en dans sur pour par avec
est sont quel quelle quels quelles qui que quoi comment ou est-ce ce cette ces mon ma mes
je tu il elle nous vous ils elles se sa son ses leur leurs y ne pas plus the of and to in is
are what which how who for on with at an it do does""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, drop stop words - used for BM25 keyword search."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Hit:
    chunk: dict
    dense: float      # cosine similarity
    keyword: float    # BM25 score
    coverage: float   # share of the query's words found in the chunk
    score: float = 0  # fused rank score


class Retriever:
    def __init__(self, index_dir=config.INDEX_DIR):
        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.embedder = SentenceTransformer(data["embedding_model"])
        tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokens)
        self.chunk_terms = [set(t) for t in tokens]

    def search(self, query: str, k=config.TOP_K, pool=30) -> list[Hit]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        dense_scores, dense_ids = self.index.search(q, min(pool, len(self.chunks)))
        dense = {int(i): float(s) for s, i in zip(dense_scores[0], dense_ids[0]) if i != -1}

        kw_scores = self.bm25.get_scores(tokenize(query))
        kw_ids = np.argsort(kw_scores)[::-1][:pool]
        keyword = {int(i): float(kw_scores[i]) for i in kw_ids if kw_scores[i] > 0}

        # Reciprocal Rank Fusion: robust way to merge two rankings with different score scales
        fused: dict[int, float] = {}
        for ranking in (sorted(dense, key=dense.get, reverse=True),
                        sorted(keyword, key=keyword.get, reverse=True)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        q_terms = set(tokenize(query))
        hits = []
        for i in top:
            coverage = len(q_terms & self.chunk_terms[i]) / len(q_terms) if q_terms else 0.0
            hits.append(Hit(self.chunks[i], dense.get(i, 0.0), keyword.get(i, 0.0), coverage, fused[i]))
        return hits

    @staticmethod
    def is_relevant(hits: list[Hit]) -> bool:
        """Gate: at least one chunk is semantically close, or contains most of the query's words."""
        return any(h.dense >= config.MIN_SCORE or h.coverage >= config.MIN_KEYWORD_COVERAGE for h in hits)


@dataclass
class Answer:
    text: str = ""
    sources: list[dict] = field(default_factory=list)  # only the sources Claude actually cited
    search_query: str = ""
    latency_s: float = 0.0


class SchoolAssistant:
    def __init__(self, retriever: Retriever | None = None, client=None):
        self.retriever = retriever or Retriever()
        self.provider = config.LLM_PROVIDER
        if self.provider == "openrouter":
            from openai import OpenAI  # OpenRouter speaks the OpenAI API
            self.client = client or OpenAI(base_url=config.OPENROUTER_BASE_URL,
                                           api_key=os.getenv("OPENROUTER_API_KEY"))
        else:
            self.client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _openrouter_create(self, **kwargs):
        """chat.completions.create with model fallback: OpenRouter's `models` list handles
        provider errors server-side, and on a 429 we also retry starting from the next model."""
        import openai
        models = config.OPENROUTER_MODELS
        for i in range(len(models)):
            order = models[i:] + models[:i]
            try:
                return self.client.chat.completions.create(
                    model=order[0], extra_body={"models": order[:3]}, **kwargs)
            except openai.RateLimitError:
                if i == len(models) - 1:
                    raise
                time.sleep(2)

    def rewrite(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up ("et pour le master ?") into a standalone search query."""
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-4:])
        if self.provider == "openrouter":
            import openai
            try:
                resp = self._openrouter_create(
                    max_tokens=200,
                    messages=[{"role": "system", "content": REWRITE_PROMPT},
                              {"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\nLast message: {question}"}],
                )
            except openai.APIError:  # rate limit or outage: search with the raw question instead
                return question
            text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            text = lines[-1].strip('"') if lines else ""
            if not text or len(text) > 300:  # reasoning models sometimes "think out loud" here
                last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                return f"{last_user} {question}".strip()
            return text
        resp = self.client.messages.create(
            model=config.REWRITE_MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\nLast message: {question}"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or question

    @staticmethod
    def format_sources(hits: list[Hit]) -> str:
        return "\n".join(
            f'<source id="{n}" title="{h.chunk["title"]}" url="{h.chunk["url"]}">\n{h.chunk["text"]}\n</source>'
            for n, h in enumerate(hits, 1)
        )

    def stream(self, question: str, history: list[dict] | None = None):
        """Yields the growing answer text, then returns the final Answer (generator return value)."""
        t0 = time.time()
        history = [m for m in (history or []) if m["role"] in ("user", "assistant")]
        history = history[-2 * config.MAX_HISTORY_TURNS:]

        answer = Answer(search_query=self.rewrite(question, history))
        hits = self.retriever.search(answer.search_query)
        if not self.retriever.is_relevant(hits):
            answer.text = NO_ANSWER
            answer.latency_s = time.time() - t0
            yield answer.text
            return answer

        user_turn = (f"<sources>\n{self.format_sources(hits)}\n</sources>\n\n"
                     f"Question : {question}")
        messages = history + [{"role": "user", "content": user_turn}]

        if self.provider == "openrouter":
            import openai
            try:
                stream = self._openrouter_create(
                    max_tokens=2000,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        answer.text += delta
                        yield answer.text
            except openai.RateLimitError:  # free models: ~20 req/min and a daily cap
                answer.text = RATE_LIMITED
            except openai.APIError:
                answer.text = API_ERROR
            if answer.text in (RATE_LIMITED, API_ERROR, ""):
                answer.text = answer.text or API_ERROR
                answer.latency_s = time.time() - t0
                yield answer.text
                return answer
        else:
            with self.client.beta.messages.stream(
                model=config.ANSWER_MODEL,
                max_tokens=16000,
                output_config={"effort": "low"},  # chat Q&A: fast answers, quality holds at low effort
                system=SYSTEM_PROMPT,
                messages=messages,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
            ) as stream:
                for delta in stream.text_stream:
                    answer.text += delta
                    yield answer.text
                final = stream.get_final_message()

            if final.stop_reason == "refusal":
                answer.text = "Désolé, je ne peux pas répondre à cette question."
                yield answer.text

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer.text) if 0 < int(n) <= len(hits)})
        seen = set()
        for n in cited:
            c = hits[n - 1].chunk
            if c["url"] not in seen:
                seen.add(c["url"])
                answer.sources.append({"n": n, "title": c["title"], "url": c["url"]})
        answer.latency_s = time.time() - t0
        return answer

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        gen = self.stream(question, history)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every Unicode char
    q = " ".join(sys.argv[1:]) or "Quelles sont les filières proposées ?"
    bot = SchoolAssistant()
    gen = bot.stream(q)
    printed = 0
    try:
        while True:
            text = next(gen)
            print(text[printed:], end="", flush=True)
            printed = len(text)
    except StopIteration as stop:
        ans = stop.value
    print("\n\nSources:")
    for s in ans.sources:
        print(f"  [{s['n']}] {s['title']} - {s['url']}")
    print(f"({ans.latency_s:.1f}s)")
