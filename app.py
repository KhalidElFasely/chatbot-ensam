"""Step 4 - Web interface (Gradio). Run:  python app.py  then open http://localhost:7860"""
import gradio as gr

from rag import SchoolAssistant

bot = SchoolAssistant()
SOURCES_MARK = "\n\n---\n**Sources**"


def message_text(content) -> str:
    """Gradio stores message content as a string or a list of parts; keep only the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def clean_history(history):
    msgs = []
    for m in history:
        text = message_text(m["content"]).split(SOURCES_MARK)[0].strip()
        if text:
            msgs.append({"role": m["role"], "content": text})
    return msgs


def respond(message, history):
    gen = bot.stream(message, clean_history(history))
    try:
        while True:
            yield next(gen)
    except StopIteration as stop:
        answer = stop.value
    if answer.sources:
        lines = "\n".join(f"{s['n']}. [{s['title']}]({s['url']})" for s in answer.sources)
        yield f"{answer.text}{SOURCES_MARK}\n{lines}"
    else:
        yield answer.text


demo = gr.ChatInterface(
    fn=respond,
    title="🎓 Assistant ENSAM Casablanca",
    description=("Posez vos questions sur l'école : filières, admissions, départements, laboratoires, actualités. "
                 "Les réponses proviennent uniquement du site officiel de l'ENSAM Casablanca, avec leurs sources."),
    examples=[
        "Quelles sont les filières d'ingénieur proposées ?",
        "Comment se passe l'admission en première année ?",
        "Quels sont les départements de l'école ?",
        "What research labs does ENSAM Casablanca have?",
    ],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
