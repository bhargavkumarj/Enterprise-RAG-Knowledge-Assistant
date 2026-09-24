"""Chat UI that shows the retrieved context alongside every answer."""

import gradio as gr

from rag import assistant, store
from rag.config import settings

EXAMPLES = [
    "Who won the IIOTY award in 2023?",
    "Which customers are signed up to Markellm and what do they pay?",
    "Summarise Avery Lancaster's career progression at the company",
    "What does the Rellm contract say about termination?",
]


def context_panel(chunks, rewritten=None) -> str:
    if not chunks:
        return "*No context retrieved.*"
    lines = []
    if rewritten:
        lines.append(f"**Search query:** `{rewritten}`\n")
    for position, chunk in enumerate(chunks, start=1):
        body = chunk.text.strip()
        if len(body) > 900:
            body = body[:900] + " …"
        lines.append(
            f"**{position}. {chunk.source}** · `{chunk.doc_type}` · score {chunk.score:.3f}\n\n"
            f"> {body.replace(chr(10), chr(10) + '> ')}"
        )
    return "\n\n---\n\n".join(lines)


def respond(message: str, history: list[dict]):
    history = history + [{"role": "user", "content": message}]
    chunks, rewritten = [], None
    for partial, retrieved in assistant.stream(message, history=history[:-1]):
        chunks = retrieved
        yield history + [{"role": "assistant", "content": partial}], context_panel(chunks, rewritten)


def build() -> gr.Blocks:
    try:
        info = store.stats()
        status = (
            f"{info['vectors']:,} chunks · {info['chunk_strategy']} chunking · "
            f"{settings.embedding_backend} embeddings · top-{settings.top_k}"
            f"{' · re-ranked' if settings.rerank else ''}"
        )
    except Exception as error:
        status = f"⚠️ {error}"

    with gr.Blocks(title=f"{settings.organisation} Knowledge Assistant", fill_height=True) as ui:
        gr.Markdown(f"## {settings.organisation} Knowledge Assistant\n{status}")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=520, show_copy_button=True)
                box = gr.Textbox(placeholder="Ask about products, contracts or people…", scale=1)
                gr.Examples(EXAMPLES, inputs=box)
                clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=2):
                context = gr.Markdown("*Retrieved context appears here.*", height=620)

        box.submit(respond, [box, chatbot], [chatbot, context]).then(lambda: "", None, box)
        clear.click(lambda: ([], "*Retrieved context appears here.*"), None, [chatbot, context])
    return ui


if __name__ == "__main__":
    build().launch(inbrowser=True)
