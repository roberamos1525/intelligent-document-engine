"""Prompt templates for chat-style LLM backends."""

from __future__ import annotations

from idengine.llm.base import LLMRequest, Task

ANSWER_SYSTEM = (
    "You answer questions using only the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [1] or [2, 3]. "
    "If the passages do not contain the answer, reply exactly: "
    '"I could not find the answer in the provided documents." '
    "Be concise and do not invent facts."
)

SUMMARY_SYSTEM = (
    "You write faithful, concise summaries of the provided text. "
    "Keep names, dates and amounts exact. Do not add information that is not in the text."
)


def render_context(request: LLMRequest) -> str:
    return "\n\n".join(f"[{p.number}] ({p.source})\n{p.text}" for p in request.passages)


def render_messages(request: LLMRequest) -> list[dict[str, str]]:
    context = render_context(request)
    if request.task is Task.ANSWER:
        user = f"Context passages:\n\n{context}\n\nQuestion: {request.instruction}\nAnswer:"
        return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
    goal = request.instruction or "Summarize the text."
    user = f"{goal}\nUse at most {request.max_sentences} sentences.\n\nText:\n\n" + "\n\n".join(
        p.text for p in request.passages
    )
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]


NOT_FOUND_ANSWER = "I could not find the answer in the provided documents."
