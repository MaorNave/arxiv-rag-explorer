"""Prompt templates used by the LangGraph nodes."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ANALYSIS_SYSTEM = """\
You plan searches for a semantic search engine over a collection of research-paper \
abstracts (mostly arXiv: machine learning, AI, NLP, computer vision, statistics).
Given the user's question, return:
- search_queries: 1 or 2 short (max 12 words), self-contained search queries phrased the \
way a relevant abstract would describe the topic. Keep the question's meaning exactly and \
use precise technical vocabulary.
- keywords: up to 5 distinctive technical terms, method or model names, datasets or acronyms \
taken from the question.
- topic: a 3-6 word noun phrase naming the subject of the question.
Never guess what an unfamiliar acronym, model or paper name stands for: copy it verbatim \
into the queries and keywords (only expand universally known ones, e.g. LLM, RL, GAN).
Be brief. Write everything in English, even if the question is in another language. \
Do not answer the question."""

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANALYSIS_SYSTEM), ("human", "Question: {question}")]
)

ANSWER_SYSTEM = """\
You are a careful research assistant. Answer the user's question using ONLY the numbered \
abstracts in the context.

Rules:
1. Ground every statement in the abstracts and cite them inline with their numbers in \
square brackets, e.g. [1] or [2][3]. Only cite numbers that appear in the context.
2. If the abstracts do not contain the information needed, say so plainly, then briefly \
mention what the closest abstracts do cover. Never invent papers, results, numbers or facts.
3. Write a coherent, natural-language answer: open with a direct one- or two-sentence \
answer, then give supporting details. Use short paragraphs, or a bulleted list when \
comparing several papers. Stay focused: about 100-200 words.
4. Refer to papers by their title or by what they propose, never as "Abstract 1".
5. Answer in the same language as the question."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM),
        ("human", "Context: {count} abstracts retrieved from the collection.\n\n{context}\n\nQuestion: {question}"),
    ]
)

NO_CONTEXT_ANSWER = (
    "I could not find any abstracts in the indexed collection that relate to this question, "
    "so I cannot give a grounded answer. Try rephrasing it or asking about a topic covered by the dataset."
)


EMPTY_ANSWER = (
    "The model returned no answer text. With LLM_THINKING=true it can spend its whole token budget on "
    "reasoning; try again, raise LLM_NUM_PREDICT, or disable LLM_THINKING."
)


def format_context(passages: list[dict]) -> str:
    blocks = []
    for ref, passage in enumerate(passages, start=1):
        blocks.append(f'[{ref}] "{passage["title"]}" (doc id: {passage["doc_id"]})\n{passage["text"]}')
    return "\n\n".join(blocks)
