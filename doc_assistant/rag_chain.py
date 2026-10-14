"""
LangChain part: LLM orchestration and response generation on top of the LlamaIndex retriever.

Concepts used (from class):
- ChatPromptTemplate / PromptTemplate                         (Day 10)
- LCEL chains:  prompt | llm | parser                         (Day 10, Day 20 sequential chain)
- RunnableParallel / RunnablePassthrough / RunnableLambda     (Day 20 - 3_rag_v1.py)
- PydanticOutputParser + StrOutputParser                      (Day 10 output parsers)
- Chat history with HumanMessage / AIMessage                  (Day 10 chatbot memory)
"""
from typing import List, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from .config import strip_think
from .indexer import get_document_nodes, retrieve


# ------------------------------------------------------------------ Output schema
class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer in Markdown, using ONLY the context. Cite sources inline like [1], [2].")
    sources_used: List[int] = Field(description="Numbers of the context chunks actually used, e.g. [1, 3]")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = clearly answered by context, medium = partially, low = not found in context")
    follow_up_questions: List[str] = Field(description="2 short follow-up questions the user could ask next")


def format_nodes(nodes) -> str:
    """Turn LlamaIndex nodes into a numbered context block with citations."""
    parts = []
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata
        page = f", page {md['page_label']}" if md.get("page_label") else ""
        parts.append(f"[{i}] (source: {md.get('file_name', 'unknown')}{page})\n{n.node.get_content().strip()}")
    return "\n\n".join(parts) if parts else "NO CONTEXT FOUND"


def nodes_to_sources(nodes) -> list:
    return [{
        "n": i,
        "file_name": n.node.metadata.get("file_name", "unknown"),
        "page": n.node.metadata.get("page_label", ""),
        "score": round(float(n.score or 0), 3),
        "text": n.node.get_content().strip(),
    } for i, n in enumerate(nodes, 1)]


class DocumentAssistant:
    """LangChain orchestration around a LlamaIndex index."""

    def __init__(self, index, llm, top_k: int = 4, similarity_cutoff: float = 0.25):
        self.index, self.llm = index, llm
        self.top_k, self.similarity_cutoff = top_k, similarity_cutoff
        self.chat_history = []                # list of HumanMessage / AIMessage
        self._build_chains()

    # -------------------------------------------------------------- chains
    def _build_chains(self):
        to_text = StrOutputParser() | RunnableLambda(strip_think)

        # 1) Condense a follow-up question into a standalone question (conversation memory)
        self.condense_chain = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's latest question as a standalone question using the chat history. "
                       "Return ONLY the rewritten question. If it is already standalone, return it unchanged."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]) | self.llm | to_text

        # 2) Grounded QA with structured output (PydanticOutputParser)
        self.answer_parser = PydanticOutputParser(pydantic_object=RAGAnswer)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an intelligent document assistant. Answer ONLY from the context below. "
             "If the answer is not in the context, say \"I could not find this in the documents.\" and set "
             "confidence to low. Cite chunk numbers like [1]. Be clear and concise.\n\n"
             "Context:\n{context}\n\n{format_instructions}"),
            ("human", "{question}"),
        ]).partial(format_instructions=self.answer_parser.get_format_instructions())

        retriever_runnable = RunnableLambda(lambda x: self._retrieve(x["question"], x.get("file_names")))
        # RunnableParallel = retrieve + pass the question through, same pattern as 3_rag_v1.py
        self.qa_chain = (
            RunnableParallel({
                "nodes": retriever_runnable,
                "question": RunnableLambda(lambda x: x["question"]),
            })
            | RunnableParallel({
                "nodes": RunnableLambda(lambda x: x["nodes"]),
                "answer": (RunnablePassthrough.assign(context=lambda x: format_nodes(x["nodes"]))
                           | qa_prompt | self.llm | to_text),
            })
        )

        # Plain-text fallback prompt (used if the model's JSON cannot be parsed)
        self.plain_qa_chain = ChatPromptTemplate.from_messages([
            ("system", "Answer ONLY from the context. Cite chunk numbers like [1]. If not found, say so.\n\n"
                       "Context:\n{context}"),
            ("human", "{question}"),
        ]) | self.llm | to_text

        # 3) Summarisation - map step and reduce step
        self.map_chain = ChatPromptTemplate.from_template(
            "Summarise the following part of the document '{file_name}' in 4-6 bullet points. "
            "Keep key facts, numbers and definitions.\n\n{text}"
        ) | self.llm | to_text
        self.reduce_chain = ChatPromptTemplate.from_template(
            "You are given partial summaries of the document '{file_name}'. Write a final summary in Markdown with:\n"
            "### Overview (2-3 sentences)\n### Key Points (5-8 bullets)\n### Conclusion (1-2 sentences)\n\n"
            "Partial summaries:\n{summaries}"
        ) | self.llm | to_text

        # 4) Multi-document comparison
        self.compare_chain = ChatPromptTemplate.from_messages([
            ("system", "You compare multiple documents. Use ONLY the context, which is grouped by document. "
                       "Answer the question for each document, then give a short comparison table in Markdown "
                       "and a conclusion. Cite as [file name]."),
            ("human", "Question: {question}\n\nContext grouped by document:\n{context}"),
        ]) | self.llm | to_text

    # -------------------------------------------------------------- helpers
    def _retrieve(self, question, file_names=None):
        return retrieve(self.index, question, self.top_k, file_names, self.similarity_cutoff)

    def standalone_question(self, question: str) -> str:
        if not self.chat_history:
            return question
        try:
            q = self.condense_chain.invoke({"history": self.chat_history[-6:], "question": question})
            return q.strip().strip('"') or question
        except Exception:
            return question

    # -------------------------------------------------------------- public API
    def ask(self, question: str, file_names: list | None = None, use_history: bool = True) -> dict:
        """Answer a question with citations. Returns a dict with answer, sources, confidence, ..."""
        standalone = self.standalone_question(question) if use_history else question
        out = self.qa_chain.invoke({"question": standalone, "file_names": file_names})
        nodes, raw = out["nodes"], out["answer"]
        try:
            parsed = self.answer_parser.parse(raw).model_dump()
        except Exception:
            # fallback: the model did not return valid JSON -> plain answer
            text = raw if raw and not raw.lstrip().startswith("{") else self.plain_qa_chain.invoke(
                {"question": standalone, "context": format_nodes(nodes)})
            parsed = {"answer": text, "sources_used": list(range(1, len(nodes) + 1)),
                      "confidence": "medium", "follow_up_questions": []}

        self.chat_history += [HumanMessage(content=question), AIMessage(content=parsed["answer"])]
        parsed.update({"question": question, "standalone_question": standalone, "sources": nodes_to_sources(nodes)})
        return parsed

    def summarize(self, file_name: str, max_parts: int = 4, chars_per_part: int = 5000) -> str:
        """Map-reduce summary of one document (limited number of parts to respect free-tier token limits)."""
        nodes = get_document_nodes(self.index, file_name)
        if not nodes:
            return f"Document '{file_name}' not found in the index."
        text = "\n".join(n.get_content() for n in nodes)
        parts = [text[i:i + chars_per_part] for i in range(0, len(text), chars_per_part)]
        if len(parts) > max_parts:                        # spread the sample across the whole document
            step = len(parts) / max_parts
            parts = [parts[int(i * step)] for i in range(max_parts)]
        partial = [self.map_chain.invoke({"file_name": file_name, "text": p}) for p in parts]
        if len(partial) == 1:
            return partial[0]
        return self.reduce_chain.invoke({"file_name": file_name, "summaries": "\n\n".join(partial)})

    def compare(self, question: str, file_names: list, k_per_doc: int = 3) -> dict:
        """Multi-document querying: retrieve separately from each document, then compare."""
        groups, sources = [], []
        for name in file_names:
            nodes = retrieve(self.index, question, k_per_doc, [name], self.similarity_cutoff)
            groups.append(f"=== {name} ===\n" + "\n---\n".join(n.node.get_content().strip() for n in nodes))
            sources += nodes_to_sources(nodes)
        answer = self.compare_chain.invoke({"question": question, "context": "\n\n".join(groups)})
        return {"answer": answer, "sources": sources}

    def reset_memory(self):
        self.chat_history = []
