"""
query/chain.py
Full RAG chain: retrieves context and generates an answer with Claude.
"""
import os
from dataclasses import dataclass

import boto3
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from query.retriever import retrieve, Chunk

load_dotenv()

REGION    = os.environ.get("AWS_REGION", "us-east-1")
LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")


PROMPT_TEMPLATE = """You are an expert assistant. Answer in the same language as the question.
Use ONLY the provided context. If the answer is not in the context, clearly state that
you could not find the information in the available documents.

At the end of your answer, list the sources used in this format:
📄 Sources: [filename, page N]

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


@dataclass
class RAGResponse:
    answer:   str
    sources:  list[Chunk]
    question: str


def build_llm() -> ChatBedrock:
    return ChatBedrock(
        model_id=LLM_MODEL,
        client=boto3.client("bedrock-runtime", region_name=REGION),
        model_kwargs={
            "max_tokens": 1024,
            "temperature": 0.2,
        },
    )


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        fname = chunk.source.split("/")[-1]
        parts.append(f"[{i}] {fname} (p.{chunk.page}):\n{chunk.text}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 5) -> RAGResponse:
    """
    Main entry point.
    Retrieves context and generates an answer with Claude.
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return RAGResponse(
            answer="No indexed documents found. Make sure to run the ingestion pipeline first.",
            sources=[],
            question=question,
        )

    # 2. Build prompt
    context = build_context(chunks)
    prompt  = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm     = build_llm()
    chain   = prompt | llm

    # 3. Generate answer
    response = chain.invoke({"context": context, "question": question})

    return RAGResponse(
        answer=response.content,
        sources=chunks,
        question=question,
    )


if __name__ == "__main__":
    result = ask("What are the system requirements?")
    print(result.answer)
