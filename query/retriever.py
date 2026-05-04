"""
query/retriever.py
Searches for the most relevant chunks in OpenSearch for a given question.
"""
import os
import json
import boto3
from dataclasses import dataclass

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()

REGION      = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT    = os.environ["OPENSEARCH_ENDPOINT"]
INDEX       = os.environ["OPENSEARCH_INDEX"]
EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
TOP_K       = 5  # chunks to retrieve


@dataclass
class Chunk:
    text:   str
    source: str
    page:   int
    score:  float


def _get_opensearch() -> OpenSearch:
    creds = boto3.Session().get_credentials()
    auth  = AWS4Auth(creds.access_key, creds.secret_key, REGION, "aoss",
                     session_token=creds.token)
    host  = ENDPOINT.replace("https://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _embed(text: str) -> list[float]:
    bedrock  = boto3.client("bedrock-runtime", region_name=REGION)
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return json.loads(response["body"].read())["embedding"]


def retrieve(question: str, top_k: int = TOP_K) -> list[Chunk]:
    """Returns the chunks most semantically similar to the question."""
    client    = _get_opensearch()
    embedding = _embed(question)

    query = {
        "size": top_k,
        "_source": ["text", "source", "page"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": top_k,
                }
            }
        },
    }

    response = client.search(index=INDEX, body=query)
    hits     = response["hits"]["hits"]

    return [
        Chunk(
            text=h["_source"]["text"],
            source=h["_source"]["source"],
            page=h["_source"]["page"],
            score=h["_score"],
        )
        for h in hits
    ]


if __name__ == "__main__":
    question = "What are the system requirements?"
    chunks   = retrieve(question)
    for c in chunks:
        print(f"\n[{c.source} — p.{c.page}] score={c.score:.3f}")
        print(c.text[:200])
