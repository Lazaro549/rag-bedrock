"""
ingesta/pipeline.py
Downloads PDFs from S3, splits them into chunks, generates embeddings
with Titan, and indexes them in OpenSearch Serverless.
"""
import os
import uuid
import json
import boto3
from io import BytesIO

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

BUCKET         = os.environ["S3_BUCKET_NAME"]
REGION         = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT       = os.environ["OPENSEARCH_ENDPOINT"]
INDEX          = os.environ["OPENSEARCH_INDEX"]
EMBED_MODEL    = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
CHUNK_SIZE     = 800   # characters per chunk
CHUNK_OVERLAP  = 150   # overlap between chunks


# ── Clients ──────────────────────────────────────────────────────────────────

def get_opensearch() -> OpenSearch:
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

bedrock   = boto3.client("bedrock-runtime", region_name=REGION)
s3        = boto3.client("s3")
os_client = get_opensearch()


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Splits text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Returns a list of (page_number, text)."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return json.loads(response["body"].read())["embedding"]


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_chunk(chunk: str, source: str, page: int) -> None:
    doc = {
        "chunk_id":  str(uuid.uuid4()),
        "text":      chunk,
        "source":    source,
        "page":      page,
        "embedding": embed(chunk),
    }
    os_client.index(index=INDEX, body=doc)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run() -> None:
    paginator    = s3.get_paginator("list_objects_v2")
    pages        = paginator.paginate(Bucket=BUCKET, Prefix="docs/")
    total_chunks = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            print(f"\nProcessing: {key}")
            pdf_bytes  = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            pages_text = extract_text_from_pdf(pdf_bytes)

            for page_num, text in pages_text:
                if not text.strip():
                    continue
                chunks = chunk_text(text)
                for chunk in chunks:
                    index_chunk(chunk, source=key, page=page_num)
                    total_chunks += 1

            print(f"  ✓ {len(pages_text)} pages → {total_chunks} total chunks so far")

    print(f"\nDone. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    run()
