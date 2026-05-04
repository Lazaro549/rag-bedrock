"""
infra/setup.py
Creates the vector index in OpenSearch Serverless.
Run once before indexing documents.
"""
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
INDEX    = os.environ["OPENSEARCH_INDEX"]
REGION   = os.environ.get("AWS_REGION", "us-east-1")
DIMS     = 1024  # Titan Embeddings v2 produces 1024 dimensions


def get_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",
        session_token=credentials.token,
    )
    host = ENDPOINT.replace("https://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def create_index(client: OpenSearch) -> None:
    if client.indices.exists(INDEX):
        print(f"Index '{INDEX}' already exists.")
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIMS,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "text":     {"type": "text"},
                "source":   {"type": "keyword"},
                "page":     {"type": "integer"},
                "chunk_id": {"type": "keyword"},
            }
        },
    }

    client.indices.create(INDEX, body=body)
    print(f"Index '{INDEX}' created with {DIMS} dimensions.")


if __name__ == "__main__":
    client = get_client()
    create_index(client)
