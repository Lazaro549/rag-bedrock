"""
ingesta/upload.py
Uploads PDFs to S3.
Usage: python ingesta/upload.py --path ./docs/
"""
import os
import argparse
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ["S3_BUCKET_NAME"]


def upload_dir(local_path: str) -> None:
    s3 = boto3.client("s3")
    path = Path(local_path)
    pdfs = list(path.rglob("*.pdf"))

    if not pdfs:
        print("No PDF files found.")
        return

    for pdf in pdfs:
        key = f"docs/{pdf.name}"
        s3.upload_file(str(pdf), BUCKET, key)
        print(f"  ✓ {pdf.name} → s3://{BUCKET}/{key}")

    print(f"\nUploaded {len(pdfs)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder containing PDFs")
    args = parser.parse_args()
    upload_dir(args.path)
