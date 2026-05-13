"""
WARC fetching and HTML-to-text conversion for NuExtract pipeline.

- Fetches WARC records from Common Crawl S3
- Converts HTML to plain text via BeautifulSoup
- Returns full untruncated text (translation handled by NuExtract via few-shot)
"""

import gzip

import boto3
from bs4 import BeautifulSoup

S3_CLIENT = boto3.client("s3")

TEXT_MIN_LENGTH = 100


def fetch_warc_record(warc_filename, offset, length):
    """Fetch a WARC record from Common Crawl S3 and extract text + title.

    Args:
        warc_filename: S3 key in the commoncrawl bucket
        offset: Byte offset into the WARC file
        length: Byte length of the record

    Returns:
        (text, title) tuple. text is the extracted plain text,
        title is the HTML <title> content or None.
    """
    response = S3_CLIENT.get_object(
        Bucket="commoncrawl",
        Key=warc_filename,
        Range=f"bytes={offset}-{offset + length - 1}",
    )

    compressed = response["Body"].read()
    decompressed = gzip.decompress(compressed)
    content = decompressed.decode("utf-8", errors="ignore")

    # Parse WARC record: WARC headers, HTTP headers, content
    parts = content.split("\r\n\r\n", 2)
    html = parts[2] if len(parts) >= 3 else parts[-1]

    # Extract text from HTML
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text, title


def fetch_and_normalize(record):
    """Fetch a WARC record and apply length validation.

    Returns the full untruncated article text. Non-English translation is
    handled by NuExtract itself via few-shot examples in the prompt.

    Args:
        record: dict with warc_filename, warc_record_offset, warc_record_length, url

    Returns:
        (text, title, error) tuple. If error is not None, text and title are None.
    """
    try:
        text, title = fetch_warc_record(
            record["warc_filename"],
            record["warc_record_offset"],
            record["warc_record_length"],
        )
    except Exception as e:
        return None, None, f"fetch_error:{str(e)[:100]}"

    if not text or len(text) < TEXT_MIN_LENGTH:
        return None, None, "too_short"

    return text, title, None
