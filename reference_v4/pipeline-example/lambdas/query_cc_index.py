"""Lambda handler: Query CommonCrawl index via Athena.

Runs the Athena query on the CC Parquet index and caches results to S3.
Returns the S3 URI of the CSV and the record count.
"""

import csv
import io
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
ATHENA_DATABASE = "ccindex"
S3_BUCKET = os.environ.get("S3_BUCKET", "lambdamap-gliner2-jobs")
ATHENA_OUTPUT_PREFIX = "athena-results/scs-grayzone"
S3_CACHE_KEY = "index/step1_results.csv"
POLL_INTERVAL = 5

athena = boto3.client("athena", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# ---------------------------------------------------------------------------
# Target domains + path filters — ported from step0/query.sql
# Domain-only matching returns 11M records; path filters narrow to ~672K
# ---------------------------------------------------------------------------

QUERY_TEMPLATE = """
SELECT
    url,
    warc_filename,
    warc_record_offset,
    warc_record_length,
    fetch_time AS crawl_date
FROM ccindex.ccindex
WHERE crawl >= '{crawl_min}'
  AND subset = 'warc'
  AND fetch_status = 200
  AND (content_mime_type = 'text/html' OR content_mime_detected = 'text/html')
  AND url_host_registered_domain IN (
    -- Australian Media
    'abc.net.au', 'sbs.com.au', 'smh.com.au', 'theage.com.au', 'afr.com',
    'theaustralian.com.au', 'news.com.au', 'theguardian.com', '9news.com.au', '7news.com.au',
    -- Chinese State Media
    'globaltimes.cn', 'chinadaily.com.cn', 'news.cn', 'cgtn.com', 'people.cn',
    -- US Media
    'nytimes.com', 'washingtonpost.com', 'cnn.com', 'reuters.com', 'apnews.com',
    'bloomberg.com', 'wsj.com', 'foxnews.com', 'thehill.com',
    -- UK Media
    'bbc.com', 'telegraph.co.uk', 'independent.co.uk', 'ft.com', 'economist.com',
    -- Asian Pacific
    'straitstimes.com', 'channelnewsasia.com', 'scmp.com', 'japantimes.co.jp', 'asahi.com',
    'thehindu.com', 'hindustantimes.com', 'thejakartapost.com', 'bangkokpost.com',
    'nzherald.co.nz', 'koreatimes.co.kr', 'koreaherald.com', 'taipeitimes.com',
    'focustaiwan.tw', 'inquirer.net',
    -- Philippine Sources
    'rappler.com', 'manilatimes.net', 'philstar.com', 'pna.gov.ph',
    -- Defense & Security
    'defensenews.com', 'navalnews.com', 'janes.com', 'maritimeexecutive.com', 'navyrecognition.com',
    'usni.org', 'c4isrnet.com', 'breakingdefense.com', 'defensepriorities.org',
    -- Think Tanks
    'aspistrategist.org.au', 'lowyinstitute.org', 'thediplomat.com', 'foreignpolicy.com',
    'foreignaffairs.com', 'cfr.org', 'csis.org', 'rand.org', 'brookings.edu',
    'heritage.org', 'aei.org', 'eastasiaforum.org', 'chathamhouse.org', 'iiss.org',
    'carnegieendowment.org', 'stimson.org', 'wilsoncenter.org',
    'iseas.edu.sg', 'rsis.edu.sg', 'idss.edu.sg',
    -- International
    'aljazeera.com', 'dw.com', 'france24.com',
    -- Magazines
    'time.com', 'newsweek.com', 'theatlantic.com', 'nationalinterest.org',
    -- Academic
    'theconversation.com', 'warontherocks.com'
  )
  AND (
    -- Australian Media paths
    (url_host_registered_domain = 'abc.net.au' AND (url_path LIKE '/news/asia%' OR url_path LIKE '/news/world%' OR url_path LIKE '/news/australia%'))
    OR (url_host_registered_domain = 'sbs.com.au' AND (url_path LIKE '/news/asia%' OR url_path LIKE '/news/world%'))
    OR (url_host_registered_domain = 'smh.com.au' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/world/north-america%' OR url_path LIKE '/politics/federal%'))
    OR (url_host_registered_domain = 'theage.com.au' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/politics/federal%'))
    OR (url_host_registered_domain = 'afr.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/politics/%'))
    OR (url_host_registered_domain = 'theaustralian.com.au' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/nation/defence%'))
    OR (url_host_registered_domain = 'news.com.au' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/world/pacific%'))
    OR (url_host_registered_domain = 'theguardian.com' AND (url_path LIKE '/world/asia-pacific%' OR url_path LIKE '/world/china%' OR url_path LIKE '/australia-news/australian-military%'))
    OR (url_host_registered_domain = '9news.com.au' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/national/defence%'))
    OR (url_host_registered_domain = '7news.com.au' AND (url_path LIKE '/news/world%' OR url_path LIKE '/news/politics%'))
    -- Chinese State Media paths
    OR (url_host_registered_domain = 'globaltimes.cn' AND url_path LIKE '/page/20%')
    OR (url_host_registered_domain = 'chinadaily.com.cn' AND url_path LIKE '/a/20%')
    OR (url_host_registered_domain = 'news.cn' AND url_host_name = 'english.news.cn' AND url_path LIKE '/20%')
    OR (url_host_registered_domain = 'cgtn.com' AND url_path LIKE '/news/20%')
    OR (url_host_registered_domain = 'people.cn' AND url_host_name = 'en.people.cn' AND url_path LIKE '/n3/20%')
    -- US Media paths
    OR (url_host_registered_domain = 'nytimes.com' AND (url_path LIKE '/20%/world/asia%' OR url_path LIKE '/20%/world/australia%' OR url_path LIKE '/section/world/asia%'))
    OR (url_host_registered_domain = 'washingtonpost.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/national-security%'))
    OR (url_host_registered_domain = 'cnn.com' AND (url_path LIKE '/asia/%' OR url_path LIKE '/world/asia%'))
    OR (url_host_registered_domain = 'reuters.com' AND (url_path LIKE '/world/asia-pacific%' OR url_path LIKE '/world/china%'))
    OR (url_host_registered_domain = 'apnews.com' AND (url_path LIKE '/article/%' OR url_path LIKE '/hub/asia-pacific%' OR url_path LIKE '/hub/china%' OR url_path LIKE '/hub/taiwan%'))
    OR (url_host_registered_domain = 'bloomberg.com' AND (url_path LIKE '/news/asia%' OR url_path LIKE '/asia/%'))
    OR (url_host_registered_domain = 'wsj.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/articles/china%' OR url_path LIKE '/articles/taiwan%'))
    OR (url_host_registered_domain = 'foxnews.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/politics/defense%'))
    OR (url_host_registered_domain = 'thehill.com' AND (url_path LIKE '/policy/defense%' OR url_path LIKE '/policy/international%'))
    -- UK Media paths
    OR (url_host_registered_domain = 'bbc.com' AND (url_path LIKE '/news/world-asia%' OR url_path LIKE '/news/world-australia%' OR url_path LIKE '/news/world-us-canada%'))
    OR (url_host_registered_domain = 'telegraph.co.uk' AND (url_path LIKE '/news/world/asia%' OR url_path LIKE '/news/world/china%'))
    OR (url_host_registered_domain = 'independent.co.uk' AND (url_path LIKE '/news/world/asia%' OR url_path LIKE '/news/world/australasia%'))
    OR (url_host_registered_domain = 'ft.com' AND (url_path LIKE '/world/asia-pacific%' OR url_path LIKE '/content/%'))
    OR (url_host_registered_domain = 'economist.com' AND (url_path LIKE '/asia/%' OR url_path LIKE '/china/%' OR url_path LIKE '/international/%'))
    -- Asian Pacific paths
    OR (url_host_registered_domain = 'straitstimes.com' AND (url_path LIKE '/asia/se-asia%' OR url_path LIKE '/asia/east-asia%' OR url_path LIKE '/asia/australianz%' OR url_path LIKE '/singapore/politics%'))
    OR (url_host_registered_domain = 'channelnewsasia.com' AND (url_path LIKE '/asia/east-asia%' OR url_path LIKE '/asia/south-east-asia%' OR url_path LIKE '/world/%'))
    OR (url_host_registered_domain = 'scmp.com' AND (url_path LIKE '/news/asia%' OR url_path LIKE '/news/china%' OR url_path LIKE '/news/hong-kong%'))
    OR (url_host_registered_domain = 'japantimes.co.jp' AND (url_path LIKE '/news/asia-pacific%' OR url_path LIKE '/news/world%'))
    OR (url_host_registered_domain = 'asahi.com' AND (url_path LIKE '/ajw/asia%' OR url_path LIKE '/ajw/articles%'))
    OR (url_host_registered_domain = 'thehindu.com' AND url_path LIKE '/news/international/%')
    OR (url_host_registered_domain = 'hindustantimes.com' AND url_path LIKE '/world-news/%')
    OR (url_host_registered_domain = 'thejakartapost.com' AND (url_path LIKE '/news/asia%' OR url_path LIKE '/news/world%'))
    OR (url_host_registered_domain = 'bangkokpost.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/world/pacific%'))
    OR (url_host_registered_domain = 'nzherald.co.nz' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/nz/politics%'))
    OR (url_host_registered_domain = 'koreatimes.co.kr' AND (url_path LIKE '/southkorea/politics%' OR url_path LIKE '/southkorea/society%' OR url_path LIKE '/foreignaffairs/%'))
    OR (url_host_registered_domain = 'koreaherald.com' AND url_path LIKE '/view.php%')
    OR (url_host_registered_domain = 'taipeitimes.com' AND (url_path LIKE '/News/front%' OR url_path LIKE '/News/taiwan%' OR url_path LIKE '/News/world%'))
    OR (url_host_registered_domain = 'focustaiwan.tw' AND (url_path LIKE '/news/politics%' OR url_path LIKE '/news/cross-strait%'))
    OR (url_host_registered_domain = 'inquirer.net' AND (url_path LIKE '/global-nation%' OR url_path LIKE '/newsinfo%'))
    -- Philippine Sources
    OR (url_host_registered_domain = 'rappler.com' AND (url_path LIKE '/world/%' OR url_path LIKE '/nation/%' OR url_path LIKE '/newsbreak/%'))
    OR (url_host_registered_domain = 'manilatimes.net' AND (url_path LIKE '/news/national%' OR url_path LIKE '/news/world%'))
    OR (url_host_registered_domain = 'philstar.com' AND (url_path LIKE '/headlines/%' OR url_path LIKE '/world/%'))
    OR (url_host_registered_domain = 'pna.gov.ph' AND url_path LIKE '/articles/%')
    -- Defense & Security paths
    OR (url_host_registered_domain = 'defensenews.com' AND (url_path LIKE '/naval/%' OR url_path LIKE '/asia-pacific/%' OR url_path LIKE '/global/asia%'))
    OR (url_host_registered_domain = 'navalnews.com' AND (url_path LIKE '/naval-news/%' OR url_path LIKE '/category/%'))
    OR (url_host_registered_domain = 'janes.com' AND url_path LIKE '/defence-news/%')
    OR (url_host_registered_domain = 'maritimeexecutive.com' AND url_path LIKE '/article/%')
    OR (url_host_registered_domain = 'navyrecognition.com' AND url_path LIKE '/naval-news/%')
    OR (url_host_registered_domain = 'usni.org' AND (url_path LIKE '/magazines/proceedings%' OR url_path LIKE '/magazines/usni-news%' OR url_path LIKE '/press/%'))
    OR (url_host_registered_domain = 'c4isrnet.com' AND (url_path LIKE '/naval/%' OR url_path LIKE '/asia-pacific/%'))
    OR (url_host_registered_domain = 'breakingdefense.com' AND url_path LIKE '/20%')
    OR (url_host_registered_domain = 'defensepriorities.org' AND (url_path LIKE '/explainers/%' OR url_path LIKE '/reports/%'))
    -- Think Tanks paths
    OR (url_host_registered_domain = 'aspistrategist.org.au' AND (url_path LIKE '/defence-%' OR url_path LIKE '/security-%' OR url_path LIKE '/china-%' OR url_path LIKE '/indo-pacific%'))
    OR (url_host_registered_domain = 'lowyinstitute.org' AND url_path LIKE '/the-interpreter/%')
    OR (url_host_registered_domain = 'thediplomat.com' AND (url_path LIKE '/category/security%' OR url_path LIKE '/category/politics%' OR url_path LIKE '/regions/east-asia%'))
    OR (url_host_registered_domain = 'foreignpolicy.com' AND (url_path LIKE '/region/asia%' OR url_path LIKE '/tag/china%' OR url_path LIKE '/tag/south-china-sea%'))
    OR (url_host_registered_domain = 'foreignaffairs.com' AND (url_path LIKE '/regions/asia%' OR url_path LIKE '/topics/security%'))
    OR (url_host_registered_domain = 'cfr.org' AND (url_path LIKE '/asia/%' OR url_path LIKE '/china/%' OR url_path LIKE '/backgrounder/%'))
    OR (url_host_registered_domain = 'csis.org' AND (url_path LIKE '/analysis/%' OR url_path LIKE '/regions/asia%'))
    OR (url_host_registered_domain = 'rand.org' AND (url_path LIKE '/topics/china%' OR url_path LIKE '/topics/asia%' OR url_path LIKE '/pubs/%'))
    OR (url_host_registered_domain = 'brookings.edu' AND (url_path LIKE '/research/china%' OR url_path LIKE '/research/asia%' OR url_path LIKE '/articles/%'))
    OR (url_host_registered_domain = 'heritage.org' AND (url_path LIKE '/asia/%' OR url_path LIKE '/defense/%'))
    OR (url_host_registered_domain = 'aei.org' AND url_path LIKE '/foreign-and-defense-policy/asia%')
    OR (url_host_registered_domain = 'eastasiaforum.org' AND (url_path LIKE '/category/security%' OR url_path LIKE '/category/politics%'))
    OR (url_host_registered_domain = 'chathamhouse.org' AND (url_path LIKE '/research/asia-pacific%' OR url_path LIKE '/2024/%' OR url_path LIKE '/2025/%' OR url_path LIKE '/2026/%'))
    OR (url_host_registered_domain = 'iiss.org' AND (url_path LIKE '/blogs/military-balance%' OR url_path LIKE '/publications/%'))
    OR (url_host_registered_domain = 'csis.org' AND url_host_name = 'amti.csis.org')
    OR (url_host_registered_domain = 'carnegieendowment.org' AND (url_path LIKE '/research/%' OR url_path LIKE '/publications/%'))
    OR (url_host_registered_domain = 'stimson.org' AND (url_path LIKE '/20%' OR url_path LIKE '/research/%'))
    OR (url_host_registered_domain = 'wilsoncenter.org' AND (url_path LIKE '/article/%' OR url_path LIKE '/blog-post/%' OR url_path LIKE '/publication/%'))
    -- Regional Think Tanks (ASEAN, Singapore, Japan)
    OR (url_host_registered_domain = 'iseas.edu.sg' AND (url_path LIKE '/articles-commentaries/%' OR url_path LIKE '/media/%'))
    OR (url_host_registered_domain = 'rsis.edu.sg' AND (url_path LIKE '/rsis-publication/%' OR url_path LIKE '/wp-content/%'))
    OR (url_host_registered_domain = 'idss.edu.sg' AND url_path LIKE '/%')
    -- International paths
    OR (url_host_registered_domain = 'aljazeera.com' AND (url_path LIKE '/news/asia-pacific%' OR url_path LIKE '/news/asia%' OR url_path LIKE '/features/%'))
    OR (url_host_registered_domain = 'dw.com' AND (url_path LIKE '/en/asia%' OR url_path LIKE '/en/top-stories%'))
    OR (url_host_registered_domain = 'france24.com' AND (url_path LIKE '/en/asia-pacific%' OR url_path LIKE '/en/tag/china%'))
    -- Magazines paths
    OR (url_host_registered_domain = 'time.com' AND (url_path LIKE '/section/world/%' OR url_path LIKE '/tag/china%' OR url_path LIKE '/tag/asia%'))
    OR (url_host_registered_domain = 'newsweek.com' AND (url_path LIKE '/world/asia%' OR url_path LIKE '/topic/china%' OR url_path LIKE '/topic/military%'))
    OR (url_host_registered_domain = 'theatlantic.com' AND (url_path LIKE '/international/%' OR url_path LIKE '/china/%'))
    OR (url_host_registered_domain = 'nationalinterest.org' AND (url_path LIKE '/blog/buzz%' OR url_path LIKE '/blog/the-reboot%' OR url_path LIKE '/feature/%'))
    -- Academic paths
    OR (url_host_registered_domain = 'theconversation.com' AND (url_path LIKE '/au/topics/asia%' OR url_path LIKE '/global/topics/china%' OR url_path LIKE '/au/topics/defence%'))
    OR (url_host_registered_domain = 'warontherocks.com' AND url_path LIKE '/20%')
  )
ORDER BY crawl_date DESC
"""


def _run_query(query: str) -> str:
    """Execute Athena query and return execution ID."""
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={
            "OutputLocation": f"s3://{S3_BUCKET}/{ATHENA_OUTPUT_PREFIX}/"
        },
    )
    execution_id = response["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(QueryExecutionId=execution_id)
        state = result["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return execution_id
        elif state in ("FAILED", "CANCELLED"):
            reason = result["QueryExecution"]["Status"].get(
                "StateChangeReason", "Unknown"
            )
            raise Exception(f"Athena query {state}: {reason}")
        time.sleep(POLL_INTERVAL)


def _setup_table():
    """Create the ccindex Athena table if needed."""
    _run_query("CREATE DATABASE IF NOT EXISTS ccindex")
    create_sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS ccindex.ccindex (
        url_surtkey STRING, url STRING, url_host_name STRING,
        url_host_tld STRING, url_host_2nd_last_part STRING,
        url_host_3rd_last_part STRING, url_host_4th_last_part STRING,
        url_host_5th_last_part STRING, url_host_registry_suffix STRING,
        url_host_registered_domain STRING, url_host_private_suffix STRING,
        url_host_private_domain STRING, url_protocol STRING, url_port INT,
        url_path STRING, url_query STRING, fetch_time TIMESTAMP,
        fetch_status SMALLINT, fetch_redirect STRING, content_digest STRING,
        content_mime_type STRING, content_mime_detected STRING,
        content_charset STRING, content_languages STRING,
        content_truncated STRING, warc_filename STRING,
        warc_record_offset INT, warc_record_length INT, warc_segment STRING
    )
    PARTITIONED BY (crawl STRING, subset STRING)
    STORED AS parquet
    LOCATION 's3://commoncrawl/cc-index/table/cc-main/warc/'
    """
    _run_query(create_sql)
    _run_query("MSCK REPAIR TABLE ccindex.ccindex")


def handler(event, context):
    """Lambda entry point.

    Input:
        crawl_min: Minimum CC crawl version (default: CC-MAIN-2024-42)
        limit: Max records to return (None = all)
        setup_table: Whether to create the Athena table first

    Output:
        csv_s3_uri: S3 URI of the results CSV
        record_count: Number of WARC records found
    """
    crawl_min = event.get("crawl_min") or "CC-MAIN-2024-42"
    limit = event.get("limit")
    setup = event.get("setup_table", False)

    if setup:
        _setup_table()

    query = QUERY_TEMPLATE.format(crawl_min=crawl_min)
    if limit:
        query += f"\nLIMIT {int(limit)}"

    execution_id = _run_query(query)

    # Count records by streaming (avoid loading entire CSV into memory)
    result_key = f"{ATHENA_OUTPUT_PREFIX}/{execution_id}.csv"
    response = s3.get_object(Bucket=S3_BUCKET, Key=result_key)
    record_count = -1  # Subtract header row
    for _ in response["Body"].iter_lines():
        record_count += 1
    record_count = max(record_count, 0)

    # Cache to well-known location
    s3.copy_object(
        CopySource={"Bucket": S3_BUCKET, "Key": result_key},
        Bucket=S3_BUCKET,
        Key=S3_CACHE_KEY,
    )

    csv_s3_uri = f"s3://{S3_BUCKET}/{S3_CACHE_KEY}"
    print(f"Query complete: {record_count} records -> {csv_s3_uri}")

    return {
        "csv_s3_uri": csv_s3_uri,
        "record_count": record_count,
    }
