"""
AWS Step Functions state machine definitions for the SIRIUS pipeline.

Pipeline 1 — Extraction (build_definition):
  1. QueryCCIndex          — Athena query on CC index (Lambda)
  2. SplitCSV / DispatchSQS — Enqueue WARC records to SQS (Lambda)
  3. LaunchGPUInstances    — EC2 direct launch (g5.xlarge on-demand via Lambda)
  4. WaitForExtraction     — Poll SQS queue depth until drained (Lambda + Wait loop)

Pipeline 2 — Fast Pipeline + Analysis (build_analysis_definition):
  1. FastPipeline          — DuckDB pipeline: ingest → entities → relations → temporal →
                             dedup → regimes → situations → XLSX (Batch, <5 min, 8GB RAM)
  2. StrategicAnalysis     — Counterfactual/implicit/Faustian inference (Batch)
  3. ExportWorkbook        — Generate Excel workbook → S3 (Lambda)
  4. BuildSQLite           — Convert DuckDB → SQLite → S3 (Lambda)
  5. MinePatterns          — Sequential mining + V2 type construction (Batch)
"""

import json


def build_definition(config: dict) -> dict:
    """Build the ASL state machine definition.

    Args:
        config: Dict with ARNs/names for Lambda functions, Batch job definitions,
                SQS queue, S3 bucket, SNS topic, etc.

    Returns:
        ASL state machine definition as a dict.
    """
    return {
        "Comment": "SIRIUS Pipeline 1: Common Crawl → GPU Extraction",
        "StartAt": "QueryCCIndex",
        "States": {

            # ----------------------------------------------------------
            # Step 0: Query CommonCrawl index via Athena
            # ----------------------------------------------------------
            "QueryCCIndex": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_query_cc_index"],
                    "Payload": {
                        "crawl_min.$": "$.crawl_min",
                        "limit.$": "$.limit",
                        "setup_table.$": "$.setup_table",
                    },
                },
                "ResultSelector": {
                    "csv_s3_uri.$": "$.Payload.csv_s3_uri",
                    "record_count.$": "$.Payload.record_count",
                },
                "ResultPath": "$.step0",
                "Retry": [{
                    "ErrorEquals": ["States.TaskFailed", "Lambda.ServiceException"],
                    "IntervalSeconds": 30,
                    "MaxAttempts": 2,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineFailed",
                }],
                "Next": "SplitCSV",
            },

            # ----------------------------------------------------------
            # Step 1a: Split CSV into chunks + create SQS queue
            # ----------------------------------------------------------
            "SplitCSV": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_split_csv"],
                    "Payload": {
                        "csv_s3_uri.$": "$.step0.csv_s3_uri",
                        "record_count.$": "$.step0.record_count",
                        "max_records.$": "$.limit",
                    },
                },
                "ResultSelector": {
                    "queue_url.$": "$.Payload.queue_url",
                    "chunks.$": "$.Payload.chunks",
                    "total_records.$": "$.Payload.total_records",
                    "chunk_count.$": "$.Payload.chunk_count",
                },
                "ResultPath": "$.split",
                "TimeoutSeconds": 900,
                "Retry": [{
                    "ErrorEquals": ["States.TaskFailed"],
                    "IntervalSeconds": 10,
                    "MaxAttempts": 2,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineFailed",
                }],
                "Next": "DispatchChunks",
            },

            # ----------------------------------------------------------
            # Step 1b: Fan-out SQS dispatch over CSV chunks (Map)
            # ----------------------------------------------------------
            "DispatchChunks": {
                "Type": "Map",
                "MaxConcurrency": 10,
                "ItemsPath": "$.split.chunks",
                "ItemSelector": {
                    "csv_s3_uri.$": "$$.Map.Item.Value.csv_s3_uri",
                    "queue_url.$": "$.split.queue_url",
                },
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "INLINE"},
                    "StartAt": "DispatchChunk",
                    "States": {
                        "DispatchChunk": {
                            "Type": "Task",
                            "Resource": "arn:aws:states:::lambda:invoke",
                            "Parameters": {
                                "FunctionName": config["lambda_dispatch_sqs"],
                                "Payload.$": "$",
                            },
                            "ResultSelector": {
                                "messages_sent.$": "$.Payload.messages_sent",
                                "record_count.$": "$.Payload.record_count",
                            },
                            "Retry": [{
                                "ErrorEquals": [
                                    "Lambda.ServiceException",
                                    "Lambda.TooManyRequestsException",
                                    "States.TaskFailed",
                                ],
                                "IntervalSeconds": 10,
                                "MaxAttempts": 2,
                                "BackoffRate": 2.0,
                            }],
                            "End": True,
                        },
                    },
                },
                "ResultPath": "$.step1_dispatch_results",
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineFailed",
                }],
                "Next": "LaunchGPUInstances",
            },

            # ----------------------------------------------------------
            # Step 1b: Launch EC2 GPU instances for NuExtract extraction
            # ----------------------------------------------------------
            "LaunchGPUInstances": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_launch_gpu_instances"],
                    "Payload": {
                        "queue_url.$": "$.split.queue_url",
                        "gpu_instance_count.$": "$.gpu_instance_count",
                    },
                },
                "ResultSelector": {
                    "instance_ids.$": "$.Payload.instance_ids",
                    "launched_count.$": "$.Payload.launched_count",
                    "instance_type.$": "$.Payload.instance_type",
                },
                "ResultPath": "$.step1_gpu",
                "TimeoutSeconds": 600,
                "Retry": [{
                    "ErrorEquals": ["States.TaskFailed", "Lambda.ServiceException"],
                    "IntervalSeconds": 30,
                    "MaxAttempts": 2,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineFailed",
                }],
                "Next": "VerifyExtractionComplete",
            },

            # ----------------------------------------------------------
            # Step 1c: Verify SQS queue is fully drained after extraction
            # ----------------------------------------------------------
            "VerifyExtractionComplete": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_check_extraction"],
                    "Payload": {
                        "queue_url.$": "$.split.queue_url",
                    },
                },
                "ResultSelector": {
                    "remaining.$": "$.Payload.remaining",
                    "results_count.$": "$.Payload.results_count",
                    "complete.$": "$.Payload.complete",
                },
                "ResultPath": "$.step1_verify",
                "Next": "IsExtractionComplete",
            },

            "IsExtractionComplete": {
                "Type": "Choice",
                "Choices": [{
                    "Variable": "$.step1_verify.complete",
                    "BooleanEquals": True,
                    "Next": "PipelineComplete",
                }],
                "Default": "WaitForExtraction",
            },

            "WaitForExtraction": {
                "Type": "Wait",
                "Seconds": 300,  # Re-check every 5 minutes
                "Next": "VerifyExtractionComplete",
            },

            # ----------------------------------------------------------
            # Terminal states
            # ----------------------------------------------------------
            "PipelineComplete": {
                "Type": "Succeed",
            },

            "PipelineFailed": {
                "Type": "Fail",
                "Error": "ExtractionPipelineFailed",
                "Cause": "One or more extraction steps failed. Check execution history for details.",
            },
        },
    }


def build_analysis_definition(config: dict) -> dict:
    """Build the ASL definition for Pipeline 2: Fast Pipeline + Analysis.

    Starts from FastPipeline (DuckDB-based, replaces legacy LanceDB ingest)
    and runs through MinePatterns.  Assumes S3 contains NuExtract JSON
    results from Pipeline 1.

    Input:
        s3_bucket: S3 bucket for results
        lancedb_uri: S3 URI for StrategicAnalysis (downstream)
    """
    return {
        "Comment": "SIRIUS Pipeline 2: Fast DuckDB Pipeline → Strategic → Export → SQLite → Mine",
        "StartAt": "FastPipeline",
        "States": {

            # ----------------------------------------------------------
            # Step 2: Fast DuckDB Pipeline (replaces legacy LanceDB ingest)
            # Stages 0-7: Ingest → Entities → Relations → Temporal →
            #             Dedup → Regimes → Situations → XLSX Export
            # Runs in <5 min with 8GB RAM (vs 4 hrs / 60GB for LanceDB)
            # ----------------------------------------------------------
            "FastPipeline": {
                "Type": "Task",
                "Resource": "arn:aws:states:::batch:submitJob.sync",
                "Parameters": {
                    "JobName": "sirius-fast-pipeline",
                    "JobDefinition": config["batch_ingest_job_def"],
                    "JobQueue": config["batch_cpu_job_queue"],
                    "ContainerOverrides": {
                        "Environment": [
                            {"Name": "NUEXTRACT_S3_BUCKET", "Value": config["s3_bucket"]},
                            {"Name": "NUEXTRACT_S3_PREFIX", "Value": "nuextract-v2-events/"},
                            {"Name": "OUTPUT_S3_PREFIX", "Value": "pipeline-output/"},
                        ],
                    },
                },
                "ResultPath": "$.step2",
                "TimeoutSeconds": 3600,  # 60 min
                "Retry": [{
                    "ErrorEquals": ["Batch.AWSBatchException"],
                    "IntervalSeconds": 60,
                    "MaxAttempts": 1,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineFailed",
                }],
                "Next": "StrategicAnalysis",
            },

            # ----------------------------------------------------------
            # Step 3: Strategic Analysis
            # ----------------------------------------------------------
            "StrategicAnalysis": {
                "Type": "Task",
                "Resource": "arn:aws:states:::batch:submitJob.sync",
                "Parameters": {
                    "JobName": "sirius-strategic-analysis",
                    "JobDefinition": config["batch_strategic_analysis_job_def"],
                    "JobQueue": config["batch_cpu_job_queue"],
                    "ContainerOverrides": {
                        "Environment": [
                            {"Name": "LANCEDB_URI", "Value": config["lancedb_uri"]},
                            {"Name": "BEDROCK_MODEL_ID", "Value": config.get(
                                "bedrock_model_id",
                                "us.anthropic.claude-sonnet-4-20250514-v1:0",
                            )},
                            {"Name": "S3_BUCKET", "Value": config["s3_bucket"]},
                            {"Name": "S3_RESULTS_PREFIX", "Value": "strategic-analysis/"},
                            {"Name": "AWS_REGION", "Value": config.get("region", "us-east-1")},
                        ],
                    },
                },
                "ResultPath": "$.step3_strategic",
                "TimeoutSeconds": 7200,
                "Retry": [{
                    "ErrorEquals": ["Batch.AWSBatchException"],
                    "IntervalSeconds": 60,
                    "MaxAttempts": 1,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "ExportWorkbook",
                }],
                "Next": "ExportWorkbook",
            },

            # ----------------------------------------------------------
            # Step 4: Export Excel workbook to S3
            # ----------------------------------------------------------
            "ExportWorkbook": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_export_workbook"],
                    "Payload": {
                        "lancedb_uri": config["lancedb_uri"],
                        "s3_bucket": config["s3_bucket"],
                        "s3_key_prefix": "exports/",
                    },
                },
                "ResultSelector": {
                    "workbook_s3_uri.$": "$.Payload.workbook_s3_uri",
                },
                "ResultPath": "$.step4",
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "BuildSQLite",
                }],
                "Next": "BuildSQLite",
            },

            # ----------------------------------------------------------
            # Step 5: Build SQLite databases from LanceDB
            # ----------------------------------------------------------
            "BuildSQLite": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": config["lambda_build_sqlite"],
                    "Payload": {
                        "lancedb_uri": config["lancedb_uri"],
                        "s3_bucket": config["s3_bucket"],
                        "s3_key_prefix": "sqlite/",
                    },
                },
                "ResultSelector": {
                    "hypergraph_db_uri.$": "$.Payload.hypergraph_db_uri",
                    "entity_kb_uri.$": "$.Payload.entity_kb_uri",
                    "stats.$": "$.Payload.stats",
                },
                "ResultPath": "$.step5",
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineComplete",
                }],
                "Next": "MinePatterns",
            },

            # ----------------------------------------------------------
            # Step 6: Sequential mining + V2 type construction (Batch)
            # ----------------------------------------------------------
            "MinePatterns": {
                "Type": "Task",
                "Resource": "arn:aws:states:::batch:submitJob.sync",
                "Parameters": {
                    "JobDefinition": config["batch_mine_patterns_job_def"],
                    "JobName": "sirius-mine-patterns",
                    "JobQueue": config["batch_cpu_queue"],
                    "ContainerOverrides": {
                        "Environment": [
                            {"Name": "HYPERGRAPH_DB_URI", "Value.$": "$.step5.hypergraph_db_uri"},
                            {"Name": "S3_BUCKET", "Value": config["s3_bucket"]},
                            {"Name": "S3_KEY_PREFIX", "Value": "sqlite/"},
                            {"Name": "AWS_REGION", "Value": config.get("region", "us-east-1")},
                        ],
                    },
                },
                "ResultPath": "$.step6",
                "TimeoutSeconds": 3600,
                "Retry": [{
                    "ErrorEquals": ["States.TaskFailed"],
                    "IntervalSeconds": 60,
                    "MaxAttempts": 1,
                    "BackoffRate": 2.0,
                }],
                "Catch": [{
                    "ErrorEquals": ["States.ALL"],
                    "ResultPath": "$.error",
                    "Next": "PipelineComplete",
                }],
                "Next": "PipelineComplete",
            },

            # ----------------------------------------------------------
            # Terminal states
            # ----------------------------------------------------------
            "PipelineComplete": {
                "Type": "Succeed",
            },

            "PipelineFailed": {
                "Type": "Fail",
                "Error": "AnalysisPipelineFailed",
                "Cause": "One or more analysis steps failed. Check execution history.",
            },
        },
    }


def to_json(config: dict) -> str:
    """Return the state machine definition as formatted JSON."""
    return json.dumps(build_definition(config), indent=2)
