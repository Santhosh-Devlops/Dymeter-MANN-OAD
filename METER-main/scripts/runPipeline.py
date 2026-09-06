from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from researchApp.core.pipeline import ResearchPipeline


def main():
    parser = argparse.ArgumentParser(description="Run a listed anomaly detection model on a dataset file.")
    parser.add_argument("datasetPath")
    parser.add_argument("--model", default="DyMETER", choices=["DyMETER", "METER", "D3R", "SARAD", "EnhancedDymeterMANN"])
    parser.add_argument("--force-numpy", action="store_true")
    parser.add_argument("--full", action="store_true", help="Print the full anomaly result JSON.")
    args = parser.parse_args()

    pipeline = ResearchPipeline()
    bundle = pipeline.runFile(args.datasetPath, args.model, {"forceNumpy": args.force_numpy})
    result = bundle.anomalyResult.toDict()
    if args.full:
        print(json.dumps(result, indent=2, default=str))
    else:
        summaryMetrics = dict(bundle.metrics)
        summaryMetrics.pop("memoryEvents", None)
        print(
            json.dumps(
                {
                    "runId": bundle.runId,
                    "datasetName": bundle.datasetName,
                    "modelName": bundle.modelName,
                    "totalRows": result["totalRows"],
                    "totalAnomalies": result["totalAnomalies"],
                    "anomalyRatio": result["anomalyRatio"],
                    "thresholdUsed": result["thresholdUsed"],
                    "topAnomalies": result["anomalies"][:5],
                    "metrics": summaryMetrics,
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
