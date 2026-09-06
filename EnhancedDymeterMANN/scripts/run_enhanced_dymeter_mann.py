from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researchApp.core.pipeline import ResearchPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EnhancedDymeterMANN on a dataset file.")
    parser.add_argument("datasetPath")
    parser.add_argument("--force-numpy", action="store_true")
    parser.add_argument("--full", action="store_true", help="Print the full anomaly result JSON.")
    args = parser.parse_args()

    pipeline = ResearchPipeline()
    bundle = pipeline.runFile(args.datasetPath, "EnhancedDymeterMANN", {"forceNumpy": args.force_numpy})
    result = bundle.anomalyResult.toDict()
    payload = result if args.full else {
        "runId": bundle.runId,
        "datasetName": bundle.datasetName,
        "modelName": bundle.modelName,
        "totalRows": result["totalRows"],
        "totalAnomalies": result["totalAnomalies"],
        "anomalyRatio": result["anomalyRatio"],
        "thresholdUsed": result["thresholdUsed"],
        "topAnomalies": result["anomalies"][:5],
        "metrics": {key: value for key, value in bundle.metrics.items() if key != "memoryEvents"},
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
