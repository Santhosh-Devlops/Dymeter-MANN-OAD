from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from researchApp.core.pipeline import ResearchPipeline


class GeneralizationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = ResearchPipeline(outputDirectory=Path(tempfile.gettempdir()) / "dymeterTestRuns")

    def assertAnomalyShape(self, result):
        payload = result.anomalyResult.toDict()
        self.assertIn("totalRows", payload)
        self.assertIn("totalAnomalies", payload)
        self.assertIn("anomalyRatio", payload)
        self.assertIn("thresholdUsed", payload)
        self.assertIn("anomalies", payload)
        self.assertIn("conceptDriftEvents", payload)
        for anomaly in payload["anomalies"]:
            self.assertIn("rowIndex", anomaly)
            self.assertIn("originalRowData", anomaly)
            self.assertIn("anomalyScore", anomaly)
            self.assertIn("reconstructionError", anomaly)
            self.assertIn("conceptUncertainty", anomaly)
            self.assertIn("detectorUsed", anomaly)
            self.assertIn("severityLevel", anomaly)
            self.assertIn("topContributingFeatures", anomaly)
            self.assertIn("naturalLanguageExplanation", anomaly)

    def testPaperBenchmarkCsvFiles(self):
        datasetNames = [
            "ambient_temperature_system_failure.csv",
            "cpu_utilization_asg_misconfiguration.csv",
            "machine_temperature_system_failure.csv",
        ]
        for datasetName in datasetNames:
            result = self.pipeline.runFile(Path("datasets") / datasetName, "DyMETER", {"forceNumpy": True})
            self.assertAnomalyShape(result)

    def testSyntheticMixedDatasets(self):
        with tempfile.TemporaryDirectory() as folder:
            folderPath = Path(folder)
            firstPath = folderPath / "synthetic_mixed_one.csv"
            secondPath = folderPath / "synthetic_mixed_two.csv"
            self.writeSynthetic(firstPath, rows=180, columns=4, includeCategory=True)
            self.writeSynthetic(secondPath, rows=220, columns=9, includeCategory=True)

            for datasetPath in [firstPath, secondPath]:
                result = self.pipeline.runFile(datasetPath, "DyMETER", {"forceNumpy": True})
                self.assertAnomalyShape(result)

    def testMannOnlyAblationIsSelectable(self):
        with tempfile.TemporaryDirectory() as folder:
            datasetPath = Path(folder) / "synthetic_mann_only.csv"
            self.writeSynthetic(datasetPath, rows=160, columns=5, includeCategory=False)

            result = self.pipeline.runFile(datasetPath, "EnhancedDymeterMANN", {"forceNumpy": True})

            self.assertEqual(result.modelName, "EnhancedDymeterMANN")
            self.assertAnomalyShape(result)
            self.assertIn("memoryEventCount", result.metrics)
            self.assertEqual(result.metrics["trainingBudget"]["maxEpochs"], 2000)
            memoryCoverage = [
                item for item in result.metrics["trainingBudget"]["architectureCoverage"]
                if item["name"] == "Long-term concept memory"
            ][0]
            self.assertEqual(memoryCoverage["status"], "implemented")

    def testMalformedDatasetStillProducesShape(self):
        with tempfile.TemporaryDirectory() as folder:
            datasetPath = Path(folder) / "malformed.csv"
            frame = pd.DataFrame(
                {
                    "id": [f"row-{index}" for index in range(90)],
                    "constant": ["same"] * 90,
                    "state": ["ok", "warn", None, "ok", "bad"] * 18,
                    "device": [f"device-{index % 30}" for index in range(90)],
                    "label": [0] * 82 + [1] * 8,
                }
            )
            frame.to_csv(datasetPath, index=False)
            result = self.pipeline.runFile(datasetPath, "DyMETER", {"forceNumpy": True})
            self.assertAnomalyShape(result)

    def writeSynthetic(self, datasetPath: Path, rows: int, columns: int, includeCategory: bool):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, size=(rows, columns))
        labels = np.zeros(rows, dtype=int)
        anomalyIndexes = random.choice(rows, size=max(6, rows // 15), replace=False)
        data[anomalyIndexes] += random.normal(5, 0.5, size=(len(anomalyIndexes), columns))
        labels[anomalyIndexes] = 1
        frame = pd.DataFrame(data, columns=[f"sensor{index}" for index in range(columns)])
        if includeCategory:
            frame["machineType"] = ["A" if index % 3 == 0 else "B" if index % 3 == 1 else "C" for index in range(rows)]
        frame["label"] = labels
        frame.to_csv(datasetPath, index=False)


if __name__ == "__main__":
    unittest.main()
