from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from researchApp.core.contracts import AnomalyItem, AnomalyResult, DriftEvent, RunBundle, TrainingLog
from researchApp.core.artifacts import buildRunArtifacts
from researchApp.core.explain import computeContributions, explainAnomaly, frameRowToDict, severityFromScore
from researchApp.core.metrics import binaryMetrics
from researchApp.core.paperResults import comparisonRows
from researchApp.core.preprocessing import AdaptivePreprocessor
from researchApp.core.profiler import profileDataset
from researchApp.models.detectors import DyMeterRunner


class ResearchPipeline:
    """Coordinates profiling, preprocessing, training, inference, and result packaging."""

    def __init__(self, outputDirectory: str | Path = "runs"):
        self.outputDirectory = Path(outputDirectory)
        self.outputDirectory.mkdir(exist_ok=True)

    def runFile(self, datasetPath: str | Path, modelName: str = "DyMETER", config: dict | None = None) -> RunBundle:
        datasetPath = Path(datasetPath)
        modelName = self.normalizeModelName(modelName)
        self.validateModelName(modelName)
        runId = str(uuid.uuid4())[:8]
        dataFrame, profile, profileLogs = profileDataset(datasetPath)

        preprocessor = AdaptivePreprocessor()
        preparedData = preprocessor.prepare(dataFrame, profile)

        logs = [
            TrainingLog("Uploading dataset", "The dataset file was received by the backend.", datasetPath.name),
            TrainingLog("Dataset loaded successfully", "Rows and columns are ready for checking.", f"{profile.rowCount} rows."),
        ]
        logs.extend(profileLogs)
        logs.extend(preparedData.logs)

        runner = DyMeterRunner(modelName, config or {})
        detectorOutput, modelLogs = runner.run(
            preparedData.featureMatrix,
            preparedData.labels,
            preparedData.preprocessingSummary,
        )
        logs.extend(modelLogs)

        anomalyResult = self.buildAnomalyResult(preparedData, detectorOutput)
        predictions = np.array([1 if detectorOutput.scores[index] > detectorOutput.thresholds[index] else 0 for index in range(len(detectorOutput.scores))])
        metrics = binaryMetrics(preparedData.labels, detectorOutput.scores, predictions)
        metrics.update(
            {
                "trainingTime": round(float(detectorOutput.trainingSeconds), 4),
                "inferenceTime": round(float(detectorOutput.inferenceSeconds), 4),
                "parameterCount": detectorOutput.parameterCount,
                "memoryEventCount": len(detectorOutput.memoryEvents),
                "memoryEvents": detectorOutput.memoryEvents[:100],
                "trainingBudget": detectorOutput.trainingBudget,
            }
        )

        localRows = [
            {
                "dataset": datasetPath.stem,
                "model": modelName,
                "aucRoc": metrics.get("aucRoc"),
                "aucPr": metrics.get("aucPr"),
                "source": "Locally Computed",
                "table": "Current run",
            }
        ]
        comparison = comparisonRows(datasetPath.stem, localRows)
        artifacts = buildRunArtifacts(
            self.outputDirectory,
            runId,
            datasetPath.stem,
            modelName,
            profile,
            preparedData.preprocessingSummary,
            metrics,
            anomalyResult,
            detectorOutput,
            comparison,
        )
        logs.append(
            TrainingLog(
                "Creating downloadable files",
                "Clear images, a report, and a zipped image folder were saved for this run.",
                f"Folder: {self.outputDirectory / 'artifacts' / runId / 'images'}",
            )
        )

        qaContext = self.makeQaContext(datasetPath.stem, profile, preparedData.preprocessingSummary, metrics, anomalyResult, comparison)
        bundle = RunBundle(
            runId=runId,
            modelName=modelName,
            datasetName=datasetPath.stem,
            profile=profile,
            preprocessing=preparedData.preprocessingSummary,
            metrics=metrics,
            anomalyResult=anomalyResult,
            pipelineLogs=logs,
            qaContext=qaContext,
            comparison=comparison,
            artifacts=artifacts,
        )
        self.saveRun(bundle)
        return bundle

    def normalizeModelName(self, modelName: str) -> str:
        normalized = modelName.strip().replace(" ", "")
        if normalized.lower() in {"enhanceddymetermann", "mann", "dymetermann"}:
            return "EnhancedDymeterMANN"
        if normalized.lower() == "dymeter":
            return "DyMETER"
        if normalized.lower() == "meter":
            return "METER"
        if normalized.lower() == "d3r":
            return "D3R"
        if normalized.lower() == "sarad":
            return "SARAD"
        return modelName

    def validateModelName(self, modelName: str) -> None:
        runnableModels = {"DyMETER", "METER", "D3R", "SARAD", "EnhancedDymeterMANN"}
        if modelName in runnableModels:
            return
        raise ValueError(
            "Unsupported model. Choose one of: DyMETER, METER, D3R, SARAD, EnhancedDymeterMANN."
        )

    def buildAnomalyResult(self, preparedData, detectorOutput) -> AnomalyResult:
        scores = detectorOutput.scores.copy()
        thresholds = detectorOutput.thresholds.copy()
        predictions = scores > thresholds

        anomalyRatio = float(np.mean(predictions)) if len(predictions) else 0.0
        if anomalyRatio > 0.5 or float(np.min(thresholds)) <= 0:
            fallbackThreshold = float(np.quantile(scores, 0.95))
            thresholds[:] = max(fallbackThreshold, 1e-8)
            predictions = scores > thresholds

        anomalies: list[AnomalyItem] = []
        for rowIndex in np.where(predictions)[0].tolist():
            rowData = frameRowToDict(preparedData.originalFrame, rowIndex)
            contributions = computeContributions(
                preparedData.featureMatrix[rowIndex],
                detectorOutput.reconstructions[rowIndex],
                preparedData.featureNames,
            )
            severityLevel, confidence = severityFromScore(float(scores[rowIndex]), float(thresholds[rowIndex]))
            explanation = explainAnomaly(rowData, contributions, float(scores[rowIndex]), float(thresholds[rowIndex]))
            anomalies.append(
                AnomalyItem(
                    rowIndex=int(rowIndex),
                    originalRowData=rowData,
                    anomalyScore=round(float(scores[rowIndex]), 6),
                    reconstructionError=round(float(scores[rowIndex]), 6),
                    conceptUncertainty=round(float(detectorOutput.conceptUncertainty[rowIndex]), 6),
                    detectorUsed=detectorOutput.detectorUsed[rowIndex],
                    severityLevel=severityLevel,
                    topContributingFeatures=contributions,
                    naturalLanguageExplanation=explanation,
                    confidence=round(float(confidence), 4),
                )
            )

        anomalies.sort(key=lambda item: item.anomalyScore, reverse=True)
        driftEvents = [
            DriftEvent(rowIndexRange=event["rowIndexRange"], description=event["description"])
            for event in detectorOutput.driftEvents
        ]

        return AnomalyResult(
            totalRows=int(len(scores)),
            totalAnomalies=int(len(anomalies)),
            anomalyRatio=round(float(len(anomalies) / max(len(scores), 1)), 6),
            thresholdUsed=round(float(np.median(thresholds)), 6),
            anomalies=anomalies,
            conceptDriftEvents=driftEvents,
        )

    def makeQaContext(self, datasetName: str, profile, preprocessing: dict, metrics: dict, anomalyResult: AnomalyResult, comparison: list[dict]) -> dict[str, Any]:
        return {
            "datasetName": datasetName,
            "datasetProfile": profile.__dict__,
            "preprocessing": preprocessing,
            "metrics": metrics,
            "topAnomalies": anomalyResult.toDict()["anomalies"][:20],
            "conceptDriftEvents": anomalyResult.toDict()["conceptDriftEvents"],
            "comparison": comparison,
            "rules": [
                "Answer only from this JSON context.",
                "If the answer is not present, say that the run context does not contain it.",
            ],
        }

    def saveRun(self, bundle: RunBundle) -> None:
        outputPath = self.outputDirectory / f"{bundle.runId}.json"
        with outputPath.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "runId": bundle.runId,
                    "modelName": bundle.modelName,
                    "datasetName": bundle.datasetName,
                    "profile": bundle.profile.__dict__,
                    "preprocessing": bundle.preprocessing,
                    "metrics": bundle.metrics,
                    "anomalyResult": bundle.anomalyResult.toDict(),
                    "pipelineLogs": [log.__dict__ for log in bundle.pipelineLogs],
                    "qaContext": bundle.qaContext,
                    "comparison": bundle.comparison,
                    "artifacts": bundle.artifacts,
                },
                file,
                indent=2,
                default=str,
            )
