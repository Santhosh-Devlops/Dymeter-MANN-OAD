from dataclasses import dataclass, field
from typing import Any


SeverityLevels = ("Normal", "Warning", "Critical", "Emergency")


@dataclass
class FeatureContribution:
    featureName: str
    contributionScore: float


@dataclass
class DriftEvent:
    rowIndexRange: list[int]
    description: str


@dataclass
class AnomalyItem:
    rowIndex: int
    originalRowData: dict[str, Any]
    anomalyScore: float
    reconstructionError: float
    conceptUncertainty: float
    detectorUsed: str
    severityLevel: str
    topContributingFeatures: list[FeatureContribution]
    naturalLanguageExplanation: str
    confidence: float = 0.0


@dataclass
class AnomalyResult:
    totalRows: int
    totalAnomalies: int
    anomalyRatio: float
    thresholdUsed: float
    anomalies: list[AnomalyItem]
    conceptDriftEvents: list[DriftEvent]

    def toDict(self) -> dict[str, Any]:
        return {
            "totalRows": self.totalRows,
            "totalAnomalies": self.totalAnomalies,
            "anomalyRatio": self.anomalyRatio,
            "thresholdUsed": self.thresholdUsed,
            "anomalies": [
                {
                    "rowIndex": item.rowIndex,
                    "originalRowData": item.originalRowData,
                    "anomalyScore": item.anomalyScore,
                    "reconstructionError": item.reconstructionError,
                    "conceptUncertainty": item.conceptUncertainty,
                    "detectorUsed": item.detectorUsed,
                    "severityLevel": item.severityLevel,
                    "topContributingFeatures": [
                        {
                            "featureName": part.featureName,
                            "contributionScore": part.contributionScore,
                        }
                        for part in item.topContributingFeatures
                    ],
                    "naturalLanguageExplanation": item.naturalLanguageExplanation,
                    "confidence": item.confidence,
                }
                for item in self.anomalies
            ],
            "conceptDriftEvents": [
                {"rowIndexRange": event.rowIndexRange, "description": event.description}
                for event in self.conceptDriftEvents
            ],
        }


@dataclass
class DatasetProfile:
    columnCount: int
    rowCount: int
    numericColumns: list[str]
    categoricalColumns: list[str]
    timestampColumns: list[str]
    idColumns: list[str]
    constantColumns: list[str]
    missingValueRatio: dict[str, float]
    labelColumn: str | None
    suspectedDataType: str
    validationMessages: list[str] = field(default_factory=list)


@dataclass
class TrainingLog:
    stage: str
    message: str
    detail: str = ""
    level: str = "info"


@dataclass
class RunBundle:
    runId: str
    modelName: str
    datasetName: str
    profile: DatasetProfile
    preprocessing: dict[str, Any]
    metrics: dict[str, Any]
    anomalyResult: AnomalyResult
    pipelineLogs: list[TrainingLog]
    qaContext: dict[str, Any]
    comparison: list[dict[str, Any]]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
