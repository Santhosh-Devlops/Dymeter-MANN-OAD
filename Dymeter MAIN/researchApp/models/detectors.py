from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from researchApp.core.contracts import TrainingLog
from researchApp.core.metrics import aucPr, aucRoc
from researchApp.core.threshold import DynamicThresholdOptimizer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torchFunction
except Exception:
    torch = None
    nn = None
    torchFunction = None


@dataclass
class DetectorOutput:
    scores: np.ndarray
    reconstructions: np.ndarray
    conceptUncertainty: np.ndarray
    detectorUsed: list[str]
    thresholds: np.ndarray
    driftEvents: list[dict]
    trainingSeconds: float
    inferenceSeconds: float
    parameterCount: int
    memoryEvents: list[dict]
    trainingBudget: dict


@dataclass(frozen=True)
class TrainingBudget:
    maxEpochs: int
    earlyStoppingPatience: int | None
    learningRate: float
    decayFactor: float


@dataclass
class TrainingState:
    stage: str
    stoppedEpoch: int
    bestEpoch: int
    validationMetric: str
    bestValidationScore: float
    checkpointSource: str


PAPER_BASELINE_MAX_EPOCHS = 2000
ENHANCED_DYMETER_MAX_EPOCHS = 20000
ENHANCED_DYMETER_EARLY_STOPPING_PATIENCE = 300
PAPER_ADAM_LEARNING_RATE = 1e-2
PAPER_EXPONENTIAL_DECAY_FACTOR = 0.96
VALIDATION_EPSILON = 1e-8
DYNAMIC_MODEL_NAMES = {"DyMETER", "EnhancedDymeterMANN"}
MEMORY_MODEL_NAMES = {"EnhancedDymeterMANN"}
EXTENDED_BUDGET_MODEL_NAMES: set[str] = set()


class NumpyStaticConceptDetector:
    """Small PCA autoencoder used when PyTorch is not installed."""

    def __init__(self, latentDimension: int):
        self.latentDimension = latentDimension
        self.meanVector: np.ndarray | None = None
        self.components: np.ndarray | None = None

    def fit(self, featureMatrix: np.ndarray) -> None:
        self.meanVector = featureMatrix.mean(axis=0)
        centeredMatrix = featureMatrix - self.meanVector
        _, _, components = np.linalg.svd(centeredMatrix, full_matrices=False)
        self.components = components[: self.latentDimension]

    def reconstruct(self, featureMatrix: np.ndarray) -> np.ndarray:
        if self.meanVector is None or self.components is None:
            raise RuntimeError("Static detector has not been trained.")
        centeredMatrix = featureMatrix - self.meanVector
        latent = centeredMatrix @ self.components.T
        return latent @ self.components + self.meanVector


if nn is not None:

    class StaticConceptAwareDetector(nn.Module):
        """SCD from Section III-B: autoencoder trained on historical normal data."""

        def __init__(self, inputDimension: int, latentDimension: int):
            super().__init__()
            hiddenDimension = max(latentDimension * 2, min(128, inputDimension * 2))
            self.encoder = nn.Sequential(
                nn.Linear(inputDimension, hiddenDimension),
                nn.ReLU(),
                nn.Linear(hiddenDimension, latentDimension),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latentDimension, hiddenDimension),
                nn.ReLU(),
                nn.Linear(hiddenDimension, inputDimension),
            )

        def forward(self, featureTensor):
            latent = self.encoder(featureTensor)
            return self.decoder(latent)


    class IntelligentEvolutionController(nn.Module):
        """IEC from Section III-C using evidence over normal/anomaly pseudo labels."""

        def __init__(self, inputDimension: int):
            super().__init__()
            hiddenDimension = max(16, min(128, inputDimension * 2))
            self.layers = nn.Sequential(
                nn.Linear(inputDimension, hiddenDimension),
                nn.ReLU(),
                nn.Linear(hiddenDimension, 2),
                nn.Softplus(),
            )

        def forward(self, featureTensor):
            evidence = self.layers(featureTensor)
            return evidence + 1.0


    class ParameterShiftHypernetwork(nn.Module):
        """DSD hypernetwork from Eq. 12-14, generating instance-aware shifts."""

        def __init__(self, inputDimension: int):
            super().__init__()
            hiddenDimension = max(16, min(128, inputDimension * 2))
            self.shift = nn.Sequential(
                nn.Linear(inputDimension, hiddenDimension),
                nn.ReLU(),
                nn.Linear(hiddenDimension, inputDimension),
                nn.Tanh(),
            )

        def forward(self, featureTensor):
            return 0.1 * self.shift(featureTensor)


class ConceptMemory:
    """MANN-style concept bank with attention-based read/write/update."""

    def __init__(self, maxCells: int = 12, temperature: float = 0.15, updateRate: float = 0.2):
        self.maxCells = maxCells
        self.temperature = max(float(temperature), 1e-6)
        self.updateRate = min(max(float(updateRate), 0.0), 1.0)
        self.cells: list[dict] = []

    def read(self, conceptVector: np.ndarray) -> tuple[dict | None, float]:
        if not self.cells:
            return None, 0.0

        similarities = np.array([self.cosineSimilarity(conceptVector, cell["key"]) for cell in self.cells], dtype=float)
        attention = self.softmax(similarities / self.temperature)
        bestIndex = int(np.argmax(attention))
        bestCell = self.cells[bestIndex]
        bestCell["reads"] += 1
        bestCell["lastAttention"] = float(attention[bestIndex])
        return bestCell, float(similarities[bestIndex])

    def write(
        self,
        conceptVector: np.ndarray,
        thresh: float,
        rowIndex: int,
        memoryValue: np.ndarray | None = None,
        scorePrototype: float | None = None,
    ) -> dict:
        cell = {
            "key": conceptVector.copy(),
            "conceptVector": conceptVector.copy(),
            "threshold": float(thresh),
            "memoryValue": None if memoryValue is None else memoryValue.copy(),
            "scorePrototype": None if scorePrototype is None else float(scorePrototype),
            "rowIndex": int(rowIndex),
            "reads": 0,
            "writes": 1,
            "lastAttention": 1.0,
        }
        self.cells.append(cell)
        if len(self.cells) > self.maxCells:
            self.cells.sort(key=lambda item: (item["reads"], item["writes"]))
            self.cells.pop(0)
        return {"type": "write", "rowIndex": int(rowIndex), "cellCount": len(self.cells)}

    def update(self, cell: dict, conceptVector: np.ndarray, thresh: float, score: float, memoryValue: np.ndarray | None = None) -> dict:
        cell["key"] = (1.0 - self.updateRate) * cell["key"] + self.updateRate * conceptVector
        cell["conceptVector"] = cell["key"].copy()
        cell["threshold"] = (1.0 - self.updateRate) * float(cell["threshold"]) + self.updateRate * float(thresh)
        if memoryValue is not None:
            if cell.get("memoryValue") is None:
                cell["memoryValue"] = memoryValue.copy()
            else:
                cell["memoryValue"] = (1.0 - self.updateRate) * cell["memoryValue"] + self.updateRate * memoryValue
        if cell.get("scorePrototype") is None:
            cell["scorePrototype"] = float(score)
        else:
            cell["scorePrototype"] = (1.0 - self.updateRate) * float(cell["scorePrototype"]) + self.updateRate * float(score)
        cell["writes"] += 1
        return {"type": "update", "rowIndex": int(cell["rowIndex"]), "cellCount": len(self.cells)}

    def cosineSimilarity(self, left: np.ndarray, right: np.ndarray) -> float:
        numerator = float(np.dot(left, right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-8)
        return numerator / denominator

    def softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - float(np.max(values))
        weights = np.exp(shifted)
        return weights / (float(np.sum(weights)) + 1e-8)


class DyMeterRunner:
    """Runs METER, DyMETER, and the MANN ablation behind one clear interface."""

    def __init__(self, modelName: str, config: dict | None = None):
        self.modelName = modelName
        self.config = config or {}

    def printTrainingProgress(
        self,
        stage: str,
        epochs: int,
        totalEpochs: int,
        lossValue: float,
        validationMetric: str,
        validationScore: float,
        bestEpochs: int,
        bestScore: float | None,
        force: bool = False,
    ) -> None:
        if not self.config.get("showEpochProgress", True):
            return
        epochLogInterval = max(1, int(self.config.get("epochLogInterval", 100)))
        if not force and epochs != 1 and epochs != totalEpochs and epochs % epochLogInterval != 0:
            return
        bestScoreText = f"{bestScore:.6f}" if bestScore is not None else "pending"
        print(
            f"[{self.displayName()}] {stage} epoch {epochs}/{totalEpochs} "
            f"loss={lossValue:.6f} {validationMetric}={validationScore:.6f} "
            f"bestEpoch={bestEpochs} best={bestScoreText}",
            flush=True,
        )

    def run(self, featureMatrix: np.ndarray, labels: np.ndarray | None, preprocessingSummary: dict) -> tuple[DetectorOutput, list[TrainingLog]]:
        logs: list[TrainingLog] = []
        startTraining = time.time()
        if self.config.get("showEpochProgress", True):
            print(f"[{self.displayName()}] training started", flush=True)
        if self.modelName == "EnhancedDymeterMANN":
            seed = self.config.get("randomSeed", 7)
            if seed is not None:
                seed = int(seed)
                np.random.seed(seed)
                if torch is not None:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                logs.append(TrainingLog("Reproducible Enhanced tuning", "EnhancedDymeterMANN uses a fixed seed for stable benchmark comparison.", f"Seed: {seed}."))

        trainRate = float(self.config.get("trainRate", 0.2))
        rowCount = featureMatrix.shape[0]
        trainCount = max(8, min(rowCount, int(rowCount * trainRate)))
        latentDimension = int(preprocessingSummary.get("latentDimension", max(1, featureMatrix.shape[1] // 2)))

        if self.modelName == "D3R":
            output = self.runD3RAdapter(featureMatrix, labels, trainCount, latentDimension, preprocessingSummary, logs, startTraining)
            return output, logs

        if self.modelName == "SARAD":
            output = self.runSARADAdapter(featureMatrix, labels, trainCount, latentDimension, preprocessingSummary, logs, startTraining)
            return output, logs

        historicalData = self.selectHistoricalData(featureMatrix, labels, trainCount)
        logs.append(TrainingLog("Preparing training data", "Clean historical rows were selected for the first concept.", f"{len(historicalData)} rows."))
        logs.append(TrainingLog("Creating Static Detector", "This model learns normal behaviour from historical data.", f"Latent dimension: {latentDimension}."))
        self.appendBudgetStartLog(logs)

        if torch is None or self.config.get("forceNumpy", False):
            output = self.runNumpy(featureMatrix, historicalData, latentDimension, preprocessingSummary, logs, startTraining)
            logs.append(TrainingLog("Dependency note", "PyTorch was not available, so the compatible PCA fallback was used.", "Install requirements for neural training.", "warning"))
            return output, logs

        output = self.runTorch(featureMatrix, labels, historicalData, trainCount, latentDimension, preprocessingSummary, logs, startTraining)
        return output, logs

    def isEnhanced(self) -> bool:
        return self.modelName in MEMORY_MODEL_NAMES

    def usesDynamicShiftDetector(self) -> bool:
        return self.modelName in DYNAMIC_MODEL_NAMES

    def usesConceptMemory(self) -> bool:
        return self.modelName in MEMORY_MODEL_NAMES

    def usesExtendedTrainingBudget(self) -> bool:
        return self.modelName in EXTENDED_BUDGET_MODEL_NAMES

    def displayName(self) -> str:
        if self.modelName == "EnhancedDymeterMANN":
            return "Enhanced Dymeter MANN"
        return self.modelName

    def trainingBudget(self) -> TrainingBudget:
        if self.usesExtendedTrainingBudget():
            return TrainingBudget(
                maxEpochs=ENHANCED_DYMETER_MAX_EPOCHS,
                earlyStoppingPatience=ENHANCED_DYMETER_EARLY_STOPPING_PATIENCE,
                learningRate=PAPER_ADAM_LEARNING_RATE,
                decayFactor=PAPER_EXPONENTIAL_DECAY_FACTOR,
            )
        return TrainingBudget(
            maxEpochs=PAPER_BASELINE_MAX_EPOCHS,
            earlyStoppingPatience=None,
            learningRate=PAPER_ADAM_LEARNING_RATE,
            decayFactor=PAPER_EXPONENTIAL_DECAY_FACTOR,
        )

    def makeBudgetSummary(self, trainingStates: list[TrainingState] | None = None) -> dict:
        budget = self.trainingBudget()
        return {
            "maxEpochs": budget.maxEpochs,
            "earlyStoppingPatience": budget.earlyStoppingPatience,
            "learningRate": budget.learningRate,
            "decayFactor": budget.decayFactor,
            "checkpointRule": "best validation checkpoint" if self.usesExtendedTrainingBudget() else "fixed baseline budget",
            "stages": [state.__dict__ for state in trainingStates or []],
            "architectureCoverage": self.architectureCoverage(),
        }

    def architectureCoverage(self) -> list[dict]:
        exactness = "research-grade implementation" if self.usesExtendedTrainingBudget() else "baseline implementation"
        coverage = [
            {"name": "Streaming multi-sensor preprocessing", "status": "implemented", "exactness": "adaptive tabular/stream preprocessing"},
            {"name": "Static Concept-aware Detector", "status": "implemented", "exactness": exactness},
            {"name": "Autoencoder reconstruction loss", "status": "implemented", "exactness": "MSE reconstruction objective"},
            {"name": "Intelligent Evolution Controller", "status": "implemented", "exactness": "Dirichlet evidence controller"},
            {"name": "Concept uncertainty", "status": "implemented", "exactness": "Dirichlet entropy minus expected data uncertainty"},
            {"name": "Dynamic Shift-aware Detector", "status": "implemented" if self.usesDynamicShiftDetector() else "not used", "exactness": "trained parameter-shift hypernetwork"},
            {"name": "Long-term concept memory", "status": "implemented" if self.usesConceptMemory() else "not used", "exactness": "attention key-value memory with read/write/update"},
            {"name": "Dynamic threshold optimization", "status": "implemented", "exactness": "online quantile threshold with uncertainty regularization"},
            {"name": "Explainable AI", "status": "implemented", "exactness": "per-row reconstruction contribution attribution"},
            {"name": "Severity prediction", "status": "implemented", "exactness": "threshold-relative severity bands"},
        ]
        if self.modelName == "EnhancedDymeterMANN":
            coverage.extend(
                [
                    {
                        "name": "Self-supervised robust initialization",
                        "status": "implemented",
                        "exactness": "denoising reconstruction with feature masking",
                    },
                    {
                        "name": "Proactive uncertainty adaptation",
                        "status": "implemented",
                        "exactness": "uncertainty trend forecast activates DSD/MANN before threshold crossing",
                    },
                ]
            )
        return coverage

    def appendBudgetStartLog(self, logs: list[TrainingLog]) -> None:
        budget = self.trainingBudget()
        patienceDetail = (
            f"early stopping patience {budget.earlyStoppingPatience}"
            if budget.earlyStoppingPatience is not None
            else "no early stopping"
        )
        logs.append(
            TrainingLog(
                "Training Budget",
                f"{self.displayName()} is fixed before run: maxEpochs={budget.maxEpochs}, {patienceDetail}.",
                f"Adam lr={budget.learningRate:g}, exponential decay={budget.decayFactor}.",
            )
        )

    def appendBudgetStopLog(self, logs: list[TrainingLog], state: TrainingState) -> None:
        logs.append(
            TrainingLog(
                "Training Budget",
                f"{state.stage} stopped at epoch {state.stoppedEpoch}; checkpoint epoch {state.bestEpoch}.",
                f"{state.validationMetric}={state.bestValidationScore:.6f}; source={state.checkpointSource}.",
            )
        )

    def selectHistoricalData(self, featureMatrix: np.ndarray, labels: np.ndarray | None, trainCount: int) -> np.ndarray:
        if labels is not None and np.sum(labels == 0) >= max(4, trainCount // 2):
            normalRows = featureMatrix[labels == 0]
            return normalRows[:trainCount]
        if self.modelName == "EnhancedDymeterMANN" and len(featureMatrix) > trainCount:
            candidateWindow = featureMatrix[: max(trainCount * 2, trainCount)]
            center = np.median(candidateWindow, axis=0, keepdims=True)
            distance = np.mean(np.abs(candidateWindow - center), axis=1)
            selectedIndexes = np.argsort(distance)[:trainCount]
            selectedIndexes.sort()
            return candidateWindow[selectedIndexes]
        return featureMatrix[:trainCount]

    def proactiveDynamicMask(self, conceptUncertainty: np.ndarray, thresh: float) -> np.ndarray:
        uncertainty = np.asarray(conceptUncertainty, dtype=float)
        if len(uncertainty) == 0:
            return np.zeros(0, dtype=bool)
        horizon = int(self.config.get("proactiveHorizon", 5))
        lookback = max(3, int(self.config.get("proactiveLookback", 8)))
        sensitivity = float(self.config.get("proactiveSensitivity", 0.9))
        mask = np.zeros(len(uncertainty), dtype=bool)
        for index in range(len(uncertainty)):
            start = max(0, index - lookback + 1)
            window = uncertainty[start : index + 1]
            if len(window) < 3:
                continue
            slope = float(np.mean(np.diff(window)))
            forecast = float(uncertainty[index] + max(slope, 0.0) * horizon)
            mask[index] = forecast >= thresh * sensitivity
        return mask

    def robustNormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) + 1e-8
        normalized = (values - median) / (1.4826 * mad)
        return np.maximum(normalized, 0.0)

    def rollingMean(self, featureMatrix: np.ndarray, windowSize: int) -> np.ndarray:
        if len(featureMatrix) == 0:
            return featureMatrix.copy()
        windowSize = max(2, min(windowSize, len(featureMatrix)))
        trends = np.zeros_like(featureMatrix, dtype=float)
        cumulative = np.vstack([np.zeros((1, featureMatrix.shape[1])), np.cumsum(featureMatrix, axis=0)])
        for index in range(len(featureMatrix)):
            start = max(0, index - windowSize + 1)
            count = index - start + 1
            trends[index] = (cumulative[index + 1] - cumulative[start]) / count
        return trends

    def groupEvents(self, mask: np.ndarray, description: str) -> list[dict]:
        events: list[dict] = []
        for index, active in enumerate(mask.astype(bool).tolist()):
            if not active:
                continue
            if not events or events[-1]["rowIndexRange"][1] != index - 1:
                events.append({"rowIndexRange": [index, index], "description": description})
            else:
                events[-1]["rowIndexRange"][1] = index
        return events

    def thresholdScores(
        self,
        scores: np.ndarray,
        conceptUncertainty: np.ndarray,
        historicalLength: int,
        preprocessingSummary: dict,
    ) -> np.ndarray:
        threshOptimizer = DynamicThresholdOptimizer(
            falsePositiveRate=float(self.config.get("falsePositiveRate", 0.05)),
            windowSize=int(preprocessingSummary.get("windowSize", 64)),
            kappa=float(self.config.get("kappa", 0.5)),
        )
        threshOptimizer.initialize(scores[: max(8, historicalLength)])
        threshValues = [
            threshOptimizer.update(float(score), float(conceptUncertainty[index]), bool(conceptUncertainty[index] > 0.65))
            for index, score in enumerate(scores)
        ]
        return np.asarray(threshValues, dtype=float)

    def baselineBudgetSummary(self, modelName: str, stages: list[dict]) -> dict:
        return {
            "maxEpochs": 1,
            "earlyStoppingPatience": None,
            "learningRate": None,
            "decayFactor": None,
            "checkpointRule": "adapter fit on historical split",
            "stages": stages,
            "architectureCoverage": [
                {"name": "Dataset preprocessing", "status": "implemented", "exactness": "shared app preprocessing"},
                {"name": modelName, "status": "implemented", "exactness": "application adapter over uploaded data"},
                {"name": "Anomaly scoring", "status": "implemented", "exactness": "dataset-dependent reconstruction/shift score"},
                {"name": "Dynamic thresholding", "status": "implemented", "exactness": "online threshold optimizer"},
                {"name": "Report artifacts", "status": "implemented", "exactness": "shared app reporting"},
            ],
        }

    def runD3RAdapter(
        self,
        featureMatrix: np.ndarray,
        labels: np.ndarray | None,
        trainCount: int,
        latentDimension: int,
        preprocessingSummary: dict,
        logs: list[TrainingLog],
        startTraining: float,
    ) -> DetectorOutput:
        logs.append(TrainingLog("Preparing D3R adapter", "Dynamic decomposition and reconstruction scoring is configured for the uploaded dataset.", f"Historical rows: {trainCount}."))
        windowSize = max(4, min(int(preprocessingSummary.get("windowSize", 64)), max(4, len(featureMatrix) // 4)))
        trends = self.rollingMean(featureMatrix, windowSize)
        stableSignals = featureMatrix - trends
        historicalStable = stableSignals[:trainCount]

        detector = NumpyStaticConceptDetector(max(1, min(latentDimension, featureMatrix.shape[1], len(historicalStable))))
        detector.fit(historicalStable)
        trainingSeconds = time.time() - startTraining
        logs.append(TrainingLog("Training D3R reconstruction", "The stable component was reconstructed after rolling trend decomposition.", f"Window size: {windowSize}."))

        startInference = time.time()
        stableRecon = detector.reconstruct(stableSignals)
        reconstructions = stableRecon + trends
        reconstructionScore = np.mean(np.abs(featureMatrix - reconstructions), axis=1)
        trendCenter = trends[:trainCount].mean(axis=0)
        trendSpread = trends[:trainCount].std(axis=0) + 1e-8
        trendShift = np.mean(np.abs((trends - trendCenter) / trendSpread), axis=1)
        scores = reconstructionScore + 0.35 * self.robustNormalize(trendShift)
        conceptUncertainty = 1.0 / (1.0 + np.exp(-(self.robustNormalize(trendShift) - 1.0)))
        thresholds = self.thresholdScores(scores, conceptUncertainty, trainCount, preprocessingSummary)
        driftEvents = self.groupEvents(conceptUncertainty > 0.65, "D3R decomposition detected a trend or reconstruction shift.")
        inferenceSeconds = time.time() - startInference
        logs.append(TrainingLog("D3R inference completed", "Rows were scored from reconstruction error plus decomposed trend shift.", f"{len(scores)} rows processed."))

        return DetectorOutput(
            scores=scores,
            reconstructions=reconstructions,
            conceptUncertainty=conceptUncertainty,
            detectorUsed=["D3R"] * len(scores),
            thresholds=thresholds,
            driftEvents=driftEvents,
            trainingSeconds=trainingSeconds,
            inferenceSeconds=inferenceSeconds,
            parameterCount=int(featureMatrix.shape[1] * max(1, latentDimension) * 2),
            memoryEvents=[],
            trainingBudget=self.baselineBudgetSummary(
                "D3R",
                [
                    {
                        "stage": "D3R adapter fit",
                        "stoppedEpoch": 1,
                        "bestEpoch": 1,
                        "validationMetric": "historical reconstruction fit",
                        "bestValidationScore": float(np.mean(reconstructionScore[:trainCount])),
                        "checkpointSource": "adapter",
                    }
                ],
            ),
        )

    def runSARADAdapter(
        self,
        featureMatrix: np.ndarray,
        labels: np.ndarray | None,
        trainCount: int,
        latentDimension: int,
        preprocessingSummary: dict,
        logs: list[TrainingLog],
        startTraining: float,
    ) -> DetectorOutput:
        logs.append(TrainingLog("Preparing SARAD adapter", "Spatial association-aware scoring is configured for the uploaded dataset.", f"Historical rows: {trainCount}."))
        historicalData = featureMatrix[:trainCount]
        detector = NumpyStaticConceptDetector(max(1, min(latentDimension, featureMatrix.shape[1], len(historicalData))))
        detector.fit(historicalData)
        trainingSeconds = time.time() - startTraining
        logs.append(TrainingLog("Training SARAD reconstruction", "A compact reconstruction model was fit before spatial association scoring.", f"Latent dimension: {detector.latentDimension}."))

        startInference = time.time()
        reconstructions = detector.reconstruct(featureMatrix)
        reconstructionScore = np.mean(np.abs(featureMatrix - reconstructions), axis=1)
        windowSize = max(4, min(int(preprocessingSummary.get("windowSize", 64)), len(featureMatrix)))
        prototype = self.spatialAssociation(historicalData)
        spatialScores = np.zeros(len(featureMatrix), dtype=float)
        for index in range(len(featureMatrix)):
            start = max(0, index - windowSize + 1)
            association = self.spatialAssociation(featureMatrix[start : index + 1])
            spatialScores[index] = float(np.mean(np.abs(association - prototype)))

        scores = reconstructionScore + 0.5 * self.robustNormalize(spatialScores)
        conceptUncertainty = 1.0 / (1.0 + np.exp(-(self.robustNormalize(spatialScores) - 1.0)))
        thresholds = self.thresholdScores(scores, conceptUncertainty, trainCount, preprocessingSummary)
        driftEvents = self.groupEvents(conceptUncertainty > 0.65, "SARAD spatial association descent changed beyond the historical pattern.")
        inferenceSeconds = time.time() - startInference
        logs.append(TrainingLog("SARAD inference completed", "Rows were scored from reconstruction error plus spatial association shift.", f"{len(scores)} rows processed."))

        return DetectorOutput(
            scores=scores,
            reconstructions=reconstructions,
            conceptUncertainty=conceptUncertainty,
            detectorUsed=["SARAD"] * len(scores),
            thresholds=thresholds,
            driftEvents=driftEvents,
            trainingSeconds=trainingSeconds,
            inferenceSeconds=inferenceSeconds,
            parameterCount=int(featureMatrix.shape[1] * max(1, latentDimension) * 2 + featureMatrix.shape[1] ** 2),
            memoryEvents=[],
            trainingBudget=self.baselineBudgetSummary(
                "SARAD",
                [
                    {
                        "stage": "SARAD adapter fit",
                        "stoppedEpoch": 1,
                        "bestEpoch": 1,
                        "validationMetric": "historical reconstruction fit",
                        "bestValidationScore": float(np.mean(reconstructionScore[:trainCount])),
                        "checkpointSource": "adapter",
                    }
                ],
            ),
        )

    def spatialAssociation(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] == 0:
            return np.zeros((1, 1), dtype=float)
        if values.shape[1] == 1:
            variance = float(np.var(values[:, 0])) if len(values) else 0.0
            return np.array([[variance]], dtype=float)
        if len(values) < 2:
            return np.eye(values.shape[1], dtype=float)
        centered = values - values.mean(axis=0, keepdims=True)
        spread = centered.std(axis=0, keepdims=True)
        active = spread.reshape(-1) > 1e-8
        standardized = np.zeros_like(centered, dtype=float)
        standardized[:, active] = centered[:, active] / spread[:, active]
        corr = standardized.T @ standardized / max(len(values) - 1, 1)
        corr = np.clip(corr, -1.0, 1.0)
        corr[np.diag_indices_from(corr)] = active.astype(float)
        return corr

    def runNumpy(
        self,
        featureMatrix: np.ndarray,
        historicalData: np.ndarray,
        latentDimension: int,
        preprocessingSummary: dict,
        logs: list[TrainingLog],
        startTraining: float,
    ) -> DetectorOutput:
        staticDetector = NumpyStaticConceptDetector(latentDimension)
        staticDetector.fit(historicalData)
        logs.append(TrainingLog("Training Autoencoder", "The static detector learned compact normal patterns.", "PCA fallback completed."))
        logs.append(
            TrainingLog(
                "Training Budget",
                "Neural epoch budget was not consumed because PCA fallback was selected.",
                "Stopped at epoch 0; no checkpoint search.",
                "warning",
            )
        )

        trainingSeconds = time.time() - startTraining
        startInference = time.time()
        staticReconstructions = staticDetector.reconstruct(featureMatrix)
        staticScores = np.mean(np.abs(featureMatrix - staticReconstructions), axis=1)
        logs.append(TrainingLog("Computing Reconstruction Errors", "Rows with larger reconstruction error are more unusual.", f"Mean score: {staticScores.mean():.4f}."))

        conceptCenter = historicalData.mean(axis=0)
        conceptSpread = np.std(historicalData, axis=0) + 1e-8
        conceptDistance = np.mean(np.abs((featureMatrix - conceptCenter) / conceptSpread), axis=1)
        conceptUncertainty = 1.0 / (1.0 + np.exp(-(conceptDistance - np.median(conceptDistance))))
        logs.append(TrainingLog("Estimating Concept Uncertainty", "The controller checks whether normal behaviour has changed.", f"Average uncertainty: {conceptUncertainty.mean():.4f}."))

        threshOptimizer = DynamicThresholdOptimizer(
            falsePositiveRate=float(self.config.get("falsePositiveRate", 0.05)),
            windowSize=int(preprocessingSummary.get("windowSize", 64)),
            kappa=float(self.config.get("kappa", 0.5)),
        )
        baseThresh = threshOptimizer.initialize(staticScores[: max(8, len(historicalData))])

        scores = staticScores.copy()
        reconstructions = staticReconstructions.copy()
        detectorUsed = ["static"] * len(scores)
        driftEvents: list[dict] = []
        memoryEvents: list[dict] = []
        memoryBank = ConceptMemory()

        if self.usesDynamicShiftDetector():
            logs.append(TrainingLog("Creating Hypernetwork", "A small network estimates how the detector should move for a changed concept.", "Numpy shift approximation active."))
            uncertaintyThresh = float(self.config.get("uncertaintyThreshold", 0.65))
            proactiveMask = self.proactiveDynamicMask(conceptUncertainty, uncertaintyThresh) if self.modelName == "EnhancedDymeterMANN" else np.zeros(len(scores), dtype=bool)
            if self.modelName == "EnhancedDymeterMANN":
                logs.append(TrainingLog("Proactive adaptation", "Uncertainty trends are forecast before hard drift activation.", f"{int(np.sum(proactiveMask))} proactive rows."))
            for rowIndex in range(len(scores)):
                driftDetected = bool(conceptUncertainty[rowIndex] > uncertaintyThresh or proactiveMask[rowIndex])
                if driftDetected:
                    detectorUsed[rowIndex] = "proactive" if proactiveMask[rowIndex] and conceptUncertainty[rowIndex] <= uncertaintyThresh else "dynamic"
                    rowShift = 0.15 * (featureMatrix[rowIndex] - conceptCenter)
                    reconstructions[rowIndex] = reconstructions[rowIndex] + rowShift
                    # Eq. 15: anomaly score is reconstruction distance after static or dynamic inference.
                    scores[rowIndex] = float(np.mean(np.abs(featureMatrix[rowIndex] - reconstructions[rowIndex])))
                    if rowIndex == 0 or detectorUsed[rowIndex - 1] == "static":
                        driftEvents.append({"rowIndexRange": [rowIndex, rowIndex], "description": "Concept uncertainty increased, so the dynamic detector was activated."})
                    else:
                        driftEvents[-1]["rowIndexRange"][1] = rowIndex

                if self.usesConceptMemory() and driftDetected:
                    conceptVector = featureMatrix[max(0, rowIndex - 8) : rowIndex + 1].mean(axis=0)
                    rememberedCell, attentionScore = memoryBank.read(conceptVector)
                    if rememberedCell is not None and attentionScore > 0.8:
                        scores[rowIndex] = min(scores[rowIndex], rememberedCell["threshold"] * 0.95)
                        memoryEvents.append({"type": "read", "rowIndex": rowIndex, "attentionScore": round(attentionScore, 4), "retrievedRowIndex": rememberedCell["rowIndex"]})
                    else:
                        memoryEvents.append(memoryBank.write(conceptVector, baseThresh, rowIndex))

        threshValues = []
        for rowIndex, score in enumerate(scores):
            threshValues.append(
                threshOptimizer.update(
                    float(score),
                    float(conceptUncertainty[rowIndex]),
                    detectorUsed[rowIndex] != "static",
                )
            )
        threshValues = np.asarray(threshValues, dtype=float)
        logs.append(TrainingLog("Computing Dynamic Threshold", "The decision boundary was recalibrated from recent scores.", f"Final threshold: {threshValues[-1]:.4f}."))

        if self.modelName == "METER":
            threshValues[:] = baseThresh

        inferenceSeconds = time.time() - startInference
        parameterCount = int(featureMatrix.shape[1] * latentDimension + latentDimension * featureMatrix.shape[1])
        if self.usesConceptMemory():
            parameterCount += int(featureMatrix.shape[1] * max(4, featureMatrix.shape[1] // 2))

        logs.append(TrainingLog("Inference completed", "Every row received a score, threshold, and decision.", f"{len(scores)} rows processed."))
        return DetectorOutput(
            scores=scores,
            reconstructions=reconstructions,
            conceptUncertainty=conceptUncertainty,
            detectorUsed=detectorUsed,
            thresholds=threshValues,
            driftEvents=driftEvents,
            trainingSeconds=trainingSeconds,
            inferenceSeconds=inferenceSeconds,
            parameterCount=parameterCount,
            memoryEvents=memoryEvents,
            trainingBudget=self.makeBudgetSummary(),
        )

    def makeTorchTrainingSplit(
        self,
        featureMatrix: np.ndarray,
        labels: np.ndarray | None,
        historicalData: np.ndarray,
        trainCount: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        validationWindow = max(8, int(trainCount * 0.2))
        validationStart = min(trainCount, max(len(featureMatrix) - 1, 0))
        validationEnd = min(len(featureMatrix), validationStart + validationWindow)
        if validationEnd - validationStart >= 2:
            validationLabels = labels[validationStart:validationEnd] if labels is not None else None
            return historicalData, featureMatrix[validationStart:validationEnd], validationLabels

        if len(historicalData) >= 4:
            validationCount = max(1, min(len(historicalData) // 5, len(historicalData) - 1))
            return historicalData[:-validationCount], historicalData[-validationCount:], None

        return historicalData, historicalData, None

    def hasUsableValidationLabels(self, validationLabels: np.ndarray | None) -> bool:
        if validationLabels is None or len(validationLabels) < 2:
            return False
        return len(np.unique(validationLabels.astype(int))) == 2

    def isBetterValidationScore(self, score: float, bestScore: float | None, maximize: bool) -> bool:
        if bestScore is None:
            return True
        if maximize:
            return score > bestScore + VALIDATION_EPSILON
        return score < bestScore - VALIDATION_EPSILON

    def evaluateReconstructionValidation(
        self,
        reconstruction,
        validationTensor,
        validationLabels: np.ndarray | None,
    ) -> tuple[float, str, bool]:
        if self.hasUsableValidationLabels(validationLabels):
            validationScores = torch.mean(torch.abs(validationTensor - reconstruction), dim=1).detach().cpu().numpy()
            validationLabels = validationLabels.astype(int)
            aucPrScore = float(aucPr(validationLabels, validationScores))
            aucRocScore = float(aucRoc(validationLabels, validationScores))
            return aucPrScore, f"validation AUCPR (AUCROC {aucRocScore:.6f})", True
        validationLoss = float(torchFunction.mse_loss(reconstruction, validationTensor).detach().cpu().item())
        return validationLoss, "validation reconstruction loss", False

    def trainStaticDetector(
        self,
        scd,
        trainTensor,
        validationTensor,
        validationLabels: np.ndarray | None,
        logs: list[TrainingLog],
    ) -> TrainingState:
        budget = self.trainingBudget()
        weightDecay = float(self.config.get("mannWeightDecay", 1e-5)) if self.modelName == "EnhancedDymeterMANN" else 0.0
        optimizer = torch.optim.Adam(scd.parameters(), lr=budget.learningRate, weight_decay=weightDecay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=budget.decayFactor)
        bestState = copy.deepcopy(scd.state_dict())
        bestScore: float | None = None
        bestEpoch = 0
        stoppedEpoch = budget.maxEpochs
        staleEpochs = 0
        validationMetric = "validation reconstruction loss"
        checkpointSource = "validation"

        if self.modelName == "EnhancedDymeterMANN":
            logs.append(
                TrainingLog(
                    "Self-supervised initialization",
                    "The MANN variant denoises masked historical rows to reduce reliance on perfectly clean initial data.",
                    f"Weight decay: {weightDecay:g}.",
                )
            )
        logs.append(TrainingLog("Training Autoencoder", "The SCD learns to rebuild normal rows.", f"Budget: {budget.maxEpochs} epochs."))
        for epochs in range(1, budget.maxEpochs + 1):
            optimizer.zero_grad()
            if self.modelName == "EnhancedDymeterMANN":
                noiseScale = float(self.config.get("selfSupervisedNoise", 0.01))
                maskRate = float(self.config.get("selfSupervisedMaskRate", 0.03))
                mask = (torch.rand_like(trainTensor) > maskRate).float()
                noisyTensor = trainTensor * mask + noiseScale * torch.randn_like(trainTensor)
                reconstruction = scd(noisyTensor)
                cleanReconstruction = scd(trainTensor)
                denoiseLoss = torchFunction.mse_loss(reconstruction, trainTensor)
                cleanLoss = torchFunction.mse_loss(cleanReconstruction, trainTensor)
                consistencyLoss = torchFunction.mse_loss(reconstruction, cleanReconstruction.detach())
                loss = denoiseLoss + 0.5 * cleanLoss + 0.1 * consistencyLoss
            else:
                reconstruction = scd(trainTensor)
                # Eq. 4: reconstruction loss for the Static Concept-aware Detector.
                loss = torchFunction.mse_loss(reconstruction, trainTensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                validationReconstruction = scd(validationTensor)
                score, validationMetric, maximize = self.evaluateReconstructionValidation(validationReconstruction, validationTensor, validationLabels)

            if self.isBetterValidationScore(score, bestScore, maximize):
                bestScore = score
                bestEpoch = epochs
                bestState = copy.deepcopy(scd.state_dict())
                staleEpochs = 0
            elif budget.earlyStoppingPatience is not None:
                staleEpochs += 1
                if staleEpochs >= budget.earlyStoppingPatience:
                    stoppedEpoch = epochs
                    self.printTrainingProgress("Autoencoder", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore, force=True)
                    break
            self.printTrainingProgress("Autoencoder", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore)

        if budget.earlyStoppingPatience is not None:
            scd.load_state_dict(bestState)
        else:
            bestEpoch = stoppedEpoch
            bestScore = score
            checkpointSource = "final epoch"

        return TrainingState(
            stage="Training Autoencoder",
            stoppedEpoch=stoppedEpoch,
            bestEpoch=bestEpoch,
            validationMetric=validationMetric,
            bestValidationScore=float(bestScore if bestScore is not None else 0.0),
            checkpointSource=checkpointSource,
        )

    def trainEvolutionController(
        self,
        iec,
        trainTensor,
        validationTensor,
        validationLabels: np.ndarray | None,
        pseudoLabels,
        validationPseudoLabels,
        logs: list[TrainingLog],
    ) -> TrainingState:
        budget = self.trainingBudget()
        optimizer = torch.optim.Adam(iec.parameters(), lr=budget.learningRate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=budget.decayFactor)
        bestState = copy.deepcopy(iec.state_dict())
        bestScore: float | None = None
        bestEpoch = 0
        stoppedEpoch = budget.maxEpochs
        staleEpochs = 0
        validationMetric = "validation IEC loss"
        checkpointSource = "validation"
        useLabelMetric = self.hasUsableValidationLabels(validationLabels)

        logs.append(TrainingLog("Training IEC", "The controller learns when a row belongs to a changed concept.", "Pseudo labels created from SCD."))
        for epochs in range(1, budget.maxEpochs + 1):
            optimizer.zero_grad()
            alpha = iec(trainTensor)
            probabilities = alpha / alpha.sum(dim=1, keepdim=True)
            # Eq. 7: focal-style controller loss for imbalanced pseudo labels.
            ceLoss = torchFunction.nll_loss(torch.log(probabilities + 1e-8), pseudoLabels)
            focalWeight = torch.pow(1.0 - probabilities[torch.arange(len(pseudoLabels), device=trainTensor.device), pseudoLabels], 2.0).mean()
            loss = ceLoss * focalWeight
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                validationAlpha = iec(validationTensor)
                validationProbabilities = validationAlpha / validationAlpha.sum(dim=1, keepdim=True)
                if useLabelMetric:
                    controllerScores = validationProbabilities[:, 1].detach().cpu().numpy()
                    labels = validationLabels.astype(int)
                    aucPrScore = float(aucPr(labels, controllerScores))
                    aucRocScore = float(aucRoc(labels, controllerScores))
                    score = aucPrScore
                    validationMetric = f"validation IEC AUCPR (AUCROC {aucRocScore:.6f})"
                    maximize = True
                else:
                    validationLoss = torchFunction.nll_loss(torch.log(validationProbabilities + 1e-8), validationPseudoLabels)
                    score = float(validationLoss.detach().cpu().item())
                    maximize = False

            if self.isBetterValidationScore(score, bestScore, maximize):
                bestScore = score
                bestEpoch = epochs
                bestState = copy.deepcopy(iec.state_dict())
                staleEpochs = 0
            elif budget.earlyStoppingPatience is not None:
                staleEpochs += 1
                if staleEpochs >= budget.earlyStoppingPatience:
                    stoppedEpoch = epochs
                    self.printTrainingProgress("IEC", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore, force=True)
                    break
            self.printTrainingProgress("IEC", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore)

        if budget.earlyStoppingPatience is not None:
            iec.load_state_dict(bestState)
        else:
            bestEpoch = stoppedEpoch
            bestScore = score
            checkpointSource = "final epoch"

        return TrainingState(
            stage="Training IEC",
            stoppedEpoch=stoppedEpoch,
            bestEpoch=bestEpoch,
            validationMetric=validationMetric,
            bestValidationScore=float(bestScore if bestScore is not None else 0.0),
            checkpointSource=checkpointSource,
        )

    def trainDynamicShiftDetector(
        self,
        scd,
        hyperNetwork,
        trainTensor,
        validationTensor,
        validationLabels: np.ndarray | None,
        pseudoLabels,
        logs: list[TrainingLog],
    ) -> TrainingState:
        budget = self.trainingBudget()
        for parameter in scd.parameters():
            parameter.requires_grad = False

        weightDecay = float(self.config.get("mannWeightDecay", 1e-5)) if self.modelName == "EnhancedDymeterMANN" else 0.0
        optimizer = torch.optim.Adam(hyperNetwork.parameters(), lr=budget.learningRate, weight_decay=weightDecay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=budget.decayFactor)
        bestState = copy.deepcopy(hyperNetwork.state_dict())
        bestScore: float | None = None
        bestEpoch = 0
        stoppedEpoch = budget.maxEpochs
        staleEpochs = 0
        validationMetric = "validation DSD reconstruction loss"
        checkpointSource = "validation"

        normalWeight = torch.where(pseudoLabels > 0, torch.tensor(1.0, device=trainTensor.device), torch.tensor(0.25, device=trainTensor.device))
        logs.append(
            TrainingLog(
                "Training Dynamic Detector",
                "The DSD learns instance-aware reconstruction shifts for changed concepts.",
                "Frozen SCD plus trained parameter-shift hypernetwork.",
            )
        )

        for epochs in range(1, budget.maxEpochs + 1):
            optimizer.zero_grad()
            staticReconstruction = scd(trainTensor)
            shift = hyperNetwork(trainTensor)
            reconstruction = staticReconstruction + shift
            perRowLoss = torch.mean(torch.square(reconstruction - trainTensor), dim=1)
            shiftRegularizer = 0.01 * torch.mean(torch.square(shift))
            loss = torch.mean(perRowLoss * normalWeight) + shiftRegularizer
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                validationReconstruction = scd(validationTensor) + hyperNetwork(validationTensor)
                score, validationMetric, maximize = self.evaluateReconstructionValidation(validationReconstruction, validationTensor, validationLabels)

            if self.isBetterValidationScore(score, bestScore, maximize):
                bestScore = score
                bestEpoch = epochs
                bestState = copy.deepcopy(hyperNetwork.state_dict())
                staleEpochs = 0
            elif budget.earlyStoppingPatience is not None:
                staleEpochs += 1
                if staleEpochs >= budget.earlyStoppingPatience:
                    stoppedEpoch = epochs
                    self.printTrainingProgress("Dynamic Detector", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore, force=True)
                    break
            self.printTrainingProgress("Dynamic Detector", epochs, budget.maxEpochs, float(loss.detach().cpu().item()), validationMetric, float(score), bestEpoch, bestScore)

        hyperNetwork.load_state_dict(bestState)
        return TrainingState(
            stage="Training Dynamic Detector",
            stoppedEpoch=stoppedEpoch,
            bestEpoch=bestEpoch,
            validationMetric=validationMetric,
            bestValidationScore=float(bestScore if bestScore is not None else 0.0),
            checkpointSource=checkpointSource,
        )

    def runTorch(
        self,
        featureMatrix: np.ndarray,
        labels: np.ndarray | None,
        historicalData: np.ndarray,
        trainCount: int,
        latentDimension: int,
        preprocessingSummary: dict,
        logs: list[TrainingLog],
        startTraining: float,
    ) -> DetectorOutput:
        inputDimension = featureMatrix.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("device", "cpu") == "cuda" else "cpu")
        scd = StaticConceptAwareDetector(inputDimension, latentDimension).to(device)
        iec = IntelligentEvolutionController(inputDimension).to(device)
        hyperNetwork = ParameterShiftHypernetwork(inputDimension).to(device)

        trainingData, validationData, validationLabels = self.makeTorchTrainingSplit(featureMatrix, labels, historicalData, trainCount)
        trainTensor = torch.tensor(trainingData, dtype=torch.float32, device=device)
        validationTensor = torch.tensor(validationData, dtype=torch.float32, device=device)
        trainingStates: list[TrainingState] = []

        staticState = self.trainStaticDetector(scd, trainTensor, validationTensor, validationLabels, logs)
        trainingStates.append(staticState)
        self.appendBudgetStopLog(logs, staticState)

        with torch.no_grad():
            historicalReconstruction = scd(trainTensor)
            trainScores = torch.mean(torch.abs(trainTensor - historicalReconstruction), dim=1)
            pseudoLabelThresh = torch.quantile(trainScores, 1.0 - float(self.config.get("pseudoLabelRate", 0.05)))
            # Eq. 5: pseudo labels are generated from high reconstruction errors.
            pseudoLabels = (trainScores > pseudoLabelThresh).long()
            validationReconstruction = scd(validationTensor)
            validationScores = torch.mean(torch.abs(validationTensor - validationReconstruction), dim=1)
            validationPseudoLabels = (validationScores > pseudoLabelThresh).long()

        iecState = self.trainEvolutionController(iec, trainTensor, validationTensor, validationLabels, pseudoLabels, validationPseudoLabels, logs)
        trainingStates.append(iecState)
        self.appendBudgetStopLog(logs, iecState)

        if self.usesDynamicShiftDetector():
            dsdState = self.trainDynamicShiftDetector(scd, hyperNetwork, trainTensor, validationTensor, validationLabels, pseudoLabels, logs)
            trainingStates.append(dsdState)
            self.appendBudgetStopLog(logs, dsdState)

        trainingSeconds = time.time() - startTraining
        startInference = time.time()
        featureTensor = torch.tensor(featureMatrix, dtype=torch.float32, device=device)
        with torch.no_grad():
            staticReconstruction = scd(featureTensor)
            alpha = iec(featureTensor)
            totalEvidence = alpha.sum(dim=1, keepdim=True)
            probabilities = alpha / totalEvidence
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            dataUncertainty = torch.sum(probabilities * (torch.digamma(totalEvidence + 1) - torch.digamma(alpha + 1)), dim=1)
            # Eq. 8: concept uncertainty is distributional uncertainty in Dirichlet space.
            conceptUncertaintyTensor = entropy - dataUncertainty
            conceptUncertainty = conceptUncertaintyTensor.detach().cpu().numpy()
            reconstructions = staticReconstruction.detach().cpu().numpy()

            if self.usesDynamicShiftDetector():
                shift = hyperNetwork(featureTensor)
                dynamicReconstruction = staticReconstruction + shift
                uncertaintyThresh = float(self.config.get("uncertaintyThreshold", 0.05))
                if self.modelName == "EnhancedDymeterMANN" and "uncertaintyThreshold" not in self.config:
                    historicalUncertainty = conceptUncertainty[: max(8, trainCount)]
                    uncertaintyThresh = max(0.08, float(np.quantile(historicalUncertainty, 0.9)))
                dynamicRows = conceptUncertaintyTensor > uncertaintyThresh
                proactiveRows = np.zeros(len(featureMatrix), dtype=bool)
                if self.modelName == "EnhancedDymeterMANN":
                    proactiveRows = self.proactiveDynamicMask(conceptUncertainty, uncertaintyThresh)
                    proactiveTensor = torch.tensor(proactiveRows, dtype=torch.bool, device=device)
                    dynamicRows = dynamicRows | proactiveTensor
                reconstructions[dynamicRows.detach().cpu().numpy()] = dynamicReconstruction[dynamicRows].detach().cpu().numpy()
                dynamicMaskArray = dynamicRows.detach().cpu().numpy()
                detectorUsed = []
                for index, value in enumerate(dynamicMaskArray):
                    if not value:
                        detectorUsed.append("static")
                    elif proactiveRows[index] and conceptUncertainty[index] <= uncertaintyThresh:
                        detectorUsed.append("proactive")
                    else:
                        detectorUsed.append("dynamic")
            else:
                dynamicReconstruction = staticReconstruction
                dynamicRows = torch.zeros(len(featureMatrix), dtype=torch.bool, device=device)
                detectorUsed = ["static"] * len(featureMatrix)

        scores = np.mean(np.abs(featureMatrix - reconstructions), axis=1)
        memoryEvents: list[dict] = []
        if self.usesConceptMemory():
            memoryBank = ConceptMemory(
                maxCells=int(self.config.get("memoryCells", 16)),
                temperature=float(self.config.get("memoryTemperature", 0.15)),
                updateRate=float(self.config.get("memoryUpdateRate", 0.2)),
            )
            memoryMatchThreshold = float(self.config.get("memoryMatchThreshold", 0.82))
            memoryWindow = max(4, min(int(preprocessingSummary.get("windowSize", 64)), len(featureMatrix)))
            baseMemoryThresh = float(np.quantile(scores[: max(4, len(historicalData))], 1.0 - float(self.config.get("falsePositiveRate", 0.05))))
            dynamicMask = dynamicRows.detach().cpu().numpy()

            for rowIndex in range(len(featureMatrix)):
                if not dynamicMask[rowIndex]:
                    continue

                conceptStart = max(0, rowIndex - memoryWindow + 1)
                conceptVector = featureMatrix[conceptStart : rowIndex + 1].mean(axis=0)
                rememberedCell, similarity = memoryBank.read(conceptVector)
                if rememberedCell is not None and similarity >= memoryMatchThreshold:
                    detectorUsed[rowIndex] = "memory"
                    memoryValue = rememberedCell.get("memoryValue")
                    if memoryValue is not None:
                        rowReconstruction = dynamicReconstruction[rowIndex].detach().cpu().numpy() + memoryValue
                        reconstructions[rowIndex] = rowReconstruction
                        scores[rowIndex] = float(np.mean(np.abs(featureMatrix[rowIndex] - rowReconstruction)))
                    memoryEvents.append(
                        {
                            "type": "read",
                            "rowIndex": rowIndex,
                            "similarity": round(float(similarity), 4),
                            "retrievedRowIndex": rememberedCell["rowIndex"],
                        }
                    )
                    residual = featureMatrix[rowIndex] - dynamicReconstruction[rowIndex].detach().cpu().numpy()
                    if scores[rowIndex] <= max(float(rememberedCell["threshold"]) * 1.25, 1e-8):
                        memoryEvents.append(memoryBank.update(rememberedCell, conceptVector, baseMemoryThresh, scores[rowIndex], residual))
                    continue

                residualWindow = featureMatrix[conceptStart : rowIndex + 1] - dynamicReconstruction[conceptStart : rowIndex + 1].detach().cpu().numpy()
                memoryValue = residualWindow.mean(axis=0)
                memoryEvents.append(memoryBank.write(conceptVector, baseMemoryThresh, rowIndex, memoryValue, scores[rowIndex]))

        threshOptimizer = DynamicThresholdOptimizer(
            falsePositiveRate=float(self.config.get("falsePositiveRate", 0.05)),
            windowSize=int(preprocessingSummary.get("windowSize", 64)),
        )
        baseThresh = threshOptimizer.initialize(scores[: len(historicalData)])
        threshValues = np.array([threshOptimizer.update(float(score), float(conceptUncertainty[index]), detectorUsed[index] != "static") for index, score in enumerate(scores)])
        if self.modelName == "METER":
            threshValues[:] = baseThresh

        driftEvents = self.makeDriftEvents(detectorUsed)
        parameterCount = sum(parameter.numel() for parameter in scd.parameters())
        parameterCount += sum(parameter.numel() for parameter in iec.parameters())
        if self.usesDynamicShiftDetector():
            parameterCount += sum(parameter.numel() for parameter in hyperNetwork.parameters())

        logs.append(TrainingLog("Adaptive route activated", "Rows with high or forecast-rising concept uncertainty used adaptive reconstruction.", f"{detectorUsed.count('dynamic')} dynamic rows; {detectorUsed.count('proactive')} proactive rows."))
        if self.usesConceptMemory():
            logs.append(TrainingLog("MANN concept memory routed", "Recurring concepts used stored memory values; new concepts were written to memory.", f"{len(memoryEvents)} memory events."))
        logs.append(TrainingLog("Threshold updated", "The anomaly boundary was recalibrated online.", f"Final threshold: {threshValues[-1]:.4f}."))

        return DetectorOutput(
            scores=scores,
            reconstructions=reconstructions,
            conceptUncertainty=conceptUncertainty,
            detectorUsed=detectorUsed,
            thresholds=threshValues,
            driftEvents=driftEvents,
            trainingSeconds=trainingSeconds,
            inferenceSeconds=time.time() - startInference,
            parameterCount=int(parameterCount),
            memoryEvents=memoryEvents,
            trainingBudget=self.makeBudgetSummary(trainingStates),
        )

    def makeDriftEvents(self, detectorUsed: list[str]) -> list[dict]:
        events: list[dict] = []
        for index, detectorName in enumerate(detectorUsed):
            if detectorName not in {"dynamic", "memory", "proactive"}:
                continue
            description = "Predictive uncertainty dynamics anticipated drift." if detectorName == "proactive" else "Concept uncertainty passed the IEC threshold."
            if not events or events[-1]["rowIndexRange"][1] != index - 1 or events[-1]["description"] != description:
                events.append({"rowIndexRange": [index, index], "description": description})
            else:
                events[-1]["rowIndexRange"][1] = index
        return events
