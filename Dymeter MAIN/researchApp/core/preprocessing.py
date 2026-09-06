from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import DatasetProfile, TrainingLog


@dataclass
class PreparedDataset:
    featureMatrix: np.ndarray
    originalFrame: pd.DataFrame
    labels: np.ndarray | None
    featureNames: list[str]
    preprocessingSummary: dict
    logs: list[TrainingLog]


class AdaptivePreprocessor:
    """Turns any supported dataset into a numeric matrix without fixed shapes."""

    def prepare(self, dataFrame: pd.DataFrame, profile: DatasetProfile) -> PreparedDataset:
        logs: list[TrainingLog] = []
        workingFrame = dataFrame.copy()

        labelValues = None
        if profile.labelColumn:
            labelValues = self.extractLabels(workingFrame[profile.labelColumn])
            workingFrame = workingFrame.drop(columns=[profile.labelColumn])

        dropColumns = sorted(set(profile.idColumns + profile.constantColumns + profile.timestampColumns))
        workingFrame = workingFrame.drop(columns=[column for column in dropColumns if column in workingFrame.columns])
        logs.append(TrainingLog("Dropping weak columns", "IDs, timestamps, and constant columns were removed from training.", ", ".join(dropColumns) or "No drops needed."))

        encodedParts: list[pd.DataFrame] = []
        scalingChoices: dict[str, str] = {}
        encodingChoices: dict[str, str] = {}

        for columnName in workingFrame.columns:
            series = workingFrame[columnName]
            if pd.api.types.is_numeric_dtype(series):
                cleanValues, scaleName = self.scaleNumeric(series)
                scalingChoices[str(columnName)] = scaleName
                encodedParts.append(pd.DataFrame({str(columnName): cleanValues}))
            else:
                encodedFrame, methodName = self.encodeCategorical(series, str(columnName))
                encodingChoices[str(columnName)] = methodName
                encodedParts.append(encodedFrame)

        if not encodedParts:
            raise ValueError("No usable feature columns remained after validation.")

        featureFrame = pd.concat(encodedParts, axis=1)
        featureFrame = featureFrame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        featureMatrix = featureFrame.to_numpy(dtype=np.float32)
        windowSize = self.chooseWindowSize(profile, featureMatrix)
        latentDimension = self.chooseLatentDimension(featureMatrix)

        logs.extend(
            [
                TrainingLog("Encoding labels", "Known labels were converted to 0 for normal and 1 for anomaly.", "No labels found." if labelValues is None else "Labels ready."),
                TrainingLog("Scaling values", "Each numeric column used a scale that fits its outlier level.", f"{len(scalingChoices)} numeric columns scaled."),
                TrainingLog("Selecting historical samples", "The first clean normal-looking records will initialize the static detector.", f"Window size: {windowSize}."),
            ]
        )

        return PreparedDataset(
            featureMatrix=featureMatrix,
            originalFrame=dataFrame,
            labels=labelValues,
            featureNames=[str(name) for name in featureFrame.columns],
            preprocessingSummary={
                "droppedColumns": dropColumns,
                "scalingMethod": scalingChoices,
                "encodingMethod": encodingChoices,
                "windowSize": windowSize,
                "latentDimension": latentDimension,
                "featureCount": int(featureMatrix.shape[1]),
            },
            logs=logs,
        )

    def extractLabels(self, series: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            values = series.fillna(0).to_numpy()
            uniqueValues = sorted(set(values.tolist()))
            if len(uniqueValues) <= 2:
                return (values == max(uniqueValues)).astype(int)
            return (values > np.median(values)).astype(int)

        cleanSeries = series.fillna("normal").astype(str).str.lower()
        anomalyWords = {"1", "true", "yes", "anomaly", "abnormal", "attack", "outlier", "fraud"}
        return cleanSeries.map(lambda value: 1 if value in anomalyWords else 0).to_numpy(dtype=int)

    def scaleNumeric(self, series: pd.Series) -> tuple[np.ndarray, str]:
        numericValues = pd.to_numeric(series, errors="coerce")
        medianValue = float(numericValues.median()) if numericValues.notna().any() else 0.0
        numericValues = numericValues.fillna(medianValue).to_numpy(dtype=np.float64)

        q1, q3 = np.quantile(numericValues, [0.25, 0.75])
        iqr = q3 - q1
        outlierRatio = float(np.mean((numericValues < q1 - 1.5 * iqr) | (numericValues > q3 + 1.5 * iqr))) if iqr > 0 else 0.0

        if outlierRatio > 0.05:
            center = np.median(numericValues)
            scale = iqr if iqr > 1e-8 else 1.0
            return ((numericValues - center) / scale).astype(np.float32), "robust"

        minimumValue = np.min(numericValues)
        maximumValue = np.max(numericValues)
        if minimumValue >= 0 and maximumValue <= 1:
            return numericValues.astype(np.float32), "minmax"

        meanValue = np.mean(numericValues)
        stdValue = np.std(numericValues)
        return ((numericValues - meanValue) / (stdValue + 1e-8)).astype(np.float32), "standard"

    def encodeCategorical(self, series: pd.Series, columnName: str) -> tuple[pd.DataFrame, str]:
        cleanSeries = series.fillna("missing").astype(str)
        cardinality = int(cleanSeries.nunique())

        if cardinality <= 12:
            encodedFrame = pd.get_dummies(cleanSeries, prefix=columnName, dtype=float)
            return encodedFrame, "one-hot"

        if cardinality <= 120:
            frequency = cleanSeries.value_counts(normalize=True)
            return pd.DataFrame({columnName + "_frequency": cleanSeries.map(frequency).astype(float)}), "frequency"

        bucketCount = min(64, max(8, int(math.sqrt(cardinality))))
        buckets = cleanSeries.map(lambda value: abs(hash(value)) % bucketCount)
        encodedFrame = pd.get_dummies(buckets, prefix=columnName + "_hash", dtype=float)
        return encodedFrame, "hashing"

    def chooseWindowSize(self, profile: DatasetProfile, featureMatrix: np.ndarray) -> int:
        rowCount = max(profile.rowCount, 1)
        baseWindow = max(16, min(256, int(math.sqrt(rowCount) * 3)))
        if profile.suspectedDataType in {"timeSeries", "sensor", "streaming"}:
            return min(rowCount, max(24, baseWindow))
        return min(rowCount, baseWindow)

    def chooseLatentDimension(self, featureMatrix: np.ndarray) -> int:
        featureCount = featureMatrix.shape[1]
        if featureCount <= 2:
            return 1

        centeredMatrix = featureMatrix - featureMatrix.mean(axis=0, keepdims=True)
        try:
            _, singularValues, _ = np.linalg.svd(centeredMatrix, full_matrices=False)
            explained = singularValues**2
            if explained.sum() <= 0:
                return max(1, featureCount // 2)
            ratio = np.cumsum(explained) / explained.sum()
            # Section IV-D: choose the smallest latent size with at least 70% explained variance.
            return int(np.searchsorted(ratio, 0.70) + 1)
        except np.linalg.LinAlgError:
            return max(1, min(featureCount - 1, int(featureCount * 0.7)))

