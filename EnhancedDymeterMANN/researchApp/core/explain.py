from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureContribution


def computeContributions(originalVector: np.ndarray, reconstructedVector: np.ndarray, featureNames: list[str], topCount: int = 5) -> list[FeatureContribution]:
    differences = np.abs(originalVector - reconstructedVector)
    if differences.sum() <= 0:
        scores = np.zeros_like(differences)
    else:
        scores = differences / differences.sum()
    order = np.argsort(-scores)[:topCount]
    return [
        FeatureContribution(featureName=featureNames[index], contributionScore=round(float(scores[index]), 6))
        for index in order
    ]


def severityFromScore(score: float, threshold: float) -> tuple[str, float]:
    if score <= threshold:
        return "Normal", 0.9
    ratio = score / max(threshold, 1e-8)
    if ratio < 1.5:
        return "Warning", min(0.95, 0.55 + ratio / 5)
    if ratio < 2.5:
        return "Critical", min(0.98, 0.65 + ratio / 6)
    return "Emergency", 0.99


def explainAnomaly(rowData: dict, contributions: list[FeatureContribution], score: float, threshold: float) -> str:
    if not contributions:
        return f"This row was flagged because its anomaly score {score:.4f} was above the threshold {threshold:.4f}."

    featureTexts = []
    for contribution in contributions[:3]:
        displayName = contribution.featureName
        rawValue = rowData.get(displayName)
        if rawValue is None and "_" in displayName:
            possibleName = displayName.split("_")[0]
            if possibleName in rowData:
                displayName = possibleName
                rawValue = rowData.get(possibleName)
        if rawValue is None:
            rawValue = "encoded value"
        featureTexts.append(f"{displayName}={rawValue}")
    joinedText = ", ".join(featureTexts)
    return (
        f"This row was flagged because its reconstruction pattern was unusual. "
        f"The strongest signals were {joinedText}. "
        f"Its score was {score:.4f}, above the active threshold {threshold:.4f}."
    )


def frameRowToDict(dataFrame: pd.DataFrame, rowIndex: int) -> dict:
    row = dataFrame.iloc[rowIndex].to_dict()
    cleanRow = {}
    for key, value in row.items():
        if pd.isna(value):
            cleanRow[str(key)] = None
        elif isinstance(value, np.generic):
            cleanRow[str(key)] = value.item()
        else:
            cleanRow[str(key)] = value
    return cleanRow
