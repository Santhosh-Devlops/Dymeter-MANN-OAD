from __future__ import annotations

import numpy as np


def binaryMetrics(labels: np.ndarray | None, scores: np.ndarray, predictions: np.ndarray) -> dict:
    if labels is None:
        proxyLabels = proxyLabelsFromScores(scores)
        proxyAucLabels = predictions.astype(int)
        if len(np.unique(proxyAucLabels)) < 2:
            proxyAucLabels = proxyLabels
        proxyPrecision, proxyRecall, proxyF1 = f1FromLabels(proxyLabels, predictions.astype(int))
        return {
            "accuracy": None,
            "precision": round(float(proxyPrecision), 4),
            "recall": round(float(proxyRecall), 4),
            "f1Score": round(float(proxyF1), 4),
            "aucRoc": round(float(aucRoc(proxyAucLabels, scores)), 4),
            "aucPr": round(float(aucPr(proxyAucLabels, scores)), 4),
            "metricMode": "unlabeledProxy",
            "metricNote": "No label column was found; F1, AUCROC, and AUCPR are proxy estimates from score and threshold separation.",
            "proxy": {
                "scoreMean": float(np.mean(scores)),
                "scoreStd": float(np.std(scores)),
                "scoreMax": float(np.max(scores)),
                "pseudoPositiveRows": int(np.sum(proxyLabels == 1)),
            },
        }

    labels = labels.astype(int)
    predictions = predictions.astype(int)
    truePositive, falsePositive, trueNegative, falseNegative = confusionCounts(labels, predictions)
    precision, recall, f1Score = f1FromLabels(labels, predictions)
    accuracy = (truePositive + trueNegative) / max(len(labels), 1)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1Score": round(float(f1Score), 4),
        "aucRoc": round(float(aucRoc(labels, scores)), 4),
        "aucPr": round(float(aucPr(labels, scores)), 4),
        "metricMode": "labelled",
        "metricNote": "Metrics were computed from the detected label column.",
        "confusionMatrix": {
            "truePositive": truePositive,
            "falsePositive": falsePositive,
            "trueNegative": trueNegative,
            "falseNegative": falseNegative,
        },
    }


def proxyLabelsFromScores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.zeros(0, dtype=int)
    threshold = np.quantile(scores, 0.95)
    labels = (scores >= threshold).astype(int)
    if len(np.unique(labels)) < 2 and len(scores) > 1:
        labels[np.argmax(scores)] = 1
    return labels


def confusionCounts(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    truePositive = int(np.sum((labels == 1) & (predictions == 1)))
    falsePositive = int(np.sum((labels == 0) & (predictions == 1)))
    trueNegative = int(np.sum((labels == 0) & (predictions == 0)))
    falseNegative = int(np.sum((labels == 1) & (predictions == 0)))
    return truePositive, falsePositive, trueNegative, falseNegative


def f1FromLabels(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    truePositive, falsePositive, _, falseNegative = confusionCounts(labels, predictions)
    precision = truePositive / max(truePositive + falsePositive, 1)
    recall = truePositive / max(truePositive + falseNegative, 1)
    f1Score = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1Score


def aucRoc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positiveScores = scores[labels == 1]
    negativeScores = scores[labels == 0]
    if len(positiveScores) == 0 or len(negativeScores) == 0:
        return 0.0
    totalPairs = len(positiveScores) * len(negativeScores)
    wins = 0.0
    for score in positiveScores:
        wins += np.sum(score > negativeScores)
        wins += 0.5 * np.sum(score == negativeScores)
    return wins / totalPairs


def aucPr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    order = np.argsort(-scores)
    sortedLabels = labels[order]
    positives = max(int(np.sum(labels == 1)), 1)
    truePositive = np.cumsum(sortedLabels == 1)
    falsePositive = np.cumsum(sortedLabels == 0)
    precision = truePositive / np.maximum(truePositive + falsePositive, 1)
    recall = truePositive / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))
