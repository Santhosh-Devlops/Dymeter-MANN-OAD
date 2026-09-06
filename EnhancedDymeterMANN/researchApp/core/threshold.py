from __future__ import annotations

from collections import deque

import numpy as np


class DynamicThresholdOptimizer:
    """Dynamic threshold optimization from DyMETER Section III-E."""

    def __init__(self, falsePositiveRate: float = 0.05, windowSize: int = 64, kappa: float = 0.5):
        self.falsePositiveRate = falsePositiveRate
        self.windowSize = windowSize
        self.kappa = kappa
        self.normalScores: deque[float] = deque(maxlen=windowSize)
        self.candidateScores: deque[float] = deque(maxlen=windowSize)
        self.currentThresh = 0.0

    def initialize(self, scores: np.ndarray) -> float:
        cleanScores = np.asarray(scores, dtype=float)
        cleanScores = cleanScores[np.isfinite(cleanScores)]
        if len(cleanScores) == 0:
            self.currentThresh = 1.0
            return self.currentThresh

        for score in cleanScores[-self.windowSize :]:
            self.normalScores.append(float(score))

        # Eq. 17: quantile threshold from recent normal score distribution.
        self.currentThresh = float(np.quantile(cleanScores, 1.0 - self.falsePositiveRate))
        return self.currentThresh

    def update(self, score: float, conceptUncertainty: float, driftDetected: bool) -> float:
        if driftDetected:
            self.normalScores.clear()
            self.candidateScores.clear()

        if conceptUncertainty < 0.5:
            self.normalScores.append(float(score))
        else:
            self.candidateScores.append(float(score))

        if len(self.normalScores) < 4:
            return self.currentThresh

        normalArray = np.array(self.normalScores, dtype=float)
        baseThresh = float(np.quantile(normalArray, 1.0 - self.falsePositiveRate))

        # Eq. 18: median candidate score acts as a robust regularizer.
        if self.candidateScores:
            candidateMedian = float(np.median(np.array(self.candidateScores, dtype=float)))
        else:
            candidateMedian = float(np.median(normalArray))

        # Eq. 19: final threshold combines quantile and uncertainty-aware regularization.
        threshRegularization = self.kappa * max(baseThresh - candidateMedian, 0.0)
        self.currentThresh = max(baseThresh + threshRegularization, 1e-8)
        return self.currentThresh
