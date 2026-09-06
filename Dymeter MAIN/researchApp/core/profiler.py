from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DatasetProfile, TrainingLog


class DatasetProfiler:
    """Detects the structure of an uploaded dataset before model training."""

    labelHints = {
        "label",
        "labels",
        "target",
        "class",
        "classes",
        "is_anomaly",
        "is_anomalous",
        "is_outlier",
        "ground_truth",
        "groundtruth",
        "truth",
        "actual",
        "anomaly",
        "anomalies",
        "attack",
        "attacks",
        "outlier",
        "outliers",
        "outlier_label",
        "anomaly_label",
        "y",
    }

    timestampHints = {"time", "timestamp", "date", "datetime", "created", "eventtime"}

    def readDataset(self, datasetPath: str | Path) -> pd.DataFrame:
        datasetPath = Path(datasetPath)
        suffix = datasetPath.suffix.lower()

        if suffix in {".csv", ".txt", ".log"}:
            try:
                return pd.read_csv(datasetPath)
            except Exception:
                return pd.read_csv(datasetPath, sep=None, engine="python")

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(datasetPath)

        if suffix == ".mat":
            try:
                import scipy.io
            except Exception as importError:
                raise RuntimeError("MAT files need scipy. Run `pip install -r requirements.txt`.") from importError
            matData = scipy.io.loadmat(datasetPath)
            if "X" not in matData:
                raise ValueError("MAT benchmark files must contain an X matrix.")
            frame = pd.DataFrame(matData["X"])
            if "y" in matData:
                frame["label"] = matData["y"].reshape(-1)
            return frame

        raise ValueError(f"Unsupported dataset file: {suffix}. Please upload CSV, TXT, LOG, MAT, XLS, or XLSX.")

    def buildProfile(self, dataFrame: pd.DataFrame) -> tuple[DatasetProfile, list[TrainingLog]]:
        logs = [TrainingLog("Reading columns", "Columns were inspected.", f"{len(dataFrame.columns)} columns found.")]

        numericColumns: list[str] = []
        categoricalColumns: list[str] = []
        timestampColumns: list[str] = []
        idColumns: list[str] = []
        constantColumns: list[str] = []
        missingValueRatio: dict[str, float] = {}
        labelColumn: str | None = None

        rowCount = len(dataFrame)
        for columnName in dataFrame.columns:
            series = dataFrame[columnName]
            cleanName = self.normalizeColumnName(columnName)
            missingValueRatio[str(columnName)] = float(series.isna().mean())

            if labelColumn is None and self.nameLooksLikeLabel(cleanName):
                labelColumn = str(columnName)
                continue

            if series.nunique(dropna=True) <= 1:
                constantColumns.append(str(columnName))

            if self.looksLikeTimestamp(cleanName, series):
                timestampColumns.append(str(columnName))
                continue

            if pd.api.types.is_numeric_dtype(series):
                numericColumns.append(str(columnName))
            else:
                categoricalColumns.append(str(columnName))

            uniqueRatio = series.nunique(dropna=True) / max(rowCount, 1)
            nameLooksLikeId = cleanName in {"id", "rowid", "recordid", "uuid"} or cleanName.endswith("_id")
            textLooksLikeId = not pd.api.types.is_numeric_dtype(series) and uniqueRatio > 0.85 and series.nunique(dropna=True) > 20
            if nameLooksLikeId or textLooksLikeId:
                idColumns.append(str(columnName))

        if labelColumn is None:
            labelColumn = self.inferBenchmarkLabelColumn(dataFrame)
            if labelColumn is not None:
                numericColumns = [column for column in numericColumns if column != labelColumn]
                categoricalColumns = [column for column in categoricalColumns if column != labelColumn]
                timestampColumns = [column for column in timestampColumns if column != labelColumn]
                idColumns = [column for column in idColumns if column != labelColumn]
                constantColumns = [column for column in constantColumns if column != labelColumn]

        suspectedDataType = self.detectDatasetType(
            dataFrame,
            numericColumns,
            categoricalColumns,
            timestampColumns,
            idColumns,
        )

        validationMessages = self.buildValidationMessages(
            rowCount,
            numericColumns,
            categoricalColumns,
            timestampColumns,
            labelColumn,
            missingValueRatio,
        )

        profile = DatasetProfile(
            columnCount=len(dataFrame.columns),
            rowCount=rowCount,
            numericColumns=numericColumns,
            categoricalColumns=categoricalColumns,
            timestampColumns=timestampColumns,
            idColumns=sorted(set(idColumns)),
            constantColumns=sorted(set(constantColumns)),
            missingValueRatio=missingValueRatio,
            labelColumn=labelColumn,
            suspectedDataType=suspectedDataType,
            validationMessages=validationMessages,
        )

        logs.extend(
            [
                TrainingLog("Checking missing values", "Missing cells were measured.", f"{len(validationMessages)} warnings."),
                TrainingLog("Finding categorical columns", "Text-like columns were separated from numbers.", f"{len(categoricalColumns)} categorical columns."),
                TrainingLog("Detecting labels", "A target column was searched for.", labelColumn or "No label column found."),
            ]
        )
        return profile, logs

    def looksLikeTimestamp(self, cleanName: str, series: pd.Series) -> bool:
        if any(hint in cleanName for hint in self.timestampHints):
            return True
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().astype(str).head(40)
            if len(sample) == 0:
                return False
            parsed = pd.to_datetime(sample, errors="coerce")
            return bool(parsed.notna().mean() > 0.8)
        return False

    def normalizeColumnName(self, columnName) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(columnName).strip().lower()).strip("_")

    def nameLooksLikeLabel(self, cleanName: str) -> bool:
        if cleanName in self.labelHints:
            return True
        tokens = set(cleanName.split("_"))
        if tokens & self.labelHints:
            return True
        return cleanName.endswith("_label") or cleanName.endswith("_target")

    def inferBenchmarkLabelColumn(self, dataFrame: pd.DataFrame) -> str | None:
        candidates: list[str] = []
        for columnName in dataFrame.columns:
            series = dataFrame[columnName].dropna()
            if len(series) == 0:
                continue
            uniqueCount = int(series.nunique(dropna=True))
            if uniqueCount < 2 or uniqueCount > max(20, min(100, len(series) // 4)):
                continue
            if pd.api.types.is_numeric_dtype(series):
                values = set(pd.to_numeric(series, errors="coerce").dropna().astype(float).unique().tolist())
                if len(values) <= 10 and (
                    values.issubset({0.0, 1.0})
                    or values.issubset({-1.0, 1.0})
                    or values.issubset({0.0, 1.0, 2.0})
                ):
                    candidates.append(str(columnName))
            else:
                lowered = set(series.astype(str).str.strip().str.lower().unique().tolist())
                knownLabelWords = {
                    "0",
                    "1",
                    "normal",
                    "nominal",
                    "benign",
                    "good",
                    "false",
                    "no",
                    "anomaly",
                    "anomalous",
                    "abnormal",
                    "attack",
                    "outlier",
                    "bad",
                    "true",
                    "yes",
                }
                if len(lowered) <= 10 and lowered.issubset(knownLabelWords):
                    candidates.append(str(columnName))
        if not candidates:
            return None
        lastColumn = str(dataFrame.columns[-1])
        if lastColumn in candidates:
            return lastColumn
        return candidates[-1]

    def detectDatasetType(
        self,
        dataFrame: pd.DataFrame,
        numericColumns: list[str],
        categoricalColumns: list[str],
        timestampColumns: list[str],
        idColumns: list[str],
    ) -> str:
        columnText = " ".join(str(column).lower() for column in dataFrame.columns)
        if re.search(r"log|message|event|level", columnText):
            return "log"
        if re.search(r"sensor|temperature|pressure|humidity|voltage|current", columnText):
            return "sensor"
        if timestampColumns:
            return "timeSeries"
        if len(idColumns) > 0 and len(numericColumns) + len(categoricalColumns) > 3:
            return "streaming"
        return "tabular"

    def buildValidationMessages(
        self,
        rowCount: int,
        numericColumns: list[str],
        categoricalColumns: list[str],
        timestampColumns: list[str],
        labelColumn: str | None,
        missingValueRatio: dict[str, float],
    ) -> list[str]:
        messages: list[str] = []
        if rowCount < 100:
            messages.append("This dataset has very few rows for reliable training. Results may be less accurate.")
        if not timestampColumns:
            messages.append("No timestamp column was found. Concept drift detection will use row order instead of real time.")
        if not labelColumn:
            messages.append("No label column was found. The app will show proxy F1 and proxy AUCROC from score separation.")
        if len(numericColumns) == 0 and len(categoricalColumns) > 0:
            messages.append("This dataset is all categorical. It will be encoded before anomaly detection.")

        highMissing = [name for name, ratio in missingValueRatio.items() if ratio > 0.4]
        if highMissing:
            messages.append("Some columns have many missing values and may be less useful: " + ", ".join(highMissing[:5]))
        return messages


def profileDataset(datasetPath: str | Path) -> tuple[pd.DataFrame, DatasetProfile, list[TrainingLog]]:
    profiler = DatasetProfiler()
    dataFrame = profiler.readDataset(datasetPath)
    profile, logs = profiler.buildProfile(dataFrame)
    return dataFrame, profile, logs
