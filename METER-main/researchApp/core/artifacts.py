from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS = (1600, 1000)
BACKGROUND = "#101820"
PANEL = "#182430"
PANEL_ALT = "#203140"
TEXT = "#f6f7f9"
MUTED = "#c9d2dc"
GRID = "#375061"
BLUE = "#52a9ff"
GREEN = "#5fd18c"
YELLOW = "#f1c75b"
RED = "#ff6b7a"
PURPLE = "#b88cff"


def buildRunArtifacts(
    outputDirectory: Path,
    runId: str,
    datasetName: str,
    modelName: str,
    profile,
    preprocessing: dict[str, Any],
    metrics: dict[str, Any],
    anomalyResult,
    detectorOutput,
    comparison: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    artifactRoot = outputDirectory / "artifacts" / runId
    imageDirectory = artifactRoot / "images"
    imageDirectory.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(detectorOutput.scores, dtype=float)
    thresholds = np.asarray(detectorOutput.thresholds, dtype=float)
    uncertainty = np.asarray(detectorOutput.conceptUncertainty, dtype=float)
    predictions = scores > thresholds
    anomalies = anomalyResult.anomalies

    imageFiles = [
        ("images/01_processing_path.png", drawProcessingPath(imageDirectory / "01_processing_path.png", modelName)),
        ("images/02_dataset_profile.png", drawDatasetProfile(imageDirectory / "02_dataset_profile.png", profile)),
        ("images/03_score_threshold_timeline.png", drawScoreThresholdTimeline(imageDirectory / "03_score_threshold_timeline.png", scores, thresholds, predictions)),
        ("images/04_uncertainty_drift_timeline.png", drawUncertaintyTimeline(imageDirectory / "04_uncertainty_drift_timeline.png", uncertainty, detectorOutput.detectorUsed)),
        ("images/05_severity_distribution.png", drawSeverityDistribution(imageDirectory / "05_severity_distribution.png", anomalyResult.totalRows, anomalies)),
        ("images/06_top_feature_contributions.png", drawFeatureContributions(imageDirectory / "06_top_feature_contributions.png", anomalies)),
        ("images/07_model_performance.png", drawPerformanceSummary(imageDirectory / "07_model_performance.png", metrics, comparison, modelName)),
    ]

    summaryPath = artifactRoot / "run_summary.json"
    with summaryPath.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "runId": runId,
                "datasetName": datasetName,
                "modelName": displayModelName(modelName),
                "profile": profile.__dict__,
                "preprocessing": preprocessing,
                "metrics": metrics,
                "anomalyResult": anomalyResult.toDict(),
                "artifactImages": [name for name, _ in imageFiles],
            },
            file,
            indent=2,
            default=str,
        )

    reportPath = artifactRoot / "enhanced_dymeter_report.pdf"
    reportCreated = writePdfReport(reportPath, datasetName, modelName, metrics, anomalyResult, imageDirectory)
    if not reportCreated:
        reportPath = artifactRoot / "enhanced_dymeter_report.html"
        writeHtmlReport(reportPath, datasetName, modelName, metrics, anomalyResult, [name for name, _ in imageFiles])

    zipPath = artifactRoot / "enhanced_dymeter_images.zip"
    with zipfile.ZipFile(zipPath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relativeName, absolutePath in imageFiles:
            archive.write(absolutePath, relativeName)

    artifacts = [
        {"name": "Complete image folder", "kind": "zip", "path": zipPath.relative_to(artifactRoot).as_posix(), "downloadName": zipPath.name},
        {"name": "Execution report", "kind": reportPath.suffix.lstrip(".").lower(), "path": reportPath.relative_to(artifactRoot).as_posix(), "downloadName": reportPath.name},
        {"name": "Run summary data", "kind": "json", "path": summaryPath.relative_to(artifactRoot).as_posix(), "downloadName": summaryPath.name},
    ]
    artifacts.extend(
        {"name": imageTitle(relativeName), "kind": "png", "path": relativeName, "downloadName": Path(relativeName).name}
        for relativeName, _ in imageFiles
    )
    return artifacts


def displayModelName(modelName: str) -> str:
    if modelName == "EnhancedDymeterMANN":
        return "Enhanced Dymeter MANN"
    return modelName


def canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, CANVAS[0] - 40, CANVAS[1] - 40), fill=PANEL, outline=GRID, width=2)
    draw.text((80, 70), title, fill=TEXT, font=font(48, bold=True))
    if subtitle:
        draw.text((82, 132), subtitle, fill=MUTED, font=font(25))
    return image, draw


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesnewroman.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save(image: Image.Image, path: Path) -> Path:
    image.save(path, quality=95)
    return path


def drawProcessingPath(path: Path, modelName: str) -> Path:
    image, draw = canvas("Enhanced Dymeter execution path", "Every block is executed after the dataset is submitted.")
    steps = [
        ("Streaming input", "multi-sensor rows, logs, or tabular values", BLUE),
        ("Preprocessing", "cleaning, sliding window sizing, feature extraction", BLUE),
        ("Static detector", "normal pattern reconstruction", GREEN),
        ("Evolution controller", "change detection and uncertainty estimate", GREEN),
        ("Concept memory", "read similar states or write a new state", YELLOW),
        ("Light adapter", "small fast adjustment for new behavior", PURPLE),
        ("Dynamic threshold", "adaptive warning boundary", BLUE),
        ("Explanation and severity", "responsible features and risk level", RED),
        ("Final decision", "normal, changed behavior, or anomaly", GREEN),
    ]
    x, y = 130, 210
    boxWidth, boxHeight = 620, 92
    for index, (name, detail, color) in enumerate(steps):
        column = index % 2
        row = index // 2
        left = x + column * 760
        top = y + row * 145
        draw.rounded_rectangle((left, top, left + boxWidth, top + boxHeight), radius=10, fill=PANEL_ALT, outline=color, width=4)
        draw.text((left + 28, top + 18), f"{index + 1}. {name}", fill=TEXT, font=font(28, bold=True))
        draw.text((left + 28, top + 54), detail, fill=MUTED, font=font(22))
        if index < len(steps) - 1:
            nextColumn = (index + 1) % 2
            nextRow = (index + 1) // 2
            start = (left + boxWidth, top + boxHeight // 2) if column == 0 else (left + boxWidth // 2, top + boxHeight)
            endLeft = x + nextColumn * 760
            endTop = y + nextRow * 145
            end = (endLeft, endTop + boxHeight // 2) if column == 0 else (endLeft + boxWidth // 2, endTop)
            draw.line((start, end), fill=GRID, width=4)
    draw.text((80, 900), f"Selected model: {displayModelName(modelName)}", fill=MUTED, font=font(26))
    return save(image, path)


def drawDatasetProfile(path: Path, profile) -> Path:
    image, draw = canvas("Dataset profile", f"{profile.rowCount} rows and {profile.columnCount} columns were prepared.")
    values = {
        "Numeric": len(profile.numericColumns),
        "Category": len(profile.categoricalColumns),
        "Time": len(profile.timestampColumns),
        "ID": len(profile.idColumns),
        "Constant": len(profile.constantColumns),
    }
    drawBarChart(draw, values, (130, 250, 1480, 780), max(values.values()) or 1, GREEN)
    missingValues = list(profile.missingValueRatio.values())
    missingAverage = sum(missingValues) / max(len(missingValues), 1)
    footer = f"Detected shape: {profile.suspectedDataType}. Average missing value ratio: {missingAverage:.4f}."
    draw.text((130, 850), footer, fill=MUTED, font=font(28))
    return save(image, path)


def drawScoreThresholdTimeline(path: Path, scores: np.ndarray, thresholds: np.ndarray, predictions: np.ndarray) -> Path:
    image, draw = canvas("Score and threshold timeline", "Rows above the threshold are marked as unusual.")
    area = (110, 220, 1500, 790)
    drawAxes(draw, area)
    maxY = float(max(np.max(scores) if len(scores) else 1, np.max(thresholds) if len(thresholds) else 1, 1e-8))
    drawLine(draw, scores, area, maxY, RED)
    drawLine(draw, thresholds, area, maxY, BLUE)
    for x, y in samplePoints(scores[predictions], np.where(predictions)[0], max(len(scores) - 1, 1), area, maxY):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=YELLOW)
    drawLegend(draw, [("Anomaly score", RED), ("Adaptive threshold", BLUE), ("Flagged rows", YELLOW)], 110, 835)
    return save(image, path)


def drawUncertaintyTimeline(path: Path, uncertainty: np.ndarray, detectorUsed: list[str]) -> Path:
    image, draw = canvas("Uncertainty and changed behavior", "Higher values show where the adaptive route was used.")
    area = (110, 220, 1500, 790)
    drawAxes(draw, area)
    maxY = float(max(np.max(uncertainty) if len(uncertainty) else 1, 1e-8))
    drawLine(draw, uncertainty, area, maxY, PURPLE)
    for index, detectorName in enumerate(detectorUsed):
        if detectorName != "dynamic":
            continue
        x = scaleX(index, max(len(detectorUsed) - 1, 1), area)
        draw.line((x, area[1], x, area[3]), fill="#35536c", width=1)
    drawLegend(draw, [("Concept uncertainty", PURPLE), ("Adaptive route", "#35536c")], 110, 835)
    return save(image, path)


def drawSeverityDistribution(path: Path, totalRows: int, anomalies: list[Any]) -> Path:
    image, draw = canvas("Severity distribution", "The final decision count is grouped by risk level.")
    counts = {"Normal": max(totalRows - len(anomalies), 0), "Warning": 0, "Critical": 0, "Emergency": 0}
    for item in anomalies:
        counts[item.severityLevel] = counts.get(item.severityLevel, 0) + 1
    drawBarChart(draw, counts, (130, 250, 1480, 780), max(counts.values()) or 1, RED)
    return save(image, path)


def drawFeatureContributions(path: Path, anomalies: list[Any]) -> Path:
    image, draw = canvas("Top responsible features", "Contribution scores are averaged across the strongest flagged rows.")
    totals: dict[str, float] = {}
    for item in anomalies[:50]:
        for part in item.topContributingFeatures:
            totals[part.featureName] = totals.get(part.featureName, 0.0) + float(part.contributionScore)
    if not totals:
        draw.text((130, 420), "No flagged rows were found in this run.", fill=MUTED, font=font(34))
        return save(image, path)
    ranked = dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:8])
    drawHorizontalBarChart(draw, ranked, (130, 250, 1480, 800), max(ranked.values()) or 1, YELLOW)
    return save(image, path)


def drawPerformanceSummary(path: Path, metrics: dict[str, Any], comparison: list[dict[str, Any]], modelName: str) -> Path:
    image, draw = canvas("Model performance summary", f"Current run: {displayModelName(modelName)}")
    metricValues = {
        "Accuracy": metricValue(metrics.get("accuracy")),
        "Precision": metricValue(metrics.get("precision")),
        "Recall": metricValue(metrics.get("recall")),
        "F1": metricValue(metrics.get("f1Score")),
        "AUC ROC": metricValue(metrics.get("aucRoc")),
        "AUC PR": metricValue(metrics.get("aucPr")),
    }
    drawBarChart(draw, metricValues, (130, 250, 1480, 650), 1.0, GREEN)
    y = 735
    draw.text((130, y), "Comparable stored runs", fill=TEXT, font=font(28, bold=True))
    y += 45
    for row in comparison[:6]:
        label = f"{displayModelName(str(row.get('model', 'Model')))}  AUC ROC: {formatMetric(row.get('aucRoc'))}  AUC PR: {formatMetric(row.get('aucPr'))}"
        draw.text((130, y), label, fill=MUTED, font=font(24))
        y += 34
    return save(image, path)


def drawAxes(draw, area: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = area
    draw.rectangle(area, outline=GRID, width=2)
    for i in range(1, 5):
        y = top + (bottom - top) * i / 5
        draw.line((left, y, right, y), fill="#263a48", width=1)


def drawLine(draw, values: np.ndarray, area: tuple[int, int, int, int], maxY: float, color: str) -> None:
    if len(values) == 0:
        return
    step = max(1, math.ceil(len(values) / 700))
    indexes = np.arange(0, len(values), step)
    points = [(scaleX(int(index), max(len(values) - 1, 1), area), scaleY(float(values[index]), maxY, area)) for index in indexes]
    if len(points) > 1:
        draw.line(points, fill=color, width=4)


def samplePoints(values: np.ndarray, indexes: np.ndarray, maxIndex: int, area: tuple[int, int, int, int], maxY: float) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    step = max(1, math.ceil(len(values) / 120))
    return [(scaleX(int(indexes[i]), maxIndex, area), scaleY(float(values[i]), maxY, area)) for i in range(0, len(values), step)]


def scaleX(index: int, maxIndex: int, area: tuple[int, int, int, int]) -> int:
    left, _, right, _ = area
    return int(left + (right - left) * index / max(maxIndex, 1))


def scaleY(value: float, maxY: float, area: tuple[int, int, int, int]) -> int:
    _, top, _, bottom = area
    return int(bottom - (bottom - top) * value / max(maxY, 1e-8))


def drawBarChart(draw, values: dict[str, float], area: tuple[int, int, int, int], maxValue: float, color: str) -> None:
    left, top, right, bottom = area
    drawAxes(draw, area)
    barArea = right - left
    gap = 28
    barWidth = max(40, int((barArea - gap * (len(values) + 1)) / max(len(values), 1)))
    for index, (label, value) in enumerate(values.items()):
        x0 = left + gap + index * (barWidth + gap)
        x1 = x0 + barWidth
        y0 = bottom - int((bottom - top - 40) * float(value) / max(maxValue, 1e-8))
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=color)
        draw.text((x0, y0 - 38), formatMetric(value), fill=TEXT, font=font(22, bold=True))
        draw.text((x0, bottom + 24), label[:14], fill=MUTED, font=font(22))


def drawHorizontalBarChart(draw, values: dict[str, float], area: tuple[int, int, int, int], maxValue: float, color: str) -> None:
    left, top, right, bottom = area
    rowHeight = max(48, int((bottom - top) / max(len(values), 1)))
    for index, (label, value) in enumerate(values.items()):
        y = top + index * rowHeight
        width = int((right - left - 420) * float(value) / max(maxValue, 1e-8))
        draw.text((left, y + 8), label[:28], fill=MUTED, font=font(24))
        draw.rounded_rectangle((left + 430, y + 8, left + 430 + width, y + 40), radius=5, fill=color)
        draw.text((left + 450 + width, y + 8), formatMetric(value), fill=TEXT, font=font(22))


def drawLegend(draw, items: list[tuple[str, str]], x: int, y: int) -> None:
    for label, color in items:
        draw.rectangle((x, y + 6, x + 26, y + 32), fill=color)
        draw.text((x + 42, y), label, fill=MUTED, font=font(24))
        x += 330


def metricValue(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def formatMetric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def imageTitle(relativeName: str) -> str:
    stem = Path(relativeName).stem
    return stem.split("_", 1)[1].replace("_", " ").title()


def writePdfReport(path: Path, datasetName: str, modelName: str, metrics: dict[str, Any], anomalyResult, imageDirectory: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas as pdfCanvas
    except Exception:
        return False

    document = pdfCanvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    document.setTitle("Enhanced Dymeter Report")
    document.setFont("Times-Bold", 22)
    document.drawString(0.75 * inch, height - 0.85 * inch, "Enhanced Dymeter Report")
    document.setFont("Times-Roman", 12)
    document.drawString(0.75 * inch, height - 1.2 * inch, f"Dataset: {datasetName}")
    document.drawString(0.75 * inch, height - 1.42 * inch, f"Model: {displayModelName(modelName)}")
    document.drawString(0.75 * inch, height - 1.64 * inch, f"Rows: {anomalyResult.totalRows}    Flagged rows: {anomalyResult.totalAnomalies}")
    document.drawString(0.75 * inch, height - 1.86 * inch, f"F1: {formatMetric(metrics.get('f1Score'))}    AUC ROC: {formatMetric(metrics.get('aucRoc'))}    AUC PR: {formatMetric(metrics.get('aucPr'))}")
    y = height - 2.25 * inch
    for imagePath in sorted(imageDirectory.glob("*.png")):
        if y < 2.8 * inch:
            document.showPage()
            y = height - 0.8 * inch
        document.setFont("Times-Bold", 13)
        document.drawString(0.75 * inch, y, imageTitle(imagePath.name))
        y -= 0.18 * inch
        document.drawImage(str(imagePath), 0.75 * inch, y - 2.15 * inch, width=6.8 * inch, height=2.0 * inch, preserveAspectRatio=True, anchor="n")
        y -= 2.35 * inch
    document.save()
    return True


def writeHtmlReport(path: Path, datasetName: str, modelName: str, metrics: dict[str, Any], anomalyResult, imageNames: list[str]) -> None:
    rows = "\n".join(f'<li><a href="{name}">{imageTitle(name)}</a></li>' for name in imageNames)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Enhanced Dymeter Report</title>
  <style>
    body {{ font-family: "Times New Roman", Times, serif; background: #101820; color: #f6f7f9; padding: 32px; }}
    a {{ color: #52a9ff; }}
  </style>
</head>
<body>
  <h1>Enhanced Dymeter Report</h1>
  <p>Dataset: {datasetName}</p>
  <p>Model: {displayModelName(modelName)}</p>
  <p>Rows: {anomalyResult.totalRows}; Flagged rows: {anomalyResult.totalAnomalies}</p>
  <p>F1: {formatMetric(metrics.get("f1Score"))}; AUC ROC: {formatMetric(metrics.get("aucRoc"))}; AUC PR: {formatMetric(metrics.get("aucPr"))}</p>
  <h2>Downloadable images</h2>
  <ul>{rows}</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
