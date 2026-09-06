from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class RunDatabase:
    """Stores run summaries and model Q&A logs in SQLite."""

    def __init__(self, databasePath: str | Path = "runs/dymeter.sqlite"):
        self.databasePath = Path(databasePath)
        self.databasePath.parent.mkdir(exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.databasePath) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    runId TEXT PRIMARY KEY,
                    datasetName TEXT,
                    modelName TEXT,
                    payload TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS questionLog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runId TEXT,
                    question TEXT,
                    answer TEXT,
                    createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def saveRunPayload(self, runId: str, datasetName: str, modelName: str, payload: dict) -> None:
        with sqlite3.connect(self.databasePath) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(runId, datasetName, modelName, payload) VALUES (?, ?, ?, ?)",
                (runId, datasetName, modelName, json.dumps(payload, default=str)),
            )

    def getRunPayload(self, runId: str) -> dict | None:
        with sqlite3.connect(self.databasePath) as connection:
            row = connection.execute("SELECT payload FROM runs WHERE runId = ?", (runId,)).fetchone()
        return json.loads(row[0]) if row else None

    def logQuestion(self, runId: str, question: str, answer: str) -> None:
        with sqlite3.connect(self.databasePath) as connection:
            connection.execute(
                "INSERT INTO questionLog(runId, question, answer) VALUES (?, ?, ?)",
                (runId, question, answer),
            )

