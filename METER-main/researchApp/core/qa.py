from __future__ import annotations

import json
import os
from typing import Any


def answerFromContext(question: str, qaContext: dict[str, Any]) -> str:
    """Grounded Q&A. Uses Anthropic if configured, otherwise a deterministic local answer."""

    apiKey = os.environ.get("ANTHROPIC_API_KEY")
    if apiKey:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=apiKey)
            response = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=500,
                system="You answer only from the provided run JSON. If the JSON does not contain the answer, say that clearly.",
                messages=[
                    {
                        "role": "user",
                        "content": "Run context:\n" + json.dumps(qaContext, default=str) + "\n\nQuestion: " + question,
                    }
                ],
            )
            return response.content[0].text
        except Exception as error:
            return "The external LLM call failed, so I answered locally. " + localAnswer(question, qaContext, str(error))

    return localAnswer(question, qaContext)


def localAnswer(question: str, qaContext: dict[str, Any], errorText: str | None = None) -> str:
    lowerQuestion = question.lower()
    anomalies = qaContext.get("topAnomalies", [])

    if "row" in lowerQuestion and anomalies:
        digits = "".join(character if character.isdigit() else " " for character in question).split()
        if digits:
            wantedRow = int(digits[0])
            for anomaly in anomalies:
                if anomaly.get("rowIndex") == wantedRow:
                    return anomaly.get("naturalLanguageExplanation", "The context contains this anomaly but no explanation text.")
            return "The run context does not contain that row in the top anomaly list."

    if "compare" in lowerQuestion or "dymeter" in lowerQuestion:
        rows = qaContext.get("comparison", [])
        if not rows:
            return "The run context does not contain comparison rows."
        textRows = []
        for row in rows:
            textRows.append(f"{row.get('model')} on {row.get('dataset')}: AUCROC={row.get('aucRoc')}, AUCPR={row.get('aucPr')} ({row.get('source')}).")
        return " ".join(textRows)

    if "drift" in lowerQuestion:
        events = qaContext.get("conceptDriftEvents", [])
        if not events:
            return "No concept drift events were recorded in this run context."
        return " ".join([f"Rows {event['rowIndexRange']} changed: {event['description']}" for event in events[:8]])

    if "threshold" in lowerQuestion:
        metrics = qaContext.get("metrics", {})
        result = qaContext.get("topAnomalies", [])
        return (
            "Lowering the threshold would usually flag more rows as anomalous. "
            f"This run recorded {len(result)} top anomaly examples and metric summary {metrics}."
        )

    if "feature" in lowerQuestion:
        if not anomalies:
            return "The run context does not contain anomaly feature contributions."
        first = anomalies[0]
        features = first.get("topContributingFeatures", [])
        return "For the strongest anomaly, the top features were: " + ", ".join([item["featureName"] for item in features])

    if errorText:
        return "The run context does not contain enough information to answer that fully. LLM error: " + errorText
    return "The run context does not contain enough information to answer that question."

