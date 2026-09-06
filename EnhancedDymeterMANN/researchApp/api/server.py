from __future__ import annotations

import json
import shutil
from pathlib import Path

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except Exception as importError:
    raise RuntimeError("FastAPI is not installed. Run `pip install -r requirements.txt` first.") from importError

from researchApp.core.pipeline import ResearchPipeline
from researchApp.storage.database import RunDatabase


app = FastAPI(title="DyMETER Research Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploadDirectory = Path("uploads")
uploadDirectory.mkdir(exist_ok=True)
pipeline = ResearchPipeline()
database = RunDatabase()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs")
async def createRun(file: UploadFile = File(...), modelName: str = Form("DyMETER"), config: str = Form("{}")):
    uploadedDatasetPath = uploadDirectory / file.filename
    with uploadedDatasetPath.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        runBundle = pipeline.runFile(uploadedDatasetPath, modelName=modelName, config=json.loads(config or "{}"))
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    # This payload is returned to the React frontend through the loopback API.
    frontendPayload = {
        "runId": runBundle.runId,
        "modelName": runBundle.modelName,
        "datasetName": runBundle.datasetName,
        "profile": runBundle.profile.__dict__,
        "preprocessing": runBundle.preprocessing,
        "metrics": runBundle.metrics,
        "anomalyResult": runBundle.anomalyResult.toDict(),
        "pipelineLogs": [log.__dict__ for log in runBundle.pipelineLogs],
        "qaContext": runBundle.qaContext,
        "comparison": runBundle.comparison,
        "artifacts": runBundle.artifacts,
    }
    database.saveRunPayload(runBundle.runId, runBundle.datasetName, runBundle.modelName, frontendPayload)
    return frontendPayload


@app.get("/runs/{runId}/artifacts/{artifactPath:path}")
async def downloadArtifact(runId: str, artifactPath: str):
    artifactRoot = (Path("runs") / "artifacts" / runId).resolve()
    targetPath = (artifactRoot / artifactPath).resolve()
    if artifactRoot not in targetPath.parents and targetPath != artifactRoot:
        return JSONResponse({"error": "Invalid artifact path."}, status_code=400)
    if not targetPath.is_file():
        return JSONResponse({"error": "Artifact not found."}, status_code=404)
    return FileResponse(targetPath, filename=targetPath.name)
