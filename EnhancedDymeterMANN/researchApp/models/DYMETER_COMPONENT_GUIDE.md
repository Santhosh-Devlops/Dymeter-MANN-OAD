# DyMETER Component Guide

This file is a student-friendly map for explaining where each DyMETER block is implemented.

## 1. SCD - Static Concept-aware Detector

Code file:

`researchApp/models/detectors.py`

Main class:

`StaticConceptAwareDetector`

Purpose:

The SCD is the base autoencoder. It learns the central normal concept from historical rows.

Main flow:

`input features -> encoder -> latent vector -> decoder -> reconstructed features`

Formula idea:

`scdLoss = mean((inputFeatures - reconstructedFeatures)^2)`

Important functions:

- `trainStaticDetector(...)`
- `runTorch(...)`
- `runNumpy(...)` for the PCA fallback

## 2. IEC - Intelligent Evolution Controller

Code file:

`researchApp/models/detectors.py`

Main class:

`IntelligentEvolutionController`

Purpose:

The IEC estimates concept uncertainty. It tells DyMETER whether a row still belongs to the learned central concept or whether dynamic adaptation may be needed.

Main flow:

`input features -> evidence network -> alpha values -> probabilities -> concept uncertainty`

Formula idea:

`probability = alpha / sum(alpha)`

`conceptUncertainty = entropy - dataUncertainty`

Important functions:

- `trainEvolutionController(...)`
- `hasUsableValidationLabels(...)`
- `evaluateReconstructionValidation(...)`

## 3. DSD - Dynamic Shift-aware Detector

Code file:

`researchApp/models/detectors.py`

Main class:

`ParameterShiftHypernetwork`

Purpose:

The DSD generates an instance-aware reconstruction shift when the IEC detects concept drift.

Main flow:

`input features -> hypernetwork -> shift -> static reconstruction + shift`

Formula idea:

`dynamicReconstruction = staticReconstruction + parameterShift`

`dsdLoss = mean((inputFeatures - dynamicReconstruction)^2) + shiftRegularizer`

Important functions:

- `trainDynamicShiftDetector(...)`
- `usesDynamicShiftDetector(...)`
- `makeDriftEvents(...)`

## 4. DTO - Dynamic Threshold Optimization

Code file:

`researchApp/core/threshold.py`

Main class:

`DynamicThresholdOptimizer`

Purpose:

The DTO updates the anomaly decision boundary while streaming rows are processed.

Main flow:

`score + concept uncertainty -> normal/candidate window -> adaptive thresh`

Formula idea:

`baseThresh = quantile(normalScores, 1 - falsePositiveRate)`

`threshRegularization = kappa * max(baseThresh - candidateMedian, 0)`

`currentThresh = baseThresh + threshRegularization`

Important functions:

- `initialize(...)`
- `update(...)`

## 5. Output Sent To Frontend Through Loopback

Backend API file:

`researchApp/api/server.py`

Main route:

`POST /runs`

Purpose:

Receives the uploaded dataset, trains the selected model, builds the run result, and returns the JSON payload to the frontend.

Returned payload includes:

- `runId`
- `modelName`
- `datasetName`
- `profile`
- `preprocessing`
- `metrics`
- `anomalyResult`
- `pipelineLogs`
- `comparison`
- `artifacts`

Frontend file:

`frontend/src/App.tsx`

Frontend call:

`fetch("/api/runs", { method: "POST", body: formData })`

Loopback proxy file:

`frontend/vite.config.ts`

Loopback address:

`http://127.0.0.1:8000`

Meaning:

The browser calls `/api/runs` on the Vite frontend server. Vite forwards that request to the FastAPI backend running on `127.0.0.1:8000`.

## 6. DyMETER vs EnhancedDymeterMANN Code Difference

Both models use SCD, IEC, DSD, and DTO.

EnhancedDymeterMANN additionally uses:

- self-supervised denoising in `trainStaticDetector(...)`
- feature masking in `trainStaticDetector(...)`
- lighter weight decay in SCD and DSD training
- MANN concept memory in `ConceptMemory`
- proactive uncertainty routing in `proactiveDynamicMask(...)`

These additions are all in:

`researchApp/models/detectors.py`
