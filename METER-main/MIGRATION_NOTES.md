# Migration Notes

## Reused As-Is

- `METER.py`: kept as the original METER implementation for baseline reference.
- `main.py`: kept as the original command-line training script.
- `data_utils.py`: kept as the original benchmark loader.
- `utils.py`: kept as the original metric helper.
- `edl_pytorch/`: kept as the original evidential learning helper code.
- `datasets/`: kept as the benchmark data source already included with the METER project.

## Extended Around Existing Code

- The new application does not replace the original METER files. It adds a parallel research application under `researchApp/` so METER, DyMETER, and `MyImprovedModel` can be compared without breaking the baseline.
- The original METER ideas are wrapped conceptually in `researchApp/models/detectors.py`, where `METER` uses the static detector path and DyMETER adds concept uncertainty, dynamic routing, and dynamic thresholding.

## New Files

- `researchApp/core/profiler.py`: automatic dataset profiler for uploaded files.
- `researchApp/core/preprocessing.py`: adaptive cleaning, scaling, categorical encoding, window sizing, and latent dimension selection using the 70% explained variance rule.
- `researchApp/models/detectors.py`: METER-compatible static detector, DyMETER dynamic detector, hypernetwork-style parameter shift, memory module, and lightweight adapter.
- `researchApp/core/threshold.py`: dynamic threshold optimization with quantile fallback.
- `researchApp/core/explain.py`: per-row feature contribution, severity, confidence, and natural-language explanation logic.
- `researchApp/core/pipeline.py`: end-to-end run orchestration and exact `anomalyResult` contract generation.
- `researchApp/core/qa.py`: grounded run-specific Q&A with Anthropic support and local fallback.
- `researchApp/api/server.py`: FastAPI backend.
- `researchApp/storage/database.py`: SQLite run and Q&A logging.
- `frontend/`: React, TypeScript, Tailwind, Framer Motion, Chart.js, and React Flow dashboard.
- `tests/testGeneralization.py`: required generalization tests for benchmark, synthetic, and malformed datasets.
- `docs/INSTALLATION_GUIDE.md`: setup and run instructions.

## Honest Limitations

- The uploaded IEEE paper PDF and presentation were not present in this workspace. Public arXiv metadata and snippets were used for structure and a few verifiable paper-reported values. Missing published table values are left unavailable instead of fabricated.
- Full neural PyTorch training requires installing `requirements.txt`. The current workspace can run the pipeline tests through the NumPy PCA fallback.

