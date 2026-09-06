# DyMETER Research Application

This project extends the existing METER folder into a full educational research app.

It supports:

- METER baseline
- DyMETER reproduction path
- `EnhancedDymeterMANN`, an ablation that keeps the DyMETER budget and adds only MANN-style concept memory
- universal dataset profiling
- adaptive preprocessing
- live-style pipeline logs
- grounded "Ask Your Model" answers from actual run JSON
- benchmark and local result separation

## Training Budget Contract

METER, DyMETER, and `EnhancedDymeterMANN` keep the paper baseline budget of `2000` epochs. `EnhancedDymeterMANN` adds only the MANN-style concept memory so it can be used as a clean ablation.

## Quick Backend Run

```powershell
python scripts/runPipeline.py datasets/ambient_temperature_system_failure.csv --model DyMETER --force-numpy
python scripts/runPipeline.py datasets/ambient_temperature_system_failure.csv --model EnhancedDymeterMANN --force-numpy
```

## Full App Run

See `docs/INSTALLATION_GUIDE.md`.

## Result Contract

Every run returns:

```json
{
  "totalRows": 0,
  "totalAnomalies": 0,
  "anomalyRatio": 0,
  "thresholdUsed": 0,
  "anomalies": [],
  "conceptDriftEvents": []
}
```

Each anomaly includes row data, score, reconstruction error, concept uncertainty, detector route, severity, per-row contributing features, and a plain-English explanation.

## Paper Reported vs Locally Computed

Paper values are stored separately in `researchApp/core/paperResults.py` and marked `Paper Reported`. Values produced by this application are marked `Locally Computed`. Negative deltas are preserved instead of hidden.
