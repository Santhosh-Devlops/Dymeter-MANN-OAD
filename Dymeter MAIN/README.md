# DyMETER MAIN

This folder contains the DyMETER backend runtime copied out of `METER-main` for a clean model-specific workspace.

## Run

```powershell
Set-Location "D:\Dymeter\Dymeter MAIN"
..\METER-main\.venv\Scripts\python.exe scripts\run_dymeter.py "..\METER-main\datasets\ionosphere.csv"
```

Use any supported dataset path in place of the example CSV.

## Included

- `researchApp/` shared backend pipeline and model runtime
- `scripts/run_dymeter.py` DyMETER-only launcher
- `requirements.txt`

The original runnable app remains in `D:\Dymeter\METER-main`.
