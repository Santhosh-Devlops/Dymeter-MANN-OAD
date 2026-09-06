# EnhancedDymeterMANN

This folder contains the EnhancedDymeterMANN backend runtime copied out of `METER-main` for a clean model-specific workspace.

## Run

```powershell
Set-Location "D:\Dymeter\EnhancedDymeterMANN"
..\METER-main\.venv\Scripts\python.exe scripts\run_enhanced_dymeter_mann.py "..\METER-main\datasets\ionosphere.csv"
```

Use any supported dataset path in place of the example CSV.

## Included

- `researchApp/` shared backend pipeline and model runtime
- `scripts/run_enhanced_dymeter_mann.py` EnhancedDymeterMANN-only launcher
- `requirements.txt`

The original runnable app remains in `D:\Dymeter\METER-main`.
