# Final Production File Manifest

This GPU-ready codebase strips away all scratch work, intermediate versions, and ambiguous duplicates. It contains exactly what is needed for the production run.

| File | Purpose | Selected From | Required? |
|------|---------|---------------|-----------|
| `run_pipeline.py` | Main orchestrator / entrypoint | New | YES |
| `eval.py` | Metric calculations (F0.5, recall) | New | YES |
| `requirements.txt` | Python dependencies | New | YES |
| `src/normalization.py` | Text/address cleaning | Existing (Member 2) | YES |
| `src/blocking.py` | Candidate pair generation | Existing (Member 2) | YES |
| `src/features.py` | 29-feature construction | Existing (Member 3)* | YES |
| `src/train.py` | LightGBM model training | Merged (Member 4 + New) | YES |
| `src/threshold.py` | t* threshold optimization | Merged (Member 4 + New) | YES |
| `src/predict.py` | Inference & aggregation | Merged (Member 4 + New) | YES |
| `utils/validate_submission.py` | Final TSV schema validation | Existing (Member 1) | YES |
| `tests/test_pipeline.py` | Local dummy test suite | New | NO (For local validation) |

*\* `src/features.py` had a minor import bug (`from normalization` vs `from src.normalization`) which was fixed in place.*
