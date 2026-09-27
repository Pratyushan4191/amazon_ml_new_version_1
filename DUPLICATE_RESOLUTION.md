# Duplicate File Resolution Report

During integration, overlapping or duplicate module implementations were analyzed and resolved to establish a single source of truth for the GPU production pipeline.

### `train.py`
* **Existing version**: `member 4 work/ber_project/src/train.py` (implemented LightGBM logic, hard-negative upsampling, entity-grouped splitting).
* **New version**: `src/train.py` (created during integration).
* **Decision**: **MERGED (Selected `src/train.py`)**
* **Reason & Compatibility**: The old version hardcoded paths and used an older column name (`candidate_id`). The new version wraps the original Member 4 training algorithms in a clean, parameterized interface (`train_model()`), explicitly aligns with Member 3's `candidate_entity_id` feature column, and is directly callable by `run_pipeline.py`.

### `threshold.py`
* **Existing version**: `member 4 work/ber_project/src/threshold.py` (F0.5 sweep logic).
* **New version**: `src/threshold.py` (created during integration).
* **Decision**: **MERGED (Selected `src/threshold.py`)**
* **Reason & Compatibility**: The old logic correctly implemented the strict F0.5 singleton rules. The new version refactors this into `sweep_threshold()`, removes hardcoded paths, dynamically detects the candidate column, and integrates with the orchestrator.

### `predict.py` / `inference.py`
* **Existing version**: `member 4 work/ber_project/src/inference.py` (prediction and aggregation).
* **New version**: `src/predict.py` (created during integration).
* **Decision**: **MERGED (Selected `src/predict.py`)**
* **Reason & Compatibility**: Functional duplication existed where `inference.py` handled scoring, threshold application, and TSV aggregation. The new `predict.py` absorbs the old `inference.py` logic verbatim, fixes the candidate column naming to align with `features.py`, parameterizes the paths, and enforces the "fail-fast" hard-rejection tests directly inside the orchestrator flow.

### `eval.py`
* **Existing version**: Empty `eval.py` in project root.
* **New version**: Fully implemented `eval.py` in project root.
* **Decision**: **NEW**
* **Reason**: The original file was just a placeholder. The new version implements `macro_f05` and `measure_blocking_recall` exactly per the BER Architecture spec.
