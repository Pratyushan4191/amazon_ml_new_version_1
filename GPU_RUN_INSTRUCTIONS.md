# GPU Execution Instructions

The pipeline trains on `dataset/train/` and predicts from `dataset/test/` by default. Submission files are written to `output/`; intermediate feature files are written to `artifacts/`.

## 1. Project Structure
The production pipeline is located in `ber_project_gpu_ready/`. This folder is completely self-contained.

## 2. Environment Setup & Dependency Installation
On your GPU machine, ensure you are running Python 3.8+ and install the dependencies:
```bash
pip install -r requirements.txt
```
*(Note: LightGBM uses the CPU by default. If your GPU machine requires LightGBM GPU acceleration, install the GPU-compiled version of lightgbm according to their docs. However, it is fully multithreaded and heavily optimized out of the box).*

## 3. Dataset Placement
Place the real datasets inside a `dataset/` folder at the root of the GPU project:
```
ber_project_gpu_ready/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

## 4. Run the Real Dataset
Run the complete pipeline from the project root:

```bash
python run_pipeline.py
```

Use `--dataset-dir` or `--output-dir` to select different locations.

## 5. Pipeline Stages
The orchestrator runs these stages sequentially:
1. Load training sources and fit TF-IDF vocabularies
2. Generate training candidates and features
3. Train LightGBM and select a validation threshold
4. Generate test candidates and features
5. Score candidates and write both submission files

## 6. Expected Outputs
Upon successful execution, the following files will be available:
- **`output/matching_results.tsv`**: Your final leaderboard submission file.
- **`output/candidate_pairs.tsv`**: The required candidate pairs file.
- `model/model.lgb`: The trained model artifact.
- `model/threshold.json`: The optimized `t*` threshold.
- `model/shap.json`: Feature importance tracking.

## 7. Troubleshooting / Validation
Validate the generated submission files with:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
