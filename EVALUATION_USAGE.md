# Evaluation Script Usage Guide

## Quick Start

The refactored `evaluate.py` evaluates a machine-generated text detector on validation data.

### Basic Usage

```bash
# Simplest form - uses output_dir from YAML config
python evaluate.py --config eval_config.yaml

# Override output directory
python evaluate.py --config eval_config.yaml --output-dir ./eval_run_1/

# Use pre-cached features (skip extraction step)
python evaluate.py --config eval_config.yaml --features-dir ./cached_features/

# Switch between different machine-support datasets
python evaluate.py --config eval_config.yaml --machine-support data/gpt4.jsonl
python evaluate.py --config eval_config.yaml --machine-support data/gpt4.jsonl data/claude.jsonl
```

## Script Workflow

### Phase 1: Feature Extraction
- Loads human and machine support documents
- Extracts features using configured extractors for each view
- Caches results to `features/` directory (or `--features-dir`)
- Subsequent runs with same data use cached features

### Phase 2: Validation Features
- Loads held-out validation set (JSONL format)
- Expects `label` field (0=human, 1=machine) or `contribution_level` field
- Extracts validation features using same extractors
- Groups documents by category if label field provided

### Phase 3: Single Evaluation
- **Training**: Gaussian Process trained on support set
  - Configurable kernel (RBF, Matern), ARD, num epochs, learning rate
  - Auto-selects variational vs exact formulation based on training set size
- **Calibration**: Threshold and uncertainty bounds set on calibration split
  - Configurable FPR target (alpha), aggregation method
- **Validation**: Scored on held-out data, produces metrics

### Phase 4: Results Output

Outputs written to `output_dir`:

```
output_dir/
├── features/                  # Cached extracted features
│   ├── <view_name>.npy
│   ├── labels.npy
│   ├── metadata.json
│   └── distance_constructor.pt
├── validation/                # Cached validation features
│   ├── <view_name>.npy
│   ├── labels.npy
│   └── categories.json
├── results.jsonl             # Metric record (one JSON object)
└── summary.txt               # Human-readable table + per-category breakdown
```

## Output Metrics

### Main Metrics (summary.txt)
- **AUROC**: Area under ROC curve (higher = better)
- **FPR@τ**: False positive rate at learned threshold (lower = better)
- **TPR@τ**: True positive rate at learned threshold (higher = better)
- **CalGap**: |FPR - target_alpha| (lower = better, shows calibration quality)
- **OOD%**: Percentage of samples flagged as out-of-distribution

### Per-Category Metrics (if validation labels provided)
- **TPR by category**: Detection rate for machine text within category
- **FPR by category**: False alarm rate for human text within category
- **AUROC by category**: ROC AUC within each category

## Configuration (eval_config.yaml)

Required sections:
```yaml
data:
  human_support: path/to/human_documents.jsonl
  machine_support: path/to/machine_documents.jsonl  # or list of paths
  validation: path/to/validation.jsonl
  text_key: "text"                    # field containing document text
  label_key: "contribution_level"     # optional field for per-category metrics

views:
  - name: gpt2_detector              # detector/feature name

evaluation:
  output_dir: ./eval_results/
```

## Example Workflow

```bash
# Run evaluation with default settings
pixi run -e dev python text_detection_baselines/evaluate.py \
  --config configs/eval_default.yaml

# Check results
cat eval_results/summary.txt

# Run with custom output directory
pixi run -e dev python text_detection_baselines/evaluate.py \
  --config configs/eval_default.yaml \
  --output-dir eval_results_v2

# Use cached features for fast re-evaluation
pixi run -e dev python text_detection_baselines/evaluate.py \
  --config configs/eval_default.yaml \
  --features-dir eval_results_v2/features
```

## Performance Considerations

- First run: ~5-10 minutes (depends on dataset size and GPU)
  - Features cached automatically
- Subsequent runs with same config: ~2-3 minutes
  - Reuses cached features
- CPU vs GPU: Feature extraction uses GPU if available (via torch)

## Troubleshooting

**Missing validation data**: Ensure `data.validation` path exists in config

**No valid AUROC**: Indicates non-OOD validation set has only one class

**Expected feature file not found**: Check feature extractor names match view names

**Out-of-memory during feature extraction**: Reduce batch sizes in feature extractors
