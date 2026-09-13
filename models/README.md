# Quantum EEG Signal Classifier — Trained Model Artifacts

This directory stores trained model artifacts.

**Do NOT commit the `.venv` or temporary files here.**

After running `python scripts/train.py`, artifacts will be saved under
a versioned subdirectory, e.g.:

```
models/
└── v1_0_0/
    ├── vqc_model.dill
    ├── pca.joblib
    ├── scaler.joblib
    ├── feature_config.json
    ├── preprocessing_config.json
    └── model_metadata.json
```
