# IMPLEMENTATION_PLAN.md
# Quantum EEG Signal Classifier for Epilepsy and Seizure Detection

**Document Version:** 1.0  
**Status:** Master Implementation Specification  
**Target IDE:** Antigravity IDE  
**Project Type:** Advanced University Major Project  
**Primary Classifier:** Variational Quantum Classifier (VQC)  
**Primary User Workflow:** Upload unseen EDF → Analyze → Predict → Report

---

# 1. MASTER DIRECTIVE

Build this project completely from scratch as a polished, advanced university-level software product.

The system must be an **inference-first EEG seizure detection application** powered by a **pre-trained Variational Quantum Classifier**.

The project has two strictly separated lifecycles:

1. **Training lifecycle** — developer/researcher only.
2. **Inference lifecycle** — end-user application.

The end user must **never be required to provide, browse, select, or train on the CHB-MIT training dataset**.

The final end-user experience must be:

```text
Open Application
      ↓
Upload NEW / UNSEEN EDF file
      ↓
Inspect recording
      ↓
Click Analyze
      ↓
Automatic EEG preprocessing
      ↓
Automatic feature extraction
      ↓
Saved PCA transformation
      ↓
Saved scaler transformation
      ↓
Saved VQC inference
      ↓
Segment-level predictions
      ↓
Recording-level seizure assessment
      ↓
EEG visualization
      ↓
Results
      ↓
Download report
```

The application must **not retrain the model when it starts** and must **not retrain for each uploaded EDF**.

---

# 2. PRODUCT VISION

Create a complete Hybrid Quantum-Classical EEG Signal Classifier capable of analyzing an unseen EEG recording and identifying segments that are likely to contain seizure activity.

The system should combine:

- EEG signal processing
- Signal feature engineering
- Classical dimensionality reduction
- Quantum feature encoding
- Variational quantum classification
- Interactive visualization
- Automated reporting

The project should be presented as a research/decision-support prototype, not as an autonomous medical diagnostic device.

The UI should clearly communicate that predictions are model outputs requiring qualified professional review.

---

# 3. CORE ARCHITECTURE

The architecture consists of two independent pipelines.

## 3.1 Training Pipeline

```text
CHB-MIT EEG Dataset
        ↓
EDF Loading
        ↓
Channel Selection
        ↓
Bandpass Filtering
        ↓
2-Second Segmentation
        ↓
Seizure Label Generation
        ↓
Feature Extraction
        ↓
16-Dimensional Feature Matrix
        ↓
Class Balancing
        ↓
Train/Test Split
        ↓
Fit PCA
        ↓
Fit Scaler
        ↓
Train VQC
        ↓
Evaluate
        ↓
Save Model Artifacts
```

## 3.2 Inference Pipeline

```text
User Uploads NEW EDF
        ↓
Validate EDF
        ↓
Load Recording Metadata
        ↓
Channel Selection
        ↓
Bandpass Filtering
        ↓
2-Second Segmentation
        ↓
Feature Extraction
        ↓
Load Saved PCA
        ↓
PCA Transform
        ↓
Load Saved Scaler
        ↓
Scaling / Quantum Input Preparation
        ↓
Load Saved VQC
        ↓
Predict Every Segment
        ↓
Aggregate Predictions
        ↓
Visualize
        ↓
Generate Report
```

---

# 4. NON-NEGOTIABLE ARCHITECTURAL RULES

## Rule 1 — Training and inference must be separate

Training code must live under a dedicated training package.

Inference code must live under application/service packages.

## Rule 2 — No built-in dataset in the user workflow

The production UI must not display the CHB-MIT dataset as selectable input.

## Rule 3 — No retraining during inference

Uploading an EDF must never invoke model fitting.

Forbidden during inference:

- `fit()`
- VQC training
- PCA fitting
- scaler fitting
- dataset balancing
- label generation from training annotations

## Rule 4 — Reuse the exact training preprocessing

Inference must use the same:

- channels
- filter frequencies
- segment duration
- feature definitions
- feature ordering
- PCA object
- scaling rules
- quantum encoding assumptions

used during training.

## Rule 5 — Save fitted preprocessing artifacts

PCA and scaler must be fitted during training and persisted.

## Rule 6 — Model artifact validation

The application must refuse to perform inference if required model artifacts are missing or incompatible.

## Rule 7 — No hardcoded machine-specific paths

Use configuration and `pathlib`.

## Rule 8 — No silent failures

Errors must be logged and shown to users in understandable language.

## Rule 9 — Quantum model remains primary

The production classifier must remain the VQC.

Classical models may optionally exist only as research baselines and must never silently replace the VQC.

## Rule 10 — New EDF means unseen input

The application must be capable of processing an EDF that was not part of the training dataset, provided its required channel/data requirements are satisfied.

---

# 5. PROJECT OBJECTIVES

## Primary Objectives

- Build an end-to-end EEG seizure classification system.
- Train a VQC using EEG-derived features.
- Persist the trained quantum model.
- Persist all fitted preprocessing transformations.
- Accept unseen EDF files.
- Produce segment-level predictions.
- Produce recording-level results.
- Provide an intuitive professional UI.
- Generate downloadable reports.

## Engineering Objectives

- Clean architecture
- Modular code
- Testable services
- Reproducible training
- Versioned model artifacts
- Configuration-driven behavior
- Strong validation
- Clear logging
- Professional UI/UX

---

# 6. TECHNOLOGY STACK

Use the following technologies unless a technical incompatibility requires a justified change.

## Core

- Python 3.11+ or a currently supported Python version compatible with the selected Qiskit stack
- NumPy
- Pandas
- SciPy
- MNE
- PyWavelets
- scikit-learn

## Quantum

- Qiskit
- Qiskit Machine Learning

## Frontend

- Streamlit

## Visualization

- Plotly
- Matplotlib where scientifically appropriate

## Reports

- ReportLab

## Configuration

- YAML and/or JSON

## Testing

- pytest

## Quality

- Ruff
- Black
- mypy where practical

## Deployment

- Docker
- Docker Compose for local multi-service development if needed

## Version Control

- Git
- GitHub

---

# 7. RECOMMENDED PROJECT STRUCTURE

Create the project from scratch using a clean architecture.

```text
Quantum-EEG-Signal-Classifier/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── pages/
│   │   ├── dashboard.py
│   │   ├── analyze.py
│   │   ├── results.py
│   │   ├── history.py
│   │   ├── model_info.py
│   │   ├── settings.py
│   │   └── about.py
│   ├── components/
│   │   ├── cards.py
│   │   ├── charts.py
│   │   ├── eeg_plot.py
│   │   ├── status.py
│   │   └── layout.py
│   ├── styles/
│   │   └── theme.css
│   └── state/
│       └── session.py
│
├── training/
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── prepare_dataset.py
│   ├── train_vqc.py
│   ├── evaluate.py
│   └── save_artifacts.py
│
├── eeg_processing/
│   ├── __init__.py
│   ├── loader.py
│   ├── channel_selector.py
│   ├── filtering.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   └── pipeline.py
│
├── quantum/
│   ├── __init__.py
│   ├── feature_map.py
│   ├── ansatz.py
│   ├── classifier.py
│   ├── trainer.py
│   ├── predictor.py
│   └── artifacts.py
│
├── inference/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predictor.py
│   ├── aggregation.py
│   └── validation.py
│
├── models/
│   ├── README.md
│   └── .gitkeep
│
├── reports/
│   ├── __init__.py
│   ├── generator.py
│   └── templates/
│
├── visualization/
│   ├── __init__.py
│   ├── eeg.py
│   ├── timeline.py
│   └── metrics.py
│
├── config/
│   ├── app.yaml
│   ├── training.yaml
│   └── features.yaml
│
├── data/
│   ├── training/
│   │   └── README.md
│   └── uploads/
│       └── .gitkeep
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── train.py
│   └── validate_model.py
│
├── docs/
├── assets/
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Do not place the training dataset inside the production application package.

---

# 8. TRAINING DATASET

The training lifecycle will use the CHB-MIT Scalp EEG Database.

The initial project training configuration is based on:

- CHB01
- CHB03
- CHB05

The selected training recordings may be configured rather than hardcoded.

The training dataset must be treated as a developer/research resource.

The end-user application must not depend on it.

---

# 9. TRAINING PIPELINE SPECIFICATION

## 9.1 EDF Loading

Use MNE to load EDF recordings.

Validate:

- file exists
- file extension
- EDF readability
- sampling frequency
- channels
- duration
- signal availability

Store metadata.

Example metadata:

```json
{
  "file_name": "training_record.edf",
  "sampling_frequency": 256,
  "duration_seconds": 3600,
  "channel_count": 23
}
```

---

# 10. CHANNEL PROCESSING

Initial training channels:

```text
FP1-F7
F3-C3
FZ-CZ
P4-O2
```

Channel configuration must be stored in `config/features.yaml`.

Do not hardcode channel names across multiple modules.

If an uploaded EDF is missing required channels, the inference UI must clearly explain which channels are missing.

Do not silently substitute unrelated channels.

---

# 11. SIGNAL FILTERING

Apply bandpass filtering:

```text
Low cutoff: 0.5 Hz
High cutoff: 40 Hz
```

Use the same filter configuration during training and inference.

The filter configuration must come from configuration.

---

# 12. SEGMENTATION

Initial segment size:

```text
2 seconds
```

For sampling frequency:

```text
256 Hz
```

window size:

```text
512 samples
```

The implementation must calculate the number of samples from the actual configured sampling frequency instead of blindly assuming 512 for every recording.

Do not silently use a different segment duration during inference.

---

# 13. TRAINING LABEL GENERATION

During training only, seizure annotations from the dataset's annotation/summary information are used to assign labels.

Labels:

```text
0 = Non-Seizure
1 = Seizure
```

The inference application must not require seizure labels because the purpose of inference is to predict them.

This distinction is mandatory.

---

# 14. FEATURE EXTRACTION

The training and inference pipelines must share the same feature extraction implementation.

For every configured EEG channel, calculate:

1. Wavelet Energy
2. Shannon Entropy
3. Delta Band Power
4. Hjorth Mobility

Initial frequency band:

```text
Delta = 0.5–4 Hz
```

Initial wavelet:

```text
db4
```

The feature order must be deterministic and stored in metadata.

For four channels and four features:

```text
4 × 4 = 16 features
```

---

# 15. FEATURE SCHEMA

Create a feature configuration that explicitly defines:

- channel order
- feature order
- wavelet
- wavelet level
- entropy method
- PSD method
- delta frequency range
- segment duration

Example:

```yaml
channels:
  - FP1-F7
  - F3-C3
  - FZ-CZ
  - P4-O2

segment_duration_seconds: 2

filter:
  low_hz: 0.5
  high_hz: 40.0

wavelet:
  name: db4
  level: 4

delta_band:
  low_hz: 0.5
  high_hz: 4.0
```

---

# 16. DATASET CONSTRUCTION

Training dataset builder should produce:

```text
X
y
metadata
```

Where:

```text
X = feature matrix
y = labels
metadata = provenance/configuration
```

Save training data in a reproducible format.

Do not use training data directly in the application UI.

---

# 17. CLASS BALANCING

The training dataset is highly imbalanced.

Use a documented balancing strategy.

Initial strategy:

- retain all minority seizure samples
- undersample non-seizure samples

The exact random seed must be configurable.

Store balancing information in training metadata.

The inference application never performs balancing.

---

# 18. TRAIN/TEST SPLIT

Use a reproducible split.

Recommended:

```text
80% training
20% test
```

Use stratification.

Set and record a random seed.

Do not use test samples during training.

---

# 19. PCA

Fit PCA during training only.

Initial target:

```text
16 → 4 components
```

The fitted PCA object must be persisted.

During inference:

```python
pca.transform(X_new)
```

Never:

```python
pca.fit_transform(X_new)
```

This distinction is mandatory.

Store:

- PCA object
- component count
- explained variance
- feature schema version

---

# 20. SCALING / QUANTUM INPUT PREPARATION

The training pipeline must fit the selected scaler or deterministic quantum input transformation.

The exact transformation must be persisted or deterministically reproduced.

The inference pipeline must apply the saved transformation.

The final quantum input range should match the training configuration.

If angle encoding uses:

```text
0 → π
```

the same mapping must be used during inference.

---

# 21. QUANTUM MODEL

Primary classifier:

```text
Variational Quantum Classifier
```

Use Qiskit Machine Learning.

Initial design:

```text
4 PCA features
        ↓
4 qubits
        ↓
ZZFeatureMap
        ↓
RealAmplitudes
        ↓
COBYLA
        ↓
VQC
```

Quantum hyperparameters must be configurable.

---

# 22. QUANTUM MODEL ARTIFACTS

The training system must persist everything required for inference.

Recommended artifact directory:

```text
models/
└── v1/
    ├── vqc_model.*
    ├── pca.joblib
    ├── scaler.joblib
    ├── feature_config.json
    ├── preprocessing_config.json
    └── model_metadata.json
```

Do not assume a particular Qiskit serialization format without verifying that it can reliably restore the trained object.

Implement a tested artifact save/load mechanism.

---

# 23. MODEL METADATA

Example:

```json
{
  "model_name": "quantum_eeg_vqc",
  "model_version": "1.0.0",
  "classifier": "VQC",
  "qubits": 4,
  "feature_count_before_pca": 16,
  "feature_count_after_pca": 4,
  "segment_duration_seconds": 2,
  "channels": [
    "FP1-F7",
    "F3-C3",
    "FZ-CZ",
    "P4-O2"
  ],
  "filter_low_hz": 0.5,
  "filter_high_hz": 40.0
}
```

Also record training metrics and software versions.

---

# 24. INFERENCE PIPELINE

The inference pipeline is the most important user-facing component.

Input:

```text
new EDF file
```

Output:

```text
prediction result
segment predictions
probabilities where supported
timeline
summary
```

The inference pipeline must:

1. Validate file.
2. Load EDF.
3. Validate required channels.
4. Read sampling frequency.
5. Apply configured preprocessing.
6. Segment signal.
7. Extract exactly the same features.
8. Apply saved PCA.
9. Apply saved scaling.
10. Load saved VQC.
11. Predict each segment.
12. Aggregate segment predictions.
13. Generate result object.

---

# 25. RECORDING-LEVEL PREDICTION

The model predicts at segment level.

The application must provide both:

## Segment-level result

```text
Segment 001 → Non-Seizure
Segment 002 → Non-Seizure
Segment 003 → Seizure
...
```

## Recording-level result

Use a clearly documented aggregation rule.

Example:

```text
seizure_segments > 0
→ Possible seizure activity detected
```

However, do not claim clinical diagnosis.

A more robust aggregation strategy may be introduced later, such as consecutive positive segments or configurable thresholds.

---

# 26. RESULT DATA MODEL

Create a structured result object containing:

```text
recording_name
duration_seconds
sampling_frequency
channels_used
total_segments
seizure_segments
non_seizure_segments
prediction_summary
segment_predictions
segment_times
model_version
processing_time
warnings
```

Do not pass unstructured dictionaries throughout the entire application if typed models can be used.

---

# 27. END-USER UI

The application must be built around one simple core action:

```text
UPLOAD EDF → ANALYZE → RESULTS
```

The user should not need to understand:

- PCA
- VQC
- feature engineering
- dataset balancing
- training

Those details can be shown in an optional technical/model information page.

---

# 28. UI PAGES

## Dashboard

Show:

- system status
- model version
- model readiness
- last analysis
- quick-start upload action
- high-level model information

## Analyze

Main user workflow:

```text
Upload EDF
↓
Validate
↓
Review metadata
↓
Analyze
```

## Results

Show:

- overall prediction
- seizure segment count
- timeline
- EEG visualization
- segment table
- model information
- processing time
- report download

## History

Optional local history of previous analyses.

Do not store sensitive data by default.

## Model Information

Show:

- VQC
- qubits
- feature count
- PCA dimensions
- model version
- training metrics

## Settings

Show safe configuration.

Do not expose dangerous model-training operations to normal users.

## About

Explain:

- project
- research objective
- technologies
- limitations
- responsible-use disclaimer

---

# 29. DASHBOARD DESIGN

Design a modern professional dashboard.

Recommended layout:

```text
┌───────────────────────────────────────────────┐
│ Quantum EEG Signal Classifier                │
│ Hybrid Quantum-Classical Seizure Detection   │
├───────────────────────────────────────────────┤
│                                               │
│ [ Upload EEG ]        [ Model Ready ● ]       │
│                                               │
├────────────┬────────────┬────────────┬─────────┤
│ Model      │ Qubits     │ Features   │ Status  │
│ VQC v1     │ 4          │ 16 → 4     │ Ready   │
├────────────┴────────────┴────────────┴─────────┤
│ Recent Analysis                               │
│                                               │
│ Upload an EDF file to begin analysis.         │
└───────────────────────────────────────────────┘
```

---

# 30. UPLOAD EXPERIENCE

Support:

- drag and drop
- `.edf` validation
- file size validation
- readable error messages
- metadata preview

Do not begin processing immediately after upload.

Let the user explicitly click:

```text
Analyze EEG
```

---

# 31. PROCESSING EXPERIENCE

Display a progress sequence:

```text
✓ EDF loaded
✓ Channels validated
✓ Signal filtered
✓ EEG segmented
✓ Features extracted
✓ PCA applied
✓ Quantum model loaded
✓ Predictions generated
✓ Results prepared
```

Do not show fake progress.

Progress must correspond to actual pipeline stages.

---

# 32. EEG VISUALIZATION

Use Plotly for interactive plots where appropriate.

Features:

- channel selector
- time range
- zoom
- pan
- seizure interval highlighting
- normal/seizure legend

Avoid rendering thousands of segments as separate traces.

Downsample or window the visualization for performance.

---

# 33. RESULTS DESIGN

Example:

```text
┌─────────────────────────────────────┐
│ Analysis Result                     │
│                                     │
│ ⚠ Possible Seizure Activity        │
│                                     │
│ Seizure Segments: 8                 │
│ Total Segments: 1799                │
│ Model: Quantum VQC v1.0             │
└─────────────────────────────────────┘
```

Use clear wording such as:

- “Possible seizure activity detected”
- “No seizure activity detected by the model”

Do not display:

- “Patient definitely has epilepsy”
- “Medical diagnosis confirmed”

---

# 34. REPORT GENERATION

Generate a PDF containing:

- application name
- analysis timestamp
- uploaded recording name
- recording duration
- sampling frequency
- channels analyzed
- total segments
- seizure segments
- non-seizure segments
- prediction summary
- model version
- selected visualizations
- disclaimer

The report must clearly identify that the output is a model prediction and not a medical diagnosis.

---

# 35. ERROR HANDLING

Handle:

- invalid file
- corrupted EDF
- missing channels
- unsupported sampling configuration
- empty signal
- insufficient duration
- missing model artifacts
- incompatible model version
- quantum runtime errors
- memory errors
- report generation errors

Every error should have:

1. technical log
2. user-friendly message
3. recovery suggestion where possible

---

# 36. LOGGING

Use Python `logging`.

Log:

- application startup
- model loading
- inference start/end
- file validation
- processing duration
- errors
- warnings

Never log sensitive raw EEG data unnecessarily.

---

# 37. MODEL READINESS

At application startup:

1. Check model artifacts.
2. Validate metadata.
3. Validate required files.
4. Load model only when safe.
5. Display model status.

If the model is missing:

```text
Model unavailable.

Please run the developer training pipeline before using inference.
```

Do not automatically train.

---

# 38. TRAINING COMMAND

Provide a dedicated developer command such as:

```bash
python scripts/train.py
```

This command may:

- load training dataset
- build features
- prepare data
- fit PCA
- fit scaler
- train VQC
- evaluate
- save artifacts

The Streamlit app must never call this command.

---

# 39. MODEL VALIDATION COMMAND

Provide:

```bash
python scripts/validate_model.py
```

It should verify:

- artifact existence
- metadata
- PCA
- scaler
- VQC loading
- expected feature count
- expected channel list
- version compatibility

---

# 40. TESTING STRATEGY

Implement tests before considering the project complete.

## Unit Tests

Test:

- EDF validation
- channel validation
- filtering
- segmentation
- each feature function
- feature ordering
- PCA transformation
- scaler transformation
- artifact loading
- prediction aggregation

## Integration Tests

Test:

```text
EDF
→ preprocessing
→ features
→ PCA
→ scaler
→ VQC
→ prediction
```

Use small test fixtures.

Do not require the full CHB-MIT dataset for ordinary unit tests.

---

# 41. TRAINING/INFERENCE CONSISTENCY TEST

This is critical.

Create a test that verifies:

```text
Training feature schema
==
Inference feature schema
```

Also verify:

```text
Training channels
==
Inference channels
```

and:

```text
Training preprocessing parameters
==
Inference preprocessing parameters
```

A schema/configuration mismatch must fail loudly.

---

# 42. REPRODUCIBILITY

Training must record:

- random seed
- Python version
- package versions
- Qiskit version
- dataset configuration
- selected EDF files
- channels
- preprocessing configuration
- feature configuration
- PCA configuration
- VQC configuration
- training metrics

---

# 43. SECURITY AND PRIVACY

The application should follow privacy-conscious defaults.

- Do not upload user EDF files to third-party services.
- Process files locally unless cloud deployment explicitly requires otherwise.
- Do not persist uploaded EEG by default.
- Provide cleanup after analysis.
- Avoid logging raw EEG.
- Avoid storing patient identifiers unnecessarily.
- Make storage behavior explicit.

---

# 44. DATA RETENTION

Default behavior:

```text
Upload
↓
Process
↓
Generate result
↓
Generate report
↓
Temporary files cleaned
```

If history is implemented, make it opt-in and configurable.

---

# 45. MODEL VERSIONING

Use model versions.

Example:

```text
models/
├── v1.0.0/
├── v1.1.0/
└── current/
```

Each model version must have metadata.

---

# 46. CONFIGURATION

Configuration must control:

- channels
- filter frequencies
- segment duration
- wavelet
- wavelet level
- delta band
- PCA components
- scaling range
- model version
- UI settings

Avoid scattered constants.

---

# 47. DEPENDENCY MANAGEMENT

Create:

```text
requirements.txt
```

and preferably:

```text
pyproject.toml
```

Pin or constrain versions where required for reproducibility.

Pay particular attention to compatibility between:

- Qiskit
- Qiskit Machine Learning
- NumPy
- SciPy
- scikit-learn

Do not blindly upgrade quantum dependencies.

---

# 48. GITIGNORE

Never commit:

- `.venv/`
- secrets
- `.env`
- temporary uploads
- caches
- large raw training datasets unless explicitly intended
- generated temporary reports

Whether trained model artifacts are committed should be decided based on repository size and licensing; document the decision.

---

# 49. DOCKER

Provide a Dockerfile for the inference application.

The container should:

- install dependencies
- copy application code
- expose Streamlit
- start the application
- use environment/configuration correctly

Do not place the training dataset inside the production container.

---

# 50. FUTURE FASTAPI ARCHITECTURE

The current implementation may use Streamlit directly.

Design services so they can later be exposed through FastAPI.

Future:

```text
React / Streamlit
       ↓
FastAPI
       ↓
Inference Service
       ↓
VQC Model
```

Do not make the current architecture dependent on Streamlit-specific business logic.

---

# 51. FUTURE IBM QUANTUM SUPPORT

Design the quantum layer so simulator and hardware execution can be separated.

Future architecture:

```text
QuantumModelInterface
        ↓
SimulatorBackend
        ↓
IBMQuantumBackend
```

The initial implementation may use a local simulator.

Do not claim hardware execution unless it is actually configured and tested.

---

# 52. PERFORMANCE REQUIREMENTS

Optimize for:

- large EDF recordings
- memory consumption
- feature extraction
- UI responsiveness
- visualization

Use caching for:

- loaded model artifacts
- immutable configuration
- expensive reusable resources

Do not cache user-specific prediction results incorrectly across users.

---

# 53. UX PRINCIPLES

The user should always know:

- what the application is doing
- what file is being analyzed
- whether the model is ready
- whether analysis succeeded
- what the prediction means
- what the limitations are

Avoid technical jargon on the primary screen.

Put quantum/technical details in the Model Information section.

---

# 54. RESPONSIBLE AI / MEDICAL DISCLAIMER

The application must clearly state:

> This system is a research and educational prototype for EEG seizure activity classification. Model predictions are not a medical diagnosis and must not replace evaluation by a qualified healthcare professional.

The wording should be visible on the About page and results/report areas.

---

# 55. COMPLETION CRITERIA

The project is considered complete only when all of the following work:

## Training

```text
Training dataset
→ feature generation
→ PCA
→ scaling
→ VQC training
→ evaluation
→ artifact persistence
```

## Inference

```text
New EDF
→ validation
→ preprocessing
→ feature extraction
→ PCA transform
→ scaling
→ VQC inference
→ segment predictions
→ recording result
```

## UI

```text
Upload
→ Analyze
→ Processing
→ Results
→ Report
```

## Engineering

- tests pass
- configuration works
- model loading works
- missing artifacts fail gracefully
- no automatic retraining
- no built-in dataset required for inference
- documentation exists
- Docker build works

---

# 56. ACCEPTANCE TEST

The most important end-to-end acceptance test is:

1. Train the model using the developer training command.
2. Confirm artifacts are generated.
3. Close the training environment.
4. Start the Streamlit application.
5. Confirm the application does not train.
6. Upload a new/unseen EDF.
7. Confirm the file is validated.
8. Click Analyze.
9. Confirm preprocessing runs.
10. Confirm features are generated.
11. Confirm saved PCA is loaded.
12. Confirm saved scaler is loaded.
13. Confirm saved VQC is loaded.
14. Confirm predictions are generated.
15. Confirm segment results are shown.
16. Confirm recording-level result is shown.
17. Confirm EEG timeline is shown.
18. Generate PDF.
19. Restart application.
20. Upload another EDF.
21. Confirm the application again performs inference without retraining.

---

# 57. ANTIGRAVITY IMPLEMENTATION WORKFLOW

Antigravity must implement this project in phases.

## Phase 0 — Architecture

Before coding:

- inspect this plan
- resolve contradictions
- propose final folder structure
- identify dependency risks
- identify Qiskit serialization strategy
- identify model artifact strategy

Do not generate the full application immediately.

## Phase 1 — Foundation

Create:

- project structure
- configuration
- logging
- dependency files
- shared data models
- exceptions

## Phase 2 — EEG Processing

Implement and test:

- EDF loader
- channel validation
- filtering
- segmentation
- feature extraction

## Phase 3 — Training

Implement:

- dataset builder
- labeling
- balancing
- train/test split
- PCA
- scaler
- VQC
- evaluation
- artifact persistence

## Phase 4 — Inference

Implement:

- upload validation
- inference pipeline
- artifact loader
- prediction
- aggregation

## Phase 5 — UI

Implement:

- dashboard
- upload page
- analysis workflow
- results page
- visualization
- model information
- report download

## Phase 6 — Testing

Implement:

- unit tests
- integration tests
- artifact tests
- training/inference consistency tests

## Phase 7 — Deployment

Implement:

- Docker
- production configuration
- documentation

---

# 58. ANTIGRAVITY CODING RULES

When implementing:

1. Read this entire plan before creating code.
2. Do not create a fake dataset.
3. Do not create hardcoded demo predictions.
4. Do not replace VQC with a classical model.
5. Do not automatically train on application startup.
6. Do not make the end user select training files.
7. Do not fit PCA during inference.
8. Do not fit the scaler during inference.
9. Do not change feature order between training and inference.
10. Do not silently ignore missing channels.
11. Do not fabricate confidence values.
12. Do not show fake processing progress.
13. Do not create medical claims.
14. Do not hardcode paths.
15. Do not duplicate preprocessing logic.
16. Do not hide exceptions.
17. Do not implement features that are not required without explaining why.
18. Preserve model reproducibility.
19. Prefer tested, maintainable implementations over clever code.
20. Keep the application usable by a non-technical end user.

---

# 59. IMPORTANT MODEL ACCURACY POLICY

Do not manipulate test data to obtain a desired accuracy.

Do not:

- leak test data into training
- tune against the test set repeatedly
- duplicate seizure samples into test data
- use labels during inference
- report fabricated metrics

If accuracy is below a desired target, improve the scientific pipeline honestly.

The project should report actual results.

---

# 60. FINAL USER EXPERIENCE

The final application should make the workflow extremely simple.

```text
                    USER
                      │
                      ▼
              ┌───────────────┐
              │ Upload EDF    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ File Validate │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Analyze EEG   │
              └───────┬───────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ Preprocess +         │
            │ Feature Extraction   │
            └──────────┬──────────┘
                       │
                       ▼
                 ┌──────────┐
                 │ PCA       │
                 │ Transform │
                 └─────┬────┘
                       │
                       ▼
                 ┌──────────┐
                 │ VQC      │
                 │ Inference│
                 └─────┬────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Results         │
              │                 │
              │ Seizure/Normal  │
              │ Timeline        │
              │ Segments        │
              └────────┬────────┘
                       │
                       ▼
                ┌────────────┐
                │ PDF Report │
                └────────────┘
```

---

# 61. FINAL DIRECTIVE TO ANTIGRAVITY

Build this project from scratch according to this document.

The **single most important product requirement** is:

> The application must contain a previously trained Quantum VQC and allow an end user to upload a new/unseen EDF recording and receive a seizure classification result without supplying the training dataset and without retraining the model.

The CHB-MIT dataset belongs exclusively to the developer/research training workflow.

The end-user application is an inference product.

Do not confuse these two workflows.

Implement the project incrementally, test every layer, maintain a clean architecture, and do not declare the project complete until the end-to-end acceptance test passes.
