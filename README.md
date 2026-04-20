# Decision-Focused Uplift Modeling (DFUM)

A research implementation of **Decision-Focused Uplift Modeling**, a framework that jointly optimizes uplift prediction and treatment allocation decisions. The model (**DFUM with Shared Layer**) is evaluated on the public CRITEO-UPLIFT v2 dataset and a private marketing dataset, and benchmarked against S-Learner, X-Learner, UpliftRank, and GRF baselines.

---

## Overview

Uplift modeling estimates the **Individual Treatment Effect (ITE / CATE)** — how much a treatment (e.g. a marketing coupon) changes an individual's outcome. Standard uplift models optimize prediction accuracy, but this project introduces a **decision-focused loss** that directly optimizes the downstream treatment allocation policy.

The core model (`XXXModel`) combines:
- A **shared layer** that processes treatment and control branches together
- A **prediction loss** (binary cross-entropy on outcome)
- A **decision loss** (policy-gradient-style ranking loss on uplift scores)
- A tunable **alpha** parameter balancing the two losses: `total_loss = prediction_loss + α * decision_loss`

---

## Repository Structure

```
Decision-Focused-Uplift-Modeling/
├── code/
│   ├── my_dfum_criteo.ipynb                     # ★ Main model — DFUM with shared layer on Criteo
│   ├── my_dfum_criteo_v4.ipynb                  # Earlier model version (v4, prediction-loss only)
│   ├── baseline_uplift_criteo.ipynb             # Baseline: S-Learner, X-Learner, UpliftRank on Criteo
│   ├── baseline_roi_criteo.ipynb                # Baseline: ROI ranking model on Criteo
│   ├── baseline_marginal_utility_criteo.ipynb   # Baseline: Marginal utility model on Criteo
│   └── marketing_data_code/
│       ├── uplift/uplift_model_train_marketing.ipynb   # Uplift baselines on marketing data
│       ├── roi/roi_model_train_marketing.ipynb         # ROI model on marketing data
│       └── MTBAP/MTBAP_train_marketing.ipynb           # MTBAP model on marketing data
├── model/
│   ├── __init__.py
│   ├── uplift_model.py       # Baseline CATE models: S-Learner, X-Learner, UpliftRank
│   ├── roi_model.py          # Baseline ROI models: ROI Rank, Direct Rank
│   └── mt_roi_model.py       # Multi-treatment marginal utility models
├── metric/
│   └── Metric.py             # Evaluation metrics: AUCC, MT-AUCC
├── model_file/
│   └── uplift/criteo/final_model/
│       ├── my_dfum/
│       │   ├── v5_total_batch_1mil_epoch_1k/         # Final weights: α ∈ {0.2, 0.4, 0.6} × 20 seeds
│       │   └── v5_total_batch_1mil_epoch_1k_iter_10/ # Variant: α ∈ {0.1, 0.2} × 10 seeds
│       ├── slearner/          # S-Learner saved weights (20 seeds)
│       ├── xlearner/          # X-Learner saved weights (tau_0 + tau_1, 20 seeds each)
│       └── upliftRank/        # UpliftRank saved weights (20 seeds, standard + 10mil variant)
├── results/
│   ├── my_dfum_avg_uplift_gain.csv    # DFUM average uplift gain curve
│   ├── grf_avg_uplift_gain.csv        # GRF baseline average uplift gain curve
│   ├── xlearner_avg_uplift_gain.csv   # X-Learner baseline average uplift gain curve
│   └── slearner_avg_uplift_gain.csv   # S-Learner baseline average uplift gain curve
├── .gitignore
└── README.md
```

---

## Model: DFUM with Shared Layer (`XXXModel`)

**Architecture:**
- Input: 12 features (CRITEO-UPLIFT v2) + treatment indicator
- Concatenates features with a fixed `0` (control) and `1` (treatment) indicator separately
- Passes both through a **shared Dense(8, ReLU)** layer
- Separate output heads predict `p(Y=1 | X, T=0)` and `p(Y=1 | X, T=1)`
- Uplift score: `τ̂ = p(Y=1|X,T=1) - p(Y=1|X,T=0)`

**Loss Function:**
```
total_loss = prediction_loss + α * decision_loss
```
- `prediction_loss`: Binary cross-entropy on outcome prediction
- `decision_loss`: Policy-gradient ranking loss — maximizes expected reward when treating individuals ranked highest by τ̂
- `α`: Trade-off parameter, swept over `{0.2, 0.4, 0.6}` (final runs) and `{0.1, 0.2}` (iter-10 runs)

**Training config (final runs):**
- Dataset: CRITEO-UPLIFT v2 (13M rows, 70/30 train-test split)
- Batch size: 1,000,000
- Epochs: 1,000
- Optimizer: Adam (lr=0.005)
- Seeds: 20 independent runs per alpha value
- Validation split: 20% of training data

---

## Baselines

| Model | Description | Source |
|---|---|---|
| S-Learner | Single model with treatment as feature | `model/uplift_model.py` |
| X-Learner | Two-stage ITE estimator (τ₀ + τ₁) | `model/uplift_model.py` |
| UpliftRank | Policy-gradient ranking on uplift | `model/uplift_model.py` |
| GRF | Generalized Random Forests (via EconML) | External |
| ROI Rank | Decision-focused ROI optimization | `model/roi_model.py` |
| Direct Rank | Direct policy ranking | `model/roi_model.py` |
| MTBAP | Multi-treatment budget allocation | `code/marketing_data_code/` |

---

## Evaluation Metric

**AUUC (Area Under the Uplift Curve)** via [CausalML](https://causalml.readthedocs.io/en/latest/):
- Measures how well the model ranks individuals by their true treatment effect
- Results averaged over 20 independent runs per model
- Uplift gain curves saved in `results/`

**AUCC (Area Under the Cost Curve)** via `metric/Metric.py`:
- Used for ROI and marginal utility models
- Plots delta reward vs. delta cost across quantiles

---

## Dataset

**CRITEO-UPLIFT v2** — public benchmark for uplift modeling.
- Download from: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Place at: `data/criteo-uplift-v2.1.csv`
- Features: 12 anonymized features (`f0`–`f11`), binary treatment, binary outcomes (`visit`, `conversion`)
- Size: ~13M rows

**Marketing data** — private business dataset (not provided due to data privacy).

---

## Setup

### Prerequisites

- Python 3.8+
- TensorFlow 2.4.1
- Keras 2.4.3

```bash
pip install tensorflow==2.4.1 keras==2.4.3 pandas numpy scikit-learn matplotlib
```

For **AUUC** evaluation:
```bash
pip install causalml
```

For **GRF** baseline:
```bash
pip install econml
```

### Running the main model

1. Download CRITEO-UPLIFT v2 and place it at `data/criteo-uplift-v2.1.csv`
2. Open `code/my_dfum_criteo.ipynb`
3. Run cells sequentially — training, evaluation, and comparison plots are all included

> **Note:** Paths in notebooks may need adjustment depending on your local directory structure (see `README.txt` note 0).

---

## Results

Evaluation results (average uplift gain curves over 20 seeds) are stored in `results/`. The DFUM model (`my_dfum_avg_uplift_gain.csv`) is compared against S-Learner, X-Learner, and GRF baselines.

---

## Dependencies Summary

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | 2.4.1 | Model training |
| Keras | 2.4.3 | Neural network layers |
| CausalML | latest | AUUC metric |
| EconML | latest | GRF baseline |
| pandas, numpy | latest | Data processing |
| scikit-learn | latest | AUC, MSE metrics |
| matplotlib | latest | Plotting |

---

## Author

Yuquan (Althea) Chen
