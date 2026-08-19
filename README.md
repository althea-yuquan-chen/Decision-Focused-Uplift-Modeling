# Decision-Focused Uplift Modeling (DFUM)

A research implementation of **Decision-Focused Uplift Modeling**, a framework that jointly optimizes uplift prediction and treatment allocation decisions under a top-K treatment budget. The model (**DFUM with Shared Layer**) is evaluated on the public CRITEO-UPLIFT v2 dataset, and benchmarked against S-Learner, X-Learner, UpliftRank, and GRF baselines.

> **Note:** the original DFUM notebook is deprecated — see `code/my_dfum_criteo_DEPRECATED_no_budget.ipynb`. Its decision loss had no top-K budget constraint, and a separate training-loop bug meant its alpha sweep never actually varied alpha. `code/my_dfum_criteo_topk.ipynb` is the current model.

---

## Overview

Uplift modeling estimates the **Individual Treatment Effect (ITE / CATE)** — how much a treatment (e.g. a marketing coupon) changes an individual's outcome. Standard uplift models optimize prediction accuracy, but this project introduces a **decision-focused loss** that directly optimizes the downstream treatment allocation policy — specifically, the policy value of treating the top-K individuals by predicted uplift, where K is a budget constraint (you can't afford to treat everyone).

The core model (`DFUMModel`, in `code/my_dfum_criteo_topk.ipynb`) combines:
- A **shared layer** that processes treatment and control branches together
- A **prediction loss** (binary cross-entropy on outcome)
- A **decision loss** derived from the Lagrangian dual of the top-K selection LP: `maximize Σx_i·τ_i s.t. Σx_i=K, 0≤x_i≤1`, whose dual `g(λ) = Σmax(0, τ_i−λ) + λK` is minimized in closed form at `λ* = the K-th largest τ̂` (an order statistic — computed directly each batch, not learned)
- A tunable **alpha** parameter balancing the two losses: `total_loss = prediction_loss + α * decision_loss`

---

## Repository Structure

```
Decision-Focused-Uplift-Modeling/
├── code/
│   ├── my_dfum_criteo_topk.ipynb                 # ★ Main model — DFUM with top-K budget-constrained decision loss
│   ├── my_dfum_criteo_DEPRECATED_no_budget.ipynb # DEPRECATED — no budget constraint; see notebook header
│   ├── my_dfum_criteo_v4.ipynb                  # Abandoned earlier draft (separate lambda_k bug, see notebook)
│   ├── baseline_uplift_criteo.ipynb             # Baseline: S-Learner, X-Learner, UpliftRank on Criteo
│   ├── baseline_roi_criteo.ipynb                # Baseline: ROI ranking model on Criteo
│   └── baseline_marginal_utility_criteo.ipynb   # Baseline: Marginal utility model on Criteo
├── model/
│   ├── __init__.py
│   ├── uplift_model.py       # Baseline CATE models: S-Learner, X-Learner, UpliftRank
│   └── roi_model.py          # Baseline ROI models: ROI Rank, Direct Rank
├── metric/
│   └── Metric.py             # Evaluation metrics: AUCC, MT-AUCC
├── model_file/
│   └── uplift/criteo/final_model/
│       ├── my_dfum_topk/       # Weights from the current top-K model (populated by my_dfum_criteo_topk.ipynb)
│       ├── slearner/          # S-Learner saved weights (20 seeds)
│       ├── xlearner/          # X-Learner saved weights (tau_0 + tau_1, 20 seeds each)
│       └── upliftRank/        # UpliftRank saved weights (20 seeds, standard + 10mil variant)
├── results/
│   ├── my_dfum_topk_avg_uplift_gain.csv # DFUM (top-K) average uplift gain curve — current model
│   ├── grf_avg_uplift_gain.csv        # GRF baseline average uplift gain curve
│   ├── xlearner_avg_uplift_gain.csv   # X-Learner baseline average uplift gain curve
│   └── slearner_avg_uplift_gain.csv   # S-Learner baseline average uplift gain curve
├── .gitignore
└── README.md
```

---

## Model: DFUM with Shared Layer (`DFUMModel`, `code/my_dfum_criteo_topk.ipynb`)

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
- `decision_loss`: top-K budget-constrained policy value. Derived from the Lagrangian dual of `maximize Σx_i·τ_i s.t. Σx_i=K, 0≤x_i≤1`, whose dual `g(λ) = Σmax(0, τ_i−λ) + λK` is minimized in closed form at `λ* = the K-th largest τ̂`. Each batch, for each budget fraction `k` in `k_fracs`: `λ` is computed directly via `tf.math.top_k` (no gradient through it), `gate = sigmoid((τ̂−λ)/temperature)` softens the top-K indicator for gradient flow, and the policy value is a self-normalized IPS estimate with fixed denominators `k·n1/N`, `k·n0/N`
- `α`: Trade-off parameter, swept over `{0.2, 0.4, 0.6}`
- `k_fracs`: Budget fractions the decision loss is averaged over, default `(0.1, 0.2, 0.3)`

**Training config (matches the deprecated model's final-run scale; not yet re-run at this full scale for the top-K version — see notebook):**
- Dataset: CRITEO-UPLIFT v2 (13M rows, 70/30 train-test split)
- Batch size: 1,000,000
- Epochs: 1,000
- Optimizer: Adam (lr=0.005)
- Seeds: 20 independent runs per alpha value
- Validation split: 20% of training data

**Validated at reduced scale** (600k-row Criteo subsample, 3 seeds, against this repo's own S-Learner/X-Learner/GRF): wins on causalml AUUC in all 3 seeds (0.864/0.816/0.824 vs. S-Learner 0.848/0.677/0.567, X-Learner 0.750/0.794/0.637, GRF 0.814/0.800/0.797). See `code/my_dfum_criteo_topk.ipynb`'s header cell for the full writeup.

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

---

## Setup

### Prerequisites

- Python 3.11
- TensorFlow 2.21 (Keras 3)

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
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
2. Open `code/my_dfum_criteo_topk.ipynb`
3. Run cells sequentially — training, evaluation, and comparison plots are all included

> **Note:** Paths in notebooks may need adjustment depending on your local directory structure (see `README.txt` note 0).

---

## Results

Evaluation results (average uplift gain curves) are stored in `results/`. The current DFUM model (`my_dfum_topk_avg_uplift_gain.csv`) is compared against S-Learner, X-Learner, and GRF baselines. The deprecated model's weights and results have been deleted (retraining is planned via `code/my_dfum_criteo_topk.ipynb`, not a fair comparison point going forward).

---

## Dependencies Summary

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | 2.21 | Model training |
| Keras | 3.x (bundled with TensorFlow) | Neural network layers |
| CausalML | latest | AUUC metric |
| EconML | latest | GRF baseline |
| pandas, numpy | latest | Data processing |
| scikit-learn | latest | AUC, MSE metrics |
| matplotlib | latest | Plotting |

---

## Author

Yuquan (Althea) Chen