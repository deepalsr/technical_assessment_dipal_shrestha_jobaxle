# Housing Price Prediction — Technical Assessment

A machine learning project that builds, evaluates, and deploys regression models to automatically estimate residential house prices from physical and categorical property features.

---

## Project Structure

```
TECHNICAL_ASSESSMENT_/
├── ai_ml/
│   ├── data/
│   │   └── Housing.csv              # Raw dataset (545 rows × 13 columns)
│   └── notebooks/
│       ├── models/
│       │   ├── dt_best.pkl          # Saved Decision Tree (max_depth=5)
│       │   ├── lr_model.pkl         # Saved Linear Regression model
│       │   └── scaler_knn.pkl       # StandardScaler used for KNN
│       └── notebook.ipynb           # Main analysis notebook
├── venv/                            # Python virtual environment
├── README.md
└── requirements.txt
```

---

## Dataset

**File:** `data/Housing.csv`  
**Shape:** 545 rows × 13 columns  
**Target:** `price` (house sale price in Indian Rupees)  
**Missing values:** None — dataset is clean on load

| Column | Type | Description |
|---|---|---|
| `price` | int64 | Sale price (target variable) |
| `area` | int64 | Total floor area (sq ft) |
| `bedrooms` | int64 | Number of bedrooms |
| `bathrooms` | int64 | Number of bathrooms |
| `stories` | int64 | Number of floors |
| `mainroad` | str | Connected to main road (yes/no) |
| `guestroom` | str | Has guest room (yes/no) |
| `basement` | str | Has basement (yes/no) |
| `hotwaterheating` | str | Has hot-water heating (yes/no) |
| `airconditioning` | str | Has air conditioning (yes/no) |
| `parking` | int64 | Number of parking spots |
| `prefarea` | str | In preferred area (yes/no) |
| `furnishingstatus` | str | furnished / semi-furnished / unfurnished |

---

## Task Implementations

### Q1 — Data Loading & Initial Exploration

Loaded `Housing.csv` using `pandas`. Confirmed shape (545 × 13), data types, and zero missing values across all columns. Displayed the first five rows to verify column names and value formats.

---

### Q2 — Data Cleaning & Preprocessing

Inspected all columns for nulls and inconsistencies. Binary yes/no string columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) were mapped to integer flags (1/0) to prepare them for modelling without inflating feature dimensionality.

---

### Q3 — Exploratory Data Analysis (EDA)

Generated four visualisations to understand the data:

- **Price Distribution** — right-skewed; most houses priced ₹2M–₹6M with a long upper tail of luxury properties.
- **Area vs Price** — scatter plot showing a moderate positive linear trend.
- **Avg Price by Furnishing Status** — Furnished > Semi-furnished > Unfurnished.
- **Correlation Heatmap** — numeric features only; `area` is the strongest predictor.

**Key correlation findings (with price):**

| Feature | Correlation |
|---|---|
| `area` | 0.536 |
| `bathrooms` | 0.518 |
| `airconditioning` | 0.453 |
| `stories` | 0.421 |
| `parking` | 0.384 |
| `bedrooms` | 0.366 |
| `hotwaterheating` | 0.093 (weakest) |

---

### Q4 — Categorical Encoding

Applied `pd.get_dummies` with `drop_first=True` to one-hot encode `furnishingstatus` (the only remaining object column after binary mapping). This produced two dummy columns (`furnishingstatus_semi-furnished`, `furnishingstatus_unfurnished`), expanding the feature set to 13 predictors.

---

### Q5 — Train / Test Split

Split the encoded dataset 80 / 20 using `train_test_split` with `random_state=42`.

| Set | Rows |
|---|---|
| Train | 436 |
| Test | 109 |

**Rationale:** An 80/20 ratio keeps a large enough training set for stable OLS parameter estimates while leaving sufficient held-out samples for a reliable performance check. This is a standard baseline for regression, tree, KNN, and clustering workflows on datasets of this size.

---

### Q6 — Linear Regression Training & Diagnostics

Trained `sklearn.linear_model.LinearRegression` on the training set. Evaluated on both splits to monitor overfitting.

| Metric | Train | Test |
|---|---|---|
| MAE | ₹719,243 | ₹970,043 |
| RMSE | ₹984,052 | ₹1,324,507 |
| R² | 0.6859 | 0.6529 |

**Diagnostics:**
- Predicted vs. actual shows a positive trend, but spread widens at the high-price end — the model systematically underpredicts luxury properties.
- Residuals histogram is centred near zero but right-skewed with a long positive tail, confirming underestimation at the upper range.

**Scaling experiment:** Applying `StandardScaler` before fitting produced identical RMSE and R² — as expected for ordinary least squares, which is scale-invariant. The unscaled version was retained for LR; scaling was reserved for KNN.

---

### Q7 — Model Evaluation (Linear Regression)

Formalised the metrics table above. The train–test R² gap (~0.03) is small, indicating low overfitting. The model explains roughly 65% of price variance on unseen data — a solid baseline for a dataset of 545 observations.

---

### Q8 — Decision Tree Depth Analysis

Trained `DecisionTreeRegressor` across eight `max_depth` values: 2, 3, 4, 5, 7, 9, 12, and unlimited.

**Findings:**
- Train R² rises monotonically towards 1.0 as depth increases.
- Test R² peaks around **depth 7** then degrades — classic overfitting signal.
- Test RMSE is minimised near depth 7; unlimited trees memorise training data without improving generalisation.
- **Selected depth = 5** as the saved model — simpler, more stable, and still close to peak test performance.

---

### Q9 — Feature Scaling Investigation

Compared Linear Regression performance with and without `StandardScaler`. Both settings produced:

- **RMSE:** 1,324,507  
- **R²:** 0.6529

Confirmed that OLS is scale-invariant; no scaling is needed for LR. Documented that distance-based models (KNN) require scaling to prevent high-magnitude features from dominating the distance metric.

---

### Q10 — KNN Regression with Hyperparameter Tuning

Applied `StandardScaler` to features, then searched over a parameter grid using 5-fold `GridSearchCV` with `neg_root_mean_squared_error` scoring.

**Search space:**
- `n_neighbors`: 3, 5, 7, 9, 11, 13, 15, 17, 19
- `weights`: uniform, distance
- `p` (Minkowski): 1 (Manhattan), 2 (Euclidean)

**Best parameters:** `n_neighbors=13`, `weights='distance'`, `p=2`

| Metric | Value |
|---|---|
| Test RMSE | ₹1,437,391 |
| Test R² | 0.5912 |

KNN with distance-weighted voting and 13 neighbours performed better than uniform-weighted or Manhattan-distance variants, but still fell short of Linear Regression.

---

### Q11 — Model Persistence

Saved three artefacts using `joblib`:

| File | Contents |
|---|---|
| `models/lr_model.pkl` | Trained Linear Regression model |
| `models/scaler_knn.pkl` | StandardScaler fitted on training data |
| `models/dt_best.pkl` | Decision Tree (max_depth=5) |

Reloaded all three and verified predictions match exactly (`np.allclose → True`). Post-reload R² values confirmed: LR 0.6529, DT 0.4602.

---

### Q12 — Prediction on Unseen Data

Constructed five synthetic houses not in the training set and ran them through both the reloaded LR and DT models after replicating the same preprocessing pipeline (binary mapping + one-hot encoding + column alignment).

| House | Area | Beds | AC | Furnishing | LR Price | DT Price |
|---|---|---|---|---|---|---|
| House 1 | 5,000 | 3 | Yes | Furnished | ~₹7.5M | — |
| House 2 | 3,200 | 2 | No | Unfurnished | lowest | lowest |
| House 3 | 7,800 | 5 | Yes | Furnished | highest | highest |
| House 4 | 4,500 | 3 | No | Semi-furnished | mid | mid |
| House 5 | 6,200 | 4 | Yes | Unfurnished | mid-high | mid-high |

All predictions fell within 50%–150% of the training price range. LR and DT agreed directionally; minor mid-range divergence reflects different model assumptions. Column alignment (`unseen_proc = unseen_proc[X.columns]`) was essential to prevent silent feature mismatches.

---

### Q13 — K-Means Clustering

Applied unsupervised clustering to segment the housing stock into natural market tiers. Used the **Elbow Method** (inertia vs. k) to select **k = 3**.

**Cluster Interpretation:**

| Cluster | Market Segment | Characteristics |
|---|---|---|
| 0 | Budget / Entry-level | Small area, fewer bedrooms, minimal amenities, low price |
| 1 | Mid-range | Medium area, ~3 bedrooms, moderate price, some amenities |
| 2 | Premium / Luxury | Large area, 4+ bedrooms, AC, preferred area, more parking, high price |

Clusters map naturally to real-world housing market tiers and can inform targeted pricing strategies or buyer recommendation systems.

---

### Q14 — Multi-Model Comparison

Side-by-side bar charts compared all three models on R², MAE, and RMSE on the held-out test set.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **₹970,043** | **₹1,324,507** | **0.6529** |
| Decision Tree (depth=5) | higher | higher | 0.4602 |
| KNN (best params) | higher | ₹1,437,391 | 0.5912 |

**Linear Regression is the best-performing model** on all three metrics for this dataset.

---

### Q15 — Real-World Problem: Automated House Price Estimation

**Problem:** A real-estate platform wants to automatically estimate a fair listing price for any property given its physical characteristics — helping sellers set competitive prices, buyers spot overpriced listings, and agents benchmark properties.

**Solution:** An end-to-end `estimate_price()` pipeline that accepts raw inputs, applies the same preprocessing steps (binary encoding → one-hot encoding → column alignment) and returns a price from the saved `lr_model.pkl`.

**Example predictions:**

| Property Type | Estimated Price |
|---|---|
| Mid-range family home (4,200 sqft, semi-furnished, AC) | ₹6,259,720 |
| Luxury bungalow (7,500 sqft, fully furnished, all amenities) | ₹10,612,951 |
| Budget apartment (2,600 sqft, unfurnished, no AC) | ₹2,529,030 |

---

## Why Linear Regression Was Chosen

| Criterion | Linear Regression | Decision Tree | KNN |
|---|:---:|:---:|:---:|
| Test R² | **0.6529** | 0.4602 | 0.5912 |
| Interpretability | Best | Good | Black-box |
| Scaling required | No | No | Yes |
| Handles non-linearity | No | Yes | Yes |
| Overfitting risk | Low | Medium | Medium |
| Inference speed | Fast | Fast | Slow at scale |

**Reasoning:**

1. **EDA confirmed approximate linearity** — the correlation heatmap showed area, bathrooms, and AC have moderate-to-strong linear relationships with price.
2. **Best test performance** — LR achieved the highest R² (0.6529) and lowest RMSE (₹1.32M) with no hyperparameter tuning.
3. **Interpretable coefficients** — each coefficient directly quantifies the marginal price impact of a feature (e.g., "every additional sq ft adds ₹X to the price"), which is essential for a client-facing pricing tool.
4. **Low complexity, easy maintenance** — no scaling required, no depth or k to tune, faster to retrain and monitor in production.
5. **Stable bias-variance trade-off** — the small train–test R² gap (~0.03) confirms the model generalises well without overfitting.

---

## Key Findings

- **Area** is the strongest single predictor of price (r = 0.54).
- **Air conditioning** adds the most value among binary amenity features.
- **Furnishing status** has a clear price hierarchy: Furnished > Semi-furnished > Unfurnished.
- **Hot water heating** has almost no predictive power (r = 0.09).
- **Linear Regression outperforms** both Decision Tree and KNN on this dataset in terms of test R², MAE, and RMSE.
- **Decision Trees overfit** rapidly beyond depth 7; depth 5 is the sweet spot.
- **KNN** requires careful scaling and hyperparameter tuning but still trails LR in accuracy.
- The dataset clusters naturally into three market segments: Budget, Mid-range, and Premium.

---

## Future Improvements

1. Collect richer features — house age, GPS location, proximity to schools and hospitals.
2. Upgrade to **Gradient Boosting (XGBoost / LightGBM)** to capture non-linear interactions while remaining interpretable via SHAP values.
3. Replace single train/test split with **k-fold cross-validation** for more reliable evaluation.
4. Deploy `lr_model.pkl` via a **REST API (FastAPI / Flask)** for real-time price estimation.
5. Add a **monitoring pipeline** to retrigger retraining when prediction drift is detected.

---

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook ai_ml/notebooks/notebook.ipynb
```

**Python version:** 3.13.9  
**Key libraries:** pandas, numpy, matplotlib, scikit-learn, joblib