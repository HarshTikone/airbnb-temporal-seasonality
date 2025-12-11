# Temporal Airbnb Seasonality and Modeling – Extra Credit (EAS 510)

This repository contains my solution to the **Temporal Airbnb Seasonality and Modeling** extra credit assignment for EAS 510 (Basics of AI).

The goal of the project is to:

1. Build a **night-level panel dataset** using InsideAirbnb calendar + listings data for multiple cities and dates.
2. Explore **seasonality patterns** in price and booking behavior.
3. Train and compare machine‑learning models to predict:

   * Nightly **price** (regression)
   * Nightly **booking probability** (classification)
4. Use **TensorBoard** to inspect neural network training, and write a short discussion of generalization and business insights.

---

## 1. Repository Structure

```text
AIRBNB-TEMPORAL-SEASONALITY/
├── data/                        # Not tracked in git; store InsideAirbnb CSVs here
│   ├── austin_2024-12-14/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── austin_2025-03-06/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── chicago_2024-12-18/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── chicago_2025-03-11/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── santacruz_2025-03-28/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── santacruz_2025-12-31/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   ├── dc_2025-03-13/
│   │   ├── calendar.csv.gz
│   │   └── listings.csv.gz
│   └── dc_2025-12-18/
│       ├── calendar.csv.gz
│       └── listings.csv.gz
│
├── images/                      # PNGs used inside the notebook
│   ├── tb_price_loss.png        # TensorBoard loss (price NN)
│   ├── tb_price_rmse.png        # TensorBoard rmse (price NN)
│   ├── tb_book_accuracy.png     # TensorBoard accuracy (booking NN)
│   └── tb_book_loss.png         # TensorBoard loss (booking NN)
│
├── logs/                        # TensorBoard log directory
│   └── ...                      # event* files created when training NNs
│
├── notebook/
│   └── airbnb_temporal_modeling.ipynb
│
├── requirements.txt
└── README.md
```

> **Note:** The `data/` folder is intentionally excluded from version control. Only the code, notebook, README, and TensorBoard images are committed.

---

## 2. Data Download and Folder Setup

1. Go to **InsideAirbnb**: [https://insideairbnb.com/get-the-data/](https://insideairbnb.com/get-the-data/).

2. For each of the following city–date combinations, download both
   `listings.csv.gz` and `calendar.csv.gz`:

   * Austin – 2024‑12‑14, 2025‑03‑06
   * Chicago – 2024‑12‑18, 2025‑03‑11
   * Santa Cruz – 2025‑03‑28, 2025‑12‑31
   * Washington DC – 2025‑03‑13, 2025‑12‑18

3. Create the `data/` folder structure shown above and place each pair of
   files in the corresponding subdirectory. **Do not unzip** the files – the
   notebook reads the `.csv.gz` files directly.

Once the data is in place, the helper function `load_city_snapshot(...)` in the
notebook will automatically find `calendar.csv.gz` and `listings.csv.gz` for
each snapshot.

---

## 3. Environment Setup

The project uses Python 3.11 and the following main libraries:

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`
* `xgboost`
* `tensorflow`
* `jupyter` (for running notebooks)

### 3.1 Create and Activate a Conda Environment (recommended)

```bash
conda create -n airbnb-extra python=3.11
conda activate airbnb-extra
pip install -r requirements.txt
```

Alternatively, you can install packages manually with `pip install` if you
prefer a different environment manager.

### 3.2 Regenerating `requirements.txt` (if needed)

If you add or change packages and want to regenerate `requirements.txt` from
your current environment, you can run:

```bash
# Inside the active environment
pip freeze > requirements.txt
```

This will capture the exact versions of all installed packages. For a smaller,
more course-friendly file, you can instead maintain it by hand and only list
high-level dependencies (as in the provided `requirements.txt`).

---

## 4. How to Run the Notebook

1. Activate your environment and navigate to the project root:

   ```bash
   conda activate airbnb-extra
   cd AIRBNB-TEMPORAL-SEASONALITY
   ```

2. Launch VS Code or Jupyter Notebook:

   ```bash
   # Option A: VS Code
   code notebook/airbnb_temporal_modeling.ipynb

   # Option B: classic Jupyter
   jupyter notebook notebook/airbnb_temporal_modeling.ipynb
   ```

3. In the notebook, select the Python kernel corresponding to the
   `airbnb-extra` environment.

4. Run all cells **from top to bottom**. The workflow is:

   1. Load city‑level snapshots (`calendar` + `listings`) and build a
      memory‑friendly night‑level panel (sampling up to 200k nights per
      snapshot).
   2. Engineer features (price cleaning, booking indicator, date features).
   3. Concatenate all snapshots into a single panel and inspect basic counts.
   4. Create seasonality plots (price and booking probability by month,
      weekend vs weekday, and by room type / city).
   5. Build a modeling dataset with one‑hot encoded room type and city dummies
      plus time features.
   6. Create a **temporal train/validation/test split** by month to avoid
      leaking future information.
   7. Train and evaluate XGBoost models for both price and bookings.
   8. Train and evaluate neural networks for both tasks and log training to
      TensorBoard.
   9. Visualize TensorBoard curves and write the final discussion.

---

## 5. Modeling Details

### 5.1 Night‑Level Panel Construction

For each city–snapshot pair, the notebook:

1. Reads `calendar.csv.gz` (subset of columns) and optionally samples up to
   200,000 nights to keep memory usage manageable.
2. Reads `listings.csv.gz` (subset of listing attributes).
3. Left‑joins calendar rows to listings on `listing_id = id`, producing one row
   per **listing‑night** with both calendar and listing information.
4. Performs cleaning and feature engineering:

   * Convert `price` from strings like `"$125.00"` to numeric `float`.
   * Create `is_booked` indicator: 1 if `available == 'f'`, 0 otherwise.
   * Parse `date` and derive `month`, `day_of_week`, `week_of_year`,
     `day_of_year`, and `is_weekend`.
   * Attach `city` and `snapshot` labels.

All city‑snapshot DataFrames are then concatenated into a single panel and used
for both EDA and modeling.

### 5.2 Temporal Train/Validation/Test Split

To mimic a realistic forecasting problem, the modeling dataset is split by
**calendar month**:

* **Train** – months 3–9 (March–September): 938,276 nights
* **Validation** – months 10–11 (October–November): 266,929 nights
* **Test** – months 12, 1, 2 (December–February): 394,795 nights

This ensures that the model is always evaluated on **later** months than those
used for training, avoiding temporal leakage.

### 5.3 Features Used

The final feature matrix includes:

* Numeric listing features:

  * `accommodates`, `bedrooms`, `beds`
  * `minimum_nights`, `maximum_nights`
  * `number_of_reviews`, `review_scores_rating`
* Time features:

  * `month`, `day_of_week`, `week_of_year`, `day_of_year`, `is_weekend`
* One‑hot encoded categorical features:

  * `room_type_*` dummies
  * `city_*` dummies

Missing numeric values are filled with the median per column, and missing
categorical values are set to "Unknown" before one‑hot encoding. All features
are combined into a single `feature_cols` list, and the resulting design
matrix is stored as `float32` for efficiency.

---

## 6. Models and Metrics

### 6.1 XGBoost Models

Two XGBoost models are trained:

1. **Price model (regression)** – `XGBRegressor`

   * Objective: predict nightly price
   * Evaluation metrics: RMSE and MAE

2. **Booking model (classification)** – `XGBClassifier`

   * Objective: predict `is_booked` (0/1)
   * Evaluation metrics: AUC and accuracy

With the chosen hyperparameters, the XGBoost models achieve approximately:

* **Price (regression)**

  * Validation RMSE ≈ **936.1**
  * Test RMSE ≈ **941.6**
  * Test MAE ≈ **263.1**

* **Bookings (classification)**

  * Validation AUC ≈ **0.759**
  * Test AUC ≈ **0.693**
  * Test accuracy ≈ **0.645**

The notebook also plots the top 15 features by importance for each model,
showing that city, room type, capacity, and time features all play major roles.

### 6.2 Neural Network Models

Two feed‑forward neural networks (built with TensorFlow/Keras) are trained on
the same feature matrix:

1. **Price NN (regression)**

   * Metric: RMSE (plus MAE printed afterwards)
   * Test RMSE ≈ **981.6**
   * Test MAE ≈ **274.9**

2. **Booking NN (classification)**

   * Metrics: AUC and accuracy
   * Test AUC ≈ **0.688**
   * Test accuracy ≈ **0.634**

Training is logged to TensorBoard, and the notebook includes screenshots of
loss/RMSE/accuracy curves for both training and validation runs.

Example TensorBoard curves (stored in the `images/` folder and referenced here):

![Price NN – loss](/images/tb_price_loss.png)
![Price NN – RMSE](/images/tb_price_rmse.png)

![Booking NN – accuracy](/images/tb_book_accuracy.png)
![Booking NN – loss](/images/tb_book_loss.png)


### 6.3 High‑Level Comparison

On this tabular data, **XGBoost** performs slightly better than the neural
networks on both tasks (lower RMSE for price, slightly higher AUC and accuracy
for bookings). TensorBoard confirms that the neural networks train stably and
show only modest gaps between training and validation curves, indicating
limited overfitting.

---

## 7. Reproducing Results

To reproduce the results:

1. Download the appropriate InsideAirbnb data and place it under `data/`.
2. Create and activate the Python environment and install dependencies
   using `requirements.txt`.
3. Open `notebook/airbnb_temporal_modeling.ipynb`.
4. Run all cells in order (or restart & run all) to:

   * Build the panel dataset
   * Generate EDA plots
   * Train XGBoost and neural network models
   * Produce evaluation metrics and TensorBoard logs

No additional configuration files are required; all paths are relative to the
project root.

---

## 8. Notes and Limitations

* For memory reasons, the notebook samples up to 200,000 nights per
  city–snapshot from the calendar files. The overall qualitative patterns
  should still be representative, but exact metrics may differ slightly from a
  full‑data run.
* The feature set is intentionally simple and focused on the variables
  discussed in class (core listing attributes + basic time features). Many
  extensions are possible (e.g., adding host features, using lagged occupancy
  rates, or more sophisticated temporal models).
* The emphasis of this project is on **temporal evaluation**, **model
  comparison**, and **interpretation**, rather than aggressive hyperparameter
  tuning.

---

## 9. Contact

If you have questions about this repository or the analysis, please contact
me at:

* Name: Harsh Mahesh Tikone
* Email: htikone@buffalo.edu
