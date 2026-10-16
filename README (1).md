# Laptop Price Tier Classification

Classifying laptops into **Cheap**, **Affordable** and **Expensive** price tiers from their specifications, using Python, pandas, scikit-learn and Plotly.

## Overview

The goal of this project is to predict which price tier a laptop belongs to based on its specifications (brand, type, screen, CPU, RAM, storage, GPU, operating system and weight). The notebook covers the full workflow: data inspection, cleaning, exploratory data analysis, feature engineering, modeling and evaluation.

## Dataset

- About 1,300 laptops with 11 original columns: `Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight` and `Price`.
- Prices are in Indian rupees (INR) and were converted to USD with an approximate rate of 0.0105 USD per INR.
- Source: public laptop price dataset from [Kaggle](https://www.kaggle.com/).
<!-- TODO: replace with the exact dataset title, owner, link and license once you have found the Kaggle page. -->

## Approach

1. **Inspect and clean:** check missing values and duplicates, convert `Ram` and `Weight` to numbers, convert prices to USD.
2. **Exploratory data analysis:** category counts, price by category, correlation, price distribution and outliers (IQR method).
   - Companies with fewer than 20 records were grouped into "Other" to avoid unreliable estimates.
   - Outliers were inspected and kept, because they are genuine high-end laptops rather than data errors.
   - Operating system labels were consolidated (for example "Mac OS X" and "macOS"), while Windows 7 was kept separate because it is priced noticeably higher.
3. **Feature engineering:**
   - Target: three price tiers (see below).
   - Features extracted from the text columns: touchscreen, IPS panel, screen resolution, CPU tier and clock speed, GPU brand, and SSD/HDD storage.
4. **Modeling:** one-hot encoding, a stratified 80/20 train/test split, Logistic Regression (with scaling) and Random Forest (with limited depth), evaluated with 5-fold cross-validation.

### Price tiers

| Tier | Price (USD) | Share of laptops |
|---|---|---|
| Cheap | below 500 | about 45% |
| Affordable | 500 to 800 | about 29% |
| Expensive | 800 and above | about 26% |

A model that always predicts "Cheap" reaches about 45% accuracy, which is the baseline to beat.

## Results

<!-- The numbers below come from an earlier run. Re-check them after re-running the notebook and update them if needed. -->

| Model | Train accuracy | Test accuracy | CV accuracy |
|---|---|---|---|
| Baseline (always "Cheap") | n/a | about 0.45 | n/a |
| Logistic Regression | 0.811 | 0.759 | 0.787 |
| Random Forest | 0.836 | 0.766 | 0.790 |

**Key findings**

- Both models perform well above the 45% baseline, at about 77% test accuracy and 79% cross-validation accuracy.
- Almost all errors (60 of 61 on the test set) occur between **neighbouring tiers**. The Affordable tier is the hardest to predict (F1 about 0.62) because it is a narrow band between the other two tiers.
- The most important features are **RAM** (by a wide margin), whether the laptop is a Notebook, SSD storage, weight, and processor features. The features extracted from the text columns appear among the top 10.
- Average price rises with processor tier: Low-end, Core i3, Core i5, Core i7.

<!-- Add your charts here after saving them as images, for example:
![Feature importance](images/feature_importance.png)
![Confusion matrix](images/confusion_matrix.png)
-->

## Repository structure

```
.
├── laptop_price_classification.ipynb   # full analysis and modeling
├── laptop_data.csv                     # dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/laptop-price-tier-classification.git
cd laptop-price-tier-classification
pip install -r requirements.txt
jupyter notebook laptop_price_classification.ipynb
```

Run all cells from top to bottom (Kernel, then Restart Kernel and Run All Cells).

## Limitations and next steps

- Prices are converted with an approximate exchange rate and the data is probably several years old, so the tier thresholds are approximate.
- The tier boundaries are a judgment call, so laptops close to a boundary are easily misclassified.
- The dataset is small (about 1,300 laptops), and the models were not tuned beyond limiting the tree depth.
- Possible next steps: predict the exact price with a regression model, tune hyperparameters, try gradient boosting, and check feature importance with permutation importance.

## Tech stack

Python, pandas, NumPy, scikit-learn, Plotly, Jupyter Notebook

## Author

[Your name](link to your LinkedIn or GitHub profile)
