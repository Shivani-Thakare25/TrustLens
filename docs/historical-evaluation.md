# Historical Model Evaluation & Empirical Benchmark Record

This document records the empirical results obtained during the development and experimental evaluation phases of **TrustLens**. In adherence to academic integrity and transparency guidelines, these results are documented exactly as measured and linked to their specific experimental conditions.

---

## 1. Fake-Review Detection Benchmarks

### Experimental Setup
* **Corpus Size**: Approximately 40,420 review records.
* **Task**: Binary classification (Genuine vs. Predicted Suspicious).
* **Validation Strategy**: Stratified group-aware train/test split preventing reviewer and product leakage across partitions.
* **Feature Engineering**:
  - Word-level TF-IDF (1–2 n-grams, sublinear scaling, top 5,000 features).
  - Character-level TF-IDF (3–5 n-grams, sublinear scaling, top 5,000 features).
  - Handcrafted stylometric features (caps ratio, punctuation density, exclamation intensity, promo ratio, lexical diversity).

### Observed Evaluation Results
| Model Configuration | Test Accuracy | Status & Engineering Observations |
| :--- | :---: | :--- |
| **Baseline Logistic Regression** | **93.60%** | Stable linear baseline; fast convergence and calibrated probabilities. |
| **Baseline Random Forest** | **90.96%** | Non-linear tree ensemble baseline with balanced class weights. |
| **Advanced Logistic Regression** | **95.60%** | Enhanced with combined word/char TF-IDF and calibrated decision threshold. |
| **Advanced Random Forest** | **91.87%** | Optimized tree depth and n_estimators with probability calibration. |
| **Advanced XGBoost** | *Omitted* | Encountered out-of-memory error on high-dimensional dense matrices on Windows hardware; gracefully skipped in production. |

---

## 2. Sentiment Analysis Benchmarks

### Experimental Setup
* **Corpus Size**: Approximately 170,675 review records.
* **Task**: 3-Class Sentiment Classification (Negative, Neutral, Positive).
* **Models Evaluated**: Logistic Regression and Linear Support Vector Classifier (Linear SVC).
* **Feature Representation**: Word-level and character-level TF-IDF features with class-weight rebalancing.

### Neutral-Class Optimization Results
A key challenge in multi-class review sentiment analysis is the **neutral class**. In consumer review corpora, ratings are overwhelmingly polarized (1-star negative or 5-star positive), making the neutral class severely underrepresented and linguistically ambiguous.

The latest neutral-optimized model reported:

| Metric | Overall Score |
| :--- | :---: |
| **Overall Accuracy** | **90.47%** |
| **Weighted F1-Score** | **90.25%** |
| **Macro F1-Score** | **74.61%** |

#### Class-Specific Breakdown for Neutral Sentiment
* **Neutral Precision**: 46.10%
* **Neutral Recall**: 40.20%
* **Neutral F1-Score**: 42.95%

### Methodological Interpretation
While overall accuracy exceeds 90%, the macro F1-score is 74.61% primarily due to the lower F1 on the neutral class (42.95%). Reporting both weighted and macro metrics is required for rigorous academic evaluation rather than relying solely on accuracy.

---

## 3. Aspect-Based Sentiment Analysis (ABSA)

### Experimental Setup
* **Source Volume**: Approximately 170,675 source reviews.
* **Target Aspects**: 34 unique extracted aspects across technology, hardware, customer support, and shipping categories.
* **Aspect Occurrences**: Approximately 99,672 occurrences identified.
* **Mean Confidence**: 0.917.

### Critical Academic Honesty Note
Ground-truth aspect annotations were **not present** in the raw source dataset. Aspect extraction operates through syntax-aware clause splitting, keyword lexicon mapping, and local negation handling. The reported confidence score (0.917) reflects rule certainty, **not ground-truth benchmark accuracy**.

---

## 4. Similarity and Duplicate Detection

* **Architecture**: Character-level TF-IDF (3–5 char n-grams) combined with word-level TF-IDF (1–2 word n-grams).
* **Similarity Metric**: Cosine similarity against precomputed reference index.
* **Default Threshold**: 0.60 for cluster grouping; exact duplicate detection via normalized text hashing.
* **Efficiency Optimization**: Feature extraction is computed once and persisted; querying uses matrix multiplication without repeated feature fitting during dashboard usage.

---

## 5. Storage and Reference

* Machine-readable summary: `reports/historical_evaluation_summary.json`
* Current runtime model metadata: `artifacts/model/metadata.json`
* Current test evaluation metrics: `reports/evaluation/metrics.json`
