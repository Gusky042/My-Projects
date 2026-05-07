# Social Network Ads Purchase Prediction

## Project Overview

A machine learning model that predicts whether a user will purchase a product based on their **gender, age, and estimated salary**. Using a **Random Forest Classifier** with hyperparameter tuning, the model achieves **92.5% accuracy** on test data.

## Model Performance

### Key Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| **Precision** | 86.7% |
| **Recall** | 92.9% |
| **F1-Score** | 89.7% |
| **Cross-Validation Accuracy** | 90.9% |

### Performance by Class (Confusion Matrix)
Predicted
Bought Not Bought
Actual Bought 48 4
Not Bought 2 26

text

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **True Negatives** | 48 | Correctly predicted "No Purchase" |
| **False Positives** | 4 | Wrongly predicted "Purchase" |
| **False Negatives** | 2 | Wrongly predicted "No Purchase" |
| **True Positives** | 26 | Correctly predicted "Purchase" |

## What These Numbers Mean

### In Plain English
> "The model correctly predicts whether a user will purchase a product **93 out of 100 times**."

### Class-Specific Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **No Purchase** | 0.96 | 0.92 | 0.94 |
| **Purchase** | 0.87 | 0.93 | 0.90 |

**Interpretation:**
- When the model says "will purchase", it's right **87%** of the time
- The model catches **93%** of actual purchasers
- Only **4 false positives** (wasted marketing spend)
- Only **2 false negatives** (missed opportunities)

## Model Architecture

```python
Pipeline:
1. Feature Engineering → Gender encoding
2. Classifier → Random Forest (optimized)
3. Cross-validation → 5-fold
4. Hyperparameter Tuning → GridSearchCV
