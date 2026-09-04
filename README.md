# 📧 Phishing Email Detection Model

A machine learning project using **Scikit-learn** to classify emails as **Phishing** or **Safe**.  
It extracts features from email text (URLs, keywords, etc.) and trains a Random Forest classifier.

## ✨ Features

- **Synthetic dataset generation** – no external data required; runs immediately.
- **Feature extraction**:
  - TF‑IDF vectorization of email text.
  - Manual features: number of URLs, suspicious words, exclamation marks, suspicious links, and text length.
- **Classification** – Random Forest with balanced class weights.
- **Evaluation** – prints accuracy, confusion matrix, classification report, and top feature importance.
- **Sample misclassified emails** – helps understand model weaknesses.

## 🚀 Usage

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/phishing-email-detector.git
cd phishing-email-detector

# Install dependencies
pip install -r requirements.txt

python phishing_detector.py

Example Output

Generating synthetic dataset...
Dataset shape: (1500, 2)
Class distribution:
0    900
1    600
Name: label, dtype: int64
Training Random Forest...

==================================================
MODEL EVALUATION
==================================================
Accuracy: 0.9567

Confusion Matrix:
[[176   4]
 [  9 111]]

Classification Report:
              precision    recall  f1-score   support

        Safe       0.95      0.98      0.96       180
    Phishing       0.97      0.93      0.95       120

    accuracy                           0.96       300
   macro avg       0.96      0.95      0.96       300
weighted avg       0.96      0.96      0.96       300

Top 10 Features:
1. suspicious_count: 0.3124
2. url_count: 0.2156
3. click: 0.1452
4. account: 0.1021
5. verify: 0.0876
...

