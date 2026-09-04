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

