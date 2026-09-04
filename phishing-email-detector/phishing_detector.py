#!/usr/bin/env python3
"""
Phishing Email Detection Model
Using Scikit-learn to classify emails as Phishing or Safe.
Includes synthetic dataset generation, feature extraction, training, and evaluation.
"""

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------- Synthetic Dataset Generation -------------------
def generate_synthetic_data(n_samples=1000):
    """Generate a synthetic dataset of emails with labels (0=Safe, 1=Phishing)."""
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        # Randomly decide label
        label = np.random.choice([0, 1], p=[0.6, 0.4])  # 60% safe, 40% phishing
        
        # Base email content
        if label == 1:  # Phishing
            subject = np.random.choice([
                "Account Suspension Alert", 
                "Verify Your Bank Account", 
                "Urgent: Update Your Password",
                "You Won a Prize!",
                "Security Alert: Unusual Activity"
            ])
            body = f"""
            Dear User,
            We detected suspicious activity on your account. Click the link below to verify your identity.
            http://fake-bank-verification.com/verify
            Failure to do so will result in account suspension.
            Please update your credentials immediately.
            Best regards,
            Security Team
            """
            # Add more phishing indicators
            body += " " + " ".join(np.random.choice(["verify", "click here", "urgent", "bank", "password", "account", "suspended", "winner"], size=5))
        else:  # Safe
            subject = np.random.choice([
                "Meeting Agenda", 
                "Project Update", 
                "Your Weekly Report",
                "Invoice #12345",
                "Team Lunch Reminder"
            ])
            body = f"""
            Hello,
            This is a regular update regarding the ongoing project.
            Please find the attached document for your reference.
            Let me know if you have any questions.
            Best,
            Colleague
            """
            # Add some normal words
            body += " " + " ".join(np.random.choice(["meeting", "report", "document", "attachment", "schedule", "feedback"], size=3))
        
        # Combine subject and body
        email_text = f"Subject: {subject}\n{body}"
        data.append({"text": email_text, "label": label})
    
    return pd.DataFrame(data)

# ------------------- Feature Extraction Functions -------------------
def count_urls(text):
    """Count number of URLs in email text."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)
    return len(urls)

def count_suspicious_words(text):
    """Count occurrences of suspicious keywords."""
    suspicious = ['verify', 'account', 'password', 'bank', 'credit', 'suspend', 'security', 'update', 'urgent', 'winner']
    count = 0
    for word in suspicious:
        count += len(re.findall(r'\b' + word + r'\b', text, re.IGNORECASE))
    return count

def count_exclamation(text):
    """Count exclamation marks."""
    return text.count('!')

def has_suspicious_link(text):
    """Check if any URL contains suspicious domain or IP."""
    # Simple heuristic: if link contains 'fake' or 'verify' etc.
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)
    suspicious_domains = ['fake', 'verify', 'secure', 'update', 'banking']
    for url in urls:
        if any(dom in url.lower() for dom in suspicious_domains):
            return 1
    return 0

# ------------------- Main Training & Evaluation -------------------
def main():
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(1500)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # Split data
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ----- Feature extraction -----
    # 1. Text features using TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')

    # 2. Manual features (applied to raw text)
    def extract_manual_features(texts):
        # texts is a Series
        urls = texts.apply(count_urls).values.reshape(-1, 1)
        suspicious = texts.apply(count_suspicious_words).values.reshape(-1, 1)
        exclamation = texts.apply(count_exclamation).values.reshape(-1, 1)
        suspicious_link = texts.apply(has_suspicious_link).values.reshape(-1, 1)
        # Also include length
        length = texts.apply(len).values.reshape(-1, 1)
        return np.hstack([urls, suspicious, exclamation, suspicious_link, length])

    # For training, extract manually
    X_train_manual = extract_manual_features(X_train)
    X_test_manual = extract_manual_features(X_test)

    # Transform text to TF-IDF
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Combine features
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_manual))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_manual))

    # Scale the manual features (optional but good for some models)
    scaler = StandardScaler()
    # We only scale the manual part (last 5 columns)
    manual_indices = list(range(X_train_tfidf.shape[1], X_train_combined.shape[1]))
    X_train_combined[:, manual_indices] = scaler.fit_transform(X_train_combined[:, manual_indices])
    X_test_combined[:, manual_indices] = scaler.transform(X_test_combined[:, manual_indices])

    # ----- Train classifier -----
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_combined, y_train)

    # Predict
    y_pred = clf.predict(X_test_combined)

    # ----- Evaluation -----
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Safe', 'Phishing'])

    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Feature importance (top 10 from TF-IDF and manual)
    importances = clf.feature_importances_
    # Get feature names from TF-IDF + manual
    feature_names = tfidf.get_feature_names_out().tolist() + ['url_count', 'suspicious_count', 'exclamation_count', 'suspicious_link', 'length']
    # Sort and display top 10
    indices = np.argsort(importances)[::-1]
    top_n = 10
    print(f"\nTop {top_n} Features:")
    for i in range(top_n):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Also show a sample of misclassified emails for inspection
    misclassified = np.where(y_pred != y_test)[0]
    if len(misclassified) > 0:
        print(f"\nSample misclassified emails (first 3):")
        for i in misclassified[:3]:
            print(f"True: {y_test.iloc[i]}, Pred: {y_pred[i]}")
            print(f"Email: {X_test.iloc[i][:200]}...\n")

if __name__ == "__main__":
    main()