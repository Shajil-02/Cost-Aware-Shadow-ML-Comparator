import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score

def train_model_A(X_train, y_train):
    # baseline model (production)
    base = LogisticRegression(max_iter=200)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model

def train_model_B(X_train, y_train):
    # candidate model (slightly different)
    base = LogisticRegression(max_iter=200, class_weight="balanced")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model

def evaluate_auc(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    return auc, proba

def find_best_threshold(y_true, proba):
    thresholds = np.linspace(0.05, 0.95, 19)
    # Calculating all F1 scores at once and finding the max
    f1_scores = [f1_score(y_true, proba >= t, zero_division=0) for t in thresholds]
    idx = np.argmax(f1_scores)
    return float(thresholds[idx]), f1_scores[idx]

def save_model(model, path):
    pickle.dump(model, open(path, "wb"))