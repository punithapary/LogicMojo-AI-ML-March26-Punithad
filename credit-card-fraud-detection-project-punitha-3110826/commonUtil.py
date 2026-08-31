import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, f1_score, 
                             precision_score, recall_score)
import lime.lime_tabular
import warnings
#### Create Evaluation Function ####
## evaluate_model can used in different ML models
#Since fraud is highly imbalanced, instead of checking accuracywe can check precision, recall, f1, roc_auc

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Probability of fraud (Class = 1)
    y_proba = model.predict_proba(X_test)[:, 1]
    #compares the actual fraud status (y_test) with the model's prediction (y_pred).
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    precision = tp / (tp + fp) # It tells us how many are actually fraud from the model when all the transcations are marked as fraud
    recall = tp / (tp + fn) #It tells us how many are sucessfully detected from the model when all the transcations are actually fraud
    f1 = 2 * (precision * recall) / (precision + recall) #F1-score combines precision and recall into one metric.
    roc_auc = roc_auc_score(y_test, y_proba) #ROC-AUC measures how model can separate fraudulent transactions from legitimate transactions.
    
    print(f"\n{name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba
    }