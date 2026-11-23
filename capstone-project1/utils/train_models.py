def evaluate_model(name, model, X_eval, y_eval, threshold=0.5):
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{'=' * 50}")
    print(name)
    print(f"Threshold: {threshold:.2f}")
    print(classification_report(
        y_eval,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    print("Average Precision (PR-AUC summary):",
          round(average_precision_score(y_eval, probabilities), 5))
    print("ROC-AUC:",
          round(roc_auc_score(y_eval, probabilities), 5))

    cm = confusion_matrix(y_eval, predictions, labels=[0, 1])

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Legitimate", "Predicted Fraud"],
        yticklabels=["Actual Legitimate", "Actual Fraud"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

    return {
        "model": name,
        "threshold": threshold,
        "precision_fraud": (
            np.sum((predictions == 1) & (y_eval.to_numpy() == 1))
            / max(np.sum(predictions == 1), 1)
        ),
        "recall_fraud": (
            np.sum((predictions == 1) & (y_eval.to_numpy() == 1))
            / max(np.sum(y_eval.to_numpy() == 1), 1)
        ),
        "pr_auc": average_precision_score(y_eval, probabilities),
        "roc_auc": roc_auc_score(y_eval, probabilities)
    }