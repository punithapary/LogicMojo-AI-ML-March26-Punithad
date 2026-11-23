import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def _finish_chart(fig, filename=None):
    """Display a chart and optionally save it."""
    fig.tight_layout()

    if filename:
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")

    plt.show()


def fraud_class_counts(df, filename=None):
    """Chart 1: Count legitimate and fraudulent transactions."""
    counts = df["isFraud"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Fraud and Legitimate Transaction Counts")
    ax.set_xlabel("isFraud (0 = non-fraud, 1 = fraud)")
    ax.set_ylabel("Number of transactions")
    ax.tick_params(axis="x", rotation=0)

    _finish_chart(fig, filename)


def transaction_amount_distribution(df, filename=None):
    """Chart 2: Distribution of transaction amounts."""
    if "amount" not in df.columns:
        raise ValueError("Dataset must contain an 'amount' column.")

    fig, ax = plt.subplots(figsize=(8, 5))
    df["amount"].dropna().plot(
        kind="hist", bins=50, ax=ax
    )

    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Transaction amount")
    ax.set_ylabel("Frequency")

    _finish_chart(fig, filename)


def amount_by_fraud_status(df, filename=None):
    """Chart 3: Compare transaction amounts by fraud status."""
    if "amount" not in df.columns:
        raise ValueError("Dataset must contain an 'amount' column.")

    fig, ax = plt.subplots(figsize=(7, 5))

    df.boxplot(
        column="amount",
        by="isFraud",
        ax=ax,
        showfliers=False
    )

    ax.set_title("Transaction Amount by Fraud Status")
    ax.set_xlabel("isFraud (0 = non-fraud, 1 = fraud)")
    ax.set_ylabel("Transaction amount")
    fig.suptitle("")

    _finish_chart(fig, filename)


def fraud_rate_by_type(df, filename=None):
    """Chart 4: Fraud percentage for each transaction type."""
    if "type" not in df.columns:
        raise ValueError("Dataset must contain a 'type' column.")

    fraud_rates = (
        df.groupby("type")["isFraud"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fraud_rates.plot(kind="bar", ax=ax)

    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_xlabel("Transaction type")
    ax.set_ylabel("Fraud rate (%)")
    ax.tick_params(axis="x", rotation=45)

    _finish_chart(fig, filename)


def transaction_type_counts(df, filename=None):
    """Chart 5: Number of transactions by type."""
    if "type" not in df.columns:
        raise ValueError("Dataset must contain a 'type' column.")

    counts = df["type"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Transaction Counts by Type")
    ax.set_xlabel("Transaction type")
    ax.set_ylabel("Number of transactions")
    ax.tick_params(axis="x", rotation=45)

    _finish_chart(fig, filename)


def numeric_correlation_heatmap(df, filename=None):
    """Chart 6: Correlations between numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    if corr.empty:
        raise ValueError("No numeric columns available for correlation.")

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.colorbar(image, ax=ax, label="Correlation")

    _finish_chart(fig, filename)
