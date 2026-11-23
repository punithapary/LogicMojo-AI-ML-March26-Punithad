
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI-Powered Financial Fraud & Risk Intelligence Platfor",
    page_icon=" ",
    layout="wide"
)

st.title("AI-Powered Financial Fraud & Risk Intelligence Platform")

st.write(
    "An educational fraud detection prototype using the "
    "PaySim synthetic financial transaction dataset."
)

# Resolve paths relative to the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_ml_model.joblib"

# ==========================================
# STEP 2: LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        loaded_model = joblib.load(MODEL_PATH)

        if not hasattr(loaded_model, "predict_proba"):
            raise ValueError(
                "Model does not support probability prediction."
            )

        if 1 not in loaded_model.classes_:
            raise ValueError(
                "Model does not contain fraud class 1."
            )

        return loaded_model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# ==========================================
# STEP 3: MODEL FEATURES
# ==========================================

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

# ==========================================
# STEP 4: TRANSACTION INPUT INTERFACE
# ==========================================

st.header("Individual Transaction Assessment")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:

        step = st.number_input(
            "1. Transaction Step",
            min_value=1,
            value=1
        )

        transaction_type = st.selectbox(
            "2. Transaction Type",
            ["CASH_IN", "CASH_OUT", "DEBIT",
             "PAYMENT", "TRANSFER"]
        )

        amount = st.number_input(
            "3. Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        oldbalanceOrg = st.number_input(
            "4. Sender Balance Before",
            min_value=0.0,
            value=5000.0
        )

    with col2:

        newbalanceOrig = st.number_input(
            "5. Sender Balance After",
            min_value=0.0,
            value=4000.0
        )

        oldbalanceDest = st.number_input(
            "6. Recipient Balance Before",
            min_value=0.0,
            value=1000.0
        )

        newbalanceDest = st.number_input(
            "7. Recipient Balance After",
            min_value=0.0,
            value=2000.0
        )

    assess_button = st.form_submit_button(
        "8. Assess Transaction",
        type="primary",
        use_container_width=True
    )

# ==========================================
# STEP 5: FRAUD PREDICTION
# ==========================================
model = load_model()
if assess_button:

    if model is None:
        st.error("Model is not loaded.")
        st.stop()

    transaction = pd.DataFrame(
        [[
            step,
            transaction_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]],
        columns=FEATURES
    )

    try:
        fraud_class_index = list(model.classes_).index(1)

        probability = model.predict_proba(
            transaction
        )[0][fraud_class_index]

        prediction = int(probability >= 0.5)

        if probability >= 0.70:
            risk = "HIGH"
            explanation = "High predicted fraud probability. Prioritise for human review."

        elif probability >= 0.30:
            risk = "MEDIUM"
            explanation = "Moderate predicted fraud probability. Further investigation is recommended."

        else:
            risk = "LOW"
            explanation = "Low predicted fraud probability. Continue normal monitoring."

        st.divider()
        st.header("Fraud Assessment Results")

        result1, result2, result3 = st.columns(3)

        result1.metric("Fraud Probability", f"{probability:.2%}")

        result2.metric(
            "Fraud Prediction",
            "Fraudulent" if prediction == 1 else "Legitimate"
        )

        result3.metric("Risk Category", risk)

        st.subheader("Risk Explanation")

        if risk == "HIGH":
            st.error(explanation)
        elif risk == "MEDIUM":
            st.warning(explanation)
        else:
            st.success(explanation)

        st.caption(
            "Educational prediction only. "
            "This is not a confirmed fraud determination."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")



# ==========================================
# STEP 6: BATCH CSV ANALYSIS
# ==========================================

st.divider()

st.header("Batch Transaction Analysis")

uploaded_file = st.file_uploader(
    "Upload PaySim transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        missing_columns = [
            col for col in FEATURES if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        else:

            if st.button("Run Batch Fraud Detection"):

                if not hasattr(model, "predict_proba"):
                    raise ValueError(
                        "The saved model must support predict_proba()."
                    )

                X = df[FEATURES]

                fraud_class_index = list(model.classes_).index(1)

                probabilities = model.predict_proba(X)[
                    :, fraud_class_index
                ]

                predictions = (probabilities >= 0.5).astype(int)

                df["Fraud_Probability"] = probabilities
                df["Fraud_Prediction"] = predictions

                df["Risk_Category"] = pd.cut(
                    probabilities,
                    bins=[-0.001, 0.30, 0.70, 1.001],
                    labels=["LOW", "MEDIUM", "HIGH"]
                )

                st.subheader("Batch Detection Results")

                st.dataframe(
                    df.head(100),
                    use_container_width=True
                )

                fraud_count = int(predictions.sum())

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Transactions", len(df))
                col2.metric("Predicted Fraud", fraud_count)
                col3.metric(
                    "Predicted Legitimate",
                    len(df) - fraud_count
                )

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Unable to process dataset: {e}")
