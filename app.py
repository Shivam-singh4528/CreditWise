
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="CreditWise | Loan Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ==================================================
# CUSTOM STYLING
# ==================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #123b65, #176b87);
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e3f3fa;
        font-size: 17px;
    }
    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 16px;
        border: 1px solid #d1d5db;
    }
    .approved {
        background-color: #e8f7ee;
        border-color: #5dbb83;
        color: #14532d;
    }
    .rejected {
        background-color: #fff0f0;
        border-color: #e58b8b;
        color: #7f1d1d;
    }
    .result-card h2 {
        margin-top: 0;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 48px;
        font-weight: 600;
    }
    .small-note {
        color: #777;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD SAVED MODEL AND SCALER
# ==================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.markdown("""
<div class="hero">
    <h1>💳 CreditWise</h1>
    <p>AI-Powered Loan Approval Assessment</p>
    <p>Enter applicant information to receive a machine-learning
    prediction and estimated approval probability.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, scaler = load_artifacts()
    FEATURES = list(scaler.feature_names_in_)
except Exception as e:
    st.error(f"Unable to load model or scaler: {e}")
    st.stop()

if len(FEATURES) != 27:
    st.warning(
        f"The saved scaler has {len(FEATURES)} features. "
        "Check that the input construction matches your training notebook."
    )

# ==================================================
# APPLICANT FORM
# ==================================================
st.header("📝 Loan Application")
st.write("Enter the applicant's details below.")

with st.form("creditwise_form"):

    st.subheader("1. Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input(
            "Applicant Income",
            min_value=0.0, value=5000.0, step=500.0
        )
        co_income = st.number_input(
            "Co-applicant Income",
            min_value=0.0, value=2000.0, step=500.0
        )
        savings = st.number_input(
            "Savings",
            min_value=0.0, value=15000.0, step=500.0
        )
        existing_loans = st.number_input(
            "Number of Existing Loans",
            min_value=0, value=1, step=1
        )

    with col2:
        loan_amount = st.number_input(
            "Requested Loan Amount",
            min_value=0.0, value=10000.0, step=500.0
        )
        loan_term = st.number_input(
            "Loan Term (as used in training)",
            min_value=1, value=12, step=1
        )
        collateral = st.number_input(
            "Collateral Value",
            min_value=0.0, value=20000.0, step=500.0
        )
        dti_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0, value=0.20, step=0.01,
            format="%.3f"
        )

    with col3:
        age = st.number_input(
            "Applicant Age",
            min_value=18, max_value=100, value=30
        )
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0, value=1, step=1
        )
        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0, value=750.0, step=10.0
        )
        education_code = st.number_input(
            "Education Level (encoded value)",
            min_value=0, max_value=100, value=0, step=1,
            help="Use the numeric category mapping from your training notebook."
        )

    st.divider()
    st.subheader("2. Applicant Profile")

    col4, col5, col6 = st.columns(3)

    with col4:
        employment = st.selectbox(
            "Employment Status",
            ["Reference category", "Salaried",
             "Self-employed", "Unemployed"]
        )
        marital = st.selectbox(
            "Marital Status",
            ["Reference category", "Single"]
        )

    with col5:
        purpose = st.selectbox(
            "Loan Purpose",
            ["Reference category", "Car", "Education",
             "Home", "Personal"]
        )
        property_area = st.selectbox(
            "Property Area",
            ["Reference category", "Semiurban", "Urban"]
        )

    with col6:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )
        employer = st.selectbox(
            "Employer Category",
            ["Reference category", "Government", "MNC",
             "Private", "Unemployed"]
        )

    submitted = st.form_submit_button(
        "🔍 Assess Loan Application",
        use_container_width=True
    )

# ==================================================
# PROCESS INPUT AND PREDICT
# ==================================================
if submitted:

    # Start with all 27 model features set to zero
    row = {feature: 0.0 for feature in FEATURES}

    # Numeric features and engineered squared features
    numeric_values = {
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_code,
        "DTI_Ratio_sq": dti_ratio ** 2,
        "Credit_Score_sq": credit_score ** 2,
    }

    for feature, value in numeric_values.items():
        if feature in row:
            row[feature] = value

    # One-hot encoded columns
    category_values = {
        f"Employment_Status_{employment}": 1,
        f"Marital_Status_{marital}": 1,
        f"Loan_Purpose_{purpose}": 1,
        f"Property_Area_{property_area}": 1,
        f"Gender_{gender}": 1,
        f"Employer_Category_{employer}": 1,
    }

    for feature, value in category_values.items():
        if feature in row:
            row[feature] = value

    input_df = pd.DataFrame([row], columns=FEATURES)

    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        # Assumption: target 1 = Approved, 0 = Not Approved.
        # Verify the target mapping in the notebook before real use.
        prediction_text = str(prediction).strip()
        is_approved = prediction_text == "1"

        st.divider()
        st.header("📊 Loan Assessment Result")

        if is_approved:
            st.markdown("""
            <div class="result-card approved">
                <h2>✅ Loan Likely to Be Approved</h2>
                <p>The model classified this application in the
                approved category.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card rejected">
                <h2>❌ Loan Not Likely to Be Approved</h2>
                <p>The model classified this application in the
                not-approved category.</p>
            </div>
            """, unsafe_allow_html=True)

        # Display class probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            classes = list(model.classes_)

            approval_probability = None
            if 1 in classes:
                approval_probability = float(
                    probabilities[classes.index(1)] * 100
                )

            st.subheader("Model Prediction Details")

            if approval_probability is not None:
                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Estimated Approval Probability",
                        f"{approval_probability:.2f}%"
                    )

                with m2:
                    st.metric(
                        "Model Classification",
                        "Approved" if is_approved else "Not Approved"
                    )

                st.progress(
                    max(0.0, min(1.0, approval_probability / 100))
                )

            with st.expander("View technical model details"):
                result_df = pd.DataFrame({
                    "Model Class": classes,
                    "Estimated Probability (%)": [
                        round(float(p) * 100, 2)
                        for p in probabilities
                    ]
                })
                st.dataframe(result_df, hide_index=True)

        with st.expander("View submitted application summary"):
            st.dataframe(
                pd.DataFrame({
                    "Input Feature": list(numeric_values.keys()),
                    "Entered Value": list(numeric_values.values())
                }),
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "Important: This is an experimental machine-learning estimate, "
            "not a guaranteed lending decision. Verify that class 1 means "
            "Approved in the training notebook and confirm all category "
            "encodings before relying on predictions."
        )

    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.markdown(
    "<p class='small-note'>CreditWise | Machine Learning Project "
    "· For demonstration and educational purposes.</p>",
    unsafe_allow_html=True
)
