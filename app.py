import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from io import StringIO

st.set_page_config(
    page_title="Student Performance PCA Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dimensionality Reduction via PCA & Machine Learning Classification</div>', unsafe_allow_html=True)

# 1. Dataset Setup
@st.cache_data
def load_and_train():
    csv_data = """gender,race/ethnicity,parental level of education,lunch,test preparation course,math score,reading score,writing score
female,group B,bachelor's degree,standard,none,72,72,74
female,group C,some college,standard,completed,69,90,88
male,group A,associate's degree,standard,none,90,86,87
female,group A,some college,free/reduced,none,55,65,62
male,group C,high school,free/reduced,completed,60,62,60
female,group D,some high school,standard,none,80,75,70
male,group B,master's degree,standard,completed,85,88,89
female,group C,some high school,standard,none,65,70,68
male,group D,high school,standard,completed,78,75,70
female,group B,associate's degree,free/reduced,none,45,50,48
female,group C,bachelor's degree,standard,completed,88,95,92
male,group E,some college,standard,none,82,78,75
male,group A,high school,free/reduced,none,48,52,50
female,group D,master's degree,standard,completed,92,90,94
male,group C,associate's degree,standard,none,70,72,68
"""
    df = pd.read_csv(StringIO(csv_data))
    
    # Target: Pass (Math >= 60) vs Fail
    df['Result'] = df['math score'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')
    y = df['Result']
    X_raw = df.drop(columns=['math score', 'Result'])
    
    # Encoding & Scaling
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Model
    model = LogisticRegression()
    model.fit(X_pca, y)
    
    return df, X_encoded.columns, scaler, pca, model, X_pca

df, feature_cols, scaler, pca, model, X_pca = load_and_train()

# 2. Sidebar Inputs for New Student
st.sidebar.header("📋 Student Profile Parameters")
st.sidebar.write("Adjust the inputs below to predict the outcome and project onto PCA space:")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], index=1)
parent_edu = st.sidebar.selectbox("Parental Education", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
], index=4)
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

st.sidebar.markdown("---")
st.sidebar.subheader("Exam Scores")
reading = st.sidebar.slider("Reading Score", 0, 100, 75)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)

# 3. Process Input & Predict
input_df = pd.DataFrame([{
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': prep,
    'reading score': reading,
    'writing score': writing
}])

input_encoded = pd.get_dummies(input_df).reindex(columns=feature_cols, fill_value=0)
input_scaled = scaler.transform(input_encoded)
input_pca = pca.transform(input_scaled)

prediction = model.predict(input_pca)[0]
probabilities = model.predict_proba(input_pca)[0]
pass_idx = list(model.classes_).index('Pass')
fail_idx = list(model.classes_).index('Fail')
confidence = np.max(probabilities) * 100

# 4. Top KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Predicted Outcome", prediction, delta="Above standard" if prediction == "Pass" else "Below standard")
with kpi2:
    st.metric("Model Confidence", f"{confidence:.1f}%")
with kpi3:
    st.metric("PC1 Coordinate", f"{input_pca[0][0]:.2f}")
with kpi4:
    st.metric("PC2 Coordinate", f"{input_pca[0][1]:.2f}")

st.write("")

# 5. Display Main Analytics & Plot
col_plot, col_info = st.columns([3, 2])

with col_plot:
    st.subheader("📍 2D Principal Component Projection (PCA)")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fdfdfd')
    
    # Plot existing students
    pass_mask = df['Result'] == 'Pass'
    ax.scatter(
        X_pca[pass_mask, 0], X_pca[pass_mask, 1],
        c='#2ECC71', label='Trained: Pass (Math ≥ 60)', s=90, edgecolors='#1e8449', alpha=0.85
    )
    ax.scatter(
        X_pca[~pass_mask, 0], X_pca[~pass_mask, 1],
        c='#E74C3C', label='Trained: Fail (Math < 60)', s=90, edgecolors='#922b21', alpha=0.85
    )
    
    # Plot the user's input student
    ax.scatter(
        input_pca[0][0], input_pca[0][1],
        c='#2980B9', s=260, marker='*', label='Current Student (Input)', edgecolors='gold', linewidth=1.8, zorder=10
    )
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% Variance Explained)", fontsize=11, fontweight='medium')
    ax.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% Variance Explained)", fontsize=11, fontweight='medium')
    ax.legend(loc='best', frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0')
    ax.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    plt.tight_layout()
    st.pyplot(fig)

with col_info:
    st.subheader("🔍 Prediction Breakdown")
    if prediction == "Pass":
        st.success(f"### 🎉 Result: PASS\nThe model predicts the student will pass mathematics with **{confidence:.1f}%** probability.")
    else:
        st.error(f"### ⚠️ Result: FAIL\nThe model predicts the student is at risk with **{confidence:.1f}%** confidence.")
    
    st.write("**Class Probabilities:**")
    st.progress(float(probabilities[pass_idx]))
    st.caption(f"Pass Probability: {probabilities[pass_idx]*100:.1f}% | Fail Probability: {probabilities[fail_idx]*100:.1f}%")
    
    st.markdown("---")
    st.write("**Explained Variance:**")
    st.info(f"Total variance retained by 2 components: **{(var_pc1 + var_pc2):.1f}%**")
    st.markdown(f"""
    - **PC1:** captures {var_pc1:.1f}% of variance across academic preparation & test metrics.
    - **PC2:** captures {var_pc2:.1f}% of demographic & preparatory background variation.
    """)

# 6. Dataset Explorer
with st.expander("📊 View Training Dataset (15 Students)", expanded=False):
    st.dataframe(df, use_container_width=True)
