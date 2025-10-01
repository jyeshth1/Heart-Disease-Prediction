import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_cols = joblib.load('columns.pkl')

# Page configuration
st.set_page_config(
    page_title="💖 Heart Disease Tracker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar info
with st.sidebar:
    st.header("Heart Disease Tracker 🩺")
    st.markdown("""
    This app predicts the **risk of heart disease** based on your health details.
    
    **Instructions:**
    - Fill all the fields accurately.
    - Press **Predict** to see your risk.
    """)
    st.image("https://images.unsplash.com/photo-1715111965882-bbdf35de510c?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fGh1bWFuJTIwaGVhcnR8ZW58MHx8MHx8fDA%3D", use_column_width=True)

st.markdown("<h1 style='text-align: center; color: red;'>💓 Heart Disease Tracker</h1>", unsafe_allow_html=True)
st.markdown("---")

# Layout input fields in columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox('Chest Pain Type', ['ATA','NAP','TA','ASY'])
    resting_bp = st.number_input('Resting Blood Pressure (mm HG)', 80, 200, 120)
    cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 600, 200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ['Normal','ST','LVH'])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ['Y','N'])
    oldpeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ['Up','Down',"Flat"])

# Colorful predict button
if st.button('💡 Predict My Risk', type='primary'):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_'+sex: 1,
        'RestingECG_'+resting_ecg: 1,
        'ExerciseAngina_'+exercise_angina: 1,
        'ST_Slope_'+st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    numeric_col = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
    input_df[numeric_col] = scaler.transform(input_df[numeric_col])

    prediction = model.predict(input_df)[0]

    # Show results in a colorful way
    if prediction == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>⚠️ High Risk of Heart Disease</h2>", unsafe_allow_html=True)
        st.warning("Please consult a doctor immediately!")
        st.progress(80)  # Optional progress bar
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ Low Risk of Heart Disease</h2>", unsafe_allow_html=True)
        st.success("Keep maintaining a healthy lifestyle!")
        st.progress(20)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color: gray;'>Designed with ❤️ by Jyeshth Joshi</p>", unsafe_allow_html=True)
