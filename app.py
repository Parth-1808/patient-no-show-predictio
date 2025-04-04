import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\bachh\patient-no-show-prediction\data\KaggleV2-May-2016.csv.csv")

# Preprocessing for visuals
df['No-show'] = df['No-show'].map({'No': 1, 'Yes': 0})

# Load model
model = joblib.load("model.pkl")

# App title
st.title("📅 Patient No-Show Prediction")
st.write("This app uses historical appointment data to predict whether a patient will miss their appointment.")

# Sidebar navigation
menu = st.sidebar.radio("Choose a section", ["📊 EDA (Visuals)", "🔮 Predict No-Show"])

# --- EDA Section ---
if menu == "📊 EDA (Visuals)":
    st.subheader("Exploratory Data Analysis")
    
    # Age Distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Gender vs No-show
    st.write("### Gender vs Show")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='No-show', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    # SMS Received vs No-show
    st.write("### SMS Received vs Show")
    fig, ax = plt.subplots()
    sns.countplot(x='SMS_received', hue='No-show', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

    # Neighbourhood-wise show rate
    st.write("### Top 20 Neighborhoods with Highest No-Show Rates")
    no_show_by_neigh = df.groupby('Neighbourhood')['No-show'].mean().sort_values(ascending=False)
    st.bar_chart(no_show_by_neigh.head(20))

# --- Prediction Section ---
elif menu == "🔮 Predict No-Show":
    st.subheader("Predict No-Show for New Patient")

    gender = st.selectbox("Gender", ["F", "M"])
    age = st.slider("Age", 0, 115, 25)
    scholarship = st.selectbox("Scholarship", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    alcoholism = st.selectbox("Alcoholism", [0, 1])
    handicap = st.selectbox("Handcap (0 = None)", [0, 1, 2, 3, 4])
    sms_received = st.selectbox("SMS Received", [0, 1])
    neighbourhood = st.selectbox("Neighbourhood", df['Neighbourhood'].unique())

    if st.button("Predict"):
        # Prepare the input like training
        input_df = pd.DataFrame({
            'Age': [age],
            'Scholarship': [scholarship],
            'Hipertension': [hypertension],
            'Diabetes': [diabetes],
            'Alcoholism': [alcoholism],
            'Handcap': [handicap],
            'SMS_received': [sms_received]
        })

        # One-hot for Gender and Neighbourhood
        input_df[f'Gender_{gender}'] = 1
        for g in ['Gender_M', 'Gender_F']:
            if g not in input_df.columns:
                input_df[g] = 0

        for col in df['Neighbourhood'].unique():
            input_df[f'Neighbourhood_{col}'] = 1 if col == neighbourhood else 0

        # Match training columns
        model_input_cols = model.feature_names_in_
        for col in model_input_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_input_cols]

        # Predict
        prediction = model.predict(input_df)[0]
        if prediction == 0:
            st.success("✅ Patient is likely to attend the appointment.")
        else:
            st.error("❌ Patient is likely to **miss** the appointment.")
