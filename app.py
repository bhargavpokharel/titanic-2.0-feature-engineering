import streamlit as st
import pandas as pd
import joblib

# ---- LOAD YOUR SAVED MODEL AND SCALER ----
model = joblib.load('titanic_model_engineered.pkl')
scaler = joblib.load('titanic_scaler_engineered.pkl')

# ---- SET UP THE APP INTERFACE ----
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter the passenger details below to find out if they would have survived!")

# ---- INPUTS (These match your 12 features) ----
# 1. Pclass
pclass = st.selectbox("Passenger Class", [1, 2, 3])

# 2. Sex
sex = st.radio("Sex", ["male", "female"])

# 3. Age
age = st.slider("Age", 0, 100, 25)

# 4. SibSp
sibsp = st.number_input("Number of Siblings / Spouses aboard", min_value=0, max_value=10, value=0)

# 5. Parch
parch = st.number_input("Number of Parents / Children aboard", min_value=0, max_value=10, value=0)

# 6. Fare
fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=50.0)

# 7. Embarked
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# ---- ENGINEERED FEATURES (The app will calculate these for you) ----
# 8. Title (We'll let the user pick)
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

# 9. FamilySize
family_size = sibsp + parch + 1

# 10. IsAlone
is_alone = 1 if family_size == 1 else 0

# 11. AgeBin (Automatically calculated based on the age slider)
if age <= 12:
    age_bin = "Child"
elif age <= 18:
    age_bin = "Teen"
elif age <= 35:
    age_bin = "Adult"
elif age <= 60:
    age_bin = "Senior"
else:
    age_bin = "Elder"

# 12. FareBin (Automatically calculated based on the fare slider)
if fare <= 7.91:
    fare_bin = "Low"
elif fare <= 14.45:
    fare_bin = "Medium"
elif fare <= 31.0:
    fare_bin = "High"
else:
    fare_bin = "Very High"

# ---- ENCODE THE INPUTS (Just like we did in the notebook) ----
sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}
title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
agebin_map = {"Child": 0, "Teen": 1, "Adult": 2, "Senior": 3, "Elder": 4}
farebin_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

# Create the feature array in the EXACT order your model expects
features = [
    pclass,
    sex_map[sex],
    age,
    sibsp,
    parch,
    fare,
    embarked_map[embarked],
    title_map[title],
    family_size,
    is_alone,
    agebin_map[age_bin],
    farebin_map[fare_bin]
]

# ---- PREDICT ----
if st.button("Predict Survival"):
    # Scale the features
    scaled_features = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    # Show result
    if prediction == 1:
        st.success("✅ This passenger would have SURVIVED! 🎉")
    else:
        st.error("❌ This passenger would NOT have survived.")