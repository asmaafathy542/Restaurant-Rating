import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

# --- إعداد وتحميل الموديلات ---
keras_model_path = "neural_model_fixed.keras"
h5_model_path = "neural_model.h5"

# لو الموديل الجديد مش موجود، نحوله من القديم
if not os.path.exists(keras_model_path):
    print("تحويل الموديل من H5 إلى Keras format...")
    old_model = load_model(h5_model_path, compile=False)
    old_model.save(keras_model_path)
    print(f"تم التحويل: {keras_model_path}")

# تحميل الموديلات
nn_model = load_model(keras_model_path, compile=False)
ml_model = joblib.load("Best_ml_model.pkl")

# --- إعداد واجهة Streamlit ---
st.set_page_config(page_title="Restaurant Ratings", layout="wide")
page = st.sidebar.selectbox("Select a page", ["Analysis", "Prediction"])

# --- الصفحة الأولى: التحليل ---
if page == "Analysis":
    st.title("Exploratory Data Analysis")

    st.markdown("### 1. Does the demography of an area matter?")
    st.image("location_resttypee.png")
    
    st.markdown("### 2. Does the location of a particular type of restaurant depend on the people living in that area?")
    st.image("rest_type_by_location_heatmap.png")

    st.markdown("### 3. Does the theme of the restaurant matter?")
    st.image("theme_resttype_heatmap.png")

    st.markdown("### 4. Is a food chain category restaurant likely to have more customers than its counterpart?")
    st.image("chain_vs_single_votes.png")

    st.markdown("### 5. Are any neighborhoods similar?")
    st.image("neighborhood_similarity_pca.png")

    st.markdown("### 6. What kind of food is more popular in a locality?")
    st.image("top_cuisines_across_locations.png")

# --- الصفحة الثانية: التنبؤ ---
elif page == "Prediction":
    st.title("Restaurant Rating Prediction")

    # إدخال المستخدم
    votes = st.number_input("Votes", min_value=0, step=1)
    approx_cost = st.number_input("Approximate Cost for Two People", min_value=0, step=1)
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Table Booking Available?", ["Yes", "No"])

    rest_type = st.selectbox("Restaurant Type", [
        "Quick Bites", "Casual Dining", "Delivery", "Dessert Parlor"
    ])

    listed_city = st.selectbox("Location (City)", [
        "Whitefield", "Banashankari", "Bannerghatta Road", "Basavanagudi"
    ])

    listed_type = st.selectbox("Listed In (Type)", [
        "Buffet", "Cafes", "Delivery", "Desserts"
    ])

    # تجهيز البيانات بنفس الأعمدة اللي اتدربت عليها الموديلات
    input_data = pd.DataFrame({
        'votes': [votes],
        'approx_cost(for two people)': [approx_cost],
        'online_order': [1 if online_order == "Yes" else 0],
        'book_table': [1 if book_table == "Yes" else 0],
        'rest_type_Quick Bites': [1 if rest_type == "Quick Bites" else 0],
        'rest_type_Casual Dining': [1 if rest_type == "Casual Dining" else 0],
        'rest_type_Delivery': [1 if rest_type == "Delivery" else 0],
        'rest_type_Dessert Parlor': [1 if rest_type == "Dessert Parlor" else 0],
        'listed_in(city)_Whitefield': [1 if listed_city == "Whitefield" else 0],
        'listed_in(city)_Banashankari': [1 if listed_city == "Banashankari" else 0],
        'listed_in(city)_Bannerghatta Road': [1 if listed_city == "Bannerghatta Road" else 0],
        'listed_in(city)_Basavanagudi': [1 if listed_city == "Basavanagudi" else 0],
        'listed_in(type)_Buffet': [1 if listed_type == "Buffet" else 0],
        'listed_in(type)_Cafes': [1 if listed_type == "Cafes" else 0],
        'listed_in(type)_Delivery': [1 if listed_type == "Delivery" else 0],
        'listed_in(type)_Desserts': [1 if listed_type == "Desserts" else 0],
    })

    # التنبؤ
    if st.button("Predict"):
        ml_pred = ml_model.predict(input_data)[0]
        nn_pred = nn_model.predict(input_data)[0][0]  # [0][0] عشان يطلع رقم صافي

        st.success(f"ML Model Prediction: **{ml_pred:.2f}**")
        st.success(f"Neural Network Prediction: **{nn_pred:.2f}**")
