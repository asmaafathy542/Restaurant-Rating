import streamlit as st
import pandas as pd
import joblib


#upload models
from tensorflow.keras.models import load_model
nn_model = load_model("neural_model.h5" , compile=False)
ml_model = joblib.load("Best_ml_model.pkl")  

#main page
st.set_page_config(page_title="Restaurant Ratings", layout="wide")


page = st.sidebar.selectbox("Select a page " , ["Analysis", "Prediction"])

#first page Analysis Page 

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



  #  st.markdown("### 3. Online Order Effect")
   # st.image("eda_charts/online_order_effect.png")

  #  st.markdown("### 4. Restaurant Type Popularity")
  #  st.image("eda_charts/rest_type_counts.png")

  #  st.markdown("> **Key Insights:**")
   # st.markdown("""
  #  - Most restaurants have a rating between 3.0 and 4.0  
   # - Restaurants allowing online orders tend to have slightly higher ratings  
  #  - Quick Bites and Casual Dining are the most common types  
  #  - Votes positively correlate with ratings  
   # """)

#second page Prediction Page 
elif page == "Prediction":
    st.title("Restaurant Rating Prediction")

  #user inputs
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

    listed_type = st.selectbox("Listed In (Type)",[
        "Buffet", "Cafes", "Delivery", "Desserts" ]
    )

    # --- تجهيز الـ input بنفس الأعمدة اللي اتدربت عليها الموديلات ---
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

 #prediction
    if st.button(" Predict"):
        ml_pred = ml_model.predict(input_data)[0]
        nn_pred = nn_model.predict(input_data)[0]

        st.success(f"ML Model Prediction: **{ml_pred:.2f}**")
        st.success(f"Neural Network Prediction: **{nn_pred:.2f}**")
