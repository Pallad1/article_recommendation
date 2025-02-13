import streamlit as st
import requests

azure_function_url = "http://articlesreco.azurewebsites.net/api/my_function/product_get"

user_id = st.text_input("Renseigner l'ID utilisateur pour lequel vous souhaitez obtenir les 5 meilleures recommendations:")

if st.button("Get Recommendations"):
    if user_id:
        try:
            # Send the HTTP request to the Azure Function
            response = requests.get(azure_function_url, params={"user_id": user_id})
            response.raise_for_status()  # Raise an exception for HTTP errors
            recommended_articles = response.json()

            # Display the recommended articles
            st.write("Recommended articles for user ID:", user_id)
            st.write(recommended_articles)
        except requests.exceptions.RequestException as e:
            st.write("An error occurred while fetching recommendations:", e)
    else:
        st.write("Please enter a user ID.")