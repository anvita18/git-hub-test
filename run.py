import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import altair as alt
import pickle
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Function to extract text from a single PDF file and preprocess it
def extract_and_process_text(file):
    try:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        # Text preprocessing
        text = re.sub(r'\W', ' ', text.lower())
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(filtered_words)
    except Exception as e:
        return str(e)

# Load the pre-trained neural network model
model = load_model('neural_network_model.h5')

# Load the TF-IDF vectorizer used during training
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the merged DataFrame
merged_df = pd.read_csv('/Users/anvitavyas/Downloads/Final_Project_2/merged_df.csv')

# Define the function to get people by category
def get_people_by_category(df, category):
    filtered_df = df[df['Main Category'] == category]
    age_groups = pd.cut(filtered_df['Age'], bins=[0, 20, 30, 40, 50, float('inf')], labels=['0-20', '21-30', '31-40', '41-50', '51+'])
    filtered_df['Age Group'] = age_groups
    town_counts = filtered_df['City'].value_counts()
    age_group_counts = age_groups.value_counts().sort_index()
    return filtered_df[['Person', 'City', 'Age', 'Age Group']], town_counts, age_group_counts

# Load the GeoDataFrame for Maine
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
maine_gdf = gdf[gdf.name == 'United States'].to_crs(epsg=4326)

# Streamlit app
st.title("CIVA Order Category Classification And Community Analysis")

# File uploader
uploaded_file = st.file_uploader("Pick a file")

if uploaded_file is not None:
    # Extract and process text from the uploaded PDF file
    processed_text = extract_and_process_text(uploaded_file)

    # Vectorize the processed text
    text_tfidf = vectorizer.transform([processed_text])

    # Predict the category using the neural network model
    y_pred = model.predict(text_tfidf.toarray())
    predicted_category_index = np.argmax(y_pred, axis=1)[0]
    categories = label_encoder.classes_
    category_name = categories[predicted_category_index]
    
    # Display the predicted category at the top as a card
    st.metric(label="Predicted Category", value=category_name)

    # Get the people in the predicted category
    people, town_counts, age_groups = get_people_by_category(merged_df, category_name)

    # Prepare the age group data for Altair
    age_group_df = age_groups.reset_index()
    age_group_df.columns = ['Age Group', 'Count']

    # Create an interactive age group bar chart using Altair
    st.write("Age Groups:")

    # Define the selection
    selection = alt.selection_multi(fields=['Age Group'])

    age_group_chart = alt.Chart(age_group_df).mark_bar().encode(
        x=alt.X('Count:Q', title='Count'),
        y=alt.Y('Age Group:N', title='Age Groups', sort=['0-20', '21-30', '31-40', '41-50', '51+']),
        color=alt.condition(selection, alt.value('steelblue'), alt.value('lightgray')),
        tooltip=['Age Group', 'Count']
    ).add_selection(
        selection
    ).properties(
        title='Distribution of Age Groups',
        width=600,
        height=400
    )

    # Display the chart
    st.altair_chart(age_group_chart, use_container_width=True)

    # Use Streamlit's session state to keep track of the selected age groups
    if 'selected_age_groups' not in st.session_state:
        st.session_state.selected_age_groups = []

    # Update session state with the selection
    selected_age_groups = st.multiselect("Select Age Groups", age_group_chart.data['Age Group'].unique().tolist())
    st.session_state.selected_age_groups = selected_age_groups

    # Filter the DataFrame based on the selected age groups
    if selected_age_groups:
        filtered_people = people[people['Age Group'].isin(selected_age_groups)]
    else:
        filtered_people = people

    col1, col2 = st.columns(2)

    with col1:
        # Display the list of people
        st.write("List of People in the Category:")
        st.write(filtered_people[['Person', 'City', 'Age']])

    with col2:
        # Display the town counts as a bar chart
        town_counts_sorted = filtered_people['City'].value_counts()
        town_counts_df = pd.DataFrame({'City': town_counts_sorted.index, 'Count': town_counts_sorted.values})

        st.write("Town Counts:")
        town_counts_chart = alt.Chart(town_counts_df).mark_bar().encode(
            x=alt.X('City:N', title='City'),
            y=alt.Y('Count:Q', title='Number of People')
        ).properties(
            title="Town Counts",
            width='container',  # Set width to container for responsive layout
            height=400  # Adjust height as needed
        )
        st.altair_chart(town_counts_chart, use_container_width=True)

    # Create the interactive map for town counts in Maine
    st.write("Interactive Map of Town Counts in Maine:")
    maine_map = folium.Map(location=[45.2538, -69.4455], zoom_start=11)  # Adjusted zoom level

    # Add town counts to the map
    for city, count in town_counts.items():
        # Dummy coordinates for demonstration, replace with actual coordinates
        coordinates = [45.2538 + np.random.uniform(-0.1, 0.1), -69.4455 + np.random.uniform(-0.1, 0.1)]
        folium.CircleMarker(
            location=coordinates,
            radius=5 + count / 10,  # Scale the circle size based on count
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f'{city}: {count} people'
        ).add_to(maine_map)

    folium_static(maine_map)

    # Create and display a word cloud
    st.write("Word Cloud of Keywords:")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
