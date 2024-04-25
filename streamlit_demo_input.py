import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Load the model from disk
model = joblib.load("<model_name.sav>")

# Load the list from the Pickle file
with open('<path_to_folder/image_features.pkl>', 'rb') as file:
    dataset_features = pickle.load(file)

# Functions about model
def load_and_preprocess_image(image_file):
    img = image.load_img(BytesIO(image_file), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_file, model):
    img_array = load_and_preprocess_image(image_file)
    features = model.predict(img_array)
    return features.flatten()

# Intro
st.write("""
# Image similarity app
Get most similar furnitures based on a furniture image!
""")

# File Upload
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # Show the image filename and image.
    st.write("### Your Input:")
    # st.write(f'filename: {uploaded_file.name}')
    st.image(bytes_data, caption=uploaded_file.name, width=200)

    # Image similarity
    st.write("### Comparison with the range:")

    # Processing similarity model
    query_features = extract_features(bytes_data, model)

    similarities = []
    for filename, features in dataset_features:
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((filename, similarity))

    # Sort the images by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print the top similar images
    names = []
    sim = []
    for i, (filename, similarity) in enumerate(similarities[:5]):
        names.append(filename)
        sim.append(similarity)

    top5 = pd.DataFrame({'filename': names, 'similarity_score': sim})
    top5['product_id'] = top5['filename'].str.extract('(\d+)')

    # Join with range lookup dataframe (Based on query outputs that contains image characteristics like unique product ID)
    range_df = pd.read_csv("<path_to_folder/query_results.csv>", converters = {'product_id': str})
    top5 = top5.merge(range_df, on = 'product_id', how='left')
    top5 = top5[['filename', 'similarity_score', 'product_id', 'product_type', 'image_url']]

    # Display the DataFrame as a table
    st.data_editor(
        top5,
        column_config={
            "image_url": st.column_config.ImageColumn("product_image", help="Streamlit app preview screenshots")
        },
        hide_index=True,
    )
