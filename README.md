# Furniture Image Similarity
This project explains how to find most similar furniture images (out of a range of images of your own) based on your input image via using convolutional neural network model.

**Please note you need a furniture image dataset for this model to work, the data I used was private, therefore it is not included here.**

## Notebooks

There are 3 notebooks;

- Range Image Downloader.ipynb: Querying a database table with image ID and image URLs, and then downloading those images from URLs into given path.
- Range Model Training.ipynb: Training the model based on downloaded furniture images and extracting the model for further usage.
- Model Usage.ipynb: Using the exported model and having an example usage; top 5 most similar furniture images based on given example picture.

In addition to that, there is also a demo file to demonstrate model via [Streamlit](https://streamlit.io/) which is "streamlit_demo_input.py" file.

## Demo Screenshots
Input file:

<img src="/images/input.png" width="500">

Results - 1:

<img src="/images/results1.png" width="500">

Results - 2:

<img src="/images/results2.png" width="500">



