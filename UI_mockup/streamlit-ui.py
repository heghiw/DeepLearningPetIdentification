import streamlit as st
import os
import base64
import random
import pathlib

# Function to get all image paths from the images folder and its subfolders
def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def load_css(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

# Load custom CSS
css_path = pathlib.Path("assets\styles.css")
load_css(css_path)

# Get all image paths from the 'images' folder
image_folder = 'images'
dog_images_path = get_image_paths(image_folder)

# Select 20 random images
dog_images = random.sample(dog_images_path, min(len(dog_images_path), 20))

st.title("Deep Learning Pet Identification")
st.subheader("Photographed dogs:")

# Display dog images in a responsive grid gallery
num_columns = 5  # Adjust the number of columns as needed
cols = st.columns(num_columns)

for i, img_path in enumerate(dog_images):
    col = cols[i % num_columns]
    with col:
        img_base64 = image_to_base64(img_path)
        img_html = f'''
        <div class="square">
            <a href="#viewbox-{i}">
                <img src="data:image/jpeg;base64,{img_base64}" />
            </a>
        </div>
        <div id="viewbox-{i}" class="viewbox">
            <div class="viewbox-content">
                <img src="data:image/jpeg;base64,{img_base64}" />
            </div>
            <a href="#" class="close_button">&#x2716;</a>
        </div>
        '''
        col.markdown(img_html, unsafe_allow_html=True)

if st.button("Upload a photo of your pet"):
    st.write("Returning a random dog from the gallery...")
    random_dog = random.choice(dog_images)
    st.image(random_dog, use_container_width=True)