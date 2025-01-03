###### Utility ########

### Add here code that can be reused in different notebooks#####



import json
from PIL import Image, ExifTags
import requests
import io
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import gdown
import pandas as pd

def get_data():
    """
    Fetches and parses JSON data from the given URL.

    Args:
        url (str): The raw GitHub URL to the JSON file.

    Returns:
        dict: The parsed JSON data as a Python dictionary.

    Raises:
        Exception: If the request or JSON parsing fails.
    """
    url = "https://raw.githubusercontent.com/avkaz/DeepLearningPetIdentification/main/pets_db.json"

    try:
        # Send a GET request to the raw URL
        response = requests.get(url)

        # Check for successful request (status code 200)
        response.raise_for_status()
        
        # Parse the response as JSON
        data = response.json()
        
        return data

    except requests.RequestException as e:
        # Handle network-related issues (connection problems, timeout, etc.)
        print(f"An error occurred while fetching data: {e}")
        raise

    except json.JSONDecodeError as e:
        # Handle issues when the response is not valid JSON
        print(f"An error occurred while parsing JSON: {e}")
        raise

# Global variable to store the detector model
detector = None

# Downloading pre-trained model for detecting pets on image and resizing image in a such way that pet will be in a center
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"


# Function to load the model
def load_detector_model():
    global detector
    if detector is None:
        print("Uploading model...")
        detector = hub.load(MODEL_URL).signatures['serving_default']
        print("Model loaded successfully.")
    else:
        pass


# Function to fix orientation using EXIF
def fix_orientation(image):
    """
    Adjust the image orientation based on its EXIF metadata to account for camera rotation.
    The function looks for the 'Orientation' tag in the EXIF data and rotates the image accordingly.

    Arguments:
    image -- The image to fix the orientation for (PIL Image object).

    Returns:
    PIL Image with corrected orientation.
    """
    try:
        # Iterate through all EXIF tags and find the 'Orientation' tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Get the EXIF data if available
        exif = image._getexif()
        if exif is not None:
            # Extract the orientation value
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If EXIF data is not present or invalid, just pass
        pass
    return image


# Function to crop and resize the image based on a bounding box
def crop_and_resize(image, bounding_box, target_size):
    """
    Crops the image using a given bounding box and then resizes it to the target size.

    Arguments:
    image -- The image to crop and resize (TensorFlow Tensor).
    bounding_box -- A tuple (x1, y1, x2, y2) specifying the coordinates of the bounding box.
    target_size -- The target size (height, width) to resize the image to.

    Returns:
    The cropped and resized image (TensorFlow Tensor).
    """
    # Convert the image to a TensorFlow tensor if it's not already
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Unpack bounding box coordinates
    x1, y1, x2, y2 = bounding_box
    # Crop the image using TensorFlow's strided_slice function
    image = tf.strided_slice(image, [int(y1), int(x1), 0], [int(y2), int(x2), 3])

    # Resize the image to the target size
    image = tf.image.resize(image, target_size)

    return image


# Function to detect pets in the image (Placeholder function, adjust as needed)
def detect_pet(image):
    """
    Detects pets (e.g., cats, dogs) in the image using a pre-trained object detection model.

    Arguments:
    image -- The image to detect pets in (TensorFlow Tensor).

    Returns:
    A bounding box (x1, y1, x2, y2) if a pet is detected, None otherwise.
    """
    load_detector_model()  # Ensure the model is loaded before detection

    # Preprocess the image for the object detection model
    input_tensor = tf.image.resize(image, [640, 640]) / 255.0  # Resize for the detector input
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add a batch dimension

    # Convert the tensor to uint8 as required by the detector
    input_tensor_uint8 = tf.cast(input_tensor * 255.0, tf.uint8)

    # Run the detector (assuming 'detector' is defined elsewhere)
    result = detector(tf.convert_to_tensor(input_tensor_uint8))  # Call the model
    result = {key: value.numpy() for key, value in result.items()}  # Convert to numpy for easier manipulation

    # Check if necessary keys exist in the detection result
    if 'detection_classes' in result and 'detection_scores' in result:
        detected_classes = result['detection_classes']
        detected_boxes = result['detection_boxes']
        detected_scores = result['detection_scores']

        # Pet classes of interest (could be different depending on model)
        pet_classes = [b"Cat", b"Dog", b"Animal"]

        # Loop through detections and check for pets with a high confidence score
        for idx in range(len(detected_classes[0])):
            detected_class = detected_classes[0][idx]  # First image in batch
            detected_score = detected_scores[0][idx]  # First image in batch
            detected_box = detected_boxes[0][idx]  # First image in batch

            if detected_class in pet_classes and detected_score > 0.5:
                return detected_box  # Return the bounding box of the first detected pet

    # If no pet detected, return None
    return None


# Function to visualize the image
def visualize_image(image, title="Processed Image", visualize=False):
    """
    Visualizes the processed image using Matplotlib.

    Arguments:
    image -- The image to visualize, can be a TensorFlow tensor or a NumPy array.
    title -- The title to display on top of the image.
    visualize -- A flag to control whether to visualize the image. Default is True.
    """
    if visualize:
        # Convert TensorFlow tensor to NumPy array if necessary
        if isinstance(image, tf.Tensor):
            image = image.numpy()

        # If it's an RGB image, clip pixel values to the range [0, 1]
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.clip(image, 0, 1)
        elif image.ndim == 2:  # If grayscale, clip to [0, 255]
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Show the image using Matplotlib
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()


# Resize the image to a smaller size to reduce the tensor's size
def resize_image(image, target_size=(128, 128)):
    """
    Resize the image to the target size.

    Args:
    image -- The image tensor to resize.
    target_size -- The target size (height, width) for resizing.

    Returns:
    Resized image tensor.
    """
    resized_image = tf.image.resize(image, target_size)
    return resized_image

# Save the image tensor in a compressed format using TFRecord
def save_tensor_as_tfrecord(tensor, filename="image_data.tfrecord"):
    """
    Saves the image tensor in TFRecord format to reduce size and improve efficiency.

    Args:
    tensor -- The image tensor to save.
    filename -- The name of the file to save the tensor in.
    """
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(filename) as writer:
        # Create a feature dictionary with the image tensor
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(tensor).numpy()]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Write the example to the TFRecord file
        writer.write(example.SerializeToString())

# Load an image from a URL and preprocess
def download_and_preprocess_image(url, target_size=(128, 128), save_to_tfrecord=False):
    """
    Download and preprocess the image, then optionally save it as a TFRecord.

    Args:
    url -- The URL to download the image.
    target_size -- The target size to resize the image.
    save_to_tfrecord -- Flag to save the image tensor as a TFRecord (default False).
    
    Returns:
    Resized image tensor.
    """
    # Assuming the `fix_orientation`, `detect_pet`, etc., are defined as before

    # Download the image (as before)
    response = requests.get(url)
    image_bytes = response.content
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = fix_orientation(pil_image)

    # Convert to tensor and normalize
    image = tf.convert_to_tensor(np.array(pil_image), dtype=tf.float32) / 255.0

    # Detect and crop the pet (optional)
    bounding_box = detect_pet(image)
    if bounding_box is not None:
        image = crop_and_resize(image, bounding_box, target_size)
    else:
        image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])

    # Resize the image to reduce its size (e.g., 128x128)
    image = resize_image(image, target_size)

    # Optionally save the image tensor to a TFRecord file
    if save_to_tfrecord:
        save_tensor_as_tfrecord(image, filename="pet_image_data.tfrecord")

    return image

# function to get dataset with paires of pets
def download_and_load_json_to_dataframe():
    """
    Downloads a JSON file from Google Drive using the file's unique ID and loads it into a pandas DataFrame.

    :param file_id: str, the unique identifier of the file on Google Drive.
                    Example: '1A2B3C4D5E6F...'
    :param output_path: str, the path to save the downloaded file. Example: './data/file.json'
    """
    # Construct the direct download URL
    url = f"https://drive.google.com/uc?id=1anGcMpqpVCCD21Az7AzJiyafN9n0fz-i&export=download"
    output_path = './pets_pair.json'
    
    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the file
    print(f"Downloading file from Google Drive: {url}")
    gdown.download(url, output_path, quiet=False)
    print(f"File saved to: {output_path}")
    
    # Load the downloaded JSON file into a pandas DataFrame
    print("Loading the JSON file into a pandas DataFrame...")
    df = pd.read_json(output_path)
    
    # Display the first few rows of the DataFrame to confirm it's loaded correctly
    print("Data loaded into DataFrame:")
    print(df.head())
    
    return df


