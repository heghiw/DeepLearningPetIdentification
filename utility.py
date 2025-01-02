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


# Function to download image and preprocess
def download_and_preprocess_image(url, target_size=(224, 224), visualize=False):
    """
    Downloads an image from the provided URL, preprocesses it, and optionally visualizes it.

    Arguments:
    url -- The URL of the image to download.
    target_size -- The size to resize the image to (default is 224x224).
    visualize -- Flag to control if the image should be visualized (default is True).

    Returns:
    The preprocessed image (TensorFlow Tensor).
    """
    # Download the image from the URL
    response = requests.get(url)
    image_bytes = response.content
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = fix_orientation(pil_image)

    # Convert the image to a TensorFlow tensor and normalize
    image = tf.convert_to_tensor(np.array(pil_image), dtype=tf.float32) / 255.0

    # Detect pets in the image
    bounding_box = detect_pet(image)

    if bounding_box is not None:
        # If a pet is detected, crop and resize around it
        image = crop_and_resize(image, bounding_box, target_size)
    else:
        # If no pet detected, resize with padding
        image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])

    # Visualize the image if needed
    visualize_image(image, title="Processed Image", visualize=visualize)

    return image
