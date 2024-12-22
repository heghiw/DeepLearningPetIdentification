###### Utility ########

### Add here code that can be reused in different notebooks#####


import requests
from PIL import Image
from io import BytesIO
import json
import firebase_admin
from firebase_admin import credentials, storage
import json
from google.colab import userdata  # Specific to Google Colab environment


def get_data(url):
    """
    Fetches and parses JSON data from the given URL.

    Args:
        url (str): The raw GitHub URL to the JSON file.

    Returns:
        dict: The parsed JSON data as a Python dictionary.

    Raises:
        Exception: If the request or JSON parsing fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing JSON: {e}")
        raise


def download_picture_as_bytes(image_url):
    """
    Downloads an image from the provided URL and returns it as bytes.

    Args:
        image_url (str): The URL of the image to download.

    Returns:
        bytes: The raw bytes of the image.

    Raises:
        Exception: If the request fails or the image cannot be downloaded.
    """
    try:
        # Fetch the image data
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Read the image content as bytes
        image_bytes = response.content
        return image_bytes
    except requests.RequestException as e:
        print(f"An error occurred while fetching the image: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def get_blobs():
    """
    We don't have anything on blob storage for now, so probably you will not need this function!
    Retrieve a list of blobs (files) from a specific folder in a Firebase storage bucket.

    Preconditions:
    - A Firebase Admin SDK JSON key must be securely stored in Google Colab's userdata.
    - The Firebase project and bucket should be set up correctly in the Firebase console.

    Returns:
        google.cloud.storage.blob.Blob instances representing files in the specified folder.

    Raises:
        ValueError: If the Firebase key is not found in Colab's userdata.
    """
    # Import required libraries for Firebase and JSON handling

    # Retrieve the Firebase key stored in Colab's userdata
    firebase_key = userdata.get('firebase')

    # Validate the presence of the Firebase key
    if not firebase_key:
        raise ValueError("Firebase key not found in Colab userdata!")

    # Convert the JSON key (string) to a dictionary for initializing Firebase
    try:
        firebase_key_dict = json.loads(firebase_key)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for Firebase key!") from e

    # Initialize Firebase Admin SDK using the provided credentials
    cred = credentials.Certificate(firebase_key_dict)
    try:
        firebase_admin.initialize_app(cred, {'storageBucket': 'codereview-22c86.appspot.com'})
    except firebase_admin.exceptions.FirebaseError as e:
        raise RuntimeError("Failed to initialize Firebase Admin SDK.") from e

    # Access the Firebase storage bucket
    bucket = storage.bucket()

    # Specify the folder to retrieve blobs from
    folder_name = "AceVentura/"

    # List all blobs (files) under the specified folder
    blobs = bucket.list_blobs(prefix=folder_name)

    # Return the list of blobs
    return blobs