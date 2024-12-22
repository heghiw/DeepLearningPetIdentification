# DeepLearningPetIdentification

This project focuses on comparing images of lost pets with those already stored in a database. It aims to help identify and match lost pets with photos in a pre-existing dataset.

## Project Structure

- **`scraper.py`**: A script used to scrape pet images and data from the website [www.psi-detective.cz](https://www.psi-detective.cz).
- **`pets_db.json`**: A database containing scraped pet data. Each entry includes a pet ID, description, and links to the corresponding images.
- **`utility.py`**: A collection of useful functions that can be utilized across multiple notebooks within the project.
- **Jupyter Notebooks**: Several notebooks, each implementing a different approach to pet classification. These notebooks demonstrate various methods for training and testing models to identify pets.
