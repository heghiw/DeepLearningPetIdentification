# DeepLearningPetIdentification

This project focuses on comparing images of lost pets with those already stored in a database. It aims to help identify and match lost pets with photos in a pre-existing dataset.

## Project Structure

- **`scraper.py`**: A script used to scrape pet images and data from the website [www.psi-detective.cz](https://www.psi-detective.cz).
- **`pets_db.json`**: A database containing scraped pet data. Each entry includes a pet ID, description, and links to the corresponding images.
- **`utility.py`**: A collection of useful functions that can be utilized across multiple notebooks within the project.
- **Jupyter Notebooks**: Several notebooks, each implementing a different approach to pet classification. These notebooks demonstrate various methods for training and testing models to identify pets.


# Collaboration Workflow

This document outlines the workflow for contributing to the project. Please follow the guidelines to ensure smooth collaboration and version control.

## Table of Contents
1. [Branch Naming Convention](#branch-naming-convention)
2. [Development on Local Machine](#development-on-local-machine)
3. [Development on Google Colab](#development-on-google-colab)
4. [Main Branch Protection](#main-branch-protection)
5. [Pull Requests](#pull-requests)
6. [Working with Pets Database](#working-with-pets-database)
7. [Utility Functions](#utility-functions)

## 1. Branch Naming Convention

When creating a new branch, follow this naming format:
- **Feature Branch**: `feat_<feature_name>_<short_description>`
- **Fix Branch**: `fix_<issue_name>_<short_description>`

This naming convention helps keep branches organized and easy to understand.

## 2. Development on Local Machine

If you are working on your local machine:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-forked-repository.git
    ```
2. Checkout to your newly created branch:
    ```bash
    git checkout feat_image_preprocessing_resize
    ```
3. Continue your development in a classic Git workflow:
    - Edit files.
    - Stage your changes:
      ```bash
      git add .
      ```
    - Commit your changes:
      ```bash
      git commit -m "Description of changes"
      ```
    - Push your changes to the remote branch:
      ```bash
      git push origin feat_image_preprocessing_resize
      ```

## 3. Development on Google Colab

If you are using Google Colab:
1. Create a new Colab notebook.
2. Save a copy of the notebook to your GitHub repository (select the branch you are working on).
    - In Colab, click on `File` → `Save a copy in GitHub`.
    - Choose the repository and the appropriate branch (e.g., `feat_image_preprocessing_resize`).
3. Work within the notebook and push your changes back to GitHub when finished.

## 4. Main Branch Protection

The `main` branch is **protected** and can only be updated via **pull requests (PRs)**. This ensures that the main codebase remains stable and any changes are reviewed before merging.

## 5. Pull Requests

When you are ready to submit your work:
1. Push your changes to your remote branch (e.g., `feat_image_preprocessing_resize`).
2. Open a **pull request** (PR) from your branch to the `main` branch.
3. **PR Approval**: Every pull request must be approved by one additional team member. This ensures the code is reviewed for quality, accuracy, and completeness.
4. **Descriptive Comments**: Each PR should include a **descriptive comment** explaining what was changed, why, and how it works.

## 6. Working with Pets Database

In the GitHub repository, there is a `pets_db.json` file that contains scraped data on pets.

- **Issue**: The dog breed "křižovec" incorrectly includes some cat entries.
- **Solution**: If you are fine-tuning the model or using the database, **exclude** all lines with the breed "křižovec" to avoid this issue.

    You can filter the data before using it:
    ```python
    import json

    with open('pets_db.json', 'r') as f:
        pets_data = json.load(f)

    pets_data_filtered = [pet for pet in pets_data if pet['breed'] != 'křižovec']
    ```

## 7. Utility Functions

The `utility.py` file contains useful functions that can be reused across different notebooks.

- If you create a new function that could be useful for other notebooks, add it to the `utility.py` file.
- Make sure to provide a **clear description** of the function in the docstring so others can understand its purpose and how to use it.

### Important: Updating `utility.py`

To use the utility functions in your notebooks, copy the top cells from `PetsDetection.ipynb`. These cells download `utility.py` and import its functions for use:

```python
# Download utility.py file
!wget https://github.com/your-username/your-repository/raw/main/utility.py

# Import utility functions
from utility import resize_image, another_function
