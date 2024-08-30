#!/bin/bash

# Desired Python version
REQUIRED_PYTHON_VERSION="3.9"

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Function to check the Python version
check_python_version() {
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ $PYTHON_VERSION == $REQUIRED_PYTHON_VERSION* ]]; then
        echo "Using Python $PYTHON_VERSION"
    else
        handle_error "Python $REQUIRED_PYTHON_VERSION is required. Current version is $PYTHON_VERSION. Please install the required version."
    fi
}

# Step 1: Check for the required Python version
echo "Checking Python version..."
check_python_version

# Step 2: Generate requirements.txt using pipreqs (if needed)
echo "Generating requirements.txt using pipreqs..."
pip install pipreqs || handle_error "Failed to install pipreqs"
pipreqs . --force || handle_error "Failed to generate requirements.txt with pipreqs"

# Step 3: Create a virtual environment using the specified Python version
echo "Creating a virtual environment named 'venv' with Python $REQUIRED_PYTHON_VERSION..."
python3 -m venv venv || handle_error "Failed to create virtual environment"

# Step 4: Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate || handle_error "Failed to activate virtual environment"

# Step 5: Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || handle_error "Failed to install dependencies from requirements.txt"

# Step 6: Install ipykernel and add the virtual environment to Jupyter
echo "Installing ipykernel in the virtual environment..."
pip install ipykernel || handle_error "Failed to install ipykernel"
python -m ipykernel install --user --name=venv_kernel --display-name="Python (venv)" || handle_error "Failed to add virtual environment to Jupyter kernels"

# Step 7: Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate || handle_error "Failed to deactivate the virtual environment"

echo "Setup complete. Please start your Jupyter notebook and select the 'Python (venv)' kernel."
