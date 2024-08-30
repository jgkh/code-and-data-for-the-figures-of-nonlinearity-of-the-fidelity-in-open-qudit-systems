#!/bin/bash

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate || handle_error "Failed to deactivate the virtual environment"

echo "Virtual environment deactivated successfully."
