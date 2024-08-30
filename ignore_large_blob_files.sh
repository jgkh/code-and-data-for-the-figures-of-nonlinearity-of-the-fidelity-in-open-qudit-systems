#!/bin/bash

# Define the directory and file type to search for
TARGET_DIR="dat/"
FILE_TYPE="*.blob"
SIZE_LIMIT="+100M"

# Find and stop tracking large files
echo "Finding and untracking .blob files larger than 100MB in the $TARGET_DIR directory..."
find -size $SIZE_LIMIT -exec git rm --cached {} \;

# Add these files to .gitignore
echo "Adding paths of large .blob files to .gitignore..."
find $TARGET_DIR -name "$FILE_TYPE" -size $SIZE_LIMIT >> .gitignore

# Commit the changes
echo "Committing the changes..."
git add .gitignore
git commit -m "Ignore .blob files larger than 100MB and update .gitignore"

# Push changes to the remote repository
echo "Pushing changes to the remote repository..."
git push origin main

echo "Process completed successfully!"
