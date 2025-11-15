#!/bin/bash

# simple_scaffold.sh - Create only the folder structure

echo "Creating ML project folder structure..."

# Create all directories
mkdir -p src/{preprocess,train,evaluate,predict,utils}
mkdir -p app
mkdir -p data/{raw,processed,external}
mkdir -p models/{saved,checkpoints}
mkdir -p visuals/{plots,examples}

# Create Python package files
touch src/__init__.py
touch src/preprocess/__init__.py
touch src/preprocess/preprocess.py
touch src/train/__init__.py
touch src/train/train.py
touch src/evaluate/__init__.py
touch src/evaluate/evaluate.py
touch src/predict/__init__.py
touch src/predict/predict.py
touch src/utils/__init__.py
touch src/utils/helpers.py

# Create other project files
touch main.py requirements.txt README.md

echo "Folder structure created successfully!"
tree -d -I '__pycache__' 2>/dev/null || find . -type d | sort
