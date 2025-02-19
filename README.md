# img_to_faiss

## Overview
This repository provides a pipeline for converting raw images into FAISS-compatible face embeddings using DeepFace and FAISS. The system extracts facial features, stores them efficiently, and enables fast similarity searches.

## Features
- Extract face embeddings using ArcFace.
- Store and index embeddings using FAISS for fast retrieval.
- Supports GPU acceleration for improved performance.
- Parallel processing for efficient embedding extraction.

## Installation
```sh
git clone https://github.com/Raghulskr12/img_to_faiss.git
cd img_to_faiss
pip install -r requirements.txt
```

## Dependencies
All required dependencies are listed in `requirements.txt`.
```txt
numpy
faiss
DeepFace
tqdm
concurrent.futures
pathlib
logging
```

## Usage
### Extract embeddings and create FAISS index
```sh
python main.py --image_folder path/to/images --output_file path/to/output.faiss
```

### Load and query FAISS index
```python
import faiss
import numpy as np

# Load index
index = faiss.read_index("path/to/output.faiss")

# Load stored image names
image_names = np.load("path/to/output_image_names.npy")

# Perform search
query_embedding = np.random.rand(512).astype(np.float32)  # Example query
faiss.normalize_L2(query_embedding.reshape(1, -1))
distances, indices = index.search(query_embedding.reshape(1, -1), k=5)

# Get matched image names
matched_images = [image_names[i] for i in indices[0]]
print("Matched Images:", matched_images)
```

## Troubleshooting
- Ensure all dependencies are installed.
- If `DeepFace.represent` fails, check if `ArcFace` model is downloaded properly.
- Use `enforce_detection=False` if faces are not always detected.

