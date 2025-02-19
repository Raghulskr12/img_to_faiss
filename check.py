import faiss
import numpy as np

# Load the FAISS index
index_file = "Training_images\\face_embeddings.faiss"  # Change this to your file path
index = faiss.read_index(index_file)

# Load the stored image names
image_names_file = index_file.replace('.faiss', '_image_names.npy')
image_names = np.load(image_names_file, allow_pickle=True)

# Print all stored image names
print("List of all stored persons:")
for i, name in enumerate(image_names):
    print(f"{i + 1}. {name}")

# Check if the count matches the FAISS index
num_stored = index.ntotal
num_expected = len(image_names)

if num_stored == num_expected:
    print(f"\n✅ All {num_stored} embeddings are correctly stored.")
else:
    print(f"\n⚠️ Mismatch! Stored: {num_stored}, Expected: {num_expected}")
