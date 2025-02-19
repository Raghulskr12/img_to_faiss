import os
import numpy as np
import faiss
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import mmap
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceIndexer:
    def __init__(
        self,
        model_name: str = "ArcFace",
        batch_size: int = 32,
        max_workers: int = None,
        distance_metric: str = "cosine"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers or os.cpu_count()
        self.distance_metric = distance_metric
        
    def _process_single_image(self, image_path: Path) -> Optional[Tuple[np.ndarray, str]]:
        try:
            embeddings = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='retinaface'
            )
            
            if isinstance(embeddings, list) and len(embeddings) > 0:
                embedding_array = np.array(embeddings[0]['embedding'], dtype=np.float32)
                return embedding_array, image_path.name
            else:
                logger.warning(f"No valid embedding for {image_path.name}")
                return None
        except Exception as e:
            logger.warning(f"Failed to process {image_path.name}: {e}")
            return None


    def extract_embeddings(self, image_folder: str) -> Tuple[np.ndarray, List[str]]:
        """Extract embeddings using parallel processing."""
        image_folder = Path(image_folder)
        image_paths = list(image_folder.glob('*.[jJpP][pPnN][gG]')) + list(image_folder.glob('*.[jJ][pP][eE][gG]')) + list(image_folder.glob('*.[bB][mM][pP]')) + list(image_folder.glob('*.[tT][iI][fF][fF]')) + list(image_folder.glob('*.[gG][iI][fF]'))
        
        embeddings = []
        image_names = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_image, path) 
                      for path in image_paths]
            
            # Show progress bar
            for future in tqdm(futures, total=len(image_paths), desc="Processing images"):
                result = future.result()
                if result is not None:
                    embedding, image_name = result
                    embeddings.append(embedding)
                    image_names.append(image_name)

        return np.array(embeddings), image_names

    def create_faiss_index(
        self,
        embeddings: np.ndarray,
        output_file: str,
        use_gpu: bool = False
    ) -> None:
        """Create and save an optimized FAISS index."""
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)
            
        # Choose appropriate index based on data size
        if len(embeddings) < 1_000_000:
            # For smaller datasets, use flat index
            index = faiss.IndexFlatIP(dimension) if self.distance_metric == "cosine" \
                else faiss.IndexFlatL2(dimension)
        else:
            # For larger datasets, use IVF index with clustering
            nlist = min(int(np.sqrt(len(embeddings))), 16384)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        
        # Use GPU if available and requested
        if use_gpu and faiss.get_num_gpus():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Add embeddings to index
        index = self._add_embeddings_batched(index, embeddings)
        
        # Convert back to CPU if necessary
        if use_gpu and faiss.get_num_gpus():
            index = faiss.index_gpu_to_cpu(index)
        
        # Save index
        faiss.write_index(index, output_file)
        logger.info(f"Index saved to {output_file}")
    
    def _add_embeddings_batched(
        self,
        index: faiss.Index,
        embeddings: np.ndarray
    ) -> faiss.Index:
        """Add embeddings to index in batches to manage memory."""
        for i in tqdm(range(0, len(embeddings), self.batch_size), desc="Building index"):
            batch = embeddings[i:i + self.batch_size]
            index.add(batch)
        return index

def main():
    # Configuration
    image_folder = "Training_images"
    output_file = Path(image_folder) / "face_embeddings.faiss"
    
    # Initialize indexer
    indexer = FaceIndexer(
        model_name="ArcFace",
        batch_size=32,
        max_workers=os.cpu_count(),
        distance_metric="cosine"
    )
    
    # Extract embeddings
    embeddings, image_names = indexer.extract_embeddings(image_folder)
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings extracted")
        return
    
    # Create and save index
    indexer.create_faiss_index(
        embeddings,
        str(output_file),
        use_gpu=True  # Will automatically fall back to CPU if GPU not available
    )
    
    # Save image names for reference
    np.save(str(output_file).replace('.faiss', '_image_names.npy'), image_names)
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()