"""
Dataset generator for cloud detection using Sentinel-2 imagery.

This module implements the core functionality for generating training datasets
from Sentinel-2 images and corresponding cloud masks from the Baetens-Hagolle
reference dataset.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CloudDatasetGenerator:
    """
    Generator for cloud detection training datasets.
    
    This class handles the extraction of patches from Sentinel-2 imagery and
    corresponding cloud masks, organizing them into training and validation sets.
    
    Attributes:
        output_dir (Path): Directory for output files
        patch_size (int): Size of extracted patches in pixels
        stride (int): Stride for patch extraction
        val_split (float): Validation set proportion
        skip_nodata (bool): Whether to skip patches with high no_data ratio
        nodata_threshold (float): Threshold for no_data pixel ratio
    """
    
    def __init__(
        self,
        output_dir: str,
        patch_size: int = 256,
        stride: int = 128,
        val_split: float = 0.2,
        skip_nodata: bool = True,
        nodata_threshold: float = 0.3,
        random_seed: int = 42
    ):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory for output files
            patch_size: Size of extracted patches in pixels
            stride: Stride for patch extraction
            val_split: Validation set proportion (0.0 - 1.0)
            skip_nodata: Whether to skip patches with high no_data ratio
            nodata_threshold: Threshold for no_data pixel ratio (0.0 - 1.0)
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.val_split = val_split
        self.skip_nodata = skip_nodata
        self.nodata_threshold = nodata_threshold
        
        np.random.seed(random_seed)
        
        # Create directory structure
        self.train_img_dir = self.output_dir / "train" / "images"
        self.train_mask_dir = self.output_dir / "train" / "masks"
        self.val_img_dir = self.output_dir / "val" / "images"
        self.val_mask_dir = self.output_dir / "val" / "masks"
        
        for dir_path in [self.train_img_dir, self.train_mask_dir,
                         self.val_img_dir, self.val_mask_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset generator initialized. Output: {self.output_dir}")
    
    def find_mask_for_image(
        self,
        image_path: Path,
        mask_base_dir: Path
    ) -> Optional[Path]:
        """
        Find corresponding cloud mask for a Sentinel-2 image.
        
        The function matches images to masks based on tile ID and acquisition date
        extracted from the filename.
        
        Args:
            image_path: Path to Sentinel-2 image
            mask_base_dir: Base directory containing reference masks
        
        Returns:
            Path to corresponding mask or None if not found
        """
        image_name = image_path.stem
        
        # Extract tile and date from filename
        # Expected format: S2_TILE_DATE.tif or S2X_TILE_DATE.tif
        parts = image_name.split('_')
        if len(parts) < 3:
            logger.warning(f"Unexpected filename format: {image_name}")
            return None
        
        tile = parts[-2]  # Tile ID (e.g., 29RPQ)
        date = parts[-1]  # Date (e.g., 20170621)
        
        # Search for matching mask directory
        for scene_dir in mask_base_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            if tile in scene_dir.name and date in scene_dir.name:
                mask_path = scene_dir / "Classification" / "classification_map.tif"
                if mask_path.exists():
                    return mask_path
        
        return None
    
    def extract_patches(
        self,
        image_path: Path,
        mask_path: Path
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Extract patches from image and mask.
        
        Args:
            image_path: Path to Sentinel-2 image
            mask_path: Path to corresponding mask
        
        Returns:
            List of tuples containing (image_patch, mask_patch, metadata)
        """
        patches = []
        
        with rasterio.open(image_path) as img_src, \
             rasterio.open(mask_path) as mask_src:
            
            # Verify dimension compatibility
            height = min(img_src.height, mask_src.height)
            width = min(img_src.width, mask_src.width)
            
            if img_src.shape != mask_src.shape:
                logger.warning(
                    f"Dimension mismatch - Image: {img_src.shape}, "
                    f"Mask: {mask_src.shape}. Using minimum dimensions."
                )
            
            # Calculate number of patches
            n_patches_h = (height - self.patch_size) // self.stride + 1
            n_patches_w = (width - self.patch_size) // self.stride + 1
            
            logger.info(
                f"Processing {image_path.name}: "
                f"{height}x{width} pixels, "
                f"{img_src.count} bands, "
                f"{n_patches_h * n_patches_w} potential patches"
            )
            
            patch_id = 0
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    row = i * self.stride
                    col = j * self.stride
                    
                    # Ensure patch doesn't exceed boundaries
                    if row + self.patch_size > height or \
                       col + self.patch_size > width:
                        continue
                    
                    # Create reading window
                    window = Window(col, row, self.patch_size, self.patch_size)
                    
                    # Read patches
                    img_patch = img_src.read(window=window)
                    mask_patch = mask_src.read(1, window=window)
                    
                    # Skip patches with high no_data ratio
                    if self.skip_nodata:
                        nodata_ratio = (mask_patch == 0).sum() / \
                                     (self.patch_size ** 2)
                        if nodata_ratio > self.nodata_threshold:
                            continue
                    
                    metadata = {
                        'patch_id': patch_id,
                        'row': row,
                        'col': col,
                        'image_name': image_path.stem,
                    }
                    
                    patches.append((img_patch, mask_patch, metadata))
                    patch_id += 1
        
        return patches
    
    def save_patches(
        self,
        patches: List[Tuple[np.ndarray, np.ndarray, Dict]]
    ):
        """
        Save patches to disk, split into training and validation sets.
        
        Args:
            patches: List of (image_patch, mask_patch, metadata) tuples
        """
        if len(patches) == 0:
            logger.warning("No patches to save")
            return
        
        # Split into train/val
        np.random.shuffle(patches)
        split_idx = int(len(patches) * (1 - self.val_split))
        train_patches = patches[:split_idx]
        val_patches = patches[split_idx:]
        
        logger.info(
            f"Saving {len(patches)} patches "
            f"(train: {len(train_patches)}, val: {len(val_patches)})"
        )
        
        # Save datasets
        self._save_patch_set(train_patches, self.train_img_dir, 
                            self.train_mask_dir)
        self._save_patch_set(val_patches, self.val_img_dir, 
                           self.val_mask_dir)
    
    def _save_patch_set(
        self,
        patches: List[Tuple[np.ndarray, np.ndarray, Dict]],
        img_dir: Path,
        mask_dir: Path
    ):
        """Save a set of patches to specified directories."""
        for img_patch, mask_patch, metadata in patches:
            base_name = (f"{metadata['image_name']}_"
                        f"patch_{metadata['patch_id']:04d}")
            
            # Save as NumPy arrays to preserve all bands and precision
            np.save(img_dir / f"{base_name}.npy", img_patch)
            np.save(mask_dir / f"{base_name}.npy", mask_patch)
    
    def process_image(
        self,
        image_path: Path,
        mask_base_dir: Path
    ) -> bool:
        """
        Process a single image and its corresponding mask.
        
        Args:
            image_path: Path to Sentinel-2 image
            mask_base_dir: Base directory containing reference masks
        
        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing image: {image_path.name}")
        
        # Find corresponding mask
        mask_path = self.find_mask_for_image(image_path, mask_base_dir)
        if mask_path is None:
            logger.error(f"No matching mask found for {image_path.name}")
            return False
        
        logger.info(f"Found mask: {mask_path.parent.parent.name}")
        
        # Extract patches
        try:
            patches = self.extract_patches(image_path, mask_path)
        except Exception as e:
            logger.error(f"Failed to extract patches: {e}")
            return False
        
        if len(patches) == 0:
            logger.warning("No valid patches extracted")
            return False
        
        logger.info(f"Extracted {len(patches)} valid patches")
        
        # Save patches
        self.save_patches(patches)
        
        return True
    
    def process_directory(
        self,
        image_dir: Path,
        mask_base_dir: Path
    ) -> Dict[str, int]:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing Sentinel-2 images
            mask_base_dir: Base directory containing reference masks
        
        Returns:
            Dictionary with processing statistics
        """
        # Find all TIFF files
        image_files = sorted(list(image_dir.glob("*.tif")) + 
                           list(image_dir.glob("*.tiff")))
        
        if len(image_files) == 0:
            logger.error(f"No TIFF files found in {image_dir}")
            return {'total': 0, 'processed': 0, 'failed': 0}
        
        logger.info(f"Found {len(image_files)} image(s) to process")
        
        stats = {'total': len(image_files), 'processed': 0, 'failed': 0}
        
        for image_path in tqdm(image_files, desc="Processing images"):
            success = self.process_image(image_path, mask_base_dir)
            if success:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def generate_summary(self):
        """Generate dataset summary statistics."""
        stats = {
            'train_images': len(list(self.train_img_dir.glob('*.npy'))),
            'train_masks': len(list(self.train_mask_dir.glob('*.npy'))),
            'val_images': len(list(self.val_img_dir.glob('*.npy'))),
            'val_masks': len(list(self.val_mask_dir.glob('*.npy'))),
        }
        
        logger.info("Dataset generation complete")
        logger.info(f"Training set: {stats['train_images']} samples")
        logger.info(f"Validation set: {stats['val_images']} samples")
        logger.info(f"Total: {stats['train_images'] + stats['val_images']} samples")
        
        # Save summary to file
        summary_file = self.output_dir / "dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Cloud Detection Dataset Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Patch size: {self.patch_size} x {self.patch_size}\n")
            f.write(f"Stride: {self.stride}\n")
            f.write(f"Validation split: {self.val_split}\n")
            f.write(f"Skip no_data: {self.skip_nodata}\n")
            if self.skip_nodata:
                f.write(f"No_data threshold: {self.nodata_threshold}\n")
            f.write(f"\nTraining samples: {stats['train_images']}\n")
            f.write(f"Validation samples: {stats['val_images']}\n")
            f.write(f"Total samples: {stats['train_images'] + stats['val_images']}\n")
            f.write("\nClass labels:\n")
            f.write("  0: no_data\n")
            f.write("  1: not_used\n")
            f.write("  2: low_clouds\n")
            f.write("  3: high_clouds\n")
            f.write("  4: cloud_shadows\n")
            f.write("  5: land\n")
            f.write("  6: water\n")
            f.write("  7: snow\n")
        
        logger.info(f"Summary saved to {summary_file}")
        
        return stats
