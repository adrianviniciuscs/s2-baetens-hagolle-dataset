"""
Dataset visualization tool for cloud detection training data.

This script provides visualization capabilities for the generated dataset,
including sample inspection and class distribution analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# Class definitions
CLASS_LABELS = {
    0: 'no_data',
    1: 'not_used',
    2: 'low_clouds',
    3: 'high_clouds',
    4: 'cloud_shadows',
    5: 'land',
    6: 'water',
    7: 'snow'
}

CLASS_COLORS = {
    0: '#000000',  # black
    1: '#808080',  # gray
    2: '#B8D4FF',  # light blue
    3: '#FFFFFF',  # white
    4: '#404040',  # dark gray
    5: '#228B22',  # forest green
    6: '#0000FF',  # blue
    7: '#FF00FF'   # magenta
}


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def normalize_bands(
    image: np.ndarray,
    band_indices: Tuple[int, int, int] = (3, 2, 1),
    percentile: int = 2
) -> np.ndarray:
    """
    Normalize multi-band image for RGB visualization.
    
    Args:
        image: Array with shape (C, H, W)
        band_indices: Indices for RGB visualization (default: bands 4,3,2)
        percentile: Percentile for contrast stretching
    
    Returns:
        Normalized RGB image (H, W, 3) in range [0, 1]
    """
    # Select bands and transpose to (H, W, C)
    if image.shape[0] > max(band_indices):
        rgb = image[band_indices, :, :]
    else:
        # Fallback to first 3 bands if indices not available
        rgb = image[:3, :, :]
    
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Normalize each band
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[:, :, i].astype(np.float32)
        p_low = np.percentile(band, percentile)
        p_high = np.percentile(band, 100 - percentile)
        band_clipped = np.clip(band, p_low, p_high)
        rgb_norm[:, :, i] = (band_clipped - p_low) / (p_high - p_low + 1e-8)
    
    return rgb_norm


def create_colored_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert class mask to RGB colored visualization.
    
    Args:
        mask: Class mask array (H, W)
    
    Returns:
        RGB image (H, W, 3) in range [0, 1]
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)
    
    for class_id, color_hex in CLASS_COLORS.items():
        # Convert hex to RGB
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        colored[mask == class_id] = [r, g, b]
    
    return colored


def visualize_sample(
    image_path: Path,
    mask_path: Path,
    output_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Visualize a single image-mask pair.
    
    Args:
        image_path: Path to image (.npy)
        mask_path: Path to corresponding mask (.npy)
        output_path: Optional path to save figure
        dpi: Output resolution
    """
    # Load data
    image = np.load(image_path)
    mask = np.load(mask_path)
    
    # Normalize image
    rgb = normalize_bands(image)
    
    # Color mask
    mask_colored = create_colored_mask(mask)
    
    # Compute class statistics
    unique_classes, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title('Sentinel-2 RGB', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_colored)
    ax2.set_title('Cloud Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = rgb * 0.6 + mask_colored * 0.4
    ax3.imshow(np.clip(overlay, 0, 1))
    ax3.set_title('Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Legend
    legend_elements = []
    for class_id in sorted(unique_classes):
        count = counts[list(unique_classes).index(class_id)]
        percentage = (count / total_pixels) * 100
        label = f'{CLASS_LABELS[class_id]:15s}: {percentage:5.2f}%'
        legend_elements.append(
            mpatches.Patch(
                facecolor=CLASS_COLORS[class_id],
                edgecolor='black',
                label=label
            )
        )
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        fontsize=9,
        frameon=True,
        title='Class Distribution'
    )
    
    # Title
    plt.suptitle(
        f'Sample: {image_path.stem}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logging.info(f"Figure saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_class_distribution(
    dataset_dir: Path,
    split: str = 'train'
) -> dict:
    """
    Analyze class distribution in dataset split.
    
    Args:
        dataset_dir: Dataset root directory
        split: 'train' or 'val'
    
    Returns:
        Dictionary with class statistics
    """
    mask_dir = dataset_dir / split / 'masks'
    
    if not mask_dir.exists():
        logging.error(f"Mask directory not found: {mask_dir}")
        return {}
    
    mask_files = list(mask_dir.glob('*.npy'))
    
    if len(mask_files) == 0:
        logging.warning(f"No mask files found in {mask_dir}")
        return {}
    
    logging.info(f"Analyzing {len(mask_files)} masks in {split} set...")
    
    # Accumulate class counts
    class_counts = {i: 0 for i in range(8)}
    
    for mask_file in mask_files:
        mask = np.load(mask_file)
        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_counts[int(class_id)] += count
    
    # Compute statistics
    total_pixels = sum(class_counts.values())
    
    stats = {
        'split': split,
        'n_samples': len(mask_files),
        'total_pixels': total_pixels,
        'class_counts': class_counts,
        'class_percentages': {
            class_id: (count / total_pixels * 100) if total_pixels > 0 else 0
            for class_id, count in class_counts.items()
        }
    }
    
    return stats


def plot_class_distribution(
    stats: dict,
    output_path: Optional[Path] = None,
    dpi: int = 150
):
    """
    Plot class distribution histogram.
    
    Args:
        stats: Statistics dictionary from analyze_class_distribution
        output_path: Optional path to save figure
        dpi: Output resolution
    """
    if not stats:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(range(8))
    percentages = [stats['class_percentages'][i] for i in classes]
    colors = [CLASS_COLORS[i] for i in classes]
    labels = [CLASS_LABELS[i] for i in classes]
    
    bars = ax.bar(classes, percentages, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{pct:.2f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Class Distribution - {stats["split"].upper()} Set\n'
        f'({stats["n_samples"]} samples, {stats["total_pixels"]:,} pixels)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(classes)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logging.info(f"Distribution plot saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_random_samples(
    dataset_dir: Path,
    split: str = 'train',
    n_samples: int = 5,
    output_dir: Optional[Path] = None,
    dpi: int = 150
):
    """
    Visualize random samples from dataset.
    
    Args:
        dataset_dir: Dataset root directory
        split: 'train' or 'val'
        n_samples: Number of samples to visualize
        output_dir: Optional directory to save figures
        dpi: Output resolution
    """
    img_dir = dataset_dir / split / 'images'
    mask_dir = dataset_dir / split / 'masks'
    
    if not img_dir.exists():
        logging.error(f"Image directory not found: {img_dir}")
        return
    
    image_files = sorted(list(img_dir.glob('*.npy')))
    
    if len(image_files) == 0:
        logging.error(f"No images found in {img_dir}")
        return
    
    # Random selection
    np.random.seed(42)
    selected = np.random.choice(
        image_files,
        size=min(n_samples, len(image_files)),
        replace=False
    )
    
    logging.info(f"Visualizing {len(selected)} random samples from {split} set...")
    
    for idx, img_path in enumerate(selected, 1):
        mask_path = mask_dir / img_path.name
        
        if not mask_path.exists():
            logging.warning(f"Mask not found: {mask_path.name}")
            continue
        
        logging.info(f"[{idx}/{len(selected)}] {img_path.name}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{split}_sample_{idx:02d}.png"
        else:
            output_path = None
        
        visualize_sample(img_path, mask_path, output_path, dpi)


def main():
    """Main entry point for visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize cloud detection dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='output/cloud_dataset',
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'both'],
        help='Dataset split to visualize'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='samples',
        choices=['samples', 'distribution', 'both'],
        help='Visualization mode'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of random samples to visualize'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output resolution (DPI)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Determine splits to process
    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Splits: {', '.join(splits)}")
    
    # Process each split
    for split in splits:
        logger.info(f"\nProcessing {split} set...")
        
        # Sample visualization
        if args.mode in ['samples', 'both']:
            split_output_dir = output_dir / split if output_dir else None
            visualize_random_samples(
                dataset_dir,
                split,
                args.n_samples,
                split_output_dir,
                args.dpi
            )
        
        # Distribution analysis
        if args.mode in ['distribution', 'both']:
            stats = analyze_class_distribution(dataset_dir, split)
            
            if stats:
                # Print statistics
                logger.info(f"\nClass distribution for {split} set:")
                logger.info(f"  Samples: {stats['n_samples']}")
                logger.info(f"  Total pixels: {stats['total_pixels']:,}")
                logger.info("  Class percentages:")
                for class_id in range(8):
                    pct = stats['class_percentages'][class_id]
                    if pct > 0:
                        logger.info(
                            f"    {class_id} ({CLASS_LABELS[class_id]:15s}): "
                            f"{pct:6.2f}%"
                        )
                
                # Plot distribution
                if output_dir:
                    output_path = output_dir / f"{split}_distribution.png"
                else:
                    output_path = None
                
                plot_class_distribution(stats, output_path, args.dpi)
    
    logger.info("\nVisualization complete")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
