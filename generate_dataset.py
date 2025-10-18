"""
Command-line interface for cloud detection dataset generation.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.dataset_generator import CloudDatasetGenerator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate cloud detection training dataset from '
                    'Sentinel-2 imagery and Baetens-Hagolle reference masks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--sentinel2-dir',
        type=str,
        help='Directory containing Sentinel-2 images (overrides config)'
    )
    
    parser.add_argument(
        '--masks-dir',
        type=str,
        help='Directory containing reference masks (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--patch-size',
        type=int,
        help='Patch size in pixels (overrides config)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        help='Stride for patch extraction (overrides config)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        help='Validation split ratio (overrides config)'
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
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.sentinel2_dir:
        config['paths']['sentinel2_images'] = args.sentinel2_dir
    if args.masks_dir:
        config['paths']['reference_masks'] = args.masks_dir
    if args.output_dir:
        config['paths']['output'] = args.output_dir
    if args.patch_size:
        config['dataset']['patch_size'] = args.patch_size
    if args.stride:
        config['dataset']['stride'] = args.stride
    if args.val_split:
        config['dataset']['val_split'] = args.val_split
    
    # Validate paths
    sentinel2_dir = Path(config['paths']['sentinel2_images'])
    masks_dir = Path(config['paths']['reference_masks'])
    output_dir = Path(config['paths']['output'])
    
    if not sentinel2_dir.exists():
        logger.error(f"Sentinel-2 directory not found: {sentinel2_dir}")
        sys.exit(1)
    
    if not masks_dir.exists():
        logger.error(f"Reference masks directory not found: {masks_dir}")
        sys.exit(1)
    
    # Create dataset generator
    generator = CloudDatasetGenerator(
        output_dir=str(output_dir),
        patch_size=config['dataset']['patch_size'],
        stride=config['dataset']['stride'],
        val_split=config['dataset']['val_split'],
        skip_nodata=config['dataset']['skip_nodata'],
        nodata_threshold=config['dataset']['nodata_threshold'],
        random_seed=config['processing']['random_seed']
    )
    
    # Process all images
    logger.info("Starting dataset generation")
    logger.info(f"Input directory: {sentinel2_dir}")
    logger.info(f"Reference masks: {masks_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    stats = generator.process_directory(sentinel2_dir, masks_dir)
    
    # Generate summary
    generator.generate_summary()
    
    # Report results
    logger.info("Processing complete")
    logger.info(f"Total images: {stats['total']}")
    logger.info(f"Successfully processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        logger.warning(
            f"{stats['failed']} image(s) failed to process. "
            "Check logs for details."
        )
    
    return 0 if stats['processed'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
