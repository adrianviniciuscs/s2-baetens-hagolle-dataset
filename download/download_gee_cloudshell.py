#!/usr/bin/env python3
"""
Sentinel-2 Image Download via Google Earth Engine

This script downloads Sentinel-2 imagery from Google Earth Engine for tiles
and dates corresponding to the Baetens-Hagolle reference cloud mask dataset.

Requirements:
    - Google Earth Engine Python API
    - Authenticated GEE account
    - products_to_download.json in same directory

Usage:
    python3 download_gee_cloudshell.py [--limit N]

Arguments:
    --limit N    Process only first N products (default: all)

Output:
    Images exported to Google Drive in 'Sentinel2_Clouds' folder
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ee


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Sentinel2Downloader:
    """
    Download Sentinel-2 imagery from Google Earth Engine.
    
    This class handles the querying and export of Sentinel-2 L1C imagery
    to Google Drive based on tile ID and acquisition date.
    """
    
    def __init__(self, output_folder: str = 'Sentinel2_Clouds'):
        """
        Initialize the downloader.
        
        Args:
            output_folder: Google Drive folder name for exports
        """
        self.output_folder = output_folder
        self.tasks = []
        
        # Initialize Earth Engine
        logger.info("Initializing Google Earth Engine...")
        try:
            ee.Initialize()
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
    
    def load_products(self, json_path: str) -> List[Dict]:
        """
        Load product list from JSON file.
        
        Args:
            json_path: Path to products JSON file
        
        Returns:
            List of product dictionaries with 'tile' and 'date' keys
        """
        try:
            with open(json_path, 'r') as f:
                products = json.load(f)
            logger.info(f"Loaded {len(products)} products from {json_path}")
            return products
        except FileNotFoundError:
            logger.error(f"Product list not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise
    
    def query_image(
        self,
        tile: str,
        date: str,
        tolerance_days: int = 1
    ) -> Optional[ee.Image]:
        """
        Query Sentinel-2 image for given tile and date.
        
        Args:
            tile: MGRS tile identifier (e.g., '29RPQ')
            date: Date in YYYYMMDD format
            tolerance_days: Days tolerance for date matching
        
        Returns:
            Earth Engine Image object or None if not found
        """
        # Format date
        try:
            date_obj = datetime.strptime(date, '%Y%m%d')
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {date}")
            return None
        
        # Remove 'T' prefix if present
        tile_clean = tile[1:] if tile.startswith('T') else tile
        
        # Query collection
        collection = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(date_str, date_str) \
            .filter(ee.Filter.eq('MGRS_TILE', tile_clean))
        
        count = collection.size().getInfo()
        
        # Try with date range if exact date not found
        if count == 0 and tolerance_days > 0:
            logger.debug(f"Exact date not found. Trying ±{tolerance_days} day range...")
            date_before = (date_obj - timedelta(days=tolerance_days)).strftime('%Y-%m-%d')
            date_after = (date_obj + timedelta(days=tolerance_days)).strftime('%Y-%m-%d')
            
            collection = ee.ImageCollection('COPERNICUS/S2') \
                .filterDate(date_before, date_after) \
                .filter(ee.Filter.eq('MGRS_TILE', tile_clean))
            
            count = collection.size().getInfo()
        
        if count == 0:
            return None
        
        return collection.first()
    
    def export_image(
        self,
        image: ee.Image,
        tile: str,
        date: str,
        bands: List[str] = None,
        scale: int = 10
    ) -> Optional[str]:
        """
        Export image to Google Drive.
        
        Args:
            image: Earth Engine Image to export
            tile: MGRS tile identifier
            date: Date in YYYYMMDD format
            bands: List of band names to export (default: B2,B3,B4,B8)
            scale: Export resolution in meters
        
        Returns:
            Task description or None if export failed
        """
        if bands is None:
            bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
        
        # Select bands
        image = image.select(bands)
        
        # Get geometry
        geometry = image.geometry()
        
        # Generate output name
        output_name = f"S2_{tile}_{date}"
        
        # Create export task
        try:
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=output_name,
                folder=self.output_folder,
                scale=scale,
                region=geometry,
                maxPixels=1e13,
                fileFormat='GeoTIFF'
            )
            
            task.start()
            self.tasks.append({
                'name': output_name,
                'task': task,
                'tile': tile,
                'date': date
            })
            
            return output_name
        except Exception as e:
            logger.error(f"Failed to export {output_name}: {e}")
            return None
    
    def process_products(
        self,
        products: List[Dict],
        limit: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process list of products and export images.
        
        Args:
            products: List of product dictionaries
            limit: Maximum number of products to process
        
        Returns:
            Dictionary with processing statistics
        """
        if limit is not None:
            products = products[:limit]
            logger.info(f"Processing limited to {limit} products")
        
        stats = {
            'total': len(products),
            'successful': 0,
            'not_found': 0,
            'failed': 0
        }
        
        logger.info(f"Starting export of {len(products)} images")
        logger.info("-" * 70)
        
        for i, product in enumerate(products, 1):
            tile = product.get('tile', '')
            date = product.get('date', '')
            
            if not tile or not date:
                logger.warning(f"[{i}/{len(products)}] Invalid product: missing tile or date")
                stats['failed'] += 1
                continue
            
            logger.info(f"[{i}/{len(products)}] Processing tile {tile}, date {date}")
            
            # Query image
            image = self.query_image(tile, date)
            
            if image is None:
                logger.warning(f"  Image not found in GEE catalog")
                stats['not_found'] += 1
                continue
            
            # Export image
            output_name = self.export_image(image, tile, date)
            
            if output_name:
                logger.info(f"  Export task started: {output_name}")
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def print_summary(self, stats: Dict[str, int]):
        """Print processing summary."""
        logger.info("-" * 70)
        logger.info("EXPORT SUMMARY")
        logger.info("-" * 70)
        logger.info(f"Total products: {stats['total']}")
        logger.info(f"Successfully exported: {stats['successful']}")
        logger.info(f"Not found in catalog: {stats['not_found']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Monitor export progress at: https://code.earthengine.google.com/tasks")
        logger.info(f"  2. Images will be saved to Google Drive folder: '{self.output_folder}'")
        logger.info(f"  3. Download images from Google Drive to local directory")
        logger.info(f"  4. Place images in data/sentinel2_images/ directory")
        logger.info("-" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 imagery from Google Earth Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of products to process (default: all)'
    )
    
    parser.add_argument(
        '--products',
        type=str,
        default='products_to_download.json',
        help='Path to products JSON file'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default='Sentinel2_Clouds',
        help='Google Drive folder name for exports'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize downloader
    try:
        downloader = Sentinel2Downloader(output_folder=args.output_folder)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Load products
    try:
        products = downloader.load_products(args.products)
    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        sys.exit(1)
    
    # Process products
    stats = downloader.process_products(products, limit=args.limit)
    
    # Print summary
    downloader.print_summary(stats)
    
    return 0 if stats['successful'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

