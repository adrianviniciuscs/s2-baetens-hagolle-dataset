# Quick Start Guide

## Prerequisites

- Python 3.8 or higher

## Installation

1. Clone or download this repository

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Option 1: Using Existing Images

If you already have Sentinel-2 images:

1. Place images in `data/sentinel2_images/`:
```bash
data/sentinel2_images/
├── S2_29RPQ_20170621.tif
├── S2_31TFJ_20170917.tif
└── ...
```

2. Ensure Baetens-Hagolle reference dataset is in `data/data_Baetens_Hagolle/`

### Option 2: Download from Google Earth Engine

If you need to download Sentinel-2 raw imagery:

1. See detailed instructions in [DOWNLOAD_GUIDE.md](docs/DOWNLOAD_GUIDE.md)

2. Quick steps:
```bash
cd download/
# Upload to Google Cloud Shell and run:
python3 download_gee_cloudshell.py
# Download from Google Drive to data/sentinel2_images/
```

**Note**: Requires Google Earth Engine access and Google Cloud Shell.

## Basic Usage

### Using Default Configuration

```bash
python generate_dataset.py
```

This will:
- Process all images in `data/sentinel2_images/`
- Match them with reference masks
- Generate dataset in `output/cloud_dataset/`

### Custom Configuration

Edit `config.yaml`:
```yaml
dataset:
  patch_size: 512    # Change patch size
  stride: 256        # Change stride
  val_split: 0.15    # 15% validation
```

### Command Line Options

```bash
# Specify custom directories
python generate_dataset.py \
    --sentinel2-dir /path/to/images \
    --masks-dir /path/to/masks \
    --output-dir /path/to/output

# Override patch parameters
python generate_dataset.py \
    --patch-size 512 \
    --stride 256

# Enable verbose logging
python generate_dataset.py --verbose
```

## Programmatic Usage

```python
from src.dataset_generator import CloudDatasetGenerator
from pathlib import Path

# Initialize generator
generator = CloudDatasetGenerator(
    output_dir="output/my_dataset",
    patch_size=256,
    stride=128,
    val_split=0.2
)

# Process images
stats = generator.process_directory(
    image_dir=Path("data/sentinel2_images"),
    mask_base_dir=Path("data/data_Baetens_Hagolle/Reference_dataset")
)

# Generate summary
generator.generate_summary()
```

## Output Structure

```
output/cloud_dataset/
├── train/
│   ├── images/        # Training images (.npy)
│   └── masks/         # Training masks (.npy)
├── val/
│   ├── images/        # Validation images (.npy)
│   └── masks/         # Validation masks (.npy)
└── dataset_summary.txt
```

## Loading Data

```python
import numpy as np

# Load a sample
image = np.load("output/cloud_dataset/train/images/S2_29RPQ_20170621_patch_0000.npy")
mask = np.load("output/cloud_dataset/train/masks/S2_29RPQ_20170621_patch_0000.npy")

print(f"Image shape: {image.shape}")  # (13, 256, 256) - 13 bands
print(f"Mask shape: {mask.shape}")    # (256, 256)
print(f"Classes in mask: {np.unique(mask)}")
```

## Visualizing Dataset

### View Random Samples

```bash
# Visualize 5 random samples from training set
python visualize_dataset.py --split train --n-samples 5

# Save visualizations to file instead of displaying
python visualize_dataset.py --split train --n-samples 5 --output-dir visualizations
```

### Analyze Class Distribution

```bash
# Analyze class distribution in training set
python visualize_dataset.py --mode distribution --split train

# Analyze both train and validation sets
python visualize_dataset.py --mode distribution --split both

# Save distribution plots
python visualize_dataset.py --mode distribution --split both --output-dir visualizations
```

### Combined Visualization

```bash
# View samples and distribution for both splits
python visualize_dataset.py --mode both --split both --output-dir visualizations
```

## Troubleshooting

**Problem:** No images found
- **Solution:** Check that .tif/.tiff files are in `data/sentinel2_images/`

**Problem:** No matching mask found
- **Solution:** Verify image filename follows format `S2_TILE_DATE.tif`
- **Solution:** Check that corresponding mask exists in reference dataset

**Problem:** ImportError: No module named 'rasterio'
- **Solution:** Install dependencies: `pip install -r requirements.txt`

## Complete Pipeline

1. **Download Images** (if needed): Use `download/download_gee_cloudshell.py` on Google Cloud Shell
2. **Generate Dataset**: Run `python generate_dataset.py`
3. **Visualize**: Run `python visualize_dataset.py --mode both --split both`
4. **Use for Training**: Load patches from `output/cloud_dataset/`

## Support

For issues or questions, please refer to the main documentation or open an issue on the project repository.
