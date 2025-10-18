# Sentinel-2 + Baetens Hagolle masks Dataset Generator

A scientific tool for generating training datasets from Sentinel-2 imagery and Baetens-Hagolle reference cloud masks for cloud detection research.

## Overview

This project provides a streamlined pipeline for preparing cloud detection datasets from Sentinel-2 satellite imagery. It processes raw Sentinel-2 images and corresponding cloud masks from the Baetens-Hagolle reference dataset, extracting patches suitable for deep learning model training.


## Features

- Automated matching of Sentinel-2 images with reference cloud masks
- Configurable patch extraction with overlapping support
- Automatic train/validation split
- Quality filtering of patches (no_data threshold)
- Batch processing of multiple images
- Comprehensive logging and dataset statistics
- Dataset visualization and analysis tools

## Reference Dataset

**Cloud Mask Reference Dataset:**

Baetens, L., Desjardins, C., & Hagolle, O. (2019). *Validation of Copernicus Sentinel-2 Cloud Masks Obtained from MAJA, Sen2Cor, and FMask Processors Using Reference Cloud Masks Generated with a Supervised Active Learning Procedure*. Remote Sensing, 11(4), 433. https://doi.org/10.3390/rs11040433

**Cloud Reference Mask Dataset Repository:**

Baetens, L., & Hagolle, O. (2018). *Sentinel-2 Reference Cloud Masks generated with a supervised active learning procedure* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1460961

### Class Labels

The reference masks contain 8 classes:
- 0: no_data
- 1: not_used  
- 2: low_clouds
- 3: high_clouds
- 4: cloud_shadows
- 5: land
- 6: water
- 7: snow

## Installation

### Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Overview

The complete workflow consists of three main stages:

1. **Data Acquisition** (Optional): Download Sentinel-2 raw imagery from Google Earth Engine
2. **Dataset Generation**: Process images and masks to create training patches
3. **Visualization**: Inspect and analyze the generated dataset

### Stage 1: Download Sentinel-2 Images (Optional)

If you need to download Sentinel-2 raw imagery corresponding to the reference masks:

```bash
# See download/README.md for instructions
cd download/
python3 download_gee_cloudshell.py
```

**Note**: This step requires Google Earth Engine access and should be run on Google Cloud Shell. 

Downloaded images should be placed in `data/sentinel2_images/`.

### Stage 2: Generate Dataset

Process Sentinel-2 images and reference masks to create training dataset:

```bash
python generate_dataset.py
```

See [Usage](#usage) section below for detailed options.

### Stage 3: Visualize Dataset

Inspect the generated dataset:

```bash
python visualize_dataset.py --mode both --split both
```

## Usage

### Directory Structure

Organize your data as follows:

```
project/
├── data/
│   ├── sentinel2_images/          # Sentinel-2 .tif files (from GEE or provided)
│   │   ├── S2_29RPQ_20170621.tif
│   │   └── ...
│   └── data_Baetens_Hagolle/      # Reference masks dataset
│       └── Reference_dataset/
│           ├── S2A_MSIL1C_20170621T110651.../
│           └── ...
├── download/                       # GEE download scripts (optional)
├── output/                         # Generated dataset output
├── config.yaml                     # Configuration file
└── generate_dataset.py             # Main script
```

### Configuration

Edit `config.yaml` to configure dataset generation parameters:

```yaml
paths:
  sentinel2_images: "data/sentinel2_images"
  reference_masks: "data/data_Baetens_Hagolle/Reference_dataset"
  output: "output/cloud_dataset"

dataset:
  patch_size: 256      # Patch size in pixels
  stride: 128          # Stride for overlapping patches
  val_split: 0.2       # Validation set proportion
  skip_nodata: true    # Filter patches with high no_data
  nodata_threshold: 0.3
```

### Running

Generate dataset using configuration file:
```bash
python generate_dataset.py
```

Override configuration via command line:
```bash
python generate_dataset.py \
    --sentinel2-dir data/sentinel2_images \
    --masks-dir data/data_Baetens_Hagolle/Reference_dataset \
    --output-dir output/cloud_dataset \
    --patch-size 512 \
    --stride 256
```

Enable verbose logging:
```bash
python generate_dataset.py --verbose
```

### Output

Generated dataset structure:
```
output/cloud_dataset/
├── train/
│   ├── images/
│   │   ├── S2_29RPQ_20170621_patch_0000.npy
│   │   └── ...
│   └── masks/
│       ├── S2_29RPQ_20170621_patch_0000.npy
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── dataset_summary.txt
```

Data format:
- Images: NumPy arrays (.npy) with shape (C, H, W) where C = number of bands
- Masks: NumPy arrays (.npy) with shape (H, W) containing class labels [0-7]

## Image Naming Convention

Sentinel-2 images must follow this naming pattern:
```
S2[A/B]_TILE_DATE.tif
```

Example: `S2_29RPQ_20170621.tif`
- Tile: 29RPQ
- Date: 20170621 (YYYYMMDD)

The script automatically matches images to masks based on tile ID and acquisition date.

## Dataset Visualization

Visualize generated datasets and analyze class distributions:

```bash
# View random samples
python visualize_dataset.py --split train --n-samples 5

# Analyze class distribution
python visualize_dataset.py --mode distribution --split both

# Save all visualizations
python visualize_dataset.py --mode both --split both --output-dir visualizations
```

Options:
- `--dataset-dir`: Path to dataset (default: `output/cloud_dataset`)
- `--split`: Dataset split to visualize (`train`, `val`, or `both`)
- `--mode`: Visualization mode (`samples`, `distribution`, or `both`)
- `--n-samples`: Number of random samples to display
- `--output-dir`: Save figures to directory instead of displaying

## Configuration Parameters

### Dataset Parameters

- `patch_size`: Size of extracted patches in pixels (e.g., 256, 512)
- `stride`: Step size for patch extraction. Values less than patch_size create overlapping patches
- `val_split`: Proportion of data reserved for validation (0.0 - 1.0)
- `skip_nodata`: Whether to discard patches with excessive no_data pixels
- `nodata_threshold`: Maximum allowed proportion of no_data pixels (0.0 - 1.0)

### Processing Parameters

- `random_seed`: Random seed for reproducible train/val splits

## Project Structure

```
.
├── src/                           # Source code
│   ├── dataset_generator.py       # Main generator class
│   └── __init__.py
├── data/                          # Data directory
│   ├── sentinel2_images/          # Sentinel-2 raw images
│   └── data_Baetens_Hagolle/      # Reference masks
├── download/                      # GEE download scripts
│   ├── download_gee_cloudshell.py
│   ├── products_to_download.json
│   └── README.md
├── docs/                          # Documentation
│   ├── DOWNLOAD_GUIDE.md          # Download guide
│   └── TECHNICAL.md               # Technical specs
├── output/                        # Generated datasets
├── config.yaml                    # Configuration
├── generate_dataset.py            # CLI tool
├── visualize_dataset.py           # Visualization tool
├── example.py                     # Example usage
└── requirements.txt               # Dependencies
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open an issue on the project repository.
